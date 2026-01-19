#include <algorithm>
#include <psynth/common.hpp>
#include <psynth/sfm/tracks.hpp>
#include <psynth/sfm/union_find.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace psynth::sfm {

Tracks::Tracks(std::vector<Track> tracks, int num_images)
    : num_images_(num_images), tracks_(std::move(tracks)) {
  RebuildIndex();
}

const std::vector<int>& Tracks::tracks_in_image(ImageId image_id) const {
  PSYNTH_REQUIRE(image_id >= 0 && image_id < num_images_, "Tracks: invalid image id");
  return tracks_per_image_[image_id];
}

void Tracks::RebuildIndex() {
  tracks_per_image_.assign(num_images_, {});
  for (const auto& t : tracks_) {
    for (const auto& obs : t.observations) {
      if (obs.image_id < 0 || obs.image_id >= num_images_) continue;
      tracks_per_image_[obs.image_id].push_back(t.id);
    }
  }
  RebuildObservationIndex();
}

void Tracks::RebuildObservationIndex() {
  obs_index_.clear();
  obs_index_.resize(tracks_.size());

  for (size_t tid = 0; tid < tracks_.size(); ++tid) {
    const auto& track = tracks_[tid];
    auto& idx = obs_index_[tid];
    idx.reserve(track.observations.size());

    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      idx[track.observations[oi].image_id] = oi;
    }
  }
}

// O(1) observation lookup using hash index
const Observation* Tracks::FindObservation(int track_id, ImageId image_id) const {
  if (track_id < 0 || track_id >= static_cast<int>(tracks_.size())) return nullptr;

  // Use hash index if available
  if (track_id < static_cast<int>(obs_index_.size())) {
    const auto& idx = obs_index_[track_id];
    auto it = idx.find(image_id);
    if (it != idx.end()) {
      return &tracks_[track_id].observations[it->second];
    }
    return nullptr;
  }

  // Fallback to linear search if index not built
  for (const auto& obs : tracks_[track_id].observations) {
    if (obs.image_id == image_id) return &obs;
  }
  return nullptr;
}

Observation* Tracks::FindObservationMutable(int track_id, ImageId image_id) {
  if (track_id < 0 || track_id >= static_cast<int>(tracks_.size())) return nullptr;

  // Use hash index if available
  if (track_id < static_cast<int>(obs_index_.size())) {
    const auto& idx = obs_index_[track_id];
    auto it = idx.find(image_id);
    if (it != idx.end()) {
      return &tracks_[track_id].observations[it->second];
    }
    return nullptr;
  }

  // Fallback to linear search
  for (auto& obs : tracks_[track_id].observations) {
    if (obs.image_id == image_id) return &obs;
  }
  return nullptr;
}

Tracks BuildTracksUnionFind(const std::vector<FeatureSet>& features, int num_images,
                            const std::vector<VerifiedPair>& verified_pairs) {
  PSYNTH_REQUIRE(static_cast<int>(features.size()) == num_images,
                 "BuildTracksUnionFind: features size mismatch");

  // Compute global keypoint offsets
  std::vector<std::size_t> offsets(num_images + 1, 0);
  for (int i = 0; i < num_images; ++i) {
    offsets[i + 1] = offsets[i] + static_cast<std::size_t>(features[i].keypoints.size());
  }
  const std::size_t total_kp = offsets[num_images];

  UnionFind uf(total_kp);

  // Pre-compute gid to image mapping for O(1) lookup instead of binary search
  // This uses O(total_kp) memory but eliminates O(log(num_images)) per lookup
  std::vector<ImageId> gid_to_image(total_kp);
  std::vector<int> gid_to_kp(total_kp);
  for (int img = 0; img < num_images; ++img) {
    const std::size_t start = offsets[img];
    const std::size_t end = offsets[img + 1];
    for (std::size_t gid = start; gid < end; ++gid) {
      gid_to_image[gid] = img;
      gid_to_kp[gid] = static_cast<int>(gid - start);
    }
  }

  // Use unordered_set directly instead of vector + sort + unique
  // This avoids O(n log n) deduplication overhead
  std::unordered_set<std::size_t> touched_set;

  // Estimate touched count: typically 2 * total_inliers
  size_t total_inliers = 0;
  for (const auto& vp : verified_pairs) {
    total_inliers += vp.inliers.size();
  }
  touched_set.reserve(total_inliers * 2);

  for (const auto& vp : verified_pairs) {
    const int i = vp.pair.i;
    const int j = vp.pair.j;
    if (i < 0 || i >= num_images || j < 0 || j >= num_images) continue;

    const std::size_t oi = offsets[i];
    const std::size_t oj = offsets[j];
    const int ni = static_cast<int>(features[i].keypoints.size());
    const int nj = static_cast<int>(features[j].keypoints.size());

    for (const auto& m : vp.inliers) {
      const int a = m.kp1;
      const int b = m.kp2;
      if (a < 0 || a >= ni) continue;
      if (b < 0 || b >= nj) continue;

      const std::size_t ga = oi + static_cast<std::size_t>(a);
      const std::size_t gb = oj + static_cast<std::size_t>(b);

      uf.Unite(ga, gb);
      touched_set.insert(ga);
      touched_set.insert(gb);
    }
  }

  if (touched_set.empty()) {
    return Tracks({}, num_images);
  }

  // Convert to vector for iteration
  std::vector<std::size_t> touched(touched_set.begin(), touched_set.end());

  // Group keypoints by their root (component)
  // Pre-compute roots and count component sizes to avoid reallocation
  std::unordered_map<std::size_t, std::size_t> comp_sizes;
  comp_sizes.reserve(touched.size() / 2);  // Estimate: avg 2 keypoints per track

  std::vector<std::size_t> roots(touched.size());
  for (size_t i = 0; i < touched.size(); ++i) {
    roots[i] = uf.Find(touched[i]);
    comp_sizes[roots[i]]++;
  }

  // Build components with pre-reserved vectors
  std::unordered_map<std::size_t, std::vector<std::size_t>> comps;
  comps.reserve(comp_sizes.size());
  for (const auto& kv : comp_sizes) {
    comps[kv.first].reserve(kv.second);
  }
  for (size_t i = 0; i < touched.size(); ++i) {
    comps[roots[i]].push_back(touched[i]);
  }

  std::vector<Track> tracks;
  tracks.reserve(comps.size());

  // Reusable buffers for the inner loop
  std::unordered_set<ImageId> seen;
  std::vector<Observation> obs;

  for (auto& kv : comps) {
    const auto& nodes = kv.second;
    if (nodes.size() < 2) continue;

    seen.clear();
    seen.reserve(nodes.size());

    obs.clear();
    obs.reserve(nodes.size());

    bool dup = false;
    for (std::size_t gid : nodes) {
      const ImageId img_id = gid_to_image[gid];
      const int kp_id = gid_to_kp[gid];

      if (!seen.insert(img_id).second) {
        dup = true;
        break;
      }

      const auto& kp = features[img_id].keypoints[kp_id];

      Observation o;
      o.image_id = img_id;
      o.keypoint_id = kp_id;
      o.u_px = static_cast<double>(kp.pt.x);
      o.v_px = static_cast<double>(kp.pt.y);
      obs.push_back(o);
    }

    if (dup) continue;
    if (obs.size() < 2) continue;

    Track t;
    t.id = static_cast<int>(tracks.size());
    t.observations = obs;  // Copy instead of move since we reuse obs buffer
    tracks.push_back(std::move(t));
  }

  return Tracks(std::move(tracks), num_images);
}

}  // namespace psynth::sfm