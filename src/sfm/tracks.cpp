#include <psynth/sfm/tracks.hpp>
#include <psynth/common.hpp>
#include <psynth/sfm/union_find.hpp>

#include <algorithm>
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
}

const Observation* Tracks::FindObservation(int track_id, ImageId image_id) const {
  if (track_id < 0 || track_id >= static_cast<int>(tracks_.size())) return nullptr;
  for (const auto& obs : tracks_[track_id].observations) {
    if (obs.image_id == image_id) return &obs;
  }
  return nullptr;
}

Observation* Tracks::FindObservationMutable(int track_id, ImageId image_id) {
  if (track_id < 0 || track_id >= static_cast<int>(tracks_.size())) return nullptr;
  for (auto& obs : tracks_[track_id].observations) {
    if (obs.image_id == image_id) return &obs;
  }
  return nullptr;
}

Tracks BuildTracksUnionFind(const std::vector<FeatureSet>& features,
                            int num_images,
                            const std::vector<VerifiedPair>& verified_pairs) {
  PSYNTH_REQUIRE(static_cast<int>(features.size()) == num_images, "BuildTracksUnionFind: features size mismatch");

  std::vector<std::size_t> offsets(num_images + 1, 0);
  for (int i = 0; i < num_images; ++i) {
    offsets[i + 1] = offsets[i] + static_cast<std::size_t>(features[i].keypoints.size());
  }
  const std::size_t total_kp = offsets[num_images];

  UnionFind uf(total_kp);

  std::vector<std::size_t> touched;
  touched.reserve(1024);

  for (const auto& vp : verified_pairs) {
    const int i = vp.pair.i;
    const int j = vp.pair.j;
    if (i < 0 || i >= num_images || j < 0 || j >= num_images) continue;

    const std::size_t oi = offsets[i];
    const std::size_t oj = offsets[j];

    for (const auto& m : vp.inliers) {
      const int a = m.kp1;
      const int b = m.kp2;
      if (a < 0 || a >= static_cast<int>(features[i].keypoints.size())) continue;
      if (b < 0 || b >= static_cast<int>(features[j].keypoints.size())) continue;

      const std::size_t ga = oi + static_cast<std::size_t>(a);
      const std::size_t gb = oj + static_cast<std::size_t>(b);

      uf.Unite(ga, gb);
      touched.push_back(ga);
      touched.push_back(gb);
    }
  }

  if (touched.empty()) {
    return Tracks({}, num_images);
  }

  std::sort(touched.begin(), touched.end());
  touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

  std::unordered_map<std::size_t, std::vector<std::size_t>> comps;
  comps.reserve(touched.size());
  for (const std::size_t gid : touched) {
    const std::size_t r = uf.Find(gid);
    comps[r].push_back(gid);
  }

  std::vector<Track> tracks;
  tracks.reserve(comps.size());

  auto DecodeGlobal = [&](std::size_t gid, ImageId* img_id, int* kp_id) {
    auto it = std::upper_bound(offsets.begin(), offsets.end(), gid);
    const int idx = static_cast<int>(std::distance(offsets.begin(), it)) - 1;
    *img_id = idx;
    *kp_id = static_cast<int>(gid - offsets[idx]);
  };

  for (auto& kv : comps) {
    const auto& nodes = kv.second;
    if (nodes.size() < 2) continue;

    std::unordered_set<ImageId> seen;
    seen.reserve(nodes.size());

    std::vector<Observation> obs;
    obs.reserve(nodes.size());

    bool dup = false;
    for (std::size_t gid : nodes) {
      ImageId img_id = -1;
      int kp_id = -1;
      DecodeGlobal(gid, &img_id, &kp_id);

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
    t.observations = std::move(obs);
    tracks.push_back(std::move(t));
  }

  return Tracks(std::move(tracks), num_images);
}

}  // namespace psynth::sfm