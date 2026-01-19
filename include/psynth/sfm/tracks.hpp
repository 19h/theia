#pragma once

#include <psynth/types.hpp>
#include <unordered_map>
#include <vector>

namespace psynth::sfm {

class Tracks {
 public:
  Tracks() = default;
  Tracks(std::vector<Track> tracks, int num_images);

  int num_images() const noexcept {
    return num_images_;
  }

  const std::vector<Track>& all() const noexcept {
    return tracks_;
  }
  std::vector<Track>& all_mut() noexcept {
    return tracks_;
  }

  const std::vector<int>& tracks_in_image(ImageId image_id) const;

  void RebuildIndex();

  // O(1) observation lookup using per-track hash index
  const Observation* FindObservation(int track_id, ImageId image_id) const;
  Observation* FindObservationMutable(int track_id, ImageId image_id);

  // Rebuild the per-track observation index (called automatically by RebuildIndex)
  void RebuildObservationIndex();

 private:
  int num_images_ = 0;
  std::vector<Track> tracks_;
  std::vector<std::vector<int>> tracks_per_image_;

  // Per-track observation index: track_id -> (image_id -> observation_index)
  // This converts FindObservation from O(observations_per_track) to O(1)
  std::vector<std::unordered_map<ImageId, size_t>> obs_index_;
};

Tracks BuildTracksUnionFind(const std::vector<FeatureSet>& features, int num_images,
                            const std::vector<VerifiedPair>& verified_pairs);

}  // namespace psynth::sfm