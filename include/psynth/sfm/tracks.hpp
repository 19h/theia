#pragma once

#include <psynth/types.hpp>

#include <vector>

namespace psynth::sfm {

class Tracks {
 public:
  Tracks() = default;
  Tracks(std::vector<Track> tracks, int num_images);

  int num_images() const { return num_images_; }

  const std::vector<Track>& all() const { return tracks_; }
  std::vector<Track>& all_mut() { return tracks_; }

  const std::vector<int>& tracks_in_image(ImageId image_id) const;

  void RebuildIndex();

  const Observation* FindObservation(int track_id, ImageId image_id) const;
  Observation* FindObservationMutable(int track_id, ImageId image_id);

 private:
  int num_images_ = 0;
  std::vector<Track> tracks_;
  std::vector<std::vector<int>> tracks_per_image_;
};

Tracks BuildTracksUnionFind(const std::vector<FeatureSet>& features,
                            int num_images,
                            const std::vector<VerifiedPair>& verified_pairs);

}  // namespace psynth::sfm