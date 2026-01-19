#pragma once

#include <psynth/sfm/tracks.hpp>
#include <psynth/types.hpp>

#include <unordered_map>

namespace psynth::sfm {

struct Reconstruction {
  std::unordered_map<ImageId, Camera> cameras;
  Tracks tracks;

  ImageId gauge_fixed_0 = -1;
  ImageId gauge_fixed_1 = -1;
};

}  // namespace psynth::sfm