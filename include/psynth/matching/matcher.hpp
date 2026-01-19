#pragma once

#include <psynth/config.hpp>
#include <psynth/types.hpp>

#include <vector>

namespace psynth::matching {

std::vector<KeypointMatch> MatchSiftSymmetric(const FeatureSet& a, const FeatureSet& b, const MatcherConfig& cfg);

}  // namespace psynth::matching