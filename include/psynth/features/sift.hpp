#pragma once

#include <psynth/config.hpp>
#include <psynth/types.hpp>

#include <opencv2/core.hpp>

namespace psynth::features {

FeatureSet ExtractSIFT(const cv::Mat& gray_u8, const SiftConfig& cfg);

}  // namespace psynth::features