#include <psynth/features/sift.hpp>
#include <psynth/common.hpp>

#include <opencv2/features2d.hpp>

namespace psynth::features {

FeatureSet ExtractSIFT(const cv::Mat& gray_u8, const SiftConfig& cfg) {
  PSYNTH_REQUIRE(!gray_u8.empty(), "ExtractSIFT: empty image");
  PSYNTH_REQUIRE(gray_u8.type() == CV_8U, "ExtractSIFT: expected CV_8U grayscale image");

  FeatureSet fs;
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create(cfg.max_features,
                                            cfg.n_octave_layers,
                                            cfg.contrast_threshold,
                                            cfg.edge_threshold,
                                            cfg.sigma);
  sift->detectAndCompute(gray_u8, cv::noArray(), fs.keypoints, fs.descriptors);
  if (!fs.descriptors.empty() && fs.descriptors.type() != CV_32F) {
    fs.descriptors.convertTo(fs.descriptors, CV_32F);
  }
  return fs;
}

}  // namespace psynth::features