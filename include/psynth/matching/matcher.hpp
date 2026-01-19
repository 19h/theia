#pragma once

#include <memory>
#include <opencv2/flann.hpp>
#include <psynth/config.hpp>
#include <psynth/types.hpp>
#include <vector>

namespace psynth::matching {

// Original function - builds FLANN index per call (simpler but slower for many pairs)
std::vector<KeypointMatch> MatchSiftSymmetric(const FeatureSet& a, const FeatureSet& b,
                                              const MatcherConfig& cfg);

// Pre-built FLANN index for a single image's features
// Avoids rebuilding the KD-tree for every pair involving this image
class FeatureIndex {
 public:
  FeatureIndex() = default;
  explicit FeatureIndex(const cv::Mat& descriptors, int flann_trees = 4);

  // Non-copyable, movable
  FeatureIndex(const FeatureIndex&) = delete;
  FeatureIndex& operator=(const FeatureIndex&) = delete;
  FeatureIndex(FeatureIndex&&) = default;
  FeatureIndex& operator=(FeatureIndex&&) = default;

  // Query k nearest neighbors
  // Returns matches where queryIdx is the index in query, trainIdx is the index in this index
  void KnnSearch(const cv::Mat& query, int k, int flann_checks,
                 std::vector<std::vector<cv::DMatch>>& matches) const;

  bool IsValid() const {
    return index_ != nullptr;
  }

 private:
  cv::Mat descriptors_;  // Keep copy of descriptors (FLANN requires data to persist)
  cv::Ptr<cv::flann::Index> index_;
};

// Feature matcher that caches FLANN indices for all images
// Use this when matching many image pairs - builds N indices instead of O(N^2)
class FeatureMatcher {
 public:
  FeatureMatcher(const std::vector<FeatureSet>& features, const MatcherConfig& cfg);

  // Build all indices (can be parallelized with OpenMP)
  void BuildIndices();

  // Match image pair using pre-built indices
  // Returns symmetric (cross-checked if enabled) matches
  std::vector<KeypointMatch> Match(int i, int j) const;

  // Check if indices are built
  bool IndicesBuilt() const {
    return indices_built_;
  }

 private:
  const std::vector<FeatureSet>& features_;
  MatcherConfig cfg_;
  std::vector<std::unique_ptr<FeatureIndex>> indices_;
  bool indices_built_ = false;
};

}  // namespace psynth::matching