#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <psynth/common.hpp>
#include <psynth/matching/matcher.hpp>
#include <vector>

#ifdef PSYNTH_USE_OPENMP
#include <omp.h>
#endif

namespace psynth::matching {

namespace {

// Structure to hold both match index and distance to avoid redundant computation
struct MatchWithDistance {
  int idx = -1;
  float distance = std::numeric_limits<float>::max();
};

// Optimized matching function that returns both best match indices AND distances
// Avoids the redundant cv::norm() calls that were recomputing already-computed distances
void BestMatchesWithRatioOptimized(const cv::Mat& desc_query, const cv::Mat& desc_train,
                                   float ratio, int flann_trees, int flann_checks,
                                   std::vector<MatchWithDistance>& out) {
  out.assign(desc_query.rows, MatchWithDistance{});
  if (desc_query.empty() || desc_train.empty()) return;

  PSYNTH_REQUIRE(desc_query.type() == CV_32F && desc_train.type() == CV_32F,
                 "MatchSiftSymmetric: descriptors must be CV_32F");

  // FLANN index params - these are lightweight, no significant allocation here
  cv::Ptr<cv::flann::IndexParams> index_params =
      cv::makePtr<cv::flann::KDTreeIndexParams>(flann_trees);
  cv::Ptr<cv::flann::SearchParams> search_params =
      cv::makePtr<cv::flann::SearchParams>(flann_checks);

  cv::FlannBasedMatcher matcher(index_params, search_params);

  // knnMatch allocates internally - this is unavoidable with OpenCV's API
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(desc_query, desc_train, knn, 2);

  const int knn_size = static_cast<int>(knn.size());
  for (int i = 0; i < knn_size; ++i) {
    if (knn[i].size() < 2) continue;
    const cv::DMatch& m1 = knn[i][0];
    const cv::DMatch& m2 = knn[i][1];
    if (m1.distance < ratio * m2.distance) {
      out[i].idx = m1.trainIdx;
      out[i].distance = m1.distance;  // Store the distance - FLANN already computed it!
    }
  }
}

// Apply ratio test to knn matches, return (index, distance) for passing matches
void ApplyRatioTest(const std::vector<std::vector<cv::DMatch>>& knn, float ratio,
                    std::vector<MatchWithDistance>& out) {
  out.assign(knn.size(), MatchWithDistance{});
  const int knn_size = static_cast<int>(knn.size());
  for (int i = 0; i < knn_size; ++i) {
    if (knn[i].size() < 2) continue;
    const cv::DMatch& m1 = knn[i][0];
    const cv::DMatch& m2 = knn[i][1];
    if (m1.distance < ratio * m2.distance) {
      out[i].idx = m1.trainIdx;
      out[i].distance = m1.distance;
    }
  }
}

}  // namespace

// ============================================================================
// FeatureIndex implementation
// ============================================================================

FeatureIndex::FeatureIndex(const cv::Mat& descriptors, int flann_trees) {
  if (descriptors.empty()) return;

  // Make a copy - FLANN requires the data to persist
  descriptors_ = descriptors.clone();

  cv::flann::KDTreeIndexParams index_params(flann_trees);
  index_ = cv::makePtr<cv::flann::Index>(descriptors_, index_params);
}

void FeatureIndex::KnnSearch(const cv::Mat& query, int k, int flann_checks,
                             std::vector<std::vector<cv::DMatch>>& matches) const {
  matches.clear();
  if (!index_ || query.empty()) return;

  cv::Mat indices(query.rows, k, CV_32S);
  cv::Mat dists(query.rows, k, CV_32F);

  cv::flann::SearchParams search_params(flann_checks);
  index_->knnSearch(query, indices, dists, k, search_params);

  matches.resize(query.rows);
  for (int i = 0; i < query.rows; ++i) {
    matches[i].resize(k);
    for (int j = 0; j < k; ++j) {
      matches[i][j].queryIdx = i;
      matches[i][j].trainIdx = indices.at<int>(i, j);
      matches[i][j].distance = dists.at<float>(i, j);
    }
  }
}

// ============================================================================
// FeatureMatcher implementation
// ============================================================================

FeatureMatcher::FeatureMatcher(const std::vector<FeatureSet>& features, const MatcherConfig& cfg)
    : features_(features), cfg_(cfg) {
  indices_.resize(features_.size());
}

void FeatureMatcher::BuildIndices() {
  const int N = static_cast<int>(features_.size());

#ifdef PSYNTH_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < N; ++i) {
    if (!features_[i].descriptors.empty()) {
      indices_[i] = std::make_unique<FeatureIndex>(features_[i].descriptors, cfg_.flann_trees);
    }
  }

  indices_built_ = true;
}

std::vector<KeypointMatch> FeatureMatcher::Match(int i, int j) const {
  std::vector<KeypointMatch> out;

  if (!indices_built_) {
    // Fall back to non-cached version if indices not built
    return MatchSiftSymmetric(features_[i], features_[j], cfg_);
  }

  if (i < 0 || j < 0 || i >= static_cast<int>(features_.size()) ||
      j >= static_cast<int>(features_.size())) {
    return out;
  }

  const auto& desc_i = features_[i].descriptors;
  const auto& desc_j = features_[j].descriptors;
  if (desc_i.empty() || desc_j.empty()) return out;

  // Match i -> j using pre-built index for j
  std::vector<std::vector<cv::DMatch>> knn_i2j;
  if (indices_[j] && indices_[j]->IsValid()) {
    indices_[j]->KnnSearch(desc_i, 2, cfg_.flann_checks, knn_i2j);
  } else {
    return out;
  }

  std::vector<MatchWithDistance> i2j;
  ApplyRatioTest(knn_i2j, cfg_.ratio, i2j);

  if (!cfg_.cross_check) {
    // No cross-check: just collect valid matches
    int valid_count = 0;
    for (const auto& m : i2j) {
      if (m.idx >= 0) ++valid_count;
    }
    out.reserve(valid_count);

    const int i2j_size = static_cast<int>(i2j.size());
    for (int qi = 0; qi < i2j_size; ++qi) {
      if (i2j[qi].idx < 0) continue;
      KeypointMatch m;
      m.kp1 = qi;
      m.kp2 = i2j[qi].idx;
      m.distance = i2j[qi].distance;
      out.push_back(m);
    }
    return out;
  }

  // Cross-check: match j -> i using pre-built index for i
  std::vector<std::vector<cv::DMatch>> knn_j2i;
  if (indices_[i] && indices_[i]->IsValid()) {
    indices_[i]->KnnSearch(desc_j, 2, cfg_.flann_checks, knn_j2i);
  } else {
    return out;
  }

  std::vector<MatchWithDistance> j2i;
  ApplyRatioTest(knn_j2i, cfg_.ratio, j2i);

  // Cross-check and collect matches
  const int i2j_size = static_cast<int>(i2j.size());
  const int j2i_size = static_cast<int>(j2i.size());

  int valid_count = 0;
  for (int qi = 0; qi < i2j_size; ++qi) {
    const int qj = i2j[qi].idx;
    if (qj >= 0 && qj < j2i_size && j2i[qj].idx == qi) {
      ++valid_count;
    }
  }
  out.reserve(valid_count);

  for (int qi = 0; qi < i2j_size; ++qi) {
    const int qj = i2j[qi].idx;
    if (qj < 0) continue;
    if (qj < j2i_size && j2i[qj].idx == qi) {
      KeypointMatch m;
      m.kp1 = qi;
      m.kp2 = qj;
      m.distance = i2j[qi].distance;
      out.push_back(m);
    }
  }
  return out;
}

// ============================================================================
// Original function (for backward compatibility and simple use cases)
// ============================================================================

std::vector<KeypointMatch> MatchSiftSymmetric(const FeatureSet& a, const FeatureSet& b,
                                              const MatcherConfig& cfg) {
  std::vector<KeypointMatch> out;
  if (a.descriptors.empty() || b.descriptors.empty()) return out;

  // Use optimized version that returns distances directly
  std::vector<MatchWithDistance> a2b;
  BestMatchesWithRatioOptimized(a.descriptors, b.descriptors, cfg.ratio, cfg.flann_trees,
                                cfg.flann_checks, a2b);

  if (!cfg.cross_check) {
    // Count valid matches first to avoid reallocation
    int valid_count = 0;
    for (const auto& m : a2b) {
      if (m.idx >= 0) ++valid_count;
    }
    out.reserve(valid_count);

    const int a2b_size = static_cast<int>(a2b.size());
    for (int i = 0; i < a2b_size; ++i) {
      if (a2b[i].idx < 0) continue;
      KeypointMatch m;
      m.kp1 = i;
      m.kp2 = a2b[i].idx;
      m.distance = a2b[i].distance;  // Use pre-computed distance - no redundant cv::norm()!
      out.push_back(m);
    }
    return out;
  }

  // Cross-check: need to match in both directions
  std::vector<MatchWithDistance> b2a;
  BestMatchesWithRatioOptimized(b.descriptors, a.descriptors, cfg.ratio, cfg.flann_trees,
                                cfg.flann_checks, b2a);

  // Count valid cross-checked matches
  int valid_count = 0;
  const int a2b_size = static_cast<int>(a2b.size());
  const int b2a_size = static_cast<int>(b2a.size());
  for (int i = 0; i < a2b_size; ++i) {
    const int j = a2b[i].idx;
    if (j >= 0 && j < b2a_size && b2a[j].idx == i) {
      ++valid_count;
    }
  }
  out.reserve(valid_count);

  for (int i = 0; i < a2b_size; ++i) {
    const int j = a2b[i].idx;
    if (j < 0) continue;
    if (j < b2a_size && b2a[j].idx == i) {
      KeypointMatch m;
      m.kp1 = i;
      m.kp2 = j;
      m.distance = a2b[i].distance;  // Use pre-computed distance
      out.push_back(m);
    }
  }
  return out;
}

}  // namespace psynth::matching