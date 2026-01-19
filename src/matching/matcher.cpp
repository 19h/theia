#include <psynth/matching/matcher.hpp>
#include <psynth/common.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

#include <vector>

namespace psynth::matching {

namespace {

std::vector<int> BestMatchesWithRatio(const cv::Mat& desc_query,
                                      const cv::Mat& desc_train,
                                      float ratio,
                                      int flann_trees,
                                      int flann_checks) {
  std::vector<int> best(desc_query.rows, -1);
  if (desc_query.empty() || desc_train.empty()) return best;

  PSYNTH_REQUIRE(desc_query.type() == CV_32F && desc_train.type() == CV_32F,
                 "MatchSiftSymmetric: descriptors must be CV_32F");

  cv::Ptr<cv::flann::IndexParams> index_params =
      cv::makePtr<cv::flann::KDTreeIndexParams>(flann_trees);
  cv::Ptr<cv::flann::SearchParams> search_params =
      cv::makePtr<cv::flann::SearchParams>(flann_checks);

  cv::FlannBasedMatcher matcher(index_params, search_params);

  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(desc_query, desc_train, knn, 2);

  for (int i = 0; i < static_cast<int>(knn.size()); ++i) {
    if (knn[i].size() < 2) continue;
    const cv::DMatch& m1 = knn[i][0];
    const cv::DMatch& m2 = knn[i][1];
    if (m1.distance < ratio * m2.distance) {
      best[i] = m1.trainIdx;
    }
  }
  return best;
}

}  // namespace

std::vector<KeypointMatch> MatchSiftSymmetric(const FeatureSet& a, const FeatureSet& b, const MatcherConfig& cfg) {
  std::vector<KeypointMatch> out;
  if (a.descriptors.empty() || b.descriptors.empty()) return out;

  const std::vector<int> a2b =
      BestMatchesWithRatio(a.descriptors, b.descriptors, cfg.ratio, cfg.flann_trees, cfg.flann_checks);

  if (!cfg.cross_check) {
    out.reserve(a2b.size());
    for (int i = 0; i < static_cast<int>(a2b.size()); ++i) {
      const int j = a2b[i];
      if (j < 0) continue;
      KeypointMatch m;
      m.kp1 = i;
      m.kp2 = j;
      m.distance = static_cast<float>(cv::norm(a.descriptors.row(i), b.descriptors.row(j), cv::NORM_L2));
      out.push_back(m);
    }
    return out;
  }

  const std::vector<int> b2a =
      BestMatchesWithRatio(b.descriptors, a.descriptors, cfg.ratio, cfg.flann_trees, cfg.flann_checks);

  out.reserve(a2b.size());
  for (int i = 0; i < static_cast<int>(a2b.size()); ++i) {
    const int j = a2b[i];
    if (j < 0) continue;
    if (j >= 0 && j < static_cast<int>(b2a.size()) && b2a[j] == i) {
      KeypointMatch m;
      m.kp1 = i;
      m.kp2 = j;
      m.distance = static_cast<float>(cv::norm(a.descriptors.row(i), b.descriptors.row(j), cv::NORM_L2));
      out.push_back(m);
    }
  }
  return out;
}

}  // namespace psynth::matching