#include <psynth/io/serialization.hpp>
#include <psynth/common.hpp>

#include <opencv2/core.hpp>

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace psynth::io {

namespace fs = std::filesystem;

void EnsureDir(const fs::path& p) {
  if (p.empty()) return;
  std::error_code ec;
  fs::create_directories(p, ec);
  PSYNTH_REQUIRE(!ec, "Failed to create directory: " + p.string() + " (" + ec.message() + ")");
}

fs::path FeaturePath(const fs::path& features_dir, ImageId id) {
  std::ostringstream oss;
  oss << std::setw(8) << std::setfill('0') << id << ".yml.gz";
  return features_dir / oss.str();
}

bool SaveFeatureSet(const fs::path& path, const FeatureSet& fs_in) {
  cv::FileStorage st(path.string(), cv::FileStorage::WRITE);
  if (!st.isOpened()) return false;

  st << "keypoints" << "[";
  for (const auto& kp : fs_in.keypoints) {
    st << "{:"
       << "x" << kp.pt.x << "y" << kp.pt.y << "size" << kp.size << "angle" << kp.angle
       << "response" << kp.response << "octave" << kp.octave << "class_id" << kp.class_id << "}";
  }
  st << "]";
  st << "descriptors" << fs_in.descriptors;
  return true;
}

bool LoadFeatureSet(const fs::path& path, FeatureSet* fs_out) {
  if (!fs_out) return false;
  cv::FileStorage st(path.string(), cv::FileStorage::READ);
  if (!st.isOpened()) return false;

  FeatureSet fs_in;

  cv::FileNode kps = st["keypoints"];
  if (!kps.empty()) {
    fs_in.keypoints.reserve(kps.size());
    for (auto it = kps.begin(); it != kps.end(); ++it) {
      cv::KeyPoint kp;
      kp.pt.x = static_cast<float>((*it)["x"]);
      kp.pt.y = static_cast<float>((*it)["y"]);
      kp.size = static_cast<float>((*it)["size"]);
      kp.angle = static_cast<float>((*it)["angle"]);
      kp.response = static_cast<float>((*it)["response"]);
      kp.octave = static_cast<int>((*it)["octave"]);
      kp.class_id = static_cast<int>((*it)["class_id"]);
      fs_in.keypoints.push_back(kp);
    }
  }

  st["descriptors"] >> fs_in.descriptors;
  if (!fs_in.descriptors.empty() && fs_in.descriptors.type() != CV_32F) {
    fs_in.descriptors.convertTo(fs_in.descriptors, CV_32F);
  }

  *fs_out = std::move(fs_in);
  return true;
}

bool SaveAllFeatures(const fs::path& features_dir, const std::vector<FeatureSet>& features) {
  EnsureDir(features_dir);
  for (int i = 0; i < static_cast<int>(features.size()); ++i) {
    const fs::path p = FeaturePath(features_dir, i);
    if (!SaveFeatureSet(p, features[i])) return false;
  }
  return true;
}

bool LoadAllFeatures(const fs::path& features_dir, int num_images, std::vector<FeatureSet>* features) {
  if (!features) return false;
  features->clear();
  features->resize(num_images);

  for (int i = 0; i < num_images; ++i) {
    const fs::path p = FeaturePath(features_dir, i);
    if (!fs::exists(p)) return false;
    FeatureSet fs_i;
    if (!LoadFeatureSet(p, &fs_i)) return false;
    (*features)[i] = std::move(fs_i);
  }
  return true;
}

bool SaveVerifiedPairs(const fs::path& path, const std::vector<VerifiedPair>& pairs) {
  cv::FileStorage st(path.string(), cv::FileStorage::WRITE);
  if (!st.isOpened()) return false;

  st << "verified_pairs" << "[";
  for (const auto& vp : pairs) {
    cv::Mat F(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        F.at<double>(r, c) = vp.F(r, c);
      }
    }

    const int n = static_cast<int>(vp.inliers.size());
    cv::Mat inliers_idx(n, 2, CV_32S);
    cv::Mat inliers_dist(n, 1, CV_32F);
    for (int k = 0; k < n; ++k) {
      inliers_idx.at<int>(k, 0) = vp.inliers[k].kp1;
      inliers_idx.at<int>(k, 1) = vp.inliers[k].kp2;
      inliers_dist.at<float>(k, 0) = vp.inliers[k].distance;
    }

    st << "{:"
       << "i" << vp.pair.i << "j" << vp.pair.j << "F" << F << "inliers_idx" << inliers_idx
       << "inliers_dist" << inliers_dist << "num_ransac_iters" << vp.num_ransac_iters
       << "inlier_threshold_px" << vp.inlier_threshold_px << "}";
  }
  st << "]";
  return true;
}

bool LoadVerifiedPairs(const fs::path& path, std::vector<VerifiedPair>* pairs_out) {
  if (!pairs_out) return false;
  cv::FileStorage st(path.string(), cv::FileStorage::READ);
  if (!st.isOpened()) return false;

  std::vector<VerifiedPair> pairs;

  cv::FileNode vps = st["verified_pairs"];
  if (vps.empty()) {
    *pairs_out = {};
    return true;
  }

  for (auto it = vps.begin(); it != vps.end(); ++it) {
    VerifiedPair vp;
    vp.pair.i = static_cast<int>((*it)["i"]);
    vp.pair.j = static_cast<int>((*it)["j"]);

    cv::Mat F;
    (*it)["F"] >> F;
    PSYNTH_REQUIRE(F.rows == 3 && F.cols == 3, "Invalid F matrix in verified_pairs");
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        vp.F(r, c) = F.at<double>(r, c);
      }
    }

    cv::Mat idx_mat;
    cv::Mat dist_mat;
    (*it)["inliers_idx"] >> idx_mat;
    (*it)["inliers_dist"] >> dist_mat;

    PSYNTH_REQUIRE(idx_mat.cols == 2, "Invalid inliers_idx matrix");
    PSYNTH_REQUIRE(dist_mat.cols == 1, "Invalid inliers_dist matrix");
    PSYNTH_REQUIRE(idx_mat.rows == dist_mat.rows, "Inlier idx/dist row mismatch");

    vp.inliers.reserve(idx_mat.rows);
    for (int k = 0; k < idx_mat.rows; ++k) {
      KeypointMatch m;
      m.kp1 = idx_mat.at<int>(k, 0);
      m.kp2 = idx_mat.at<int>(k, 1);
      m.distance = dist_mat.at<float>(k, 0);
      vp.inliers.push_back(m);
    }

    const cv::FileNode iters = (*it)["num_ransac_iters"];
    const cv::FileNode thr = (*it)["inlier_threshold_px"];
    if (!iters.empty()) iters >> vp.num_ransac_iters;
    if (!thr.empty()) thr >> vp.inlier_threshold_px;

    pairs.push_back(std::move(vp));
  }

  *pairs_out = std::move(pairs);
  return true;
}

}  // namespace psynth::io