#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace psynth {

using ImageId = int;

struct Intrinsics {
  int width_px = 0;
  int height_px = 0;
  double f_px = 0.0;
  double cx_px = 0.0;
  double cy_px = 0.0;
  double k1 = 0.0;
  double k2 = 0.0;
};

struct Pose {
  // World-to-camera: X_cam = R * X_world + t
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

inline Eigen::Vector3d CameraCenterWorld(const Pose& pose) {
  return -pose.R.transpose() * pose.t;
}

struct Camera {
  Intrinsics intr;
  Pose pose;
};

struct ImageInfo {
  ImageId id = -1;
  std::string path;
  Intrinsics intr;
};

struct FeatureSet {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;  // CV_32F (OpenCV SIFT output)
};

struct KeypointMatch {
  int kp1 = -1;
  int kp2 = -1;
  float distance = 0.0f;
};

struct ImagePair {
  ImageId i = -1;
  ImageId j = -1;
};

struct VerifiedPair {
  ImagePair pair;
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  std::vector<KeypointMatch> inliers;
  int num_ransac_iters = 0;
  double inlier_threshold_px = 0.0;
};

struct Observation {
  ImageId image_id = -1;
  int keypoint_id = -1;
  double u_px = 0.0;
  double v_px = 0.0;
};

struct Track {
  int id = -1;
  std::vector<Observation> observations;

  bool triangulated = false;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  cv::Vec3b color_bgr = cv::Vec3b(0, 0, 0);
};

}  // namespace psynth