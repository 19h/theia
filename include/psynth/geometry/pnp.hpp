#pragma once

#include <psynth/types.hpp>

#include <Eigen/Core>

#include <vector>

namespace psynth::geometry {

struct PnPRansacOptions {
  int max_iterations = 2000;
  double reprojection_error_px = 4.0;
  double confidence = 0.99;
  int min_inliers = 25;
  bool use_ap3p = true;
};

struct PnPRansacResult {
  bool success = false;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  std::vector<int> inlier_indices;
};

PnPRansacResult SolvePnPRansacOpenCV(const Intrinsics& intr,
                                     const std::vector<Eigen::Vector3d>& X_world,
                                     const std::vector<Eigen::Vector2d>& uv_px,
                                     const PnPRansacOptions& opt);

}  // namespace psynth::geometry