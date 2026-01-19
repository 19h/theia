#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace psynth::geometry {

struct FundamentalRansacOptions {
  int max_iterations = 2000;
  double confidence = 0.99;
  double inlier_threshold_px = 2.0;
  int min_inliers = 30;
  uint32_t rng_seed = 0xC0FFEEu;
};

struct FundamentalRansacResult {
  bool success = false;
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  std::vector<int> inlier_indices;
  int iterations_run = 0;
  double inlier_threshold_px = 0.0;
};

Eigen::Matrix3d EstimateFundamentalNormalized8Point(const std::vector<Eigen::Vector2d>& x1_px,
                                                    const std::vector<Eigen::Vector2d>& x2_px);

double SampsonDistanceSquared(const Eigen::Matrix3d& F, const Eigen::Vector2d& x1_px, const Eigen::Vector2d& x2_px);

FundamentalRansacResult EstimateFundamentalRansac(const std::vector<Eigen::Vector2d>& x1_px,
                                                  const std::vector<Eigen::Vector2d>& x2_px,
                                                  const FundamentalRansacOptions& opt);

}  // namespace psynth::geometry