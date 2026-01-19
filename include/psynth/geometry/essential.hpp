#pragma once

#include <psynth/types.hpp>

#include <Eigen/Core>

#include <vector>

namespace psynth::geometry {

struct RelativePose {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

Eigen::Matrix3d EssentialFromFundamental(const Eigen::Matrix3d& F, const Intrinsics& intr1, const Intrinsics& intr2);

std::vector<RelativePose> DecomposeEssential(const Eigen::Matrix3d& E);

struct CheiralityResult {
  RelativePose pose;
  int num_positive_depth = 0;
};

CheiralityResult ChooseRelativePoseCheirality(const Eigen::Matrix3d& E,
                                              const Intrinsics& intr1,
                                              const Intrinsics& intr2,
                                              const std::vector<Eigen::Vector2d>& x1_px,
                                              const std::vector<Eigen::Vector2d>& x2_px);

}  // namespace psynth::geometry