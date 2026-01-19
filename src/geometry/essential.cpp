#include <psynth/geometry/essential.hpp>
#include <psynth/common.hpp>
#include <psynth/geometry/triangulation.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <limits>

namespace psynth::geometry {

Eigen::Matrix3d EssentialFromFundamental(const Eigen::Matrix3d& F,
                                         const Intrinsics& intr1,
                                         const Intrinsics& intr2) {
  const Eigen::Matrix3d K1 = IntrinsicsMatrix(intr1);
  const Eigen::Matrix3d K2 = IntrinsicsMatrix(intr2);
  Eigen::Matrix3d E = K2.transpose() * F * K1;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  if (U.determinant() < 0) U.col(2) *= -1.0;
  if (V.determinant() < 0) V.col(2) *= -1.0;

  const Eigen::Vector3d s = svd.singularValues();
  const double a = 0.5 * (s(0) + s(1));
  E = U * Eigen::Vector3d(a, a, 0.0).asDiagonal() * V.transpose();

  return E;
}

std::vector<RelativePose> DecomposeEssential(const Eigen::Matrix3d& E) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  if (U.determinant() < 0) U.col(2) *= -1.0;
  if (V.determinant() < 0) V.col(2) *= -1.0;

  Eigen::Matrix3d W;
  W << 0, -1, 0,
       1,  0, 0,
       0,  0, 1;

  Eigen::Matrix3d R1 = U * W * V.transpose();
  Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
  Eigen::Vector3d t = U.col(2);

  if (R1.determinant() < 0) R1 = -R1;
  if (R2.determinant() < 0) R2 = -R2;

  std::vector<RelativePose> poses;
  poses.reserve(4);
  poses.push_back(RelativePose{R1,  t});
  poses.push_back(RelativePose{R1, -t});
  poses.push_back(RelativePose{R2,  t});
  poses.push_back(RelativePose{R2, -t});
  return poses;
}

CheiralityResult ChooseRelativePoseCheirality(const Eigen::Matrix3d& E,
                                              const Intrinsics& intr1,
                                              const Intrinsics& intr2,
                                              const std::vector<Eigen::Vector2d>& x1_px,
                                              const std::vector<Eigen::Vector2d>& x2_px) {
  PSYNTH_REQUIRE(x1_px.size() == x2_px.size(), "ChooseRelativePoseCheirality: size mismatch");
  PSYNTH_REQUIRE(x1_px.size() >= 8, "ChooseRelativePoseCheirality: need >=8 correspondences");

  const std::vector<RelativePose> poses = DecomposeEssential(E);

  const int N = static_cast<int>(x1_px.size());
  const int max_test = std::min(N, 200);

  // Early exit threshold: if we find a pose with >80% positive depth, use it
  const int early_exit_threshold = static_cast<int>(max_test * 0.8);

  // Pre-compute P1 (identity projection)
  static const Eigen::Matrix<double, 3, 4> P1 =
      (Eigen::Matrix<double, 3, 4>() << 1, 0, 0, 0,
                                        0, 1, 0, 0,
                                        0, 0, 1, 0)
          .finished();

  // Pre-compute undistorted normalized coordinates (once, used for all poses)
  // This eliminates 4x redundant undistortion for each point
  thread_local std::vector<Eigen::Vector2d> n1_cache, n2_cache;
  n1_cache.resize(max_test);
  n2_cache.resize(max_test);

  for (int i = 0; i < max_test; ++i) {
    n1_cache[i] = PixelToUndistortedNormalized(intr1, x1_px[i]);
    n2_cache[i] = PixelToUndistortedNormalized(intr2, x2_px[i]);
  }

  CheiralityResult best;
  best.num_positive_depth = -1;

  for (const auto& pose : poses) {
    Eigen::Matrix<double, 3, 4> P2;
    P2.block<3, 3>(0, 0) = pose.R;
    P2.col(3) = pose.t;

    // Extract R row 2 and t[2] for fast z2 computation
    const double R20 = pose.R(2, 0), R21 = pose.R(2, 1), R22 = pose.R(2, 2);
    const double t2 = pose.t.z();

    int positive = 0;
    for (int i = 0; i < max_test; ++i) {
      Eigen::Vector3d X;
      if (!TriangulateDLT(P1, P2, n1_cache[i], n2_cache[i], &X)) continue;

      // z1 = X.z() for identity camera
      // z2 = R[2,:] * X + t[2] - computed directly without full matrix multiply
      const double z1 = X.z();
      const double z2 = R20 * X.x() + R21 * X.y() + R22 * X.z() + t2;

      if (z1 > 0 && z2 > 0) positive++;
    }

    if (positive > best.num_positive_depth) {
      best.pose = pose;
      best.num_positive_depth = positive;

      // Early exit: if this pose has enough positive depth, use it
      if (positive >= early_exit_threshold) {
        return best;
      }
    }
  }

  return best;
}

}  // namespace psynth::geometry