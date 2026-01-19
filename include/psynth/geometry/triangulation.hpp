#pragma once

#include <psynth/types.hpp>

#include <Eigen/Core>

namespace psynth::geometry {

Eigen::Matrix3d IntrinsicsMatrix(const Intrinsics& intr);

Eigen::Matrix<double, 3, 4> ExtrinsicsMatrix(const Pose& pose);

Eigen::Matrix<double, 3, 4> ProjectionMatrix(const Camera& cam);

Eigen::Vector2d PixelToNormalized(const Intrinsics& intr, const Eigen::Vector2d& uv_px);

Eigen::Vector2d PixelToUndistortedNormalized(const Intrinsics& intr, const Eigen::Vector2d& uv_px, int iterations = 5);

bool TriangulateDLT(const Eigen::Matrix<double, 3, 4>& P1,
                    const Eigen::Matrix<double, 3, 4>& P2,
                    const Eigen::Vector2d& x1,
                    const Eigen::Vector2d& x2,
                    Eigen::Vector3d* X_out);

double TriangulationAngleRad(const Eigen::Vector3d& C1, const Eigen::Vector3d& C2, const Eigen::Vector3d& X);

Eigen::Vector2d ProjectPointPx(const Camera& cam, const Eigen::Vector3d& X_world);

double ReprojectionErrorPx(const Camera& cam, const Eigen::Vector3d& X_world, const Eigen::Vector2d& uv_obs_px);

}  // namespace psynth::geometry