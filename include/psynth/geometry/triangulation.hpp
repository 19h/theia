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

// Batch reprojection error computation with SIMD optimization
// Computes reprojection errors for multiple 3D points against a single camera
// Returns errors in the output vector (resized to match input size)
void BatchReprojectionErrorPx(const Camera& cam,
                              const double* points_xyz,  // Interleaved: x0,y0,z0,x1,y1,z1,...
                              const double* obs_uv,      // Interleaved: u0,v0,u1,v1,...
                              int num_points,
                              double* errors_out);

// Batch point projection (world to pixel coordinates)
// Input: points_xyz as x0,y0,z0,x1,y1,z1,...
// Output: proj_uv as u0,v0,u1,v1,...
void BatchProjectPoints(const Camera& cam,
                        const double* points_xyz,
                        int num_points,
                        double* proj_uv);

// Compute reprojection errors for a single 3D point observed by multiple cameras
// This is the hot path in outlier filtering where one track is checked against all its observations
// Input: X_world - single 3D point, cameras - array of camera pointers, obs_uv - observation coords
// Output: errors_out - reprojection error per camera (size = num_cameras)
void SinglePointMultiCameraErrors(const Eigen::Vector3d& X_world,
                                  const Camera* const* cameras,  // Array of pointers
                                  const double* obs_uv,          // u0,v0,u1,v1,...
                                  int num_cameras,
                                  double* errors_out);

}  // namespace psynth::geometry