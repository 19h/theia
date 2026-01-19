#include <psynth/geometry/triangulation.hpp>
#include <psynth/common.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>

namespace psynth::geometry {

Eigen::Matrix3d IntrinsicsMatrix(const Intrinsics& intr) {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = intr.f_px;
  K(1, 1) = intr.f_px;
  K(0, 2) = intr.cx_px;
  K(1, 2) = intr.cy_px;
  return K;
}

Eigen::Matrix<double, 3, 4> ExtrinsicsMatrix(const Pose& pose) {
  Eigen::Matrix<double, 3, 4> Rt;
  Rt.block<3, 3>(0, 0) = pose.R;
  Rt.col(3) = pose.t;
  return Rt;
}

Eigen::Matrix<double, 3, 4> ProjectionMatrix(const Camera& cam) {
  const Eigen::Matrix3d K = IntrinsicsMatrix(cam.intr);
  return K * ExtrinsicsMatrix(cam.pose);
}

Eigen::Vector2d PixelToNormalized(const Intrinsics& intr, const Eigen::Vector2d& uv_px) {
  return Eigen::Vector2d((uv_px.x() - intr.cx_px) / intr.f_px, (uv_px.y() - intr.cy_px) / intr.f_px);
}

Eigen::Vector2d PixelToUndistortedNormalized(const Intrinsics& intr, const Eigen::Vector2d& uv_px, int iterations) {
  const Eigen::Vector2d x0 = PixelToNormalized(intr, uv_px);
  if (intr.k1 == 0.0 && intr.k2 == 0.0) return x0;

  Eigen::Vector2d x = x0;
  for (int i = 0; i < iterations; ++i) {
    const double r2 = x.squaredNorm();
    const double d = 1.0 + intr.k1 * r2 + intr.k2 * r2 * r2;
    if (std::abs(d) <= std::numeric_limits<double>::epsilon()) break;
    x = x0 / d;
  }
  return x;
}

bool TriangulateDLT(const Eigen::Matrix<double, 3, 4>& P1,
                    const Eigen::Matrix<double, 3, 4>& P2,
                    const Eigen::Vector2d& x1,
                    const Eigen::Vector2d& x2,
                    Eigen::Vector3d* X_out) {
  if (!X_out) return false;

  Eigen::Matrix4d A;
  A.row(0) = x1.x() * P1.row(2) - P1.row(0);
  A.row(1) = x1.y() * P1.row(2) - P1.row(1);
  A.row(2) = x2.x() * P2.row(2) - P2.row(0);
  A.row(3) = x2.y() * P2.row(2) - P2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix4d> svdA(A, Eigen::ComputeFullV);
  const Eigen::Vector4d Xh = svdA.matrixV().col(3);
  if (std::abs(Xh(3)) <= std::numeric_limits<double>::epsilon()) return false;

  *X_out = Xh.head<3>() / Xh(3);
  return std::isfinite((*X_out)(0)) && std::isfinite((*X_out)(1)) && std::isfinite((*X_out)(2));
}

double TriangulationAngleRad(const Eigen::Vector3d& C1, const Eigen::Vector3d& C2, const Eigen::Vector3d& X) {
  // Use atan2(cross_mag, dot) instead of acos(dot/(n1*n2)) for better numerical stability
  // This avoids:
  // 1. Division by small norms
  // 2. Clamping artifacts near +/-1
  // 3. Loss of precision for small angles

  const Eigen::Vector3d v1 = X - C1;
  const Eigen::Vector3d v2 = X - C2;

  const double dot = v1.dot(v2);
  const double cross_mag = v1.cross(v2).norm();

  // atan2 handles all edge cases gracefully
  return std::atan2(cross_mag, dot);
}

Eigen::Vector2d ProjectPointPx(const Camera& cam, const Eigen::Vector3d& X_world) {
  const Eigen::Vector3d Xc = cam.pose.R * X_world + cam.pose.t;

  const double X = Xc.x();
  const double Y = Xc.y();
  const double Z = Xc.z();

  if (Z <= std::numeric_limits<double>::epsilon() || !std::isfinite(Z)) {
    return Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
  }

  const double x = X / Z;
  const double y = Y / Z;

  const double r2 = x * x + y * y;
  const double d = 1.0 + cam.intr.k1 * r2 + cam.intr.k2 * r2 * r2;

  const double xd = x * d;
  const double yd = y * d;

  const double u = cam.intr.f_px * xd + cam.intr.cx_px;
  const double v = cam.intr.f_px * yd + cam.intr.cy_px;

  return Eigen::Vector2d(u, v);
}

double ReprojectionErrorPx(const Camera& cam, const Eigen::Vector3d& X_world, const Eigen::Vector2d& uv_obs_px) {
  const Eigen::Vector2d uv_pred = ProjectPointPx(cam, X_world);
  if (!std::isfinite(uv_pred.x()) || !std::isfinite(uv_pred.y())) return std::numeric_limits<double>::max();
  return (uv_pred - uv_obs_px).norm();
}

}  // namespace psynth::geometry