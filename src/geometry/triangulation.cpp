#include <psynth/geometry/triangulation.hpp>
#include <psynth/common.hpp>
#include <psynth/simd/simd.hpp>

#include <Eigen/SVD>

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

void BatchProjectPoints(const Camera& cam,
                        const double* points_xyz,
                        int num_points,
                        double* proj_uv) {
  if (num_points <= 0) return;

  // Extract camera parameters for vectorized computation
  const double r00 = cam.pose.R(0, 0), r01 = cam.pose.R(0, 1), r02 = cam.pose.R(0, 2);
  const double r10 = cam.pose.R(1, 0), r11 = cam.pose.R(1, 1), r12 = cam.pose.R(1, 2);
  const double r20 = cam.pose.R(2, 0), r21 = cam.pose.R(2, 1), r22 = cam.pose.R(2, 2);
  const double tx = cam.pose.t.x(), ty = cam.pose.t.y(), tz = cam.pose.t.z();
  const double f = cam.intr.f_px;
  const double cx = cam.intr.cx_px, cy = cam.intr.cy_px;
  const double k1 = cam.intr.k1, k2 = cam.intr.k2;

#if defined(PSYNTH_SIMD_NEON) || defined(PSYNTH_SIMD_AVX2)
  using namespace simd;

  // SIMD constants
  const Double4 vr00(r00), vr01(r01), vr02(r02);
  const Double4 vr10(r10), vr11(r11), vr12(r12);
  const Double4 vr20(r20), vr21(r21), vr22(r22);
  const Double4 vtx(tx), vty(ty), vtz(tz);
  const Double4 vf(f), vcx(cx), vcy(cy);
  const Double4 vk1(k1), vk2(k2);
  const Double4 vone(1.0);
  const Double4 veps(std::numeric_limits<double>::epsilon());

  // Convert interleaved XYZ to separate arrays for SIMD (thread-local to avoid allocation)
  thread_local std::vector<double> xs, ys, zs;
  xs.resize(num_points);
  ys.resize(num_points);
  zs.resize(num_points);

  for (int i = 0; i < num_points; ++i) {
    xs[i] = points_xyz[i * 3 + 0];
    ys[i] = points_xyz[i * 3 + 1];
    zs[i] = points_xyz[i * 3 + 2];
  }

  const int n4 = num_points & ~3;
  for (int i = 0; i < n4; i += 4) {
    // Prefetch next batch
    if (i + 16 < num_points) {
      prefetch_read(&xs[i + 16]);
      prefetch_read(&ys[i + 16]);
      prefetch_read(&zs[i + 16]);
    }

    const Double4 X = Double4::load(&xs[i]);
    const Double4 Y = Double4::load(&ys[i]);
    const Double4 Z = Double4::load(&zs[i]);

    // Transform: Xc = R * X_world + t
    const Double4 Xc = fmadd(vr00, X, fmadd(vr01, Y, fmadd(vr02, Z, vtx)));
    const Double4 Yc = fmadd(vr10, X, fmadd(vr11, Y, fmadd(vr12, Z, vty)));
    const Double4 Zc = fmadd(vr20, X, fmadd(vr21, Y, fmadd(vr22, Z, vtz)));

    // Safe division (handle points behind camera)
    const Double4 safe_z = max(Zc, veps);
    const Double4 inv_z = vone / safe_z;
    const Double4 x = Xc * inv_z;
    const Double4 y = Yc * inv_z;

    // Radial distortion: d = 1 + k1*r² + k2*r⁴
    const Double4 r2 = fmadd(x, x, y * y);
    const Double4 r4 = r2 * r2;
    const Double4 d = fmadd(vk1, r2, fmadd(vk2, r4, vone));

    // Final projection
    const Double4 u = fmadd(vf * d, x, vcx);
    const Double4 v = fmadd(vf * d, y, vcy);

    // Store interleaved UV
    alignas(32) double u_arr[4], v_arr[4];
    u.store(u_arr);
    v.store(v_arr);

    for (int j = 0; j < 4; ++j) {
      proj_uv[(i + j) * 2 + 0] = u_arr[j];
      proj_uv[(i + j) * 2 + 1] = v_arr[j];
    }
  }

  // Handle remainder with scalar code
  for (int i = n4; i < num_points; ++i) {
#else
  for (int i = 0; i < num_points; ++i) {
#endif
    const double X = points_xyz[i * 3 + 0];
    const double Y = points_xyz[i * 3 + 1];
    const double Z = points_xyz[i * 3 + 2];

    const double Xc = r00 * X + r01 * Y + r02 * Z + tx;
    const double Yc = r10 * X + r11 * Y + r12 * Z + ty;
    const double Zc = r20 * X + r21 * Y + r22 * Z + tz;

    if (Zc <= std::numeric_limits<double>::epsilon()) {
      proj_uv[i * 2 + 0] = std::numeric_limits<double>::quiet_NaN();
      proj_uv[i * 2 + 1] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    const double inv_z = 1.0 / Zc;
    const double x = Xc * inv_z;
    const double y = Yc * inv_z;
    const double r2 = x * x + y * y;
    const double d = 1.0 + k1 * r2 + k2 * r2 * r2;

    proj_uv[i * 2 + 0] = f * d * x + cx;
    proj_uv[i * 2 + 1] = f * d * y + cy;
#if defined(PSYNTH_SIMD_NEON) || defined(PSYNTH_SIMD_AVX2)
  }
#else
  }
#endif
}

void BatchReprojectionErrorPx(const Camera& cam,
                              const double* points_xyz,
                              const double* obs_uv,
                              int num_points,
                              double* errors_out) {
  if (num_points <= 0) return;

  // Extract camera parameters
  const double r00 = cam.pose.R(0, 0), r01 = cam.pose.R(0, 1), r02 = cam.pose.R(0, 2);
  const double r10 = cam.pose.R(1, 0), r11 = cam.pose.R(1, 1), r12 = cam.pose.R(1, 2);
  const double r20 = cam.pose.R(2, 0), r21 = cam.pose.R(2, 1), r22 = cam.pose.R(2, 2);
  const double tx = cam.pose.t.x(), ty = cam.pose.t.y(), tz = cam.pose.t.z();
  const double f = cam.intr.f_px;
  const double cx = cam.intr.cx_px, cy = cam.intr.cy_px;
  const double k1 = cam.intr.k1, k2 = cam.intr.k2;

#if defined(PSYNTH_SIMD_NEON) || defined(PSYNTH_SIMD_AVX2)
  using namespace simd;

  const Double4 vr00(r00), vr01(r01), vr02(r02);
  const Double4 vr10(r10), vr11(r11), vr12(r12);
  const Double4 vr20(r20), vr21(r21), vr22(r22);
  const Double4 vtx(tx), vty(ty), vtz(tz);
  const Double4 vf(f), vcx(cx), vcy(cy);
  const Double4 vk1(k1), vk2(k2);
  const Double4 vone(1.0);
  const Double4 veps(std::numeric_limits<double>::epsilon());
  const Double4 vmax_err(std::numeric_limits<double>::max());

  // Convert interleaved to SoA for SIMD
  thread_local std::vector<double> xs, ys, zs, us, vs;
  xs.resize(num_points);
  ys.resize(num_points);
  zs.resize(num_points);
  us.resize(num_points);
  vs.resize(num_points);

  for (int i = 0; i < num_points; ++i) {
    xs[i] = points_xyz[i * 3 + 0];
    ys[i] = points_xyz[i * 3 + 1];
    zs[i] = points_xyz[i * 3 + 2];
    us[i] = obs_uv[i * 2 + 0];
    vs[i] = obs_uv[i * 2 + 1];
  }

  const int n4 = num_points & ~3;
  for (int i = 0; i < n4; i += 4) {
    const Double4 X = Double4::load(&xs[i]);
    const Double4 Y = Double4::load(&ys[i]);
    const Double4 Z = Double4::load(&zs[i]);
    const Double4 obs_u = Double4::load(&us[i]);
    const Double4 obs_v = Double4::load(&vs[i]);

    // Transform
    const Double4 Xc = fmadd(vr00, X, fmadd(vr01, Y, fmadd(vr02, Z, vtx)));
    const Double4 Yc = fmadd(vr10, X, fmadd(vr11, Y, fmadd(vr12, Z, vty)));
    const Double4 Zc = fmadd(vr20, X, fmadd(vr21, Y, fmadd(vr22, Z, vtz)));

    // Safe division
    const Double4 safe_z = max(Zc, veps);
    const Double4 inv_z = vone / safe_z;
    const Double4 x = Xc * inv_z;
    const Double4 y = Yc * inv_z;

    // Distortion
    const Double4 r2 = fmadd(x, x, y * y);
    const Double4 r4 = r2 * r2;
    const Double4 d = fmadd(vk1, r2, fmadd(vk2, r4, vone));

    // Projection
    const Double4 u_proj = fmadd(vf * d, x, vcx);
    const Double4 v_proj = fmadd(vf * d, y, vcy);

    // Error computation
    const Double4 du = u_proj - obs_u;
    const Double4 dv = v_proj - obs_v;
    const Double4 err2 = fmadd(du, du, dv * dv);
    const Double4 err = sqrt(err2);

    // Handle invalid points (Zc <= eps)
    // Note: For simplicity, we just compute normally and let the caller handle NaN/large values
    err.store(&errors_out[i]);
  }

  // Remainder
  for (int i = n4; i < num_points; ++i) {
#else
  for (int i = 0; i < num_points; ++i) {
#endif
    const double X = points_xyz[i * 3 + 0];
    const double Y = points_xyz[i * 3 + 1];
    const double Z = points_xyz[i * 3 + 2];

    const double Xc = r00 * X + r01 * Y + r02 * Z + tx;
    const double Yc = r10 * X + r11 * Y + r12 * Z + ty;
    const double Zc = r20 * X + r21 * Y + r22 * Z + tz;

    if (Zc <= std::numeric_limits<double>::epsilon()) {
      errors_out[i] = std::numeric_limits<double>::max();
      continue;
    }

    const double inv_z = 1.0 / Zc;
    const double x = Xc * inv_z;
    const double y = Yc * inv_z;
    const double r2 = x * x + y * y;
    const double d = 1.0 + k1 * r2 + k2 * r2 * r2;

    const double u_proj = f * d * x + cx;
    const double v_proj = f * d * y + cy;

    const double du = u_proj - obs_uv[i * 2 + 0];
    const double dv = v_proj - obs_uv[i * 2 + 1];
    errors_out[i] = std::sqrt(du * du + dv * dv);
#if defined(PSYNTH_SIMD_NEON) || defined(PSYNTH_SIMD_AVX2)
  }
#else
  }
#endif
}

void SinglePointMultiCameraErrors(const Eigen::Vector3d& X_world,
                                  const Camera* const* cameras,
                                  const double* obs_uv,
                                  int num_cameras,
                                  double* errors_out) {
  if (num_cameras <= 0) return;

  const double Xw = X_world.x();
  const double Yw = X_world.y();
  const double Zw = X_world.z();

  for (int ci = 0; ci < num_cameras; ++ci) {
    const Camera& cam = *cameras[ci];

    // Extract camera params
    const double r00 = cam.pose.R(0, 0), r01 = cam.pose.R(0, 1), r02 = cam.pose.R(0, 2);
    const double r10 = cam.pose.R(1, 0), r11 = cam.pose.R(1, 1), r12 = cam.pose.R(1, 2);
    const double r20 = cam.pose.R(2, 0), r21 = cam.pose.R(2, 1), r22 = cam.pose.R(2, 2);
    const double tx = cam.pose.t.x(), ty = cam.pose.t.y(), tz = cam.pose.t.z();

    // Transform to camera coords
    const double Xc = r00 * Xw + r01 * Yw + r02 * Zw + tx;
    const double Yc = r10 * Xw + r11 * Yw + r12 * Zw + ty;
    const double Zc = r20 * Xw + r21 * Yw + r22 * Zw + tz;

    if (Zc <= std::numeric_limits<double>::epsilon()) {
      errors_out[ci] = std::numeric_limits<double>::max();
      continue;
    }

    const double inv_z = 1.0 / Zc;
    const double x = Xc * inv_z;
    const double y = Yc * inv_z;

    // Distortion
    const double r2 = x * x + y * y;
    const double d = 1.0 + cam.intr.k1 * r2 + cam.intr.k2 * r2 * r2;

    // Project
    const double u_proj = cam.intr.f_px * d * x + cam.intr.cx_px;
    const double v_proj = cam.intr.f_px * d * y + cam.intr.cy_px;

    // Error
    const double du = u_proj - obs_uv[ci * 2];
    const double dv = v_proj - obs_uv[ci * 2 + 1];
    errors_out[ci] = std::sqrt(du * du + dv * dv);
  }
}

}  // namespace psynth::geometry