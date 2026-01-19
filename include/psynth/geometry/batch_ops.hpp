#pragma once

// Batch geometry operations optimized with SIMD
// These functions process multiple points simultaneously for better performance

#include <psynth/simd/simd.hpp>
#include <Eigen/Core>
#include <vector>
#include <cmath>
#include <limits>

namespace psynth::geometry {

// ============================================================================
// Aligned data structures for batch operations
// ============================================================================

// SoA layout for 2D points - cache-friendly and SIMD-friendly
struct alignas(32) PointBatch2D {
  std::vector<double> x;
  std::vector<double> y;

  void resize(size_t n) {
    x.resize(n);
    y.resize(n);
  }

  void reserve(size_t n) {
    x.reserve(n);
    y.reserve(n);
  }

  size_t size() const { return x.size(); }

  void push_back(double px, double py) {
    x.push_back(px);
    y.push_back(py);
  }

  void push_back(const Eigen::Vector2d& p) {
    x.push_back(p.x());
    y.push_back(p.y());
  }

  void clear() {
    x.clear();
    y.clear();
  }
};

// SoA layout for 3D points
struct alignas(32) PointBatch3D {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  void resize(size_t n) {
    x.resize(n);
    y.resize(n);
    z.resize(n);
  }

  size_t size() const { return x.size(); }

  void clear() {
    x.clear();
    y.clear();
    z.clear();
  }
};

// ============================================================================
// Batch Sampson distance computation (SIMD-optimized)
// ============================================================================

// Count inliers using SIMD - processes 4 correspondences at a time
// Returns count and optionally fills inlier_indices
inline int CountInliersSIMD(
    const double* F,  // 9 elements: F[0..2] = row 0, F[3..5] = row 1, F[6..8] = row 2
    const double* x1, const double* y1,  // First image points
    const double* x2, const double* y2,  // Second image points
    int N,
    double thresh2,
    std::vector<int>* inlier_indices = nullptr) {

  using namespace simd;

  // Extract F matrix elements
  const Double4 F00(F[0]), F01(F[1]), F02(F[2]);
  const Double4 F10(F[3]), F11(F[4]), F12(F[5]);
  const Double4 F20(F[6]), F21(F[7]), F22(F[8]);
  const Double4 vthresh2(thresh2);
  const Double4 eps(std::numeric_limits<double>::epsilon());

  int count = 0;

  if (inlier_indices) {
    inlier_indices->clear();
    inlier_indices->reserve(N);
  }

  // Process 4 at a time
  const int N4 = N & ~3;
  for (int i = 0; i < N4; i += 4) {
    prefetch_read(x1 + i + 16);
    prefetch_read(x2 + i + 16);

    const Double4 px1 = Double4::load(x1 + i);
    const Double4 py1 = Double4::load(y1 + i);
    const Double4 px2 = Double4::load(x2 + i);
    const Double4 py2 = Double4::load(y2 + i);

    // Fx1 = F * [x1, y1, 1]^T
    const Double4 Fx1_0 = fmadd(F00, px1, fmadd(F01, py1, F02));
    const Double4 Fx1_1 = fmadd(F10, px1, fmadd(F11, py1, F12));
    const Double4 Fx1_2 = fmadd(F20, px1, fmadd(F21, py1, F22));

    // Ftx2 = F^T * [x2, y2, 1]^T
    const Double4 Ftx2_0 = fmadd(F00, px2, fmadd(F10, py2, F20));
    const Double4 Ftx2_1 = fmadd(F01, px2, fmadd(F11, py2, F21));

    // x2^T * F * x1
    const Double4 x2tFx1 = fmadd(px2, Fx1_0, fmadd(py2, Fx1_1, Fx1_2));

    // Denominator
    const Double4 denom = fmadd(Fx1_0, Fx1_0, fmadd(Fx1_1, Fx1_1,
                          fmadd(Ftx2_0, Ftx2_0, Ftx2_1 * Ftx2_1)));

    // Check: x2tFx1^2 < thresh2 * denom && denom > eps
    const Double4 num2 = x2tFx1 * x2tFx1;
    const Double4 rhs = vthresh2 * denom;

    // Get comparison results as a bitmask
#if defined(PSYNTH_SIMD_AVX2) || defined(PSYNTH_SIMD_AVX512) || defined(PSYNTH_SIMD_SSE42)
    const Double4 valid_denom = denom > eps;
    const Double4 valid_dist = num2 < rhs;
    const Double4 is_inlier = valid_denom & valid_dist;
    const int mask = is_inlier.movemask();

    // Count bits and collect indices
    count += __builtin_popcount(mask);

    if (inlier_indices) {
      if (mask & 1) inlier_indices->push_back(i);
      if (mask & 2) inlier_indices->push_back(i + 1);
      if (mask & 4) inlier_indices->push_back(i + 2);
      if (mask & 8) inlier_indices->push_back(i + 3);
    }
#else
    // Scalar fallback
    for (int j = 0; j < 4; ++j) {
      const double d = denom[j];
      const double n2 = num2[j];
      if (d > std::numeric_limits<double>::epsilon() && n2 < thresh2 * d) {
        count++;
        if (inlier_indices) inlier_indices->push_back(i + j);
      }
    }
#endif
  }

  // Handle remainder
  for (int i = N4; i < N; ++i) {
    const double px1_i = x1[i], py1_i = y1[i];
    const double px2_i = x2[i], py2_i = y2[i];

    const double Fx1_0 = F[0] * px1_i + F[1] * py1_i + F[2];
    const double Fx1_1 = F[3] * px1_i + F[4] * py1_i + F[5];
    const double Fx1_2 = F[6] * px1_i + F[7] * py1_i + F[8];

    const double Ftx2_0 = F[0] * px2_i + F[3] * py2_i + F[6];
    const double Ftx2_1 = F[1] * px2_i + F[4] * py2_i + F[7];

    const double x2tFx1 = px2_i * Fx1_0 + py2_i * Fx1_1 + Fx1_2;
    const double denom = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1;
    const double num2 = x2tFx1 * x2tFx1;

    if (denom > std::numeric_limits<double>::epsilon() && num2 < thresh2 * denom) {
      count++;
      if (inlier_indices) inlier_indices->push_back(i);
    }
  }

  return count;
}

// Convenience wrapper for Eigen input
inline int CountInliersSIMD(
    const Eigen::Matrix3d& F,
    const std::vector<Eigen::Vector2d>& x1_px,
    const std::vector<Eigen::Vector2d>& x2_px,
    double thresh2,
    std::vector<int>& inlier_indices) {

  const int N = static_cast<int>(x1_px.size());
  if (N == 0) return 0;

  // Convert to SoA layout (one-time cost amortized over many RANSAC iterations)
  thread_local std::vector<double> x1_soa, y1_soa, x2_soa, y2_soa;
  x1_soa.resize(N);
  y1_soa.resize(N);
  x2_soa.resize(N);
  y2_soa.resize(N);

  for (int i = 0; i < N; ++i) {
    x1_soa[i] = x1_px[i].x();
    y1_soa[i] = x1_px[i].y();
    x2_soa[i] = x2_px[i].x();
    y2_soa[i] = x2_px[i].y();
  }

  // Extract F in row-major order
  double F_flat[9] = {
    F(0,0), F(0,1), F(0,2),
    F(1,0), F(1,1), F(1,2),
    F(2,0), F(2,1), F(2,2)
  };

  return CountInliersSIMD(F_flat, x1_soa.data(), y1_soa.data(),
                          x2_soa.data(), y2_soa.data(), N, thresh2, &inlier_indices);
}

// ============================================================================
// Batch reprojection error computation
// ============================================================================

// Compute reprojection errors for multiple points
// Uses SoA layout internally for SIMD efficiency
inline void ComputeReprojectionErrorsBatch(
    const double* R,     // 9 elements row-major
    const double* t,     // 3 elements
    double f, double cx, double cy,
    double k1, double k2,
    const double* X_x, const double* X_y, const double* X_z,  // 3D points (SoA)
    const double* u_obs, const double* v_obs,                  // Observed 2D points
    int N,
    double* errors_out) {  // Output: squared errors

  using namespace simd;

  const Double4 vR00(R[0]), vR01(R[1]), vR02(R[2]);
  const Double4 vR10(R[3]), vR11(R[4]), vR12(R[5]);
  const Double4 vR20(R[6]), vR21(R[7]), vR22(R[8]);
  const Double4 vt0(t[0]), vt1(t[1]), vt2(t[2]);
  const Double4 vf(f), vcx(cx), vcy(cy);
  const Double4 vk1(k1), vk2(k2);
  const Double4 veps(1e-12);
  const Double4 one(1.0);

  const int N4 = N & ~3;

  for (int i = 0; i < N4; i += 4) {
    // Load 3D points
    const Double4 Xx = Double4::load(X_x + i);
    const Double4 Xy = Double4::load(X_y + i);
    const Double4 Xz = Double4::load(X_z + i);

    // Transform to camera coordinates: Xc = R * X + t
    const Double4 Xc_x = fmadd(vR00, Xx, fmadd(vR01, Xy, fmadd(vR02, Xz, vt0)));
    const Double4 Xc_y = fmadd(vR10, Xx, fmadd(vR11, Xy, fmadd(vR12, Xz, vt1)));
    const Double4 Xc_z = fmadd(vR20, Xx, fmadd(vR21, Xy, fmadd(vR22, Xz, vt2)));

    // Perspective division with epsilon for stability
    const Double4 inv_z = one / (Xc_z + veps);
    const Double4 x = Xc_x * inv_z;
    const Double4 y = Xc_y * inv_z;

    // Radial distortion
    const Double4 r2 = fmadd(x, x, y * y);
    const Double4 r4 = r2 * r2;
    const Double4 radial = fmadd(vk1, r2, fmadd(vk2, r4, one));

    // Project to pixels
    const Double4 u_pred = fmadd(vf * radial, x, vcx);
    const Double4 v_pred = fmadd(vf * radial, y, vcy);

    // Load observed points
    const Double4 u = Double4::load(u_obs + i);
    const Double4 v = Double4::load(v_obs + i);

    // Compute squared error
    const Double4 du = u_pred - u;
    const Double4 dv = v_pred - v;
    const Double4 err2 = fmadd(du, du, dv * dv);

    err2.store(errors_out + i);
  }

  // Handle remainder
  for (int i = N4; i < N; ++i) {
    const double Xx = X_x[i], Xy = X_y[i], Xz = X_z[i];

    const double Xc_x = R[0]*Xx + R[1]*Xy + R[2]*Xz + t[0];
    const double Xc_y = R[3]*Xx + R[4]*Xy + R[5]*Xz + t[1];
    const double Xc_z = R[6]*Xx + R[7]*Xy + R[8]*Xz + t[2];

    const double inv_z = 1.0 / (Xc_z + 1e-12);
    const double x = Xc_x * inv_z;
    const double y = Xc_y * inv_z;

    const double r2 = x*x + y*y;
    const double radial = 1.0 + k1*r2 + k2*r2*r2;

    const double u_pred = f * radial * x + cx;
    const double v_pred = f * radial * y + cy;

    const double du = u_pred - u_obs[i];
    const double dv = v_pred - v_obs[i];
    errors_out[i] = du*du + dv*dv;
  }
}

// ============================================================================
// Batch triangulation angle computation
// ============================================================================

// Compute triangulation angles for multiple points using atan2 for numerical stability
// Returns angles in radians
inline void ComputeTriangulationAngles(
    const double* C1_x, const double* C1_y, const double* C1_z,  // Camera 1 centers (can be single point broadcast)
    const double* C2_x, const double* C2_y, const double* C2_z,  // Camera 2 centers
    const double* X_x, const double* X_y, const double* X_z,     // 3D points
    int N,
    double* angles_out) {

  for (int i = 0; i < N; ++i) {
    // Vectors from cameras to point
    const double v1x = X_x[i] - C1_x[i];
    const double v1y = X_y[i] - C1_y[i];
    const double v1z = X_z[i] - C1_z[i];

    const double v2x = X_x[i] - C2_x[i];
    const double v2y = X_y[i] - C2_y[i];
    const double v2z = X_z[i] - C2_z[i];

    // Dot product
    const double dot = v1x*v2x + v1y*v2y + v1z*v2z;

    // Cross product magnitude (for atan2)
    const double cx = v1y*v2z - v1z*v2y;
    const double cy = v1z*v2x - v1x*v2z;
    const double cz = v1x*v2y - v1y*v2x;
    const double cross_mag = std::sqrt(cx*cx + cy*cy + cz*cz);

    // Use atan2 for numerical stability - better than acos(dot/(n1*n2))
    angles_out[i] = std::atan2(cross_mag, dot);
  }
}

// Single camera version (camera centers are the same for all points)
inline void ComputeTriangulationAnglesSingleCam(
    double C1x, double C1y, double C1z,
    double C2x, double C2y, double C2z,
    const double* X_x, const double* X_y, const double* X_z,
    int N,
    double* angles_out) {

  using namespace simd;

  const Double4 vC1x(C1x), vC1y(C1y), vC1z(C1z);
  const Double4 vC2x(C2x), vC2y(C2y), vC2z(C2z);

  const int N4 = N & ~3;

  for (int i = 0; i < N4; i += 4) {
    const Double4 Xx = Double4::load(X_x + i);
    const Double4 Xy = Double4::load(X_y + i);
    const Double4 Xz = Double4::load(X_z + i);

    // Vectors from cameras to points
    const Double4 v1x = Xx - vC1x;
    const Double4 v1y = Xy - vC1y;
    const Double4 v1z = Xz - vC1z;

    const Double4 v2x = Xx - vC2x;
    const Double4 v2y = Xy - vC2y;
    const Double4 v2z = Xz - vC2z;

    // Dot products
    const Double4 dot = fmadd(v1x, v2x, fmadd(v1y, v2y, v1z * v2z));

    // Cross product magnitudes
    const Double4 cx = fmsub(v1y, v2z, v1z * v2y);
    const Double4 cy = fmsub(v1z, v2x, v1x * v2z);
    const Double4 cz = fmsub(v1x, v2y, v1y * v2x);
    const Double4 cross_mag2 = fmadd(cx, cx, fmadd(cy, cy, cz * cz));
    const Double4 cross_mag = sqrt(cross_mag2);

    // Store intermediate results and compute atan2 (no SIMD atan2 available)
    alignas(32) double dot_arr[4], cross_arr[4];
    dot.store(dot_arr);
    cross_mag.store(cross_arr);

    for (int j = 0; j < 4; ++j) {
      angles_out[i + j] = std::atan2(cross_arr[j], dot_arr[j]);
    }
  }

  // Remainder
  for (int i = N4; i < N; ++i) {
    const double v1x = X_x[i] - C1x;
    const double v1y = X_y[i] - C1y;
    const double v1z = X_z[i] - C1z;

    const double v2x = X_x[i] - C2x;
    const double v2y = X_y[i] - C2y;
    const double v2z = X_z[i] - C2z;

    const double dot = v1x*v2x + v1y*v2y + v1z*v2z;

    const double cx = v1y*v2z - v1z*v2y;
    const double cy = v1z*v2x - v1x*v2z;
    const double cz = v1x*v2y - v1y*v2x;
    const double cross_mag = std::sqrt(cx*cx + cy*cy + cz*cz);

    angles_out[i] = std::atan2(cross_mag, dot);
  }
}

// ============================================================================
// Batch depth check for cheirality
// ============================================================================

// Check if points are in front of both cameras
// Returns count of points with positive depth in both views
inline int CountPositiveDepthBatch(
    const double* R,  // Rotation matrix (row-major 9 elements)
    const double* t,  // Translation (3 elements)
    const double* X_x, const double* X_y, const double* X_z,
    int N) {

  using namespace simd;

  const Double4 vR20(R[6]), vR21(R[7]), vR22(R[8]);
  const Double4 vt2(t[2]);
  const Double4 zero(0.0);

  int count = 0;
  const int N4 = N & ~3;

  for (int i = 0; i < N4; i += 4) {
    const Double4 Xx = Double4::load(X_x + i);
    const Double4 Xy = Double4::load(X_y + i);
    const Double4 Xz = Double4::load(X_z + i);

    // z1 = Xz (for P1 = [I|0])
    // z2 = R[2,:] * X + t[2]
    const Double4 z1 = Xz;
    const Double4 z2 = fmadd(vR20, Xx, fmadd(vR21, Xy, fmadd(vR22, Xz, vt2)));

#if defined(PSYNTH_SIMD_AVX2) || defined(PSYNTH_SIMD_AVX512) || defined(PSYNTH_SIMD_SSE42)
    const Double4 valid = (z1 > zero) & (z2 > zero);
    count += __builtin_popcount(valid.movemask());
#else
    for (int j = 0; j < 4; ++j) {
      if (z1[j] > 0 && z2[j] > 0) count++;
    }
#endif
  }

  // Remainder
  for (int i = N4; i < N; ++i) {
    const double z1 = X_z[i];
    const double z2 = R[6]*X_x[i] + R[7]*X_y[i] + R[8]*X_z[i] + t[2];
    if (z1 > 0 && z2 > 0) count++;
  }

  return count;
}

// ============================================================================
// Batch undistortion
// ============================================================================

// Undistort multiple points from pixels to normalized coordinates
inline void UndistortPointsBatch(
    double f, double cx, double cy, double k1, double k2,
    const double* u_px, const double* v_px,
    int N,
    double* x_out, double* y_out,
    int iterations = 5) {

  const double inv_f = 1.0 / f;

  for (int i = 0; i < N; ++i) {
    // Initial normalized coordinates (no distortion)
    double x = (u_px[i] - cx) * inv_f;
    double y = (v_px[i] - cy) * inv_f;

    if (k1 == 0.0 && k2 == 0.0) {
      x_out[i] = x;
      y_out[i] = y;
      continue;
    }

    // Iterative undistortion
    const double x0 = x;
    const double y0 = y;

    for (int iter = 0; iter < iterations; ++iter) {
      const double r2 = x*x + y*y;
      const double d = 1.0 + k1*r2 + k2*r2*r2;
      if (std::abs(d) < std::numeric_limits<double>::epsilon()) break;
      x = x0 / d;
      y = y0 / d;
    }

    x_out[i] = x;
    y_out[i] = y;
  }
}

}  // namespace psynth::geometry
