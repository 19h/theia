#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <psynth/common.hpp>
#include <psynth/geometry/fundamental.hpp>
#include <random>
#include <unordered_set>
#include <vector>

namespace psynth::geometry {

namespace {

struct Normalization {
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  std::vector<Eigen::Vector3d> x;
};

Normalization NormalizePoints(const std::vector<Eigen::Vector2d>& pts) {
  Normalization n;
  const int N = static_cast<int>(pts.size());
  PSYNTH_REQUIRE(N > 0, "NormalizePoints: empty input");

  Eigen::Vector2d centroid(0.0, 0.0);
  for (const auto& p : pts) centroid += p;
  centroid /= static_cast<double>(N);

  double rms = 0.0;
  for (const auto& p : pts) {
    const Eigen::Vector2d d = p - centroid;
    rms += d.squaredNorm();
  }
  rms = std::sqrt(rms / static_cast<double>(N));
  const double scale = (rms > 0.0) ? (std::sqrt(2.0) / rms) : 1.0;

  n.T << scale, 0.0, -scale * centroid.x(), 0.0, scale, -scale * centroid.y(), 0.0, 0.0, 1.0;

  n.x.reserve(N);
  for (const auto& p : pts) {
    n.x.emplace_back(n.T * Eigen::Vector3d(p.x(), p.y(), 1.0));
  }
  return n;
}

// Fisher-Yates partial shuffle for sampling k unique indices from [0, N)
// Uses pre-allocated buffers to avoid heap allocations in RANSAC hot loop
void SampleUniqueIndicesFast(std::mt19937& rng, int N, int k, std::vector<int>& sample,
                             std::vector<int>& indices) {
  // Ensure indices array is properly sized and initialized
  if (static_cast<int>(indices.size()) != N) {
    indices.resize(N);
    for (int i = 0; i < N; ++i) indices[i] = i;
  }

  sample.clear();
  sample.reserve(k);

  // Fisher-Yates partial shuffle: swap k elements to the front
  for (int i = 0; i < k; ++i) {
    std::uniform_int_distribution<int> dist(i, N - 1);
    const int j = dist(rng);
    std::swap(indices[i], indices[j]);
    sample.push_back(indices[i]);
  }

  // Restore the swapped elements for next iteration (cheaper than full reinit)
  // This keeps indices in a "mostly sorted" state for locality
}

// Legacy function for backward compatibility
std::vector<int> SampleUniqueIndices(std::mt19937& rng, int N, int k) {
  PSYNTH_REQUIRE(N >= k, "SampleUniqueIndices: N < k");

  std::unordered_set<int> used;
  used.reserve(static_cast<std::size_t>(k) * 2);

  std::uniform_int_distribution<int> dist(0, N - 1);

  std::vector<int> sample;
  sample.reserve(k);
  while (static_cast<int>(sample.size()) < k) {
    const int idx = dist(rng);
    if (used.insert(idx).second) {
      sample.push_back(idx);
    }
  }
  return sample;
}

}  // namespace

Eigen::Matrix3d EstimateFundamentalNormalized8Point(const std::vector<Eigen::Vector2d>& x1_px,
                                                    const std::vector<Eigen::Vector2d>& x2_px) {
  PSYNTH_REQUIRE(x1_px.size() == x2_px.size(), "EstimateFundamental: size mismatch");
  const int N = static_cast<int>(x1_px.size());
  PSYNTH_REQUIRE(N >= 8, "EstimateFundamental: need >= 8 correspondences");

  const Normalization n1 = NormalizePoints(x1_px);
  const Normalization n2 = NormalizePoints(x2_px);

  Eigen::MatrixXd A(N, 9);
  for (int i = 0; i < N; ++i) {
    const double x = n1.x[i](0);
    const double y = n1.x[i](1);
    const double xp = n2.x[i](0);
    const double yp = n2.x[i](1);

    A(i, 0) = xp * x;
    A(i, 1) = xp * y;
    A(i, 2) = xp;
    A(i, 3) = yp * x;
    A(i, 4) = yp * y;
    A(i, 5) = yp;
    A(i, 6) = x;
    A(i, 7) = y;
    A(i, 8) = 1.0;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeFullV);
  const Eigen::VectorXd f = svdA.matrixV().col(8);

  Eigen::Matrix3d F;
  F << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);

  Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d s = svdF.singularValues();
  s(2) = 0.0;
  const Eigen::Matrix3d Frank2 = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

  Eigen::Matrix3d Fdenorm = n2.T.transpose() * Frank2 * n1.T;

  const double frob = Fdenorm.norm();
  if (frob > 0.0) Fdenorm /= frob;

  return Fdenorm;
}

// Optimized Sampson distance - avoids redundant matrix operations and temporaries
// Uses direct scalar computation to avoid Eigen expression template overhead
double SampsonDistanceSquared(const Eigen::Matrix3d& F, const Eigen::Vector2d& x1_px,
                              const Eigen::Vector2d& x2_px) {
  const double x1 = x1_px.x(), y1 = x1_px.y();
  const double x2 = x2_px.x(), y2 = x2_px.y();

  // Fx1 = F * [x1, y1, 1]^T
  const double Fx1_0 = F(0, 0) * x1 + F(0, 1) * y1 + F(0, 2);
  const double Fx1_1 = F(1, 0) * x1 + F(1, 1) * y1 + F(1, 2);
  const double Fx1_2 = F(2, 0) * x1 + F(2, 1) * y1 + F(2, 2);

  // Ftx2 = F^T * [x2, y2, 1]^T (only need first two components for denominator)
  const double Ftx2_0 = F(0, 0) * x2 + F(1, 0) * y2 + F(2, 0);
  const double Ftx2_1 = F(0, 1) * x2 + F(1, 1) * y2 + F(2, 1);

  // x2^T * F * x1 = x2 * Fx1_0 + y2 * Fx1_1 + 1 * Fx1_2
  const double x2tFx1 = x2 * Fx1_0 + y2 * Fx1_1 + Fx1_2;

  const double denom = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1;

  if (denom <= std::numeric_limits<double>::epsilon()) {
    return std::numeric_limits<double>::infinity();
  }
  return (x2tFx1 * x2tFx1) / denom;
}

// Batch Sampson distance computation for SIMD-friendly inlier counting
// Returns count of inliers and populates inlier_indices
int CountInliersFast(const Eigen::Matrix3d& F, const std::vector<Eigen::Vector2d>& x1_px,
                     const std::vector<Eigen::Vector2d>& x2_px, double thresh2,
                     std::vector<int>& inlier_indices) {
  inlier_indices.clear();
  const int N = static_cast<int>(x1_px.size());

  // Extract F matrix elements for scalar computation
  const double F00 = F(0, 0), F01 = F(0, 1), F02 = F(0, 2);
  const double F10 = F(1, 0), F11 = F(1, 1), F12 = F(1, 2);
  const double F20 = F(2, 0), F21 = F(2, 1), F22 = F(2, 2);

  for (int i = 0; i < N; ++i) {
    const double x1 = x1_px[i].x(), y1 = x1_px[i].y();
    const double x2 = x2_px[i].x(), y2 = x2_px[i].y();

    // Fx1 = F * [x1, y1, 1]^T
    const double Fx1_0 = F00 * x1 + F01 * y1 + F02;
    const double Fx1_1 = F10 * x1 + F11 * y1 + F12;
    const double Fx1_2 = F20 * x1 + F21 * y1 + F22;

    // Ftx2 = F^T * [x2, y2, 1]^T
    const double Ftx2_0 = F00 * x2 + F10 * y2 + F20;
    const double Ftx2_1 = F01 * x2 + F11 * y2 + F21;

    // x2^T * F * x1
    const double x2tFx1 = x2 * Fx1_0 + y2 * Fx1_1 + Fx1_2;

    const double denom = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1;

    // Avoid division when possible - multiply instead
    // d2 < thresh2  =>  x2tFx1^2 < thresh2 * denom
    if (denom > std::numeric_limits<double>::epsilon() && x2tFx1 * x2tFx1 < thresh2 * denom) {
      inlier_indices.push_back(i);
    }
  }
  return static_cast<int>(inlier_indices.size());
}

FundamentalRansacResult EstimateFundamentalRansac(const std::vector<Eigen::Vector2d>& x1_px,
                                                  const std::vector<Eigen::Vector2d>& x2_px,
                                                  const FundamentalRansacOptions& opt) {
  FundamentalRansacResult res;
  res.inlier_threshold_px = opt.inlier_threshold_px;

  PSYNTH_REQUIRE(x1_px.size() == x2_px.size(), "EstimateFundamentalRansac: size mismatch");
  const int N = static_cast<int>(x1_px.size());
  if (N < 8) {
    res.success = false;
    return res;
  }

  std::mt19937 rng(opt.rng_seed);

  const double thresh2 = opt.inlier_threshold_px * opt.inlier_threshold_px;
  constexpr int sample_size = 8;

  int best_inliers = -1;
  Eigen::Matrix3d bestF = Eigen::Matrix3d::Zero();
  std::vector<int> best_inlier_idx;
  best_inlier_idx.reserve(N);

  int max_iters = opt.max_iterations;
  int iters_run = 0;

  // Pre-allocate buffers for RANSAC loop to eliminate per-iteration heap allocations
  std::vector<int> sample;
  sample.reserve(sample_size);
  std::vector<int> indices;  // For Fisher-Yates sampling
  indices.reserve(N);
  std::vector<Eigen::Vector2d> s1(sample_size);
  std::vector<Eigen::Vector2d> s2(sample_size);
  std::vector<int> inliers;
  inliers.reserve(N);

  while (iters_run < max_iters) {
    // Use fast Fisher-Yates sampling with pre-allocated buffers
    SampleUniqueIndicesFast(rng, N, sample_size, sample, indices);

    // Copy sample points directly into pre-allocated arrays
    for (int k = 0; k < sample_size; ++k) {
      const int idx = sample[k];
      s1[k] = x1_px[idx];
      s2[k] = x2_px[idx];
    }

    Eigen::Matrix3d F;
    try {
      F = EstimateFundamentalNormalized8Point(s1, s2);
    } catch (...) {
      iters_run++;
      continue;
    }

    // Use optimized batch inlier counting
    const int num_inliers = CountInliersFast(F, x1_px, x2_px, thresh2, inliers);

    if (num_inliers > best_inliers) {
      best_inliers = num_inliers;
      bestF = F;
      best_inlier_idx = inliers;  // Copy (could use swap for slightly better perf)

      const double w = static_cast<double>(best_inliers) / static_cast<double>(N);
      const double wn = std::pow(w, static_cast<double>(sample_size));
      const double p_no_good = 1.0 - wn;

      if (p_no_good > std::numeric_limits<double>::epsilon() && p_no_good < 1.0) {
        const double k_real = std::log(1.0 - opt.confidence) / std::log(p_no_good);
        const int k_int = static_cast<int>(std::ceil(k_real));
        if (k_int > 0) max_iters = std::min(max_iters, k_int);
      }
    }

    iters_run++;
  }

  res.iterations_run = iters_run;

  if (best_inliers >= opt.min_inliers) {
    // Final refinement: reuse s1/s2 if they're big enough, otherwise allocate
    const size_t inlier_count = best_inlier_idx.size();
    std::vector<Eigen::Vector2d> in1;
    std::vector<Eigen::Vector2d> in2;
    in1.reserve(inlier_count);
    in2.reserve(inlier_count);
    for (const int idx : best_inlier_idx) {
      in1.push_back(x1_px[idx]);
      in2.push_back(x2_px[idx]);
    }
    res.F = EstimateFundamentalNormalized8Point(in1, in2);
    res.inlier_indices = std::move(best_inlier_idx);
    res.success = true;
    return res;
  }

  res.success = false;
  res.F = bestF;
  res.inlier_indices = std::move(best_inlier_idx);
  return res;
}

}  // namespace psynth::geometry