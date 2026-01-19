#include <psynth/geometry/fundamental.hpp>
#include <psynth/common.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
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

  n.T << scale, 0.0, -scale * centroid.x(),
         0.0, scale, -scale * centroid.y(),
         0.0, 0.0, 1.0;

  n.x.reserve(N);
  for (const auto& p : pts) {
    n.x.emplace_back(n.T * Eigen::Vector3d(p.x(), p.y(), 1.0));
  }
  return n;
}

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
  F << f(0), f(1), f(2),
       f(3), f(4), f(5),
       f(6), f(7), f(8);

  Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d s = svdF.singularValues();
  s(2) = 0.0;
  const Eigen::Matrix3d Frank2 = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

  Eigen::Matrix3d Fdenorm = n2.T.transpose() * Frank2 * n1.T;

  const double frob = Fdenorm.norm();
  if (frob > 0.0) Fdenorm /= frob;

  return Fdenorm;
}

double SampsonDistanceSquared(const Eigen::Matrix3d& F, const Eigen::Vector2d& x1_px, const Eigen::Vector2d& x2_px) {
  const Eigen::Vector3d x1(x1_px.x(), x1_px.y(), 1.0);
  const Eigen::Vector3d x2(x2_px.x(), x2_px.y(), 1.0);

  const Eigen::Vector3d Fx1 = F * x1;
  const Eigen::Vector3d Ftx2 = F.transpose() * x2;
  const double x2tFx1 = x2.transpose() * F * x1;

  const double denom =
      Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) + Ftx2(0) * Ftx2(0) + Ftx2(1) * Ftx2(1);

  if (denom <= std::numeric_limits<double>::epsilon()) {
    return std::numeric_limits<double>::infinity();
  }
  return (x2tFx1 * x2tFx1) / denom;
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
  const int sample_size = 8;

  int best_inliers = -1;
  Eigen::Matrix3d bestF = Eigen::Matrix3d::Zero();
  std::vector<int> best_inlier_idx;

  int max_iters = opt.max_iterations;
  int iters_run = 0;

  while (iters_run < max_iters) {
    const std::vector<int> sample = SampleUniqueIndices(rng, N, sample_size);

    std::vector<Eigen::Vector2d> s1;
    std::vector<Eigen::Vector2d> s2;
    s1.reserve(sample_size);
    s2.reserve(sample_size);
    for (const int idx : sample) {
      s1.push_back(x1_px[idx]);
      s2.push_back(x2_px[idx]);
    }

    Eigen::Matrix3d F;
    try {
      F = EstimateFundamentalNormalized8Point(s1, s2);
    } catch (...) {
      iters_run++;
      continue;
    }

    std::vector<int> inliers;
    inliers.reserve(N);
    for (int i = 0; i < N; ++i) {
      const double d2 = SampsonDistanceSquared(F, x1_px[i], x2_px[i]);
      if (d2 < thresh2) inliers.push_back(i);
    }

    if (static_cast<int>(inliers.size()) > best_inliers) {
      best_inliers = static_cast<int>(inliers.size());
      bestF = F;
      best_inlier_idx = std::move(inliers);

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
    std::vector<Eigen::Vector2d> in1;
    std::vector<Eigen::Vector2d> in2;
    in1.reserve(best_inlier_idx.size());
    in2.reserve(best_inlier_idx.size());
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