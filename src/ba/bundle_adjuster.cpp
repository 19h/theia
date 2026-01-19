#include <psynth/ba/bundle_adjuster.hpp>
#include <psynth/common.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace psynth::ba {

namespace {

struct ReprojectionResidual {
  ReprojectionResidual(double u_obs, double v_obs, double cx, double cy)
      : u_obs_(u_obs), v_obs_(v_obs), cx_(cx), cy_(cy) {}

  template <typename T>
  bool operator()(const T* const pose, const T* const intr, const T* const point, T* residuals) const {
    const T* aa = pose;
    const T* t = pose + 3;

    T p[3] = {T(point[0]), T(point[1]), T(point[2])};
    T p_rot[3];
    ceres::AngleAxisRotatePoint(aa, p, p_rot);

    p_rot[0] += t[0];
    p_rot[1] += t[1];
    p_rot[2] += t[2];

    const T X = p_rot[0];
    const T Y = p_rot[1];
    const T Z = p_rot[2];

    const T inv_z = T(1.0) / (Z + T(1e-12));

    const T x = X * inv_z;
    const T y = Y * inv_z;

    const T r2 = x * x + y * y;
    const T k1 = intr[1];
    const T k2 = intr[2];
    const T radial = T(1.0) + k1 * r2 + k2 * r2 * r2;

    const T f = intr[0];
    const T u = f * radial * x + T(cx_);
    const T v = f * radial * y + T(cy_);

    residuals[0] = u - T(u_obs_);
    residuals[1] = v - T(v_obs_);
    return true;
  }

  static ceres::CostFunction* Create(double u_obs, double v_obs, double cx, double cy) {
    return new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 6, 3, 3>(
        new ReprojectionResidual(u_obs, v_obs, cx, cy));
  }

  double u_obs_;
  double v_obs_;
  double cx_;
  double cy_;
};

Eigen::Vector3d AngleAxisFromRotationMatrix(const Eigen::Matrix3d& R) {
  Eigen::AngleAxisd aa(R);
  const double angle = aa.angle();
  const Eigen::Vector3d axis = aa.axis();
  if (!std::isfinite(angle)) return Eigen::Vector3d::Zero();
  return axis * angle;
}

Eigen::Matrix3d RotationMatrixFromAngleAxis(const double aa[3]) {
  double R[9];
  ceres::AngleAxisToRotationMatrix(aa, R);
  Eigen::Matrix3d M;
  M << R[0], R[1], R[2],
       R[3], R[4], R[5],
       R[6], R[7], R[8];
  return M;
}

}  // namespace

void BundleAdjuster::Adjust(const io::ImageDataset& dataset, sfm::Reconstruction* rec) const {
  PSYNTH_REQUIRE(rec, "BundleAdjuster::Adjust: rec is null");
  if (rec->cameras.size() < 2) return;

  std::vector<ImageId> cam_ids;
  cam_ids.reserve(rec->cameras.size());
  for (const auto& kv : rec->cameras) cam_ids.push_back(kv.first);
  std::sort(cam_ids.begin(), cam_ids.end());

  const int M = static_cast<int>(cam_ids.size());

  std::vector<std::array<double, 6>> pose_params(M);
  std::vector<std::array<double, 3>> intr_params(M);

  std::unordered_map<ImageId, int> cam_to_idx;
  cam_to_idx.reserve(cam_ids.size());

  for (int ci = 0; ci < M; ++ci) {
    const ImageId id = cam_ids[ci];
    cam_to_idx[id] = ci;

    const Camera& cam = rec->cameras.at(id);
    const Eigen::Vector3d aa = AngleAxisFromRotationMatrix(cam.pose.R);

    pose_params[ci] = {aa.x(), aa.y(), aa.z(), cam.pose.t.x(), cam.pose.t.y(), cam.pose.t.z()};
    intr_params[ci] = {cam.intr.f_px, cam.intr.k1, cam.intr.k2};
  }

  std::vector<int> track_ids;
  track_ids.reserve(rec->tracks.all().size());
  for (const auto& t : rec->tracks.all()) {
    if (t.triangulated) track_ids.push_back(t.id);
  }
  if (track_ids.empty()) return;

  std::vector<std::array<double, 3>> points(track_ids.size());
  std::unordered_map<int, int> track_to_pidx;
  track_to_pidx.reserve(track_ids.size());

  for (int pi = 0; pi < static_cast<int>(track_ids.size()); ++pi) {
    const Track& t = rec->tracks.all()[track_ids[pi]];
    points[pi] = {t.xyz.x(), t.xyz.y(), t.xyz.z()};
    track_to_pidx[t.id] = pi;
  }

  ceres::Problem problem;

  for (int ci = 0; ci < M; ++ci) {
    problem.AddParameterBlock(pose_params[ci].data(), 6);
    problem.AddParameterBlock(intr_params[ci].data(), 3);
    problem.SetParameterLowerBound(intr_params[ci].data(), 0, 1.0);
    if (!opt_.optimize_intrinsics) {
      problem.SetParameterBlockConstant(intr_params[ci].data());
    }
  }
  for (auto& p : points) {
    problem.AddParameterBlock(p.data(), 3);
  }

  std::vector<int> fixed_indices;
  fixed_indices.reserve(2);

  auto try_fix = [&](ImageId id) {
    auto it = cam_to_idx.find(id);
    if (it == cam_to_idx.end()) return;
    const int idx = it->second;
    if (std::find(fixed_indices.begin(), fixed_indices.end(), idx) == fixed_indices.end()) {
      fixed_indices.push_back(idx);
    }
  };

  try_fix(rec->gauge_fixed_0);
  try_fix(rec->gauge_fixed_1);

  if (fixed_indices.empty() && M >= 1) fixed_indices.push_back(0);
  if (fixed_indices.size() == 1 && M >= 2) fixed_indices.push_back(1);

  for (const int idx : fixed_indices) {
    problem.SetParameterBlockConstant(pose_params[idx].data());
  }

  int residuals_added = 0;
  for (const int tid : track_ids) {
    const Track& t = rec->tracks.all()[tid];
    const int pidx = track_to_pidx.at(tid);

    for (const auto& obs : t.observations) {
      const auto it = cam_to_idx.find(obs.image_id);
      if (it == cam_to_idx.end()) continue;

      const int ci = it->second;
      const double cx = dataset.image(obs.image_id).intr.cx_px;
      const double cy = dataset.image(obs.image_id).intr.cy_px;

      ceres::CostFunction* cost = ReprojectionResidual::Create(obs.u_px, obs.v_px, cx, cy);
      ceres::LossFunction* loss = nullptr;
      if (opt_.huber_loss_px > 0.0) {
        loss = new ceres::HuberLoss(opt_.huber_loss_px);
      }

      problem.AddResidualBlock(cost, loss, pose_params[ci].data(), intr_params[ci].data(), points[pidx].data());
      residuals_added++;
    }
  }

  if (residuals_added == 0) return;

  ceres::Solver::Options options;
  options.max_num_iterations = opt_.max_iterations;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cerr << "[psynth] BA: " << summary.BriefReport() << "\n";

  for (int ci = 0; ci < M; ++ci) {
    const ImageId id = cam_ids[ci];
    Camera& cam = rec->cameras[id];

    cam.pose.R = RotationMatrixFromAngleAxis(pose_params[ci].data());
    cam.pose.t = Eigen::Vector3d(pose_params[ci][3], pose_params[ci][4], pose_params[ci][5]);

    if (opt_.optimize_intrinsics) {
      cam.intr.f_px = intr_params[ci][0];
      cam.intr.k1 = intr_params[ci][1];
      cam.intr.k2 = intr_params[ci][2];
    }
  }

  for (int pi = 0; pi < static_cast<int>(track_ids.size()); ++pi) {
    const int tid = track_ids[pi];
    Track& t = rec->tracks.all_mut()[tid];
    t.xyz = Eigen::Vector3d(points[pi][0], points[pi][1], points[pi][2]);
  }
}

}  // namespace psynth::ba