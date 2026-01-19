#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <psynth/ba/bundle_adjuster.hpp>
#include <psynth/common.hpp>
#include <psynth/geometry/essential.hpp>
#include <psynth/geometry/pnp.hpp>
#include <psynth/geometry/triangulation.hpp>
#include <psynth/sfm/incremental_sfm.hpp>
#include <unordered_set>
#include <vector>

#ifdef PSYNTH_USE_OPENMP
#include <omp.h>
#endif

namespace psynth::sfm {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Computes median in-place (modifies input vector)
// Pass by reference to avoid unnecessary copy - caller should std::move if they don't need the data
double MedianInPlace(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[mid];
}

}  // namespace

IncrementalSfM::IncrementalSfM(const io::ImageDataset& dataset,
                               const std::vector<FeatureSet>& features,
                               const std::vector<VerifiedPair>& verified_pairs, Tracks tracks,
                               PipelineConfig cfg)
    : dataset_(dataset),
      features_(features),
      verified_pairs_(verified_pairs),
      tracks_(std::move(tracks)),
      cfg_(std::move(cfg)) {}

Reconstruction IncrementalSfM::Run() {
  Reconstruction rec;
  rec.tracks = std::move(tracks_);

  const int init_idx = SelectInitialPairIndex();
  PSYNTH_REQUIRE(init_idx >= 0, "SfM initialization failed: no suitable initial pair found");
  const VerifiedPair& vp = verified_pairs_[init_idx];

  InitializeFromPair(vp.pair.i, vp.pair.j, vp, &rec);

  const int N = dataset_.size();
  std::vector<bool> registered(N, false);
  for (const auto& kv : rec.cameras) {
    if (kv.first >= 0 && kv.first < N) registered[kv.first] = true;
  }

  while (true) {
    const ImageId next = SelectNextImage(rec, registered);
    if (next < 0) break;

    const bool ok = RegisterImage(next, &rec);
    registered[next] = ok;
    if (!ok) continue;

    TriangulateNewTracks(&rec);

    if (cfg_.sfm.bundle_every_n_images > 0 &&
        static_cast<int>(rec.cameras.size()) % cfg_.sfm.bundle_every_n_images == 0) {
      BundleAdjust(&rec);
      FilterTrackOutliers(&rec);
    }
  }

  BundleAdjust(&rec);
  FilterTrackOutliers(&rec);

  return rec;
}

int IncrementalSfM::SelectInitialPairIndex() const {
  const int num_pairs = static_cast<int>(verified_pairs_.size());
  if (num_pairs == 0) return -1;

  // Pre-allocate identity projection matrix (constant, no need to recreate each iteration)
  static const Eigen::Matrix<double, 3, 4> P1 =
      (Eigen::Matrix<double, 3, 4>() << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0).finished();

  int best_idx = -1;
  double best_score = -1.0;

#ifdef PSYNTH_USE_OPENMP
  const int max_threads = omp_get_max_threads();
  std::vector<int> local_best_idx(max_threads, -1);
  std::vector<double> local_best_score(max_threads, -1.0);

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();

    // Thread-local buffers to avoid repeated heap allocations
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<Eigen::Vector2d> x1_px, x2_px;
    std::vector<double> angles_deg;

#pragma omp for schedule(dynamic, 4)
    for (int idx = 0; idx < num_pairs; ++idx) {
      const auto& vp = verified_pairs_[idx];
      const int i = vp.pair.i;
      const int j = vp.pair.j;
      if (i < 0 || j < 0) continue;
      if (i >= dataset_.size() || j >= dataset_.size()) continue;

      const int nF = static_cast<int>(vp.inliers.size());
      if (nF < cfg_.sfm.init_min_inliers) continue;

      int nH = 0;
      {
        pts1.clear();
        pts2.clear();
        pts1.reserve(vp.inliers.size());
        pts2.reserve(vp.inliers.size());
        for (const auto& m : vp.inliers) {
          const auto& k1 = features_[i].keypoints[m.kp1];
          const auto& k2 = features_[j].keypoints[m.kp2];
          pts1.emplace_back(k1.pt.x, k1.pt.y);
          pts2.emplace_back(k2.pt.x, k2.pt.y);
        }
        cv::Mat mask;
        cv::Mat H =
            cv::findHomography(pts1, pts2, cv::RANSAC, cfg_.fund_ransac.inlier_threshold_px, mask);
        if (!H.empty() && !mask.empty()) nH = cv::countNonZero(mask);
      }

      const double homography_ratio =
          (nF > 0) ? (static_cast<double>(nH) / static_cast<double>(nF)) : 1.0;
      if (homography_ratio > cfg_.sfm.init_max_homography_ratio) continue;

      x1_px.clear();
      x2_px.clear();
      x1_px.reserve(vp.inliers.size());
      x2_px.reserve(vp.inliers.size());
      for (const auto& m : vp.inliers) {
        const auto& k1 = features_[i].keypoints[m.kp1];
        const auto& k2 = features_[j].keypoints[m.kp2];
        x1_px.emplace_back(k1.pt.x, k1.pt.y);
        x2_px.emplace_back(k2.pt.x, k2.pt.y);
      }

      const Intrinsics intr1 = dataset_.image(i).intr;
      const Intrinsics intr2 = dataset_.image(j).intr;

      const Eigen::Matrix3d E = geometry::EssentialFromFundamental(vp.F, intr1, intr2);
      const auto cheirality = geometry::ChooseRelativePoseCheirality(E, intr1, intr2, x1_px, x2_px);
      if (cheirality.num_positive_depth <= 0) continue;

      geometry::RelativePose rel = cheirality.pose;
      const double tnorm = rel.t.norm();
      if (tnorm <= std::numeric_limits<double>::epsilon()) continue;
      rel.t /= tnorm;

      const Eigen::Vector3d C1 = Eigen::Vector3d::Zero();
      const Eigen::Vector3d C2 = -rel.R.transpose() * rel.t;

      angles_deg.clear();
      angles_deg.reserve(200);

      Eigen::Matrix<double, 3, 4> P2;
      P2.block<3, 3>(0, 0) = rel.R;
      P2.col(3) = rel.t;

      const int max_test = std::min(static_cast<int>(vp.inliers.size()), 200);
      for (int k = 0; k < max_test; ++k) {
        const Eigen::Vector2d n1 = geometry::PixelToUndistortedNormalized(intr1, x1_px[k]);
        const Eigen::Vector2d n2 = geometry::PixelToUndistortedNormalized(intr2, x2_px[k]);

        Eigen::Vector3d X;
        if (!geometry::TriangulateDLT(P1, P2, n1, n2, &X)) continue;

        const double z1 = X.z();
        const double z2 = (rel.R * X + rel.t).z();
        if (z1 <= 0 || z2 <= 0) continue;

        const double ang_rad = geometry::TriangulationAngleRad(C1, C2, X);
        const double ang_deg = ang_rad * (180.0 / kPi);
        angles_deg.push_back(ang_deg);
      }

      const double median_angle_deg = MedianInPlace(angles_deg);
      if (median_angle_deg < cfg_.sfm.init_min_median_triangulation_angle_deg) continue;

      const double score = static_cast<double>(nF) * median_angle_deg;
      if (score > local_best_score[tid]) {
        local_best_score[tid] = score;
        local_best_idx[tid] = idx;
      }
    }
  }

  // Reduction: find global best from thread-local results
  for (int t = 0; t < max_threads; ++t) {
    if (local_best_score[t] > best_score) {
      best_score = local_best_score[t];
      best_idx = local_best_idx[t];
    }
  }
#else
  // Sequential fallback when OpenMP is not available
  std::vector<cv::Point2f> pts1, pts2;
  std::vector<Eigen::Vector2d> x1_px, x2_px;
  std::vector<double> angles_deg;

  for (int idx = 0; idx < num_pairs; ++idx) {
    const auto& vp = verified_pairs_[idx];
    const int i = vp.pair.i;
    const int j = vp.pair.j;
    if (i < 0 || j < 0) continue;
    if (i >= dataset_.size() || j >= dataset_.size()) continue;

    const int nF = static_cast<int>(vp.inliers.size());
    if (nF < cfg_.sfm.init_min_inliers) continue;

    int nH = 0;
    {
      pts1.clear();
      pts2.clear();
      pts1.reserve(vp.inliers.size());
      pts2.reserve(vp.inliers.size());
      for (const auto& m : vp.inliers) {
        const auto& k1 = features_[i].keypoints[m.kp1];
        const auto& k2 = features_[j].keypoints[m.kp2];
        pts1.emplace_back(k1.pt.x, k1.pt.y);
        pts2.emplace_back(k2.pt.x, k2.pt.y);
      }
      cv::Mat mask;
      cv::Mat H =
          cv::findHomography(pts1, pts2, cv::RANSAC, cfg_.fund_ransac.inlier_threshold_px, mask);
      if (!H.empty() && !mask.empty()) nH = cv::countNonZero(mask);
    }

    const double homography_ratio =
        (nF > 0) ? (static_cast<double>(nH) / static_cast<double>(nF)) : 1.0;
    if (homography_ratio > cfg_.sfm.init_max_homography_ratio) continue;

    x1_px.clear();
    x2_px.clear();
    x1_px.reserve(vp.inliers.size());
    x2_px.reserve(vp.inliers.size());
    for (const auto& m : vp.inliers) {
      const auto& k1 = features_[i].keypoints[m.kp1];
      const auto& k2 = features_[j].keypoints[m.kp2];
      x1_px.emplace_back(k1.pt.x, k1.pt.y);
      x2_px.emplace_back(k2.pt.x, k2.pt.y);
    }

    const Intrinsics intr1 = dataset_.image(i).intr;
    const Intrinsics intr2 = dataset_.image(j).intr;

    const Eigen::Matrix3d E = geometry::EssentialFromFundamental(vp.F, intr1, intr2);
    const auto cheirality = geometry::ChooseRelativePoseCheirality(E, intr1, intr2, x1_px, x2_px);
    if (cheirality.num_positive_depth <= 0) continue;

    geometry::RelativePose rel = cheirality.pose;
    const double tnorm = rel.t.norm();
    if (tnorm <= std::numeric_limits<double>::epsilon()) continue;
    rel.t /= tnorm;

    const Eigen::Vector3d C1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d C2 = -rel.R.transpose() * rel.t;

    angles_deg.clear();
    angles_deg.reserve(200);

    Eigen::Matrix<double, 3, 4> P2;
    P2.block<3, 3>(0, 0) = rel.R;
    P2.col(3) = rel.t;

    const int max_test = std::min(static_cast<int>(vp.inliers.size()), 200);
    for (int k = 0; k < max_test; ++k) {
      const Eigen::Vector2d n1 = geometry::PixelToUndistortedNormalized(intr1, x1_px[k]);
      const Eigen::Vector2d n2 = geometry::PixelToUndistortedNormalized(intr2, x2_px[k]);

      Eigen::Vector3d X;
      if (!geometry::TriangulateDLT(P1, P2, n1, n2, &X)) continue;

      const double z1 = X.z();
      const double z2 = (rel.R * X + rel.t).z();
      if (z1 <= 0 || z2 <= 0) continue;

      const double ang_rad = geometry::TriangulationAngleRad(C1, C2, X);
      const double ang_deg = ang_rad * (180.0 / kPi);
      angles_deg.push_back(ang_deg);
    }

    const double median_angle_deg = MedianInPlace(angles_deg);
    if (median_angle_deg < cfg_.sfm.init_min_median_triangulation_angle_deg) continue;

    const double score = static_cast<double>(nF) * median_angle_deg;
    if (score > best_score) {
      best_score = score;
      best_idx = idx;
    }
  }
#endif

  if (best_idx >= 0) return best_idx;

  // Fallback: pick pair with most inliers if no pair passes all criteria
  int fallback = -1;
  int best_inliers = -1;
  for (int idx = 0; idx < num_pairs; ++idx) {
    const int nF = static_cast<int>(verified_pairs_[idx].inliers.size());
    if (nF > best_inliers) {
      best_inliers = nF;
      fallback = idx;
    }
  }
  return fallback;
}

void IncrementalSfM::InitializeFromPair(ImageId i, ImageId j, const VerifiedPair& vp,
                                        Reconstruction* rec) {
  PSYNTH_REQUIRE(rec, "InitializeFromPair: rec is null");
  PSYNTH_REQUIRE(i >= 0 && j >= 0, "InitializeFromPair: invalid ids");

  rec->gauge_fixed_0 = i;
  rec->gauge_fixed_1 = j;

  const Intrinsics intr1 = dataset_.image(i).intr;
  const Intrinsics intr2 = dataset_.image(j).intr;

  std::vector<Eigen::Vector2d> x1_px;
  std::vector<Eigen::Vector2d> x2_px;
  x1_px.reserve(vp.inliers.size());
  x2_px.reserve(vp.inliers.size());

  std::unordered_set<uint64_t> inlier_pair_set;
  inlier_pair_set.reserve(vp.inliers.size() * 2);

  for (const auto& m : vp.inliers) {
    const auto& k1 = features_[i].keypoints[m.kp1];
    const auto& k2 = features_[j].keypoints[m.kp2];
    x1_px.emplace_back(k1.pt.x, k1.pt.y);
    x2_px.emplace_back(k2.pt.x, k2.pt.y);

    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(m.kp1)) << 32) ^
                         static_cast<uint64_t>(static_cast<uint32_t>(m.kp2));
    inlier_pair_set.insert(key);
  }

  const Eigen::Matrix3d E = geometry::EssentialFromFundamental(vp.F, intr1, intr2);
  const auto cheirality = geometry::ChooseRelativePoseCheirality(E, intr1, intr2, x1_px, x2_px);
  PSYNTH_REQUIRE(cheirality.num_positive_depth > 0, "InitializeFromPair: cheirality failed");

  geometry::RelativePose rel = cheirality.pose;
  const double tnorm = rel.t.norm();
  PSYNTH_REQUIRE(tnorm > std::numeric_limits<double>::epsilon(),
                 "InitializeFromPair: degenerate translation");
  rel.t /= tnorm;

  Camera cam_i;
  cam_i.intr = intr1;
  cam_i.pose.R = Eigen::Matrix3d::Identity();
  cam_i.pose.t = Eigen::Vector3d::Zero();
  rec->cameras[i] = cam_i;

  Camera cam_j;
  cam_j.intr = intr2;
  cam_j.pose.R = rel.R;
  cam_j.pose.t = rel.t;
  rec->cameras[j] = cam_j;

  const Eigen::Matrix<double, 3, 4> P1 =
      (Eigen::Matrix<double, 3, 4>() << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0).finished();

  Eigen::Matrix<double, 3, 4> P2;
  P2.block<3, 3>(0, 0) = rel.R;
  P2.col(3) = rel.t;

  cv::Mat color_img = dataset_.ReadColor(i);

  int triangulated = 0;
  for (auto& track : rec->tracks.all_mut()) {
    const Observation* oi = rec->tracks.FindObservation(track.id, i);
    const Observation* oj = rec->tracks.FindObservation(track.id, j);
    if (!oi || !oj) continue;

    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(oi->keypoint_id)) << 32) ^
                         static_cast<uint64_t>(static_cast<uint32_t>(oj->keypoint_id));
    if (inlier_pair_set.find(key) == inlier_pair_set.end()) continue;

    const Eigen::Vector2d n1 =
        geometry::PixelToUndistortedNormalized(intr1, Eigen::Vector2d(oi->u_px, oi->v_px));
    const Eigen::Vector2d n2 =
        geometry::PixelToUndistortedNormalized(intr2, Eigen::Vector2d(oj->u_px, oj->v_px));

    Eigen::Vector3d X;
    if (!geometry::TriangulateDLT(P1, P2, n1, n2, &X)) continue;

    const double z1 = X.z();
    const double z2 = (rel.R * X + rel.t).z();
    if (z1 <= 0 || z2 <= 0) continue;

    const double e1 =
        geometry::ReprojectionErrorPx(rec->cameras[i], X, Eigen::Vector2d(oi->u_px, oi->v_px));
    const double e2 =
        geometry::ReprojectionErrorPx(rec->cameras[j], X, Eigen::Vector2d(oj->u_px, oj->v_px));
    if (e1 > cfg_.sfm.max_reprojection_error_px || e2 > cfg_.sfm.max_reprojection_error_px)
      continue;

    track.triangulated = true;
    track.xyz = X;

    const int u =
        std::max(0, std::min(color_img.cols - 1, static_cast<int>(std::lround(oi->u_px))));
    const int v =
        std::max(0, std::min(color_img.rows - 1, static_cast<int>(std::lround(oi->v_px))));
    track.color_bgr = color_img.at<cv::Vec3b>(v, u);

    triangulated++;
  }

  rec->tracks.RebuildIndex();

  std::cerr << "[psynth] init pair (" << i << "," << j
            << ") cameras=2 tracks=" << rec->tracks.all().size() << " triangulated=" << triangulated
            << "\n";
}

ImageId IncrementalSfM::SelectNextImage(const Reconstruction& rec,
                                        const std::vector<bool>& registered) const {
  const int N = dataset_.size();
  ImageId best = -1;
  int best_count = -1;

  for (ImageId img = 0; img < N; ++img) {
    if (registered[img]) continue;

    int count = 0;
    for (const int tid : rec.tracks.tracks_in_image(img)) {
      const auto& t = rec.tracks.all()[tid];
      if (!t.triangulated) continue;
      count++;
    }

    if (count > best_count) {
      best_count = count;
      best = img;
    }
  }

  if (best_count < cfg_.sfm.min_pnp_correspondences) return -1;
  return best;
}

bool IncrementalSfM::RegisterImage(ImageId img, Reconstruction* rec) {
  PSYNTH_REQUIRE(rec, "RegisterImage: rec is null");
  PSYNTH_REQUIRE(img >= 0 && img < dataset_.size(), "RegisterImage: invalid image id");

  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> uv;
  X.reserve(1024);
  uv.reserve(1024);

  for (const int tid : rec->tracks.tracks_in_image(img)) {
    const auto& t = rec->tracks.all()[tid];
    if (!t.triangulated) continue;

    const Observation* o = rec->tracks.FindObservation(tid, img);
    if (!o) continue;

    X.push_back(t.xyz);
    uv.emplace_back(o->u_px, o->v_px);
  }

  if (static_cast<int>(X.size()) < cfg_.sfm.min_pnp_correspondences) return false;

  geometry::PnPRansacOptions opt;
  opt.max_iterations = 2000;
  opt.reprojection_error_px = cfg_.sfm.max_reprojection_error_px;
  opt.confidence = 0.99;
  opt.min_inliers = cfg_.sfm.min_pnp_inliers;
  opt.use_ap3p = true;

  const Intrinsics intr = dataset_.image(img).intr;
  const auto pnp = geometry::SolvePnPRansacOpenCV(intr, X, uv, opt);
  if (!pnp.success) return false;

  Camera cam;
  cam.intr = intr;
  cam.pose.R = pnp.R;
  cam.pose.t = pnp.t;
  rec->cameras[img] = cam;

  std::cerr << "[psynth] registered image " << img << " with inliers=" << pnp.inlier_indices.size()
            << " / " << X.size() << "\n";

  return true;
}

void IncrementalSfM::TriangulateNewTracks(Reconstruction* rec) {
  PSYNTH_REQUIRE(rec, "TriangulateNewTracks: rec is null");

  int new_points = 0;

  // Pre-cache camera centers for all registered cameras to avoid recomputation in O(K^2) baseline
  // selection
  std::unordered_map<ImageId, Eigen::Vector3d> camera_centers;
  camera_centers.reserve(rec->cameras.size());
  for (const auto& kv : rec->cameras) {
    camera_centers[kv.first] = CameraCenterWorld(kv.second.pose);
  }

  // Image color cache to avoid repeated disk I/O
  // CRITICAL: Without caching, ReadColor() is called for EVERY newly triangulated track,
  // causing potentially thousands of redundant image reads
  std::unordered_map<ImageId, cv::Mat> color_cache;

  // Reusable buffer for observations
  std::vector<const Observation*> obs_reg;

  for (auto& track : rec->tracks.all_mut()) {
    if (track.triangulated) continue;

    // Gather registered observations
    obs_reg.clear();
    obs_reg.reserve(track.observations.size());
    for (const auto& obs : track.observations) {
      if (rec->cameras.find(obs.image_id) != rec->cameras.end()) {
        obs_reg.push_back(&obs);
      }
    }
    if (obs_reg.size() < 2) continue;

    // Find widest baseline pair using cached camera centers
    double best_baseline2 = -1.0;
    const Observation* oa = nullptr;
    const Observation* ob = nullptr;

    const size_t n_obs = obs_reg.size();
    for (size_t a = 0; a < n_obs; ++a) {
      const Eigen::Vector3d& Ca = camera_centers.at(obs_reg[a]->image_id);
      for (size_t b = a + 1; b < n_obs; ++b) {
        const Eigen::Vector3d& Cb = camera_centers.at(obs_reg[b]->image_id);
        const double d2 = (Ca - Cb).squaredNorm();
        if (d2 > best_baseline2) {
          best_baseline2 = d2;
          oa = obs_reg[a];
          ob = obs_reg[b];
        }
      }
    }
    if (!oa || !ob) continue;

    const auto& cam1 = rec->cameras.at(oa->image_id);
    const auto& cam2 = rec->cameras.at(ob->image_id);

    const Eigen::Vector2d x1 =
        geometry::PixelToUndistortedNormalized(cam1.intr, Eigen::Vector2d(oa->u_px, oa->v_px));
    const Eigen::Vector2d x2 =
        geometry::PixelToUndistortedNormalized(cam2.intr, Eigen::Vector2d(ob->u_px, ob->v_px));

    const Eigen::Matrix<double, 3, 4> P1 = geometry::ExtrinsicsMatrix(cam1.pose);
    const Eigen::Matrix<double, 3, 4> P2 = geometry::ExtrinsicsMatrix(cam2.pose);

    Eigen::Vector3d X;
    if (!geometry::TriangulateDLT(P1, P2, x1, x2, &X)) continue;

    const double z1 = (cam1.pose.R * X + cam1.pose.t).z();
    const double z2 = (cam2.pose.R * X + cam2.pose.t).z();
    if (z1 <= 0 || z2 <= 0) continue;

    const double e1 = geometry::ReprojectionErrorPx(cam1, X, Eigen::Vector2d(oa->u_px, oa->v_px));
    const double e2 = geometry::ReprojectionErrorPx(cam2, X, Eigen::Vector2d(ob->u_px, ob->v_px));
    if (e1 > cfg_.sfm.max_reprojection_error_px || e2 > cfg_.sfm.max_reprojection_error_px)
      continue;

    track.triangulated = true;
    track.xyz = X;

    // Use cached image for color lookup - avoids repeated disk I/O
    auto& img_color = color_cache[oa->image_id];
    if (img_color.empty()) {
      img_color = dataset_.ReadColor(oa->image_id);
    }
    const int u =
        std::max(0, std::min(img_color.cols - 1, static_cast<int>(std::lround(oa->u_px))));
    const int v =
        std::max(0, std::min(img_color.rows - 1, static_cast<int>(std::lround(oa->v_px))));
    track.color_bgr = img_color.at<cv::Vec3b>(v, u);

    new_points++;
  }

  if (new_points > 0) {
    rec->tracks.RebuildIndex();
    std::cerr << "[psynth] triangulated new tracks: " << new_points << "\n";
  }
}

void IncrementalSfM::BundleAdjust(Reconstruction* rec) {
  PSYNTH_REQUIRE(rec, "BundleAdjust: rec is null");
  if (rec->cameras.size() < 2) return;

  ba::BundleAdjusterOptions opt;
  opt.max_iterations = cfg_.sfm.ba_max_iterations;
  opt.huber_loss_px = cfg_.sfm.ba_huber_loss_px;
  opt.optimize_intrinsics = cfg_.sfm.ba_optimize_intrinsics;

  ba::BundleAdjuster ba(opt);
  ba.Adjust(dataset_, rec);
}

void IncrementalSfM::FilterTrackOutliers(Reconstruction* rec) {
  PSYNTH_REQUIRE(rec, "FilterTrackOutliers: rec is null");

  auto& all_tracks = rec->tracks.all_mut();
  const size_t num_tracks = all_tracks.size();

  std::atomic<int> removed_obs{0};
  std::atomic<int> disabled_tracks{0};

#ifdef PSYNTH_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
  for (size_t ti = 0; ti < num_tracks; ++ti) {
    auto& track = all_tracks[ti];
    if (!track.triangulated) continue;

    std::vector<Observation> kept;
    kept.reserve(track.observations.size());

    int local_removed = 0;
    for (const auto& obs : track.observations) {
      const auto it_cam = rec->cameras.find(obs.image_id);
      if (it_cam == rec->cameras.end()) {
        kept.push_back(obs);
        continue;
      }

      const Eigen::Vector2d uv(obs.u_px, obs.v_px);
      const double e = geometry::ReprojectionErrorPx(it_cam->second, track.xyz, uv);
      if (e <= cfg_.sfm.max_reprojection_error_px) {
        kept.push_back(obs);
      } else {
        local_removed++;
      }
    }

    track.observations = std::move(kept);
    removed_obs.fetch_add(local_removed, std::memory_order_relaxed);

    int reg_after = 0;
    for (const auto& obs : track.observations) {
      if (rec->cameras.find(obs.image_id) != rec->cameras.end()) reg_after++;
    }
    if (reg_after < 2) {
      track.triangulated = false;
      disabled_tracks.fetch_add(1, std::memory_order_relaxed);
    }
  }

  rec->tracks.RebuildIndex();

  const int total_removed = removed_obs.load();
  const int total_disabled = disabled_tracks.load();
  if (total_removed > 0 || total_disabled > 0) {
    std::cerr << "[psynth] outlier filter: removed_obs=" << total_removed
              << " disabled_tracks=" << total_disabled << "\n";
  }
}

}  // namespace psynth::sfm