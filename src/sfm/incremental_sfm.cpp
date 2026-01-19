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
  std::vector<bool> tried(N, false);  // Track images we've attempted to register
  for (const auto& kv : rec.cameras) {
    if (kv.first >= 0 && kv.first < N) {
      registered[kv.first] = true;
      tried[kv.first] = true;
    }
  }

  while (true) {
    ImageId next = SelectNextImage(rec, registered, tried);

    // If no image can be registered via normal PnP, try via verified pair
    if (next < 0) {
      next = TryRegisterViaVerifiedPair(registered, tried, &rec);
      if (next >= 0) {
        registered[next] = true;
        tried[next] = true;
        // Triangulate new tracks after relative pose registration
        TriangulateNewTracks(&rec);
        // Continue to next iteration to try more registrations
        continue;
      }
      break;  // No more images can be registered
    }

    tried[next] = true;  // Mark as tried BEFORE attempting registration
    const bool ok = RegisterImage(next, &rec);
    if (!ok) {
      std::cerr << "[psynth] failed to register image " << next << "\n";
      continue;
    }
    registered[next] = true;

    // Count triangulated tracks before
    size_t tracks_before = 0;
    for (const auto& t : rec.tracks.all()) {
      if (t.triangulated) tracks_before++;
    }

    TriangulateNewTracks(&rec);

    // Count triangulated tracks after
    size_t tracks_after = 0;
    for (const auto& t : rec.tracks.all()) {
      if (t.triangulated) tracks_after++;
    }

    // Reset tried list if we triangulated new points - failed images might now have enough correspondences
    if (tracks_after > tracks_before) {
      std::fill(tried.begin(), tried.end(), false);
      for (ImageId img = 0; img < N; ++img) {
        if (registered[img]) tried[img] = true;
      }
    }

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

      // Build temporary cameras for reprojection check
      Camera test_cam1;
      test_cam1.intr = intr1;
      test_cam1.pose.R = Eigen::Matrix3d::Identity();
      test_cam1.pose.t = Eigen::Vector3d::Zero();

      Camera test_cam2;
      test_cam2.intr = intr2;
      test_cam2.pose.R = rel.R;
      test_cam2.pose.t = rel.t;

      const int max_test = std::min(static_cast<int>(vp.inliers.size()), 200);
      int good_reproj = 0;
      for (int k = 0; k < max_test; ++k) {
        const Eigen::Vector2d n1 = geometry::PixelToUndistortedNormalized(intr1, x1_px[k]);
        const Eigen::Vector2d n2 = geometry::PixelToUndistortedNormalized(intr2, x2_px[k]);

        Eigen::Vector3d X;
        if (!geometry::TriangulateDLT(P1, P2, n1, n2, &X)) continue;

        const double z1 = X.z();
        const double z2 = (rel.R * X + rel.t).z();
        if (z1 <= 0 || z2 <= 0) continue;

        // Check reprojection error - must be reasonable for both views
        const double e1 = geometry::ReprojectionErrorPx(test_cam1, X, x1_px[k]);
        const double e2 = geometry::ReprojectionErrorPx(test_cam2, X, x2_px[k]);
        if (e1 > cfg_.sfm.max_reprojection_error_px || e2 > cfg_.sfm.max_reprojection_error_px) continue;

        good_reproj++;

        const double ang_rad = geometry::TriangulationAngleRad(C1, C2, X);
        const double ang_deg = ang_rad * (180.0 / kPi);
        angles_deg.push_back(ang_deg);
      }

      // Require at least 50% of tested points to have good reprojection
      if (good_reproj < max_test / 2) continue;

      const double median_angle_deg = MedianInPlace(angles_deg);
      if (median_angle_deg < cfg_.sfm.init_min_median_triangulation_angle_deg) continue;

      // Score favors both inlier count and reprojection quality
      const double score = static_cast<double>(good_reproj) * median_angle_deg;

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

    // Build temporary cameras for reprojection check
    Camera test_cam1;
    test_cam1.intr = intr1;
    test_cam1.pose.R = Eigen::Matrix3d::Identity();
    test_cam1.pose.t = Eigen::Vector3d::Zero();

    Camera test_cam2;
    test_cam2.intr = intr2;
    test_cam2.pose.R = rel.R;
    test_cam2.pose.t = rel.t;

    const int max_test = std::min(static_cast<int>(vp.inliers.size()), 200);
    int good_reproj = 0;
    for (int k = 0; k < max_test; ++k) {
      const Eigen::Vector2d n1 = geometry::PixelToUndistortedNormalized(intr1, x1_px[k]);
      const Eigen::Vector2d n2 = geometry::PixelToUndistortedNormalized(intr2, x2_px[k]);

      Eigen::Vector3d X;
      if (!geometry::TriangulateDLT(P1, P2, n1, n2, &X)) continue;

      const double z1 = X.z();
      const double z2 = (rel.R * X + rel.t).z();
      if (z1 <= 0 || z2 <= 0) continue;

      // Check reprojection error - must be reasonable for both views
      const double e1 = geometry::ReprojectionErrorPx(test_cam1, X, x1_px[k]);
      const double e2 = geometry::ReprojectionErrorPx(test_cam2, X, x2_px[k]);
      if (e1 > cfg_.sfm.max_reprojection_error_px || e2 > cfg_.sfm.max_reprojection_error_px) continue;

      good_reproj++;

      const double ang_rad = geometry::TriangulationAngleRad(C1, C2, X);
      const double ang_deg = ang_rad * (180.0 / kPi);
      angles_deg.push_back(ang_deg);
    }

    // Require at least 50% of tested points to have good reprojection
    if (good_reproj < max_test / 2) continue;

    const double median_angle_deg = MedianInPlace(angles_deg);
    if (median_angle_deg < cfg_.sfm.init_min_median_triangulation_angle_deg) continue;

    // Score favors both inlier count and reprojection quality
    const double score = static_cast<double>(good_reproj) * median_angle_deg;

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
  int dbg_no_obs = 0, dbg_no_inlier = 0, dbg_tri_fail = 0, dbg_depth_fail = 0, dbg_reproj_fail = 0;
  for (auto& track : rec->tracks.all_mut()) {
    const Observation* oi = rec->tracks.FindObservation(track.id, i);
    const Observation* oj = rec->tracks.FindObservation(track.id, j);
    if (!oi || !oj) { dbg_no_obs++; continue; }

    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(oi->keypoint_id)) << 32) ^
                         static_cast<uint64_t>(static_cast<uint32_t>(oj->keypoint_id));
    if (inlier_pair_set.find(key) == inlier_pair_set.end()) { dbg_no_inlier++; continue; }

    const Eigen::Vector2d n1 =
        geometry::PixelToUndistortedNormalized(intr1, Eigen::Vector2d(oi->u_px, oi->v_px));
    const Eigen::Vector2d n2 =
        geometry::PixelToUndistortedNormalized(intr2, Eigen::Vector2d(oj->u_px, oj->v_px));

    Eigen::Vector3d X;
    if (!geometry::TriangulateDLT(P1, P2, n1, n2, &X)) { dbg_tri_fail++; continue; }

    const double z1 = X.z();
    const double z2 = (rel.R * X + rel.t).z();
    if (z1 <= 0 || z2 <= 0) { dbg_depth_fail++; continue; }

    const double e1 =
        geometry::ReprojectionErrorPx(rec->cameras[i], X, Eigen::Vector2d(oi->u_px, oi->v_px));
    const double e2 =
        geometry::ReprojectionErrorPx(rec->cameras[j], X, Eigen::Vector2d(oj->u_px, oj->v_px));
    if (e1 > cfg_.sfm.max_reprojection_error_px || e2 > cfg_.sfm.max_reprojection_error_px) {
      dbg_reproj_fail++;
      continue;
    }

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
            << " (no_obs=" << dbg_no_obs << " no_inlier=" << dbg_no_inlier
            << " tri_fail=" << dbg_tri_fail << " depth_fail=" << dbg_depth_fail
            << " reproj_fail=" << dbg_reproj_fail << ")\n";
}

ImageId IncrementalSfM::SelectNextImage(const Reconstruction& rec,
                                        const std::vector<bool>& registered,
                                        const std::vector<bool>& tried) const {
  const int N = dataset_.size();
  ImageId best = -1;
  int best_count = -1;

  // For debugging: track why images are skipped
  int num_registered = 0, num_tried = 0, num_below_threshold = 0;

  for (ImageId img = 0; img < N; ++img) {
    if (registered[img]) {
      num_registered++;
      continue;
    }
    if (tried[img]) {
      num_tried++;
      continue;
    }

    int count = 0;
    for (const int tid : rec.tracks.tracks_in_image(img)) {
      const auto& t = rec.tracks.all()[tid];
      if (!t.triangulated) continue;
      count++;
    }

    if (count < cfg_.sfm.min_pnp_correspondences) {
      num_below_threshold++;
    }

    if (count > best_count) {
      best_count = count;
      best = img;
    }
  }

  if (best_count < cfg_.sfm.min_pnp_correspondences) {
    std::cerr << "[psynth] SelectNextImage: no candidate (registered=" << num_registered
              << " tried=" << num_tried << " below_threshold=" << num_below_threshold
              << " best_count=" << best_count
              << " min_correspondences=" << cfg_.sfm.min_pnp_correspondences << ")\n";
    return -1;
  }
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
    // Optimization: For tracks with many observations (>10), use heuristic instead of O(n²)
    double best_baseline2 = -1.0;
    const Observation* oa = nullptr;
    const Observation* ob = nullptr;

    const size_t n_obs = obs_reg.size();

    if (n_obs <= 10) {
      // Small number of observations: full O(n²) search is fast
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
    } else {
      // Many observations: use O(n) heuristic based on bounding box extremes
      // Find cameras at min/max x, y, z positions - one of these pairs likely has widest baseline
      const Observation* min_x = obs_reg[0];
      const Observation* max_x = obs_reg[0];
      const Observation* min_y = obs_reg[0];
      const Observation* max_y = obs_reg[0];
      const Observation* min_z = obs_reg[0];
      const Observation* max_z = obs_reg[0];

      double mx = camera_centers.at(obs_reg[0]->image_id).x();
      double Mx = mx, my = camera_centers.at(obs_reg[0]->image_id).y();
      double My = my, mz = camera_centers.at(obs_reg[0]->image_id).z();
      double Mz = mz;

      for (size_t i = 1; i < n_obs; ++i) {
        const Eigen::Vector3d& C = camera_centers.at(obs_reg[i]->image_id);
        if (C.x() < mx) { mx = C.x(); min_x = obs_reg[i]; }
        if (C.x() > Mx) { Mx = C.x(); max_x = obs_reg[i]; }
        if (C.y() < my) { my = C.y(); min_y = obs_reg[i]; }
        if (C.y() > My) { My = C.y(); max_y = obs_reg[i]; }
        if (C.z() < mz) { mz = C.z(); min_z = obs_reg[i]; }
        if (C.z() > Mz) { Mz = C.z(); max_z = obs_reg[i]; }
      }

      // Check the 3 candidate pairs (min/max for each axis)
      auto check_pair = [&](const Observation* a, const Observation* b) {
        if (a == b) return;
        const Eigen::Vector3d& Ca = camera_centers.at(a->image_id);
        const Eigen::Vector3d& Cb = camera_centers.at(b->image_id);
        const double d2 = (Ca - Cb).squaredNorm();
        if (d2 > best_baseline2) {
          best_baseline2 = d2;
          oa = a;
          ob = b;
        }
      };

      check_pair(min_x, max_x);
      check_pair(min_y, max_y);
      check_pair(min_z, max_z);
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

  const double max_err = cfg_.sfm.max_reprojection_error_px;

#ifdef PSYNTH_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
  for (size_t ti = 0; ti < num_tracks; ++ti) {
    auto& track = all_tracks[ti];
    if (!track.triangulated) continue;

    const int num_obs = static_cast<int>(track.observations.size());
    if (num_obs == 0) continue;

    // Thread-local buffers for batch computation
    thread_local std::vector<const Camera*> cam_ptrs;
    thread_local std::vector<double> obs_uv;
    thread_local std::vector<double> errors;
    thread_local std::vector<int> obs_indices;  // Maps batch index to original obs index

    cam_ptrs.clear();
    obs_uv.clear();
    obs_indices.clear();
    cam_ptrs.reserve(num_obs);
    obs_uv.reserve(num_obs * 2);
    obs_indices.reserve(num_obs);

    // Collect observations that have registered cameras
    for (int oi = 0; oi < num_obs; ++oi) {
      const auto& obs = track.observations[oi];
      const auto it_cam = rec->cameras.find(obs.image_id);
      if (it_cam != rec->cameras.end()) {
        cam_ptrs.push_back(&it_cam->second);
        obs_uv.push_back(obs.u_px);
        obs_uv.push_back(obs.v_px);
        obs_indices.push_back(oi);
      }
    }

    const int num_to_check = static_cast<int>(cam_ptrs.size());

    // Batch compute reprojection errors
    errors.resize(num_to_check);
    if (num_to_check > 0) {
      geometry::SinglePointMultiCameraErrors(track.xyz, cam_ptrs.data(), obs_uv.data(),
                                             num_to_check, errors.data());
    }

    // Build kept observations
    std::vector<Observation> kept;
    kept.reserve(num_obs);

    int local_removed = 0;
    int batch_idx = 0;

    for (int oi = 0; oi < num_obs; ++oi) {
      const auto& obs = track.observations[oi];
      const auto it_cam = rec->cameras.find(obs.image_id);

      if (it_cam == rec->cameras.end()) {
        // Unregistered camera - keep observation
        kept.push_back(obs);
      } else {
        // Check error from batch computation
        if (errors[batch_idx] <= max_err) {
          kept.push_back(obs);
        } else {
          local_removed++;
        }
        batch_idx++;
      }
    }

    track.observations = std::move(kept);
    removed_obs.fetch_add(local_removed, std::memory_order_relaxed);

    // Count remaining registered observations
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

ImageId IncrementalSfM::TryRegisterViaVerifiedPair(const std::vector<bool>& registered,
                                                    const std::vector<bool>& tried,
                                                    Reconstruction* rec) {
  PSYNTH_REQUIRE(rec, "TryRegisterViaVerifiedPair: rec is null");

  // Find the best verified pair where one image is registered and one is not
  int best_pair_idx = -1;
  int best_inliers = -1;
  ImageId best_registered = -1;
  ImageId best_unregistered = -1;

  for (int pi = 0; pi < static_cast<int>(verified_pairs_.size()); ++pi) {
    const auto& vp = verified_pairs_[pi];
    const ImageId i = vp.pair.i;
    const ImageId j = vp.pair.j;

    const bool i_reg = (i >= 0 && i < static_cast<int>(registered.size()) && registered[i]);
    const bool j_reg = (j >= 0 && j < static_cast<int>(registered.size()) && registered[j]);
    const bool i_tried = (i >= 0 && i < static_cast<int>(tried.size()) && tried[i]);
    const bool j_tried = (j >= 0 && j < static_cast<int>(tried.size()) && tried[j]);

    // We need exactly one registered and one unregistered (and not tried)
    if (i_reg && !j_reg && !j_tried) {
      const int n_inliers = static_cast<int>(vp.inliers.size());
      if (n_inliers > best_inliers) {
        best_inliers = n_inliers;
        best_pair_idx = pi;
        best_registered = i;
        best_unregistered = j;
      }
    } else if (j_reg && !i_reg && !i_tried) {
      const int n_inliers = static_cast<int>(vp.inliers.size());
      if (n_inliers > best_inliers) {
        best_inliers = n_inliers;
        best_pair_idx = pi;
        best_registered = j;
        best_unregistered = i;
      }
    }
  }

  if (best_pair_idx < 0) {
    std::cerr << "[psynth] TryRegisterViaVerifiedPair: no suitable pair found\n";
    return -1;
  }

  const auto& vp = verified_pairs_[best_pair_idx];
  std::cerr << "[psynth] TryRegisterViaVerifiedPair: attempting (" << best_registered << ", "
            << best_unregistered << ") with " << best_inliers << " inliers\n";

  // Get the registered camera - use its optimized intrinsics
  const auto& cam_reg = rec->cameras.at(best_registered);
  const Intrinsics& intr_reg = cam_reg.intr;  // Use optimized intrinsics from reconstruction
  const Intrinsics& intr_unreg = dataset_.image(best_unregistered).intr;

  // Build pixel coordinates from inliers
  std::vector<Eigen::Vector2d> x_reg_px, x_unreg_px;
  x_reg_px.reserve(vp.inliers.size());
  x_unreg_px.reserve(vp.inliers.size());

  for (const auto& m : vp.inliers) {
    const auto& k1 = features_[vp.pair.i].keypoints[m.kp1];
    const auto& k2 = features_[vp.pair.j].keypoints[m.kp2];
    if (vp.pair.i == best_registered) {
      x_reg_px.emplace_back(k1.pt.x, k1.pt.y);
      x_unreg_px.emplace_back(k2.pt.x, k2.pt.y);
    } else {
      x_reg_px.emplace_back(k2.pt.x, k2.pt.y);
      x_unreg_px.emplace_back(k1.pt.x, k1.pt.y);
    }
  }

  // Compute essential matrix and relative pose
  // Important: F is computed for pair (i, j) with convention x_j^T * F * x_i = 0
  // If we swap the image order (registered != i), we need to transpose F
  const Eigen::Matrix3d F_ordered =
      (vp.pair.i == best_registered) ? vp.F : vp.F.transpose();
  const Eigen::Matrix3d E =
      geometry::EssentialFromFundamental(F_ordered, intr_reg, intr_unreg);
  const auto cheirality = geometry::ChooseRelativePoseCheirality(E, intr_reg, intr_unreg, x_reg_px, x_unreg_px);
  if (cheirality.num_positive_depth <= 0) {
    std::cerr << "[psynth] TryRegisterViaVerifiedPair: cheirality test failed\n";
    return -1;
  }

  geometry::RelativePose rel = cheirality.pose;
  const double tnorm = rel.t.norm();
  if (tnorm <= std::numeric_limits<double>::epsilon()) {
    std::cerr << "[psynth] TryRegisterViaVerifiedPair: degenerate translation\n";
    return -1;
  }

  // The relative pose is: x_unreg_cam = rel.R * x_reg_cam + rel.t
  // We need to compose this with the registered camera's pose to get world-to-camera for unreg

  // Registered camera: X_reg_cam = R_reg * X_world + t_reg
  // Relative: X_unreg_cam = R_rel * X_reg_cam + t_rel (after normalization)
  // Combined: X_unreg_cam = R_rel * (R_reg * X_world + t_reg) + t_rel
  //                       = (R_rel * R_reg) * X_world + (R_rel * t_reg + t_rel)

  // The relative translation is unit length - we need to find the scale by matching
  // against existing triangulated 3D points visible in the registered camera

  // Build index: (image_id, kp_idx) -> (track_id, 3D point) for triangulated tracks
  std::unordered_map<uint64_t, std::pair<int, Eigen::Vector3d>> kp_to_track;
  const auto& all_tracks = rec->tracks.all();
  for (int ti = 0; ti < static_cast<int>(all_tracks.size()); ++ti) {
    const auto& track = all_tracks[ti];
    if (!track.triangulated) continue;
    for (const auto& obs : track.observations) {
      if (obs.image_id == best_registered) {
        // Pack (image_id, kp_idx) into a 64-bit key
        const uint64_t key =
            (static_cast<uint64_t>(obs.image_id) << 32) | static_cast<uint64_t>(obs.keypoint_id);
        kp_to_track[key] = {ti, track.xyz};
      }
    }
  }

  // Create a temporary projection matrix for relative pose (unit scale)
  static const Eigen::Matrix<double, 3, 4> P_identity =
      (Eigen::Matrix<double, 3, 4>() << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0).finished();
  Eigen::Matrix<double, 3, 4> P_unreg_rel;
  P_unreg_rel.block<3, 3>(0, 0) = rel.R;
  P_unreg_rel.col(3) = rel.t;

  // Collect scale ratios by comparing triangulated points with existing 3D points
  std::vector<double> scale_ratios;
  scale_ratios.reserve(vp.inliers.size());

  for (size_t k = 0; k < vp.inliers.size(); ++k) {
    const auto& m = vp.inliers[k];
    // Get keypoint index in registered image
    const int kp_reg = (vp.pair.i == best_registered) ? m.kp1 : m.kp2;

    // Check if this keypoint is part of a triangulated track
    const uint64_t key =
        (static_cast<uint64_t>(best_registered) << 32) | static_cast<uint64_t>(kp_reg);
    auto it = kp_to_track.find(key);
    if (it == kp_to_track.end()) continue;

    const Eigen::Vector3d& X_world_existing = it->second.second;

    // Triangulate this correspondence at unit scale
    const Eigen::Vector2d n_reg =
        geometry::PixelToUndistortedNormalized(intr_reg, x_reg_px[k]);
    const Eigen::Vector2d n_unreg =
        geometry::PixelToUndistortedNormalized(intr_unreg, x_unreg_px[k]);

    Eigen::Vector3d X_reg_cam;
    if (!geometry::TriangulateDLT(P_identity, P_unreg_rel, n_reg, n_unreg, &X_reg_cam)) continue;

    const double z_reg = X_reg_cam.z();
    const double z_unreg = (rel.R * X_reg_cam + rel.t).z();
    if (z_reg <= 0.01 || z_unreg <= 0.01) continue;

    // Transform triangulated point to world coordinates (at unit scale)
    const Eigen::Vector3d X_world_unit =
        cam_reg.pose.R.transpose() * X_reg_cam + CameraCenterWorld(cam_reg.pose);

    // Compute scale ratio: distance from camera to existing point / distance to unit point
    const Eigen::Vector3d C_reg = CameraCenterWorld(cam_reg.pose);
    const double dist_existing = (X_world_existing - C_reg).norm();
    const double dist_unit = (X_world_unit - C_reg).norm();

    if (dist_unit > 0.001) {
      scale_ratios.push_back(dist_existing / dist_unit);
    }
  }

  double scale = 1.0;
  if (scale_ratios.size() >= 5) {
    // Use median scale ratio
    std::nth_element(scale_ratios.begin(), scale_ratios.begin() + scale_ratios.size() / 2,
                     scale_ratios.end());
    scale = scale_ratios[scale_ratios.size() / 2];
  } else {
    // Fallback: estimate scale from average depth of existing 3D points visible from reg camera
    double total_depth = 0.0;
    int depth_count = 0;
    for (const auto& track : all_tracks) {
      if (!track.triangulated) continue;
      for (const auto& obs : track.observations) {
        if (obs.image_id == best_registered) {
          const Eigen::Vector3d X_cam = cam_reg.pose.R * track.xyz + cam_reg.pose.t;
          if (X_cam.z() > 0.1) {
            total_depth += X_cam.z();
            depth_count++;
          }
          break;
        }
      }
    }

    if (depth_count >= 10) {
      // Estimate scale so triangulated points have similar depth
      const double avg_existing_depth = total_depth / depth_count;

      // Compute average depth of unit-scale triangulated points
      double total_unit_depth = 0.0;
      int unit_count = 0;
      for (int k = 0; k < std::min(50, static_cast<int>(x_reg_px.size())); ++k) {
        const Eigen::Vector2d n_reg =
            geometry::PixelToUndistortedNormalized(intr_reg, x_reg_px[k]);
        const Eigen::Vector2d n_unreg =
            geometry::PixelToUndistortedNormalized(intr_unreg, x_unreg_px[k]);

        Eigen::Vector3d X_reg_cam;
        if (!geometry::TriangulateDLT(P_identity, P_unreg_rel, n_reg, n_unreg, &X_reg_cam))
          continue;

        if (X_reg_cam.z() > 0.01) {
          total_unit_depth += X_reg_cam.z();
          unit_count++;
        }
      }

      if (unit_count >= 10) {
        const double avg_unit_depth = total_unit_depth / unit_count;
        scale = avg_existing_depth / avg_unit_depth;
      }
    }
  }

  if (scale <= 0.001 || scale > 1000.0) {
    std::cerr << "[psynth] TryRegisterViaVerifiedPair: invalid scale " << scale << "\n";
    return -1;
  }

  // Compose the world-to-camera transformation
  Camera cam_unreg;
  cam_unreg.intr = intr_unreg;
  cam_unreg.pose.R = rel.R * cam_reg.pose.R;
  cam_unreg.pose.t = rel.R * cam_reg.pose.t + rel.t * scale;

  // Verify by checking reprojection error on existing 3D points and triangulated points
  int good_reproj = 0;
  int total_checked = 0;

  // First check against existing 3D points
  for (size_t k = 0; k < vp.inliers.size() && total_checked < 50; ++k) {
    const auto& m = vp.inliers[k];
    const int kp_reg = (vp.pair.i == best_registered) ? m.kp1 : m.kp2;

    const uint64_t key =
        (static_cast<uint64_t>(best_registered) << 32) | static_cast<uint64_t>(kp_reg);
    auto it = kp_to_track.find(key);
    if (it == kp_to_track.end()) continue;

    const Eigen::Vector3d& X_world = it->second.second;
    const double e_reg = geometry::ReprojectionErrorPx(cam_reg, X_world, x_reg_px[k]);
    const double e_unreg = geometry::ReprojectionErrorPx(cam_unreg, X_world, x_unreg_px[k]);

    total_checked++;
    if (e_reg < cfg_.sfm.max_reprojection_error_px * 2 &&
        e_unreg < cfg_.sfm.max_reprojection_error_px * 2) {
      good_reproj++;
    }
  }

  // If not enough existing 3D points, also check freshly triangulated points
  if (total_checked < 30) {
    Eigen::Matrix<double, 3, 4> P_unreg_rel_scaled;
    P_unreg_rel_scaled.block<3, 3>(0, 0) = rel.R;
    P_unreg_rel_scaled.col(3) = rel.t * scale;

    for (int k = 0; k < static_cast<int>(x_reg_px.size()) && total_checked < 50; ++k) {
      const Eigen::Vector2d n_reg =
          geometry::PixelToUndistortedNormalized(intr_reg, x_reg_px[k]);
      const Eigen::Vector2d n_unreg =
          geometry::PixelToUndistortedNormalized(intr_unreg, x_unreg_px[k]);

      Eigen::Vector3d X_reg_cam;
      if (!geometry::TriangulateDLT(P_identity, P_unreg_rel_scaled, n_reg, n_unreg, &X_reg_cam))
        continue;

      const double z_reg = X_reg_cam.z();
      const double z_unreg = (rel.R * X_reg_cam + rel.t * scale).z();
      if (z_reg <= 0.01 || z_unreg <= 0.01) continue;

      // Transform to world coordinates
      const Eigen::Vector3d X_world =
          cam_reg.pose.R.transpose() * X_reg_cam + CameraCenterWorld(cam_reg.pose);

      const double e_reg = geometry::ReprojectionErrorPx(cam_reg, X_world, x_reg_px[k]);
      const double e_unreg = geometry::ReprojectionErrorPx(cam_unreg, X_world, x_unreg_px[k]);

      total_checked++;
      if (e_reg < cfg_.sfm.max_reprojection_error_px * 2 &&
          e_unreg < cfg_.sfm.max_reprojection_error_px * 2) {
        good_reproj++;
      }
    }
  }

  const int min_good = std::max(5, total_checked / 3);
  if (good_reproj < min_good) {
    std::cerr << "[psynth] TryRegisterViaVerifiedPair: failed (" << good_reproj << "/"
              << total_checked << " good, need " << min_good << ")\n";
    return -1;
  }

  // Add the camera to reconstruction
  rec->cameras[best_unregistered] = cam_unreg;
  std::cerr << "[psynth] registered image " << best_unregistered << " via verified pair with "
            << best_registered << "\n";

  return best_unregistered;
}

}  // namespace psynth::sfm