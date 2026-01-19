#pragma once

#include <string>

namespace psynth {

struct SiftConfig {
  int max_features = 0;          // 0 preserves OpenCV default behavior
  int n_octave_layers = 3;
  double contrast_threshold = 0.04;
  double edge_threshold = 10.0;
  double sigma = 1.6;
};

struct MatcherConfig {
  float ratio = 0.8f;
  bool cross_check = true;
  int flann_trees = 8;
  int flann_checks = 200;
  int min_matches = 20;
};

struct FundamentalRansacConfig {
  int max_iterations = 2000;
  double confidence = 0.99;
  double inlier_threshold_px = 2.0;
  int min_inliers = 30;
};

struct SfMConfig {
  int init_min_inliers = 100;
  double init_max_homography_ratio = 0.6;
  double init_min_median_triangulation_angle_deg = 5.0;

  int min_pnp_correspondences = 30;
  int min_pnp_inliers = 25;

  double max_reprojection_error_px = 4.0;

  int bundle_every_n_images = 5;

  int ba_max_iterations = 50;
  double ba_huber_loss_px = 4.0;
  bool ba_optimize_intrinsics = true;
};

struct PMVSConfig {
  bool enabled = false;
  std::string pmvs_root = "pmvs";

  int level = 1;
  int csize = 2;
  int minImageNum = 3;
  double threshold = 0.7;
  int wsize = 7;
  int CPU = 4;

  bool undistort = true;

  bool write_vis_dat = true;
  int vis_min_shared_tracks = 10;
};

struct PipelineConfig {
  double initial_focal_px_factor = 0.9;
  double initial_k1 = 0.0;
  double initial_k2 = 0.0;

  SiftConfig sift;
  MatcherConfig matcher;
  FundamentalRansacConfig fund_ransac;
  SfMConfig sfm;
  PMVSConfig pmvs;

  static PipelineConfig FromYAML(const std::string& path);
  void SaveYAML(const std::string& path) const;
};

}  // namespace psynth