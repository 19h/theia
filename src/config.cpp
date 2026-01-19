#include <psynth/config.hpp>
#include <psynth/common.hpp>

#include <opencv2/core.hpp>

namespace psynth {

namespace {

template <typename T>
void ReadIfPresent(const cv::FileNode& node, const char* key, T* value) {
  const cv::FileNode v = node[key];
  if (!v.empty()) {
    v >> *value;
  }
}

void ReadBoolIfPresent(const cv::FileNode& node, const char* key, bool* value) {
  const cv::FileNode v = node[key];
  if (!v.empty()) {
    int tmp = 0;
    v >> tmp;
    *value = (tmp != 0);
  }
}

}  // namespace

PipelineConfig PipelineConfig::FromYAML(const std::string& path) {
  PipelineConfig cfg;
  if (path.empty()) {
    return cfg;
  }

  cv::FileStorage fs(path, cv::FileStorage::READ);
  PSYNTH_REQUIRE(fs.isOpened(), "Failed to open config file: " + path);

  ReadIfPresent(fs.root(), "initial_focal_px_factor", &cfg.initial_focal_px_factor);
  ReadIfPresent(fs.root(), "initial_k1", &cfg.initial_k1);
  ReadIfPresent(fs.root(), "initial_k2", &cfg.initial_k2);

  {
    cv::FileNode sift = fs["sift"];
    if (!sift.empty()) {
      ReadIfPresent(sift, "max_features", &cfg.sift.max_features);
      ReadIfPresent(sift, "n_octave_layers", &cfg.sift.n_octave_layers);
      ReadIfPresent(sift, "contrast_threshold", &cfg.sift.contrast_threshold);
      ReadIfPresent(sift, "edge_threshold", &cfg.sift.edge_threshold);
      ReadIfPresent(sift, "sigma", &cfg.sift.sigma);
    }
  }

  {
    cv::FileNode m = fs["matcher"];
    if (!m.empty()) {
      ReadIfPresent(m, "ratio", &cfg.matcher.ratio);
      ReadBoolIfPresent(m, "cross_check", &cfg.matcher.cross_check);
      ReadIfPresent(m, "flann_trees", &cfg.matcher.flann_trees);
      ReadIfPresent(m, "flann_checks", &cfg.matcher.flann_checks);
      ReadIfPresent(m, "min_matches", &cfg.matcher.min_matches);
    }
  }

  {
    cv::FileNode r = fs["fund_ransac"];
    if (!r.empty()) {
      ReadIfPresent(r, "max_iterations", &cfg.fund_ransac.max_iterations);
      ReadIfPresent(r, "confidence", &cfg.fund_ransac.confidence);
      ReadIfPresent(r, "inlier_threshold_px", &cfg.fund_ransac.inlier_threshold_px);
      ReadIfPresent(r, "min_inliers", &cfg.fund_ransac.min_inliers);
    }
  }

  {
    cv::FileNode s = fs["sfm"];
    if (!s.empty()) {
      ReadIfPresent(s, "init_min_inliers", &cfg.sfm.init_min_inliers);
      ReadIfPresent(s, "init_max_homography_ratio", &cfg.sfm.init_max_homography_ratio);
      ReadIfPresent(s, "init_min_median_triangulation_angle_deg",
                    &cfg.sfm.init_min_median_triangulation_angle_deg);
      ReadIfPresent(s, "min_pnp_correspondences", &cfg.sfm.min_pnp_correspondences);
      ReadIfPresent(s, "min_pnp_inliers", &cfg.sfm.min_pnp_inliers);
      ReadIfPresent(s, "max_reprojection_error_px", &cfg.sfm.max_reprojection_error_px);
      ReadIfPresent(s, "bundle_every_n_images", &cfg.sfm.bundle_every_n_images);
      ReadIfPresent(s, "ba_max_iterations", &cfg.sfm.ba_max_iterations);
      ReadIfPresent(s, "ba_huber_loss_px", &cfg.sfm.ba_huber_loss_px);
      ReadBoolIfPresent(s, "ba_optimize_intrinsics", &cfg.sfm.ba_optimize_intrinsics);
    }
  }

  {
    cv::FileNode p = fs["pmvs"];
    if (!p.empty()) {
      ReadBoolIfPresent(p, "enabled", &cfg.pmvs.enabled);
      ReadIfPresent(p, "pmvs_root", &cfg.pmvs.pmvs_root);
      ReadIfPresent(p, "level", &cfg.pmvs.level);
      ReadIfPresent(p, "csize", &cfg.pmvs.csize);
      ReadIfPresent(p, "minImageNum", &cfg.pmvs.minImageNum);
      ReadIfPresent(p, "threshold", &cfg.pmvs.threshold);
      ReadIfPresent(p, "wsize", &cfg.pmvs.wsize);
      ReadIfPresent(p, "CPU", &cfg.pmvs.CPU);
      ReadBoolIfPresent(p, "undistort", &cfg.pmvs.undistort);
      ReadBoolIfPresent(p, "write_vis_dat", &cfg.pmvs.write_vis_dat);
      ReadIfPresent(p, "vis_min_shared_tracks", &cfg.pmvs.vis_min_shared_tracks);
    }
  }

  return cfg;
}

void PipelineConfig::SaveYAML(const std::string& path) const {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  PSYNTH_REQUIRE(fs.isOpened(), "Failed to open config file for write: " + path);

  fs << "initial_focal_px_factor" << initial_focal_px_factor;
  fs << "initial_k1" << initial_k1;
  fs << "initial_k2" << initial_k2;

  fs << "sift"
     << "{"
     << "max_features" << sift.max_features << "n_octave_layers" << sift.n_octave_layers
     << "contrast_threshold" << sift.contrast_threshold << "edge_threshold" << sift.edge_threshold
     << "sigma" << sift.sigma << "}";

  fs << "matcher"
     << "{"
     << "ratio" << matcher.ratio << "cross_check" << (matcher.cross_check ? 1 : 0) << "flann_trees"
     << matcher.flann_trees << "flann_checks" << matcher.flann_checks << "min_matches"
     << matcher.min_matches << "}";

  fs << "fund_ransac"
     << "{"
     << "max_iterations" << fund_ransac.max_iterations << "confidence" << fund_ransac.confidence
     << "inlier_threshold_px" << fund_ransac.inlier_threshold_px << "min_inliers"
     << fund_ransac.min_inliers << "}";

  fs << "sfm"
     << "{"
     << "init_min_inliers" << sfm.init_min_inliers << "init_max_homography_ratio"
     << sfm.init_max_homography_ratio << "init_min_median_triangulation_angle_deg"
     << sfm.init_min_median_triangulation_angle_deg << "min_pnp_correspondences"
     << sfm.min_pnp_correspondences << "min_pnp_inliers" << sfm.min_pnp_inliers
     << "max_reprojection_error_px" << sfm.max_reprojection_error_px << "bundle_every_n_images"
     << sfm.bundle_every_n_images << "ba_max_iterations" << sfm.ba_max_iterations
     << "ba_huber_loss_px" << sfm.ba_huber_loss_px << "ba_optimize_intrinsics"
     << (sfm.ba_optimize_intrinsics ? 1 : 0) << "}";

  fs << "pmvs"
     << "{"
     << "enabled" << (pmvs.enabled ? 1 : 0) << "pmvs_root" << pmvs.pmvs_root << "level" << pmvs.level
     << "csize" << pmvs.csize << "minImageNum" << pmvs.minImageNum << "threshold" << pmvs.threshold
     << "wsize" << pmvs.wsize << "CPU" << pmvs.CPU << "undistort" << (pmvs.undistort ? 1 : 0)
     << "write_vis_dat" << (pmvs.write_vis_dat ? 1 : 0) << "vis_min_shared_tracks"
     << pmvs.vis_min_shared_tracks << "}";
}

}  // namespace psynth