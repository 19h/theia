#include <psynth/geometry/pnp.hpp>
#include <psynth/common.hpp>

#include <opencv2/calib3d.hpp>

namespace psynth::geometry {

PnPRansacResult SolvePnPRansacOpenCV(const Intrinsics& intr,
                                     const std::vector<Eigen::Vector3d>& X_world,
                                     const std::vector<Eigen::Vector2d>& uv_px,
                                     const PnPRansacOptions& opt) {
  PnPRansacResult out;

  PSYNTH_REQUIRE(X_world.size() == uv_px.size(), "SolvePnPRansacOpenCV: size mismatch");
  if (X_world.size() < 4) {
    out.success = false;
    return out;
  }

  const size_t N = X_world.size();

  // Use thread-local buffers to avoid repeated allocation across calls
  // This is particularly important in incremental SfM where PnP is called per-image
  thread_local std::vector<cv::Point3d> obj;
  thread_local std::vector<cv::Point2d> img;

  obj.resize(N);
  img.resize(N);

  // Direct assignment is faster than emplace_back with re-allocation checks
  for (size_t i = 0; i < N; ++i) {
    obj[i].x = X_world[i].x();
    obj[i].y = X_world[i].y();
    obj[i].z = X_world[i].z();
    img[i].x = uv_px[i].x();
    img[i].y = uv_px[i].y();
  }

  // Pre-allocate intrinsic matrix - use thread-local to avoid repeated allocation
  thread_local cv::Mat K(3, 3, CV_64F);
  thread_local cv::Mat dist(1, 5, CV_64F);

  double* K_data = K.ptr<double>();
  K_data[0] = intr.f_px; K_data[1] = 0;          K_data[2] = intr.cx_px;
  K_data[3] = 0;         K_data[4] = intr.f_px;  K_data[5] = intr.cy_px;
  K_data[6] = 0;         K_data[7] = 0;          K_data[8] = 1;

  double* dist_data = dist.ptr<double>();
  dist_data[0] = intr.k1;
  dist_data[1] = intr.k2;
  dist_data[2] = 0.0;
  dist_data[3] = 0.0;
  dist_data[4] = 0.0;

  cv::Mat rvec, tvec, inliers;

  const int flags = opt.use_ap3p ? cv::SOLVEPNP_AP3P : cv::SOLVEPNP_EPNP;

  const bool ok = cv::solvePnPRansac(obj,
                                     img,
                                     K,
                                     dist,
                                     rvec,
                                     tvec,
                                     false,
                                     opt.max_iterations,
                                     opt.reprojection_error_px,
                                     opt.confidence,
                                     inliers,
                                     flags);

  if (!ok || inliers.empty() || inliers.rows < opt.min_inliers) {
    out.success = false;
    return out;
  }

  // Use thread-local rotation matrix
  thread_local cv::Mat Rcv(3, 3, CV_64F);
  cv::Rodrigues(rvec, Rcv);

  // Direct copy from cv::Mat to Eigen - avoid element-by-element access
  const double* R_data = Rcv.ptr<double>();
  out.R << R_data[0], R_data[1], R_data[2],
           R_data[3], R_data[4], R_data[5],
           R_data[6], R_data[7], R_data[8];

  out.success = true;
  out.t = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  out.inlier_indices.resize(inliers.rows);
  const int* inlier_data = inliers.ptr<int>();
  for (int i = 0; i < inliers.rows; ++i) {
    out.inlier_indices[i] = inlier_data[i];
  }

  return out;
}

}  // namespace psynth::geometry