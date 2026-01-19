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

  std::vector<cv::Point3d> obj;
  std::vector<cv::Point2d> img;
  obj.reserve(X_world.size());
  img.reserve(uv_px.size());

  for (size_t i = 0; i < X_world.size(); ++i) {
    obj.emplace_back(X_world[i].x(), X_world[i].y(), X_world[i].z());
    img.emplace_back(uv_px[i].x(), uv_px[i].y());
  }

  cv::Mat K = (cv::Mat_<double>(3, 3) << intr.f_px, 0, intr.cx_px,
                0, intr.f_px, intr.cy_px,
                0, 0, 1);

  cv::Mat dist = (cv::Mat_<double>(1, 5) << intr.k1, intr.k2, 0.0, 0.0, 0.0);

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

  cv::Mat Rcv;
  cv::Rodrigues(rvec, Rcv);

  Eigen::Matrix3d R;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R(r, c) = Rcv.at<double>(r, c);
    }
  }

  out.success = true;
  out.R = R;
  out.t = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  out.inlier_indices.reserve(inliers.rows);
  for (int i = 0; i < inliers.rows; ++i) {
    out.inlier_indices.push_back(inliers.at<int>(i, 0));
  }

  return out;
}

}  // namespace psynth::geometry