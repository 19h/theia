#include <psynth/io/export.hpp>
#include <psynth/common.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>

namespace psynth::io {

namespace {

std::vector<ImageId> SortedCameraIds(const sfm::Reconstruction& rec) {
  std::vector<ImageId> ids;
  ids.reserve(rec.cameras.size());
  for (const auto& kv : rec.cameras) ids.push_back(kv.first);
  std::sort(ids.begin(), ids.end());
  return ids;
}

}  // namespace

bool ExportSparsePLY(const std::string& path, const sfm::Reconstruction& rec) {
  std::vector<const Track*> pts;
  pts.reserve(rec.tracks.all().size());
  for (const auto& t : rec.tracks.all()) {
    if (!t.triangulated) continue;
    pts.push_back(&t);
  }

  std::ofstream out(path);
  if (!out.is_open()) return false;

  out << "ply\nformat ascii 1.0\n";
  out << "element vertex " << pts.size() << "\n";
  out << "property float x\nproperty float y\nproperty float z\n";
  out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  out << "end_header\n";

  out << std::setprecision(9);
  for (const auto* t : pts) {
    const int r = static_cast<int>(t->color_bgr[2]);
    const int g = static_cast<int>(t->color_bgr[1]);
    const int b = static_cast<int>(t->color_bgr[0]);
    out << static_cast<float>(t->xyz.x()) << " " << static_cast<float>(t->xyz.y()) << " "
        << static_cast<float>(t->xyz.z()) << " " << r << " " << g << " " << b << "\n";
  }
  return true;
}

bool ExportCamerasPLY(const std::string& path, const sfm::Reconstruction& rec) {
  const std::vector<ImageId> cam_ids = SortedCameraIds(rec);

  std::ofstream out(path);
  if (!out.is_open()) return false;

  out << "ply\nformat ascii 1.0\n";
  out << "element vertex " << cam_ids.size() << "\n";
  out << "property float x\nproperty float y\nproperty float z\n";
  out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  out << "end_header\n";

  out << std::setprecision(9);
  for (const ImageId id : cam_ids) {
    const Camera& cam = rec.cameras.at(id);
    const Eigen::Vector3d C = CameraCenterWorld(cam.pose);
    out << static_cast<float>(C.x()) << " " << static_cast<float>(C.y()) << " "
        << static_cast<float>(C.z()) << " 255 255 255\n";
  }
  return true;
}

bool ExportBundleOut(const std::string& path, const ImageDataset& dataset, const sfm::Reconstruction& rec) {
  const std::vector<ImageId> cam_ids = SortedCameraIds(rec);

  std::map<ImageId, int> cam_index;
  for (int i = 0; i < static_cast<int>(cam_ids.size()); ++i) {
    cam_index[cam_ids[i]] = i;
  }

  std::vector<const Track*> pts;
  pts.reserve(rec.tracks.all().size());
  for (const auto& t : rec.tracks.all()) {
    if (!t.triangulated) continue;
    int reg_views = 0;
    for (const auto& obs : t.observations) {
      if (cam_index.find(obs.image_id) != cam_index.end()) reg_views++;
    }
    if (reg_views >= 2) pts.push_back(&t);
  }

  std::ofstream out(path);
  if (!out.is_open()) return false;

  out << "# Bundle file v0.3\n";
  out << cam_ids.size() << " " << pts.size() << "\n";
  out << std::setprecision(17);

  const Eigen::Matrix3d S = (Eigen::Matrix3d() << 1, 0, 0, 0, -1, 0, 0, 0, -1).finished();

  for (const ImageId id : cam_ids) {
    const Camera& cam = rec.cameras.at(id);

    out << cam.intr.f_px << " " << cam.intr.k1 << " " << cam.intr.k2 << "\n";

    const Eigen::Matrix3d Rb = S * cam.pose.R;
    const Eigen::Vector3d tb = S * cam.pose.t;

    for (int r = 0; r < 3; ++r) {
      out << Rb(r, 0) << " " << Rb(r, 1) << " " << Rb(r, 2) << "\n";
    }
    out << tb.x() << " " << tb.y() << " " << tb.z() << "\n";
  }

  for (const Track* t : pts) {
    out << t->xyz.x() << " " << t->xyz.y() << " " << t->xyz.z() << "\n";
    out << static_cast<int>(t->color_bgr[2]) << " " << static_cast<int>(t->color_bgr[1]) << " "
        << static_cast<int>(t->color_bgr[0]) << "\n";

    struct View {
      int cam_idx;
      int key_idx;
      double x;
      double y;
    };
    std::vector<View> views;
    views.reserve(t->observations.size());

    for (const auto& obs : t->observations) {
      auto it = cam_index.find(obs.image_id);
      if (it == cam_index.end()) continue;

      const auto& intr = dataset.image(obs.image_id).intr;
      const double x = obs.u_px - intr.cx_px;
      const double y = intr.cy_px - obs.v_px;

      views.push_back(View{it->second, obs.keypoint_id, x, y});
    }

    out << views.size();
    for (const auto& v : views) {
      out << " " << v.cam_idx << " " << v.key_idx << " " << v.x << " " << v.y;
    }
    out << "\n";
  }

  return true;
}

}  // namespace psynth::io