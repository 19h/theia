#include <psynth/mvs/pmvs_export.hpp>
#include <psynth/common.hpp>
#include <psynth/geometry/triangulation.hpp>
#include <psynth/io/serialization.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace psynth::mvs {

namespace fs = std::filesystem;

namespace {

std::string EightDigits(int idx, const std::string& ext) {
  std::ostringstream oss;
  oss << std::setw(8) << std::setfill('0') << idx << ext;
  return oss.str();
}

cv::Mat CvK(const Intrinsics& intr) {
  return (cv::Mat_<double>(3, 3) << intr.f_px, 0, intr.cx_px,
          0, intr.f_px, intr.cy_px,
          0, 0, 1);
}

cv::Mat CvDist(const Intrinsics& intr) {
  return (cv::Mat_<double>(1, 5) << intr.k1, intr.k2, 0.0, 0.0, 0.0);
}

}  // namespace

bool ExportPMVS(const std::string& pmvs_root,
                const io::ImageDataset& dataset,
                const sfm::Reconstruction& rec,
                const PipelineConfig& cfg) {
  if (rec.cameras.empty()) return false;

  const fs::path root(pmvs_root);
  const fs::path vis_dir = root / "visualize";
  const fs::path txt_dir = root / "txt";

  io::EnsureDir(vis_dir);
  io::EnsureDir(txt_dir);

  std::vector<ImageId> cam_ids;
  cam_ids.reserve(rec.cameras.size());
  for (const auto& kv : rec.cameras) cam_ids.push_back(kv.first);
  std::sort(cam_ids.begin(), cam_ids.end());

  const int M = static_cast<int>(cam_ids.size());

  std::unordered_map<ImageId, int> pmvs_index;
  pmvs_index.reserve(cam_ids.size());
  for (int i = 0; i < M; ++i) pmvs_index[cam_ids[i]] = i;

  for (int pi = 0; pi < M; ++pi) {
    const ImageId id = cam_ids[pi];
    const Camera& cam = rec.cameras.at(id);

    cv::Mat img = dataset.ReadColor(id);

    cv::Mat img_out = img;
    Intrinsics intr_out = cam.intr;

    if (cfg.pmvs.undistort && (std::abs(cam.intr.k1) > 0.0 || std::abs(cam.intr.k2) > 0.0)) {
      cv::Mat K = CvK(cam.intr);
      cv::Mat dist = CvDist(cam.intr);
      cv::Mat undist;
      cv::undistort(img, undist, K, dist, K);
      img_out = undist;

      intr_out.k1 = 0.0;
      intr_out.k2 = 0.0;
    }

    const fs::path img_path = vis_dir / EightDigits(pi, ".jpg");
    PSYNTH_REQUIRE(cv::imwrite(img_path.string(), img_out),
                   "Failed to write PMVS image: " + img_path.string());

    const Eigen::Matrix3d K_e = geometry::IntrinsicsMatrix(intr_out);
    const Eigen::Matrix<double, 3, 4> Rt = geometry::ExtrinsicsMatrix(cam.pose);
    const Eigen::Matrix<double, 3, 4> P = K_e * Rt;

    const fs::path cam_path = txt_dir / EightDigits(pi, ".txt");
    std::ofstream cam_out(cam_path);
    PSYNTH_REQUIRE(cam_out.is_open(), "Failed to write PMVS camera: " + cam_path.string());

    cam_out << "CONTOUR\n";
    cam_out << std::setprecision(17);
    for (int r = 0; r < 3; ++r) {
      cam_out << P(r, 0) << " " << P(r, 1) << " " << P(r, 2) << " " << P(r, 3) << "\n";
    }
  }

  {
    const fs::path opt_path = root / "options.txt";
    std::ofstream opt_out(opt_path);
    PSYNTH_REQUIRE(opt_out.is_open(), "Failed to write PMVS options: " + opt_path.string());

    opt_out << "# psynth-generated PMVS2 options\n";
    opt_out << "timages -1 0 " << M << "\n";
    opt_out << "oimages -1 0 " << M << "\n";
    opt_out << "level " << cfg.pmvs.level << "\n";
    opt_out << "csize " << cfg.pmvs.csize << "\n";
    opt_out << "threshold " << cfg.pmvs.threshold << "\n";
    opt_out << "wsize " << cfg.pmvs.wsize << "\n";
    opt_out << "minImageNum " << cfg.pmvs.minImageNum << "\n";
    opt_out << "CPU " << cfg.pmvs.CPU << "\n";
    opt_out << "useVisData " << (cfg.pmvs.write_vis_dat ? 1 : 0) << "\n";
  }

  if (cfg.pmvs.write_vis_dat) {
    std::unordered_map<uint64_t, int> pair_counts;
    pair_counts.reserve(1024);

    for (const auto& t : rec.tracks.all()) {
      if (!t.triangulated) continue;

      std::vector<int> cams;
      cams.reserve(t.observations.size());
      for (const auto& obs : t.observations) {
        auto it = pmvs_index.find(obs.image_id);
        if (it == pmvs_index.end()) continue;
        cams.push_back(it->second);
      }

      std::sort(cams.begin(), cams.end());
      cams.erase(std::unique(cams.begin(), cams.end()), cams.end());
      if (cams.size() < 2) continue;

      for (size_t a = 0; a < cams.size(); ++a) {
        for (size_t b = a + 1; b < cams.size(); ++b) {
          const uint32_t u = static_cast<uint32_t>(cams[a]);
          const uint32_t v = static_cast<uint32_t>(cams[b]);
          const uint64_t key = (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
          pair_counts[key] += 1;
        }
      }
    }

    std::vector<std::vector<int>> neighbors(M);
    for (const auto& kv : pair_counts) {
      const int count = kv.second;
      if (count < cfg.pmvs.vis_min_shared_tracks) continue;

      const uint64_t key = kv.first;
      const int a = static_cast<int>(key >> 32);
      const int b = static_cast<int>(key & 0xffffffffu);
      if (a < 0 || a >= M || b < 0 || b >= M) continue;
      neighbors[a].push_back(b);
      neighbors[b].push_back(a);
    }

    for (auto& nb : neighbors) {
      std::sort(nb.begin(), nb.end());
      nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }

    const fs::path vis_path = root / "vis.dat";
    std::ofstream vis_out(vis_path);
    PSYNTH_REQUIRE(vis_out.is_open(), "Failed to write PMVS vis.dat: " + vis_path.string());

    vis_out << "VISDATA\n";
    vis_out << M << "\n";
    for (int i = 0; i < M; ++i) {
      vis_out << i << " " << neighbors[i].size();
      for (int j : neighbors[i]) vis_out << " " << j;
      vis_out << "\n";
    }
  }

  return true;
}

}  // namespace psynth::mvs