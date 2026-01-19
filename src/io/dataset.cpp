#include <psynth/io/dataset.hpp>
#include <psynth/common.hpp>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

namespace psynth::io {

namespace fs = std::filesystem;

namespace {

std::string ToLower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool HasImageExtension(const fs::path& p) {
  const std::string ext = ToLower(p.extension().string());
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff" ||
         ext == ".bmp";
}

}  // namespace

ImageDataset ImageDataset::FromDirectory(const std::string& dir, const PipelineConfig& cfg) {
  ImageDataset ds;
  ds.base_dir_ = fs::path(dir);

  PSYNTH_REQUIRE(fs::exists(ds.base_dir_), "images_dir does not exist: " + ds.base_dir_.string());
  PSYNTH_REQUIRE(fs::is_directory(ds.base_dir_),
                 "images_dir is not a directory: " + ds.base_dir_.string());

  std::vector<fs::path> paths;
  for (const auto& ent : fs::directory_iterator(ds.base_dir_)) {
    if (!ent.is_regular_file()) continue;
    if (!HasImageExtension(ent.path())) continue;
    paths.push_back(ent.path());
  }
  std::sort(paths.begin(), paths.end());

  PSYNTH_REQUIRE(!paths.empty(), "No images found in directory: " + ds.base_dir_.string());

  ds.images_.reserve(paths.size());
  for (int i = 0; i < static_cast<int>(paths.size()); ++i) {
    const fs::path& p = paths[i];
    cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
    PSYNTH_REQUIRE(!img.empty(), "Failed to read image: " + p.string());

    ImageInfo info;
    info.id = i;
    info.path = p.string();
    info.intr.width_px = img.cols;
    info.intr.height_px = img.rows;
    info.intr.cx_px = 0.5 * static_cast<double>(img.cols);
    info.intr.cy_px = 0.5 * static_cast<double>(img.rows);
    info.intr.f_px =
        cfg.initial_focal_px_factor * static_cast<double>(std::max(img.cols, img.rows));
    info.intr.k1 = cfg.initial_k1;
    info.intr.k2 = cfg.initial_k2;

    ds.images_.push_back(info);
  }

  return ds;
}

const ImageInfo& ImageDataset::image(ImageId id) const {
  PSYNTH_REQUIRE(id >= 0 && id < static_cast<ImageId>(images_.size()), "Invalid image id");
  return images_[id];
}

cv::Mat ImageDataset::ReadColor(ImageId id) const {
  const auto& info = image(id);
  cv::Mat img = cv::imread(info.path, cv::IMREAD_COLOR);
  PSYNTH_REQUIRE(!img.empty(), "Failed to read image: " + info.path);
  return img;
}

cv::Mat ImageDataset::ReadGray(ImageId id) const {
  const auto& info = image(id);
  cv::Mat img = cv::imread(info.path, cv::IMREAD_GRAYSCALE);
  PSYNTH_REQUIRE(!img.empty(), "Failed to read image: " + info.path);
  return img;
}

}  // namespace psynth::io