#pragma once

#include <psynth/config.hpp>
#include <psynth/types.hpp>

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace psynth::io {

class ImageDataset {
 public:
  static ImageDataset FromDirectory(const std::string& dir, const PipelineConfig& cfg);

  int size() const { return static_cast<int>(images_.size()); }

  const ImageInfo& image(ImageId id) const;

  const std::vector<ImageInfo>& images() const { return images_; }

  cv::Mat ReadColor(ImageId id) const;
  cv::Mat ReadGray(ImageId id) const;

 private:
  std::filesystem::path base_dir_;
  std::vector<ImageInfo> images_;
};

}  // namespace psynth::io