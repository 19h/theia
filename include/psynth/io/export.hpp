#pragma once

#include <psynth/io/dataset.hpp>
#include <psynth/sfm/reconstruction.hpp>

#include <string>

namespace psynth::io {

bool ExportSparsePLY(const std::string& path, const sfm::Reconstruction& rec);
bool ExportCamerasPLY(const std::string& path, const sfm::Reconstruction& rec);

bool ExportBundleOut(const std::string& path, const ImageDataset& dataset, const sfm::Reconstruction& rec);

}  // namespace psynth::io