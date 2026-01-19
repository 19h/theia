#pragma once

#include <psynth/config.hpp>
#include <psynth/io/dataset.hpp>
#include <psynth/sfm/reconstruction.hpp>

#include <string>

namespace psynth::mvs {

bool ExportPMVS(const std::string& pmvs_root,
                const io::ImageDataset& dataset,
                const sfm::Reconstruction& rec,
                const PipelineConfig& cfg);

}  // namespace psynth::mvs