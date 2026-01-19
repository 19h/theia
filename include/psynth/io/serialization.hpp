#pragma once

#include <psynth/types.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace psynth::io {

void EnsureDir(const std::filesystem::path& p);

std::filesystem::path FeaturePath(const std::filesystem::path& features_dir, ImageId id);

bool SaveFeatureSet(const std::filesystem::path& path, const FeatureSet& fs);
bool LoadFeatureSet(const std::filesystem::path& path, FeatureSet* fs);

bool SaveAllFeatures(const std::filesystem::path& features_dir,
                     const std::vector<FeatureSet>& features);

bool LoadAllFeatures(const std::filesystem::path& features_dir,
                     int num_images,
                     std::vector<FeatureSet>* features);

bool SaveVerifiedPairs(const std::filesystem::path& path, const std::vector<VerifiedPair>& pairs);
bool LoadVerifiedPairs(const std::filesystem::path& path, std::vector<VerifiedPair>* pairs);

}  // namespace psynth::io