#pragma once

#include <psynth/config.hpp>
#include <psynth/io/dataset.hpp>
#include <psynth/sfm/reconstruction.hpp>
#include <psynth/sfm/tracks.hpp>
#include <psynth/types.hpp>

#include <vector>

namespace psynth::sfm {

class IncrementalSfM {
 public:
  IncrementalSfM(const io::ImageDataset& dataset,
                 const std::vector<FeatureSet>& features,
                 const std::vector<VerifiedPair>& verified_pairs,
                 Tracks tracks,
                 PipelineConfig cfg);

  Reconstruction Run();

 private:
  const io::ImageDataset& dataset_;
  const std::vector<FeatureSet>& features_;
  std::vector<VerifiedPair> verified_pairs_;
  Tracks tracks_;
  PipelineConfig cfg_;

  int SelectInitialPairIndex() const;

  void InitializeFromPair(ImageId i, ImageId j, const VerifiedPair& vp, Reconstruction* rec);

  ImageId SelectNextImage(const Reconstruction& rec, const std::vector<bool>& registered) const;

  bool RegisterImage(ImageId img, Reconstruction* rec);

  void TriangulateNewTracks(Reconstruction* rec);

  void BundleAdjust(Reconstruction* rec);

  void FilterTrackOutliers(Reconstruction* rec);
};

}  // namespace psynth::sfm