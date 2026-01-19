#pragma once

#include <psynth/io/dataset.hpp>
#include <psynth/sfm/reconstruction.hpp>

namespace psynth::ba {

struct BundleAdjusterOptions {
  int max_iterations = 50;
  double huber_loss_px = 4.0;
  bool optimize_intrinsics = true;
};

class BundleAdjuster {
 public:
  explicit BundleAdjuster(BundleAdjusterOptions opt) : opt_(opt) {}

  void Adjust(const io::ImageDataset& dataset, sfm::Reconstruction* rec) const;

 private:
  BundleAdjusterOptions opt_;
};

}  // namespace psynth::ba