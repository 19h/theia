#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace psynth::sfm {

class UnionFind {
 public:
  explicit UnionFind(std::size_t n) : parent_(n), rank_(n, 0) {
    for (std::size_t i = 0; i < n; ++i) parent_[i] = i;
  }

  std::size_t Find(std::size_t x) {
    const std::size_t p = parent_[x];
    if (p == x) return x;
    parent_[x] = Find(p);
    return parent_[x];
  }

  void Unite(std::size_t a, std::size_t b) {
    a = Find(a);
    b = Find(b);
    if (a == b) return;

    if (rank_[a] < rank_[b]) {
      parent_[a] = b;
    } else if (rank_[a] > rank_[b]) {
      parent_[b] = a;
    } else {
      parent_[b] = a;
      rank_[a]++;
    }
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<uint8_t> rank_;
};

}  // namespace psynth::sfm