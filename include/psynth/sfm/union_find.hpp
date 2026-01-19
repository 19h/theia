#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace psynth::sfm {

// High-performance Union-Find (Disjoint Set Union) with:
// - Iterative path compression (no recursion overhead, no stack usage)
// - Union by rank for optimal tree depth
// - Compact rank storage (uint8_t - rank never exceeds log2(n))
class UnionFind {
 public:
  explicit UnionFind(std::size_t n) : parent_(n), rank_(n, 0) {
    for (std::size_t i = 0; i < n; ++i) parent_[i] = i;
  }

  // Iterative Find with two-pass path compression
  // Pass 1: Find root
  // Pass 2: Compress path (set all nodes to point directly to root)
  // This eliminates recursion overhead and is more cache-friendly
  std::size_t Find(std::size_t x) noexcept {
    // Find root
    std::size_t root = x;
    while (parent_[root] != root) {
      root = parent_[root];
    }

    // Path compression: make all nodes on path point directly to root
    while (parent_[x] != root) {
      const std::size_t next = parent_[x];
      parent_[x] = root;
      x = next;
    }

    return root;
  }

  // Union by rank - always attach smaller tree under root of larger tree
  void Unite(std::size_t a, std::size_t b) noexcept {
    a = Find(a);
    b = Find(b);
    if (a == b) return;

    // Union by rank
    if (rank_[a] < rank_[b]) {
      parent_[a] = b;
    } else if (rank_[a] > rank_[b]) {
      parent_[b] = a;
    } else {
      parent_[b] = a;
      rank_[a]++;
    }
  }

  // Check if two elements are in the same set (without path compression side effects)
  bool Connected(std::size_t a, std::size_t b) noexcept {
    return Find(a) == Find(b);
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<uint8_t>
      rank_;  // rank_ is bounded by log2(n), so uint8_t is sufficient for n < 2^255
};

}  // namespace psynth::sfm