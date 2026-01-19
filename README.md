# psynth

psynth is a C++17 reference implementation of a Photosynth / Photo-Tourism style reconstruction pipeline:

SIFT (OpenCV) → symmetric descriptor matching → fundamental-matrix RANSAC → track building (union-find) → incremental SfM (PnP + triangulation) → bundle adjustment (Ceres) → export to Bundler + PLY + PMVS2 input layout.

## Build

Dependencies:
- CMake >= 3.20
- OpenCV (core, imgcodecs, imgproc, features2d, calib3d, flann)
- Eigen3
- Ceres Solver

Example (out-of-source):
```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j