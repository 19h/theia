# Architecture

## Pipeline graph

1. `io::ImageDataset`
   - Enumerates image files and initializes intrinsics (width/height/cx/cy/f/k1/k2).
2. `features::ExtractSIFT` (OpenCV SIFT)
   - Produces `FeatureSet{keypoints, descriptors}` per image.
3. `matching::MatchSiftSymmetric`
   - FLANN KNN matching + Lowe ratio + symmetric cross-check.
4. `geometry::EstimateFundamentalRansac`
   - Normalized 8-point fit + Sampson-distance inlier scoring under RANSAC.
5. `sfm::BuildTracksUnionFind`
   - Builds multi-view tracks from verified inlier matches using union-find.
6. `sfm::IncrementalSfM`
   - Initialization from a verified seed pair.
   - Incremental registration using PnP-RANSAC (OpenCV `solvePnPRansac`).
   - Triangulation of new tracks by DLT from best-baseline registered views.
   - Global bundle adjustment with gauge anchors.
7. Export:
   - `io::ExportBundleOut` (Bundler v0.3)
   - `io::ExportSparsePLY` / `io::ExportCamerasPLY`
   - `mvs::ExportPMVS` (PMVS2 input layout)

## Coordinate conventions

- World-to-camera: `X_cam = R * X_world + t` (OpenCV-style, `+Z` forward).
- Pixel coordinates `(u,v)` use origin at top-left and `+v` down.
- Bundle export applies a fixed axis reflection to match Bundler's `-Z` viewing direction.

## Invariants

- `Track.id` equals index in `Tracks::all()` and is stable after construction.
- `Track.triangulated == true` implies `Track.xyz` is valid in the current gauge and at least two registered observations exist after filtering.
- `Reconstruction.gauge_fixed_{0,1}` store the seed pair used as gauge anchors for bundle adjustment.

## Complexity (dominant terms)

Let:
- `N` images,
- `F_i` features per image,
- `P = N(N-1)/2` pairs in exhaustive mode,
- `m_p` tentative matches in pair `p`,
- `k_F` RANSAC iterations per pair,
- `T` tracks,
- `O` total track observations.

Dominant costs:
- Feature extraction: `O(Σ cost_SIFT(image_i))`.
- Matching: `O(P * (F log F))` (approximate; FLANN-dependent).
- F RANSAC per pair: `O(k_F * m_p)` for scoring + `O(8^3)` for each model fit (SVD on small matrices).
- Track building (union-find): `O(U α(V))` where `U` unions, `V` matched keypoints, `α` inverse-Ackermann.
- Incremental SfM:
  - PnP-RANSAC: `O(k_PnP * c)` for `c` correspondences.
  - Triangulation: `O(T * s^2)` worst-case for choosing best baseline among `s` registered views per track (typically small).
- Bundle adjustment:
  - Solved by Ceres sparse Schur complement; scaling depends on camera count and observation sparsity.