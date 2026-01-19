# Data formats

## Feature cache

Directory: `<out_dir>/features/`

File per image:
- `00000000.yml.gz`
- `00000001.yml.gz`
- …

Contents:
- `keypoints`: sequence of `{x,y,size,angle,response,octave,class_id}`
- `descriptors`: OpenCV matrix (float32), rows correspond to keypoints

## Verified pair cache

File: `<out_dir>/verified_pairs.yml.gz`

Contents:
- `verified_pairs`: sequence of entries:
  - `i`, `j`: image ids
  - `F`: 3x3 matrix (float64)
  - `inliers_idx`: Nx2 matrix (int32) containing `(kp_i, kp_j)`
  - `inliers_dist`: Nx1 matrix (float32) containing descriptor distances
  - `num_ransac_iters`, `inlier_threshold_px`

## Sparse export

- `<out_dir>/sparse.ply`: triangulated points with RGB from a reference observation (ASCII PLY).
- `<out_dir>/cameras.ply`: camera centers as vertices (ASCII PLY).

## Bundler export

- `<out_dir>/bundle.out` in “Bundle file v0.3” format:
  - Per camera: `<f> <k1> <k2>`, then `R` rows, then `t`.
  - Per point: `X`, `RGB`, and a list of `(camera_idx, keypoint_id, x, y)` observations.

The camera coordinate transform is applied to match Bundler’s `-Z` looking convention.

## PMVS2 input export

Root: `<out_dir>/<pmvs_root>/`

Subdirs:
- `visualize/00000000.jpg`, …
- `txt/00000000.txt`, …

Files:
- `options.txt`: PMVS2 options.
- `vis.dat` (optional): visibility adjacency list for image graph (VISDATA format).