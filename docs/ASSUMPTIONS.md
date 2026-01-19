# Assumptions

A1. Intrinsics initialization uses a heuristic focal length
- Statement: `f = initial_focal_px_factor * max(width,height)` and `k1=k2=initial_k*` when no EXIF/calibration is provided.
- Stress test: replace intrinsics with EXIF-derived focal length; compare (i) cheirality success rates, (ii) BA reprojection RMS (px), (iii) fraction of tracks triangulated.

A2. OpenCV SIFT and FLANN implementations
- Statement: Feature detection/description and approximate NN search use OpenCV algorithms.
- Stress test: validate match repeatability via inlier count distributions and spatial coverage; swap to a different implementation and compare verified-pair graph connectivity.

A3. Radial distortion model is 2-parameter Brown with fixed principal point
- Statement: `k1,k2` are optimized (optional), principal point `(cx,cy)` is fixed at image center by dataset initialization.
- Stress test: observe residual patterns vs radius; if systematic, extend to `(cx,cy)` optimization and/or tangential terms.

A4. PnP uses OpenCV solvePnPRansac (AP3P/EPNP backend)
- Statement: Registration uses RANSAC PnP as implemented by OpenCV.
- Stress test: replace with a separate PnP solver; compare registration success and pose stability under BA.

A5. Track building assumes at most one keypoint per image per track; conflicting components are discarded
- Statement: Union-find components with duplicate image ids are removed.
- Stress test: track-conflict rate vs dataset size; implement track splitting if conflict rate reduces connectivity.

A6. BA gauge anchors are fixed to the seed pair
- Statement: Two camera poses are fixed to remove similarity ambiguity.
- Stress test: vary anchor choice and check coordinate drift and convergence stability.

A7. Pair selection is exhaustive over all image pairs
- Statement: no retrieval / vocabulary-tree candidate pruning is implemented.
- Stress test: measure runtime scaling vs N; insert retrieval stage for large N.

A8. Color assignment uses a single reference observation
- Statement: Track RGB is sampled from the first used observation (nearest pixel in a chosen image).
- Stress test: compare to multi-view color fusion (median in RGB/HSV) to reduce view-dependent artifacts.