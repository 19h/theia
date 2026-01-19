# Mathematical notes (implementation-aligned)

## Fundamental matrix estimation (normalized 8-point)

Given pixel correspondences `x1_px[i], x2_px[i]` in homogeneous coordinates:
1. Normalize each point set with a similarity transform `T` such that centroid is at origin and RMS distance is `sqrt(2)`.
2. Solve `A f = 0` by SVD (smallest right singular vector).
3. Enforce `rank(F)=2` by setting the smallest singular value of `F` to zero.
4. Denormalize: `F = T2^T * F_norm * T1`.

The inlier error metric is the Sampson distance:
`d^2 = (x2^T F x1)^2 / ( (F x1)_0^2 + (F x1)_1^2 + (F^T x2)_0^2 + (F^T x2)_1^2 )`.

## Essential matrix and relative pose

With intrinsics `K1, K2`:
`E = K2^T * F * K1`.

The code projects `E` back to the essential manifold (two equal singular values, one zero) and decomposes:
- `R = U W V^T` or `U W^T V^T`
- `t = ± U[:,2]`

A cheirality test selects the solution maximizing positive depth counts after DLT triangulation in normalized coordinates.

## Camera model

World-to-camera:
`X_cam = R * X_world + t`.

Projection (Brown radial, 2 coefficients):
- `x = X/Z`, `y = Y/Z`
- `r^2 = x^2 + y^2`
- `d = 1 + k1 r^2 + k2 r^4`
- `u = f * d * x + cx`, `v = f * d * y + cy`

Undistorted normalized coordinates for triangulation are obtained by fixed-point inversion:
`x0 = (u-cx)/f`, iterate `x <- x0 / (1 + k1 r^2 + k2 r^4)`.

## Triangulation (DLT)

Given two projection matrices `P1, P2` that map world points to normalized image coordinates, and normalized observations `(x1,y1)`, `(x2,y2)`:

Solve `A X = 0` with:
- row1: `x1 * P1_3^T - P1_1^T`
- row2: `y1 * P1_3^T - P1_2^T`
- row3: `x2 * P2_3^T - P2_1^T`
- row4: `y2 * P2_3^T - P2_2^T`

`X` is the last right singular vector (homogeneous), then dehomogenized.

## Bundle adjustment

Ceres optimizes:
- per-camera pose `(angle-axis, t)` and optional intrinsics `(f,k1,k2)`
- per-point `(X,Y,Z)`

Residuals are 2D reprojection errors in pixels with Huber robust loss.
Gauge ambiguity is removed by fixing the seed pair camera poses.