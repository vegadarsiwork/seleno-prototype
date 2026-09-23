# Model-flexibility experiment (offline, saved fit points only)

Every model is refitted by the same RANSAC (threshold 3 reference px, as in the fine
stage) and least-squares refit on the saved fit-fold points (`matches.csv`), then
scored on the frozen test-cell points (`evaluation.json`). Nothing is filtered.
"exported" is the shipped affine plus segments. The systematic/random split uses the
same neighbour method as `diagnostics/`. `register.py` is unchanged.

## Reconstruction check

The fit points are rebuilt from original source pixels. The rebuilt points should
reproduce the pipeline's own inlier flags under the exported matrix. ECC changes the
matrix after the flags are set, so IIRS is expected to differ slightly. The sharper test
is a least-squares affine on the flagged inliers: OpenCV's refinement converges to it,
so without ECC it should reproduce the exported matrix.

| Pair | Fit points | Pinned, excluded | Rebuilt | Inlier flag reproduced | Inliers within threshold | LSQ affine on flagged inliers vs exported, max working px | ECC adopted |
|---|---:|---:|---:|---:|---:|---:|---|
| OHRC → NAC | 15905 | 0 | 15905 | 96.8% | 96.8% | 2.84e-06 | no |
| TMC-2 → SELENE | 497 | 13 | 484 | 96.3% | 96.6% | 1.89e-02 | no |
| IIRS 2025-07-29 → WAC | 169 | 0 | 169 | 92.9% | 91.7% | 1.18e+00 | yes |
| IIRS 2024-01-20 → WAC | 137 | 7 | 126 | 98.4% | 95.5% | 4.96e-01 | yes |

## Fit-point conditioning (exported model's fit inliers, working grid)

| Pair | Inliers | Hull covers overlap | Axis ratio minor/major | Fit spread vs overlap spread, minor axis | major axis |
|---|---:|---:|---:|---:|---:|
| OHRC → NAC | 6104 | 19.9% | 0.284 | 0.253 | 0.588 |
| TMC-2 → SELENE | 266 | 21.0% | 0.208 | 0.370 | 0.411 |
| IIRS 2025-07-29 → WAC | 48 | 17.9% | 0.006 | 0.240 | 0.858 |
| IIRS 2024-01-20 → WAC | 22 | 11.3% | 0.009 | 0.190 | 0.854 |

Spread vs overlap is the fit cloud's standard deviation along its own principal axis
divided by the overlap's along ITS principal axis (sorted, not aligned).

## OHRC → NAC — 6220 held-out points

| Model | Fit inliers | Median ref px | p90 ref px | RMSE ref px | Median src px | p90 src px | ≤1 ref px | ≤2 ref px | ≤3 ref px | Systematic ref px | Random ref px | Scale | Rotation ° | sx / sy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exported (affine + segments) | 6104 | 6.57 | 13.30 | 25.55 | 27.38 | 55.40 | 4.0% | 15.6% | 27.8% | 6.56 | 0.95 | 1.04548 | -0.6114 | 0.9925 / 1.1014 |
| translation | 602 | 30.58 | 75.69 | 50.44 | 127.43 | 315.38 | 0.0% | 0.0% | 0.0% | 29.98 | 1.92 | 1.00000 | 0.0000 | 1.0000 / 1.0000 |
| rigid | 1323 | 63.36 | 112.24 | 77.00 | 264.00 | 467.69 | 0.0% | 0.0% | 0.0% | 63.03 | 1.88 | 1.00000 | -1.9589 | 1.0000 / 1.0000 |
| similarity | 1692 | 48.61 | 98.12 | 65.10 | 202.55 | 408.84 | 0.1% | 0.5% | 1.4% | 47.71 | 1.86 | 0.99419 | -1.3979 | 0.9942 / 0.9942 |
| affine **(wins)** | 6399 | 5.90 | 11.36 | 25.29 | 24.57 | 47.35 | 5.0% | 12.0% | 25.7% | 5.79 | 0.93 | 1.05099 | -0.4104 | 0.9929 / 1.1125 |

## TMC-2 → SELENE — 479 held-out points

| Model | Fit inliers | Median ref px | p90 ref px | RMSE ref px | Median src px | p90 src px | ≤1 ref px | ≤2 ref px | ≤3 ref px | Systematic ref px | Random ref px | Scale | Rotation ° | sx / sy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exported (affine + segments) | 267 | 4.46 | 11.81 | 7.27 | 6.03 | 15.96 | 8.6% | 21.3% | 36.1% | 3.74 | 1.01 | 1.00658 | -0.0089 | 1.0133 / 0.9999 |
| translation | 171 | 7.53 | 17.69 | 10.54 | 10.18 | 23.90 | 1.3% | 4.4% | 12.1% | 6.85 | 0.98 | 1.00000 | 0.0000 | 1.0000 / 1.0000 |
| rigid | 189 | 9.66 | 21.07 | 12.54 | 13.05 | 28.46 | 5.2% | 12.5% | 19.8% | 9.19 | 1.01 | 1.00000 | -0.1143 | 1.0000 / 1.0000 |
| similarity | 189 | 9.62 | 21.00 | 12.51 | 13.00 | 28.37 | 5.8% | 13.8% | 20.5% | 9.10 | 1.00 | 1.00006 | -0.1146 | 1.0001 / 1.0001 |
| affine **(wins)** | 279 | 3.74 | 13.11 | 7.74 | 5.05 | 17.72 | 14.8% | 30.5% | 43.8% | 2.96 | 0.95 | 1.00814 | 0.0190 | 1.0165 / 0.9999 |

## IIRS 2025-07-29 → WAC — 166 held-out points

| Model | Fit inliers | Median ref px | p90 ref px | RMSE ref px | Median src px | p90 src px | ≤1 ref px | ≤2 ref px | ≤3 ref px | Systematic ref px | Random ref px | Scale | Rotation ° | sx / sy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exported (affine + segments) | 48 | 10.16 | 16.03 | 10.88 | 11.92 | 18.81 | 1.2% | 5.4% | 10.2% | 10.62 | 4.79 | 0.99297 | -0.2153 | 0.9883 / 0.9977 |
| translation | 25 | 14.88 | 22.11 | 15.87 | 17.47 | 25.96 | 0.0% | 0.0% | 0.6% | 12.69 | 5.29 | 1.00000 | 0.0000 | 1.0000 / 1.0000 |
| rigid | 30 | 15.82 | 25.61 | 17.64 | 18.57 | 30.07 | 0.0% | 0.0% | 1.2% | 13.31 | 5.30 | 1.00000 | 0.0585 | 1.0000 / 1.0000 |
| similarity | 55 | 11.40 | 17.25 | 11.91 | 13.38 | 20.25 | 3.0% | 9.0% | 13.9% | 10.78 | 5.19 | 0.99753 | 0.0093 | 0.9975 / 0.9975 |
| affine **(wins)** | 109 | 2.14 | 11.30 | 7.87 | 2.51 | 13.26 | 15.7% | 47.6% | 66.3% | 1.31 | 1.79 | 0.92542 | 0.2103 | 0.8585 / 0.9976 |

## IIRS 2024-01-20 → WAC — 127 held-out points

| Model | Fit inliers | Median ref px | p90 ref px | RMSE ref px | Median src px | p90 src px | ≤1 ref px | ≤2 ref px | ≤3 ref px | Systematic ref px | Random ref px | Scale | Rotation ° | sx / sy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exported (affine + segments) | 22 | 10.96 | 21.30 | 13.29 | 19.31 | 37.55 | 2.4% | 7.9% | 15.0% | 2.74 | 5.43 | 0.99607 | -0.3128 | 0.9925 / 0.9996 |
| translation | 20 | 10.75 | 19.93 | 12.66 | 18.95 | 35.13 | 2.4% | 4.7% | 8.7% | 5.54 | 5.90 | 1.00000 | 0.0000 | 1.0000 / 1.0000 |
| rigid | 25 | 13.87 | 22.11 | 14.79 | 24.44 | 38.97 | 1.6% | 8.7% | 13.4% | 13.26 | 5.90 | 1.00000 | 0.0682 | 1.0000 / 1.0000 |
| similarity | 25 | 13.42 | 22.33 | 14.80 | 23.66 | 39.36 | 1.6% | 7.9% | 13.4% | 13.21 | 5.89 | 0.99977 | 0.0660 | 0.9998 / 0.9998 |
| affine **(wins)** | 54 | 9.40 | 33.34 | 18.35 | 16.56 | 58.76 | 3.1% | 14.2% | 25.2% | 6.56 | 5.44 | 0.93067 | 0.3853 | 0.8669 / 0.9992 |

## Winner per pair (lowest held-out median, p90 as tie-break)

| Pair | Winner | Ranking |
|---|---|---|
| OHRC → NAC | affine | affine < translation < similarity < rigid |
| TMC-2 → SELENE | affine | affine < translation < similarity < rigid |
| IIRS 2025-07-29 → WAC | affine | affine < similarity < translation < rigid |
| IIRS 2024-01-20 → WAC | affine | affine < translation < similarity < rigid |
