## Real-pair check points

| Pair | Unit | RMSE | Median | p90 | ≤1 unit | Invalid / n |
|---|---|---|---|---|---|---|
| ohrc_nac | source | 3.840 | 3.086 | 5.844 | 8.0% | 0/412 |
| ohrc_nac | reference | 1.057 | 0.847 | 1.642 | 62.4% | 0/412 |
| ohrc_nac | metres | 1.057 | 0.847 | 1.642 | 62.4% | 0/412 |
| tmc2_selene | source | 2.336 | 1.393 | 3.443 | 33.6% | 0/792 |
| tmc2_selene | reference | 1.746 | 1.015 | 2.540 | 49.2% | 0/792 |
| tmc2_selene | metres | 12.923 | 7.513 | 18.806 | 2.1% | 0/792 |
| iirs_20250729_wac | source | 1.870 | 1.236 | 2.596 | 36.3% | 0/292 |
| iirs_20250729_wac | reference | 1.616 | 1.062 | 2.133 | 46.6% | 0/292 |
| iirs_20250729_wac | metres | 161.572 | 106.214 | 213.313 | 0.0% | 0/292 |
| iirs_20240120_wac | source | 1.312 | 0.743 | 2.468 | 67.4% | 0/648 |
| iirs_20240120_wac | reference | 1.191 | 0.698 | 2.203 | 69.1% | 0/648 |
| iirs_20240120_wac | metres | 119.141 | 69.818 | 220.327 | 0.0% | 0/648 |
| tmc2_selene_morning | source | — | — | — | — | 0/0 |
| tmc2_selene_morning | reference | — | — | — | — | 0/0 |

## Spatial support and unscreened evidence

| Pair | Coverage | Extrapolation | Check-point retention | All-held source RMSE | All-held invalid | Fit inliers | Native points | Patch | Terms | Accepted |
|---|---|---|---|---|---|---|---|---|---|---|
| ohrc_nac | 67.7% | 52.9% | 99.3% | 7.280 | 0 | 1259 | 1986 | 41 | affine + terrain | False |
| tmc2_selene | 78.3% | 31.7% | 99.4% | 2.594 | 0 | 1249 | 2302 | 41 | affine + field | False |
| iirs_20250729_wac | 69.0% | 56.9% | 98.0% | 2.038 | 0 | 474 | 858 | 41 | affine + field | False |
| iirs_20240120_wac | 75.3% | 47.9% | 98.9% | 1.487 | 0 | 1256 | 2052 | 61 | affine + field | False |
| tmc2_selene_morning | — | — | — | — | 0 | — | — | — | none (refused) | False |

## Synthetic controls

| Case | Sun Δ° | Status | Truth RMSE (src px) | Corner RMSE / max (ref px) | Acceptance |
|---|---|---|---|---|---|
| translation | — | pass | 0.011 | 0.026 / 0.037 | True |
| rotation_60 | — | pass | 0.008 | 0.018 / 0.031 | True |
| scale_1p5_no_metadata | — | pass | 0.184 | 0.130 / 0.150 | True |
| scale_0p5_no_metadata | — | warning | 0.399 | 2.954 / 4.032 | False |
| gamma | — | pass | 0.008 | 0.017 / 0.027 | True |
| contrast_inversion | — | pass | 0.007 | 0.013 / 0.023 | True |
| rotation_120_scale_1p5 | — | warning | 0.764 | 0.496 / 0.512 | False |
| scale4_georeferenced | — | pass | 0.000 | 0.000 / 0.000 | True |
| decimated_identity | — | pass | 0.008 | 0.010 / 0.014 | True |

## Illumination controls

| Case | Sun Δ° | Status | Truth RMSE (src px) | Corner RMSE / max (ref px) | Acceptance |
|---|---|---|---|---|---|
| lroc_005_055_r6144_c27648 | 50 | warning | 0.191 | 0.356 / 0.454 | False |
| lroc_005_055_r4096_c23552 | 50 | warning | 0.232 | 0.370 / 0.544 | False |
| lroc_095_145_r4096_c43008 | 50 | warning | 0.443 | 0.383 / 0.548 | False |
| lroc_095_145_r9216_c43008 | 50 | warning | 0.442 | 0.623 / 0.773 | False |
| lroc_145_235_r9216_c41984 | 90 | warning | 0.551 | 1.186 / 1.771 | False |
| lroc_235_325_r1024_c43008 | 90 | warning | 0.232 | 0.358 / 0.536 | False |
| lroc_235_325_r3072_c41984 | 90 | warning | 0.294 | 0.566 / 0.751 | False |
| lroc_055_145_r8192_c43008 | 90 | warning | 0.230 | 0.597 / 1.020 | False |
| lroc_055_145_r5120_c43008 | 90 | warning | 0.553 | 0.582 / 0.968 | False |
| lroc_145_325_r3072_c43008 | 180 | failed | — | — / — | False |
| lroc_145_325_r12288_c32768 | 180 | failed | — | — / — | False |
| lroc_055_235_r10240_c43008 | 180 | failed | — | — / — | False |
| lroc_005_235_r6144_c27648_neg | 130 | failed (negative) | — | — / — | True |
| lroc_005_325_r4096_c23552_neg | 40 | failed (negative) | — | — / — | True |

## Illumination by sun-angle difference

Corner aggregates include exported positive cases only; refusal counts are shown beside them. Bins include their upper endpoint; adjacent lower endpoints are excluded.

| Sun Δ° | Positive cases | Exported | Accepted | Median / worst corner RMSE (ref px) | Negative rejected or flagged / n |
|---|---|---|---|---|---|
| 0–60 | 4 | 4 | 0 | 0.376 / 0.623 | 1/1 |
| 60–90 | 5 | 5 | 0 | 0.582 / 1.186 | 0/0 |
| 90–135 | 0 | 0 | 0 | — / — | 1/1 |
| 135–180 | 3 | 0 | 0 | — / — | 0/0 |
