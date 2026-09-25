# Registration results — 2026-09-25

Fresh final runs. Held-out check points are matcher measurements, not surveyed controls. Source errors use backward transfer; reference/metre errors use forward transfer. Percentages in this table use ≤1 source pixel. All real cases remain unverified independently.

| Pair | Role | Source RMSE / median / p90 (px) | ≤1 src px | Reference RMSE (px) | RMSE (m) | Coverage / extrapolation | Check / held | Patch | Model | Accepted |
|---|---|---|---|---|---|---|---|---|---|---|
| ohrc_nac | primary | 3.840 / 3.086 / 5.844 | 8.0% | 1.057 | 1.057 | 67.7% / 52.9% | 412/415 | 41 | affine + terrain | False |
| tmc2_selene | primary | 2.336 / 1.393 / 3.443 | 33.6% | 1.746 | 12.923 | 78.3% / 31.7% | 792/797 | 41 | affine + field | False |
| iirs_20250729_wac | primary | 1.870 / 1.236 / 2.596 | 36.3% | 1.616 | 161.572 | 69.0% / 56.9% | 292/298 | 41 | affine + field | False |
| iirs_20240120_wac | primary | 1.312 / 0.743 / 2.468 | 67.4% | 1.191 | 119.141 | 75.3% / 47.9% | 648/655 | 61 | affine + field | False |
| tmc2_selene_morning | secondary cross-illumination | — / — / — | — | — | — | — / — | 0/0 | — | none (refused) | False |

[Full report](REPORT.md) · [CSV](deck_summary.csv) · [Raw real results](real.json)
