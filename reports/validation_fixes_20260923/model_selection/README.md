# Model flexibility: result

**The hypothesis is rejected.** On all four real pairs, a more constrained model scores
worse than affine on the same frozen test cells. The full tables are in
[flex.md](flex.md); the script is [flex.py](flex.py).

| Pair | Affine median / p90 ref px | Best constrained model | Its median / p90 ref px | Inliers, affine vs translation-only |
|---|---:|---|---:|---:|
| OHRC → NAC | 5.90 / 11.36 | translation | 30.58 / 75.69 | 6399 vs 602 |
| TMC-2 → SELENE | 3.74 / 13.11 | translation | 7.53 / 17.69 | 279 vs 171 |
| IIRS 2025-07-29 → WAC | 2.14 / 11.30 | similarity | 11.40 / 17.25 | 109 vs 25 |
| IIRS 2024-01-20 → WAC | 9.40 / 33.34 | translation | 10.75 / 19.93 | 54 vs 20 |

- **The fit data demand the extra freedom.** Under translation-only, RANSAC finds a
  tenth of the OHRC inliers it finds under affine. The OHRC stretch reappears in an
  independent refit (sy 1.1125 against the exported 1.1014), and so does the TMC-2
  cross-track scale (sx 1.0165 against 1.0133). The metadata lattice does **not**
  already carry the scale: the correction is not mostly translation.
- **IIRS used a constrained model and lost to affine.** Both IIRS exports are
  similarity models (plus ECC). `_auto_model` returns `similarity` when the coarse
  stage has fewer than 30 candidates, and the IIRS coarse stages had 29 and 28. A
  global affine refit on the same saved fit points scores 2.14 ref px median against
  the exported 10.16 on IIRS 2025-07-29, with 109 inliers against 48.
- **IIRS 2024-01-20 is not a clean win.** Affine has the lowest median (9.40) but a far
  worse p90 (33.34, against 19.93 for translation-only). Its fit points are nearly
  collinear: the axis ratio is 0.009 and there are 22 inliers.
- **Segments do not help.** A global affine refit beats the exported affine plus 6
  segments on median for OHRC (5.90 vs 6.57) and TMC-2 (3.74 vs 4.46). TMC-2 p90 is
  slightly worse (13.11 vs 11.81).

So step 2 of the plan, which was conditional on a constrained model winning, was not
implemented, and `register.py` is unchanged.

## Findings made along the way

- **The lattice thinning drops the swath edge.** `_lattice_interpolators` thins the
  geometry lattice with one step, derived from its row count, on both axes. On narrow
  lattices this drops the last columns from the triangulation:

  | Product | Triangulation ends at sample | Source width |
  |---|---:|---:|
  | TMC-2 | 3500 | 4000 (the last 12.5% of the swath is unplaced) |
  | IIRS | 240 | 250 |
  | OHRC | 11999 | 12000 |
- **Pinned source coordinates in matches.csv.** 13 of 497 TMC-2 fit records have
  `src_x` = 3500.0 exactly, and 7 of 137 IIRS 2024-01-20 records have 230.0. They do
  not record where the point was, so they are excluded here (see the reconstruction
  check in flex.md).
