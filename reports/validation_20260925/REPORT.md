# Dense registration validation — 2026-09-25

This report uses the fresh final automatic runs, including full raster exports.
The review/deck table is [deck_summary.md](deck_summary.md), with
[CSV](deck_summary.csv) for import. All changes remain uncommitted.
Development sweep winners are not substituted for these final results.

## What the numbers measure

Real-pair errors are measured on sealed held-out matcher correspondences, not
surveyed control points. Headline check points pass a neighbour-consistency
screen relative to the frozen coarse model. The screen never consults the
exported model. Unscreened errors, excluded counts and invalid predictions remain
reported beside the headline values. At least 80% of held-out measurements must
survive the screen. The test fold does not select patches, methods, prewarp
retries, model terms, ECC or refits.

Source errors below are backward transfer through the complete export and native
geometry, in original detector pixels. Reference errors are forward transfer in
native reference pixels. Metre errors are that forward displacement multiplied
by reference sampling; they are not independent geodetic measurements. These two
directional errors need not equal a nominal GSD conversion of one another.
Legacy `rmse_reference_px`/`rmse_m` headline fields still convert the symmetric
working-grid transfer distance; the tables use the explicit directional
`check_point_reference_*` assessment. Table percentages mean **≤1 stated unit**;
the acceptance gate retains its existing strict **<1 source pixel** comparison.
Statistics over valid predictions remain visible when others are invalid, but
**any invalid held-out prediction still fails acceptance**.

Acceptance requires at least 12 check points, source RMSE and p90 below one pixel,
at least 95% below one pixel, bias below 0.5 pixel, at least 50% eligible-cell fit
coverage, and at most 25% extrapolation, in addition to check-point retention and
validity. Export completion and acceptance are separate outcomes. Independent
verification also requires adequate external controls and uncertainty ≤0.25
source pixel.

For known-transform controls, grid errors are backward errors in source pixels.
Corner errors evaluate the complete serialized model at all four source corners
in reference pixels, including corners outside the measured overlap. Illumination
controls use map alignment plus a known image warp; they are not surveyed detector
points. Their declared 0.8-source-pixel reference uncertainty prevents independent
subpixel certification even when measured displacement is small.

## Changes and regression evidence

1. Coarse candidates use a-contrario NFA significance and overlap-wide fold,
   scale, anisotropy and area-scale checks. Unrelated-image regressions still
   produce declared failures.
2. Coarse dense measurements and the native stage use lattice seeds. Native
   tiles span predicted overlap, patches shrink near masks/borders, and outward
   retries use fit-derived affine prewarps. Retry decisions now compare fit-fold
   counts only; test-only tiles follow that fit-derived schedule.
3. Dense fitting combines loose RANSAC, neighbour consistency and cell-balanced
   least squares. DEM height and B-spline terms earn adoption through spatial
   cross-validation over fit cells. They replace segment fitting when adopted.
4. The complete model is shared by export, previews and evaluation, with
   serializable `local_field` and `parallax` terms and frame conjugation.
5. Spatial folds follow elongated overlap geometry. Both raw held-out and
   screened check-point statistics are exported, along with reproducible evidence.
6. Export is cropped, streams every source band, and uses local-Jacobian
   supersampling for finer sources. All-NaN tiles no longer invoke `nanmedian`.
   Lazy raster slices release full read blocks and grid budgeting respects cgroups.
7. Geometry subsampling retains the final row/column. Newton inversion of the
   bilinear forward lattice rejects curved-edge hull slivers. New tests caught
   and fixed stale pointer-based inverse caching and four-node extrapolation.
8. New terrain tests cover pixel centres, bilinear heights, borders, invalid
   points and composed frame conjugation. Lattice measurement tests use a known
   continuous subpixel translation. Patch-CV tests poison validation/test data
   and verify an unchanged choice; coverage loss cannot earn a larger patch.
9. The accuracy panel exposes check points, invalid predictions, selected patch,
   field and terrain adoption. Documentation describes evidence files, complete
   transforms, memory limits and the distinction between consistency and truth.

## Patch choice and development sweep

The earlier sweep changed `_FINE_PATCH`, but `_measure_seeds` had captured 41 as
a Python default argument. Its purported “61” variants changed padding and seed
spacing without changing the correlation measurement patch. Those results do
not establish that a measured 61-pixel patch is best for OHRC or IIRS.

After fixing that bug and the geometry regressions, the requested IIRS 2025
development sweep measured both patches with current lattice inversion:

| Effective patch | Native measurements | Check points | Source RMSE / median / p90 (px) | <1 src px | Coverage | Extrapolation |
|---|---:|---:|---|---:|---:|---:|
| 41 | 864 | 297 | 1.743 / 1.238 / 2.472 | 36.4% | 70.7% | 56.0% |
| 61 | 454 | 157 | 1.287 / 0.899 / 2.035 | 54.8% | 57.3% | 64.1% |

This development run stubs raster export; the final runs below export all bands.
The sweep also changes minimum seed spacing and tile padding, so its point-count
loss cannot be attributed to patch measurement alone. The 61 configuration
improves this development residual while providing fewer points and less support.
The final selector compares 41/61 on identical probe tiles and common fit-fold probe
seeds, by whole-cell CV of an inverse affine. It requires ≥5% median improvement,
no more than 5% tail worsening, and retention of ≥80% of the default's fit
measurements. Too few common points/cells keeps 41. Thresholds were fixed before
final validation; held-out scores were not used to revise them. Per-pair decisions
and candidate tables are recorded in `fine_stage.patch_cv`.

No global measurement relaxation was adopted. The default correlation threshold
and forward/backward limit remain **0.55 and 0.35**. Earlier development sweeps
found relaxed thresholds particularly harmful on OHRC. Evening SELENE
(`TCO_MAPe04`) remains the primary TMC-2 reference; the low-sun morning map is a
separate secondary cross-illumination case.

## Results at a glance

**Real pairs (four primary).** All four exported with zero invalid held-out
predictions. None meets the fixed acceptance gates. Every one fails the source
RMSE, p90 and ≤1 px gates and has more than 25% of its area outside fit-point
support (32–57%).

| Pair | Check-point source RMSE / median (px) | Reference RMSE / median (px) | Terms adopted (fit-cell CV) | Patch |
|---|---|---|---|---:|
| OHRC → NAC | 3.84 / 3.09 | 1.06 / 0.85 (≈ 1.06 m / 0.85 m) | affine + terrain parallax | 41 |
| TMC-2 → SELENE (evening) | 2.34 / 1.39 | 1.75 / 1.01 (12.9 m RMSE) | affine + B-spline field | 41 |
| IIRS 2025-07-29 → WAC | 1.87 / 1.24 | 1.62 / 1.06 | affine + B-spline field | 41 |
| IIRS 2024-01-20 → WAC | 1.31 / 0.74 (67% ≤ 1 px) | 1.19 / 0.70 | affine + B-spline field | 61 |

- On OHRC, the terrain term cut the fit-cell CV median from 1.45 to 0.26
  working px. The field was then not adopted, because the model without it had
  the lower CV (0.15 vs ≥0.18). TMC-2 and both IIRS pairs have no DEM, and each
  adopted a field.
- The patch selector chose 61 only for IIRS 2024: CV median 1.06 → 0.87 and
  p90 2.71 → 1.84. It kept 41 on TMC-2, where 61 was 4% better on median and
  10% better on p90, below the fixed 5% median requirement. It also kept 41 on
  OHRC and IIRS 2025, where the medians were within 1%.
- The check-point screen removed 3–7 points per pair (98.0–99.4% retained).
  Unscreened source RMSEs are in the support table below. OHRC's 7.28 px comes
  from a few isolated mismatches: the unscreened median is unchanged at 3.09 px.
- The secondary morning-SELENE TMC-2 case was refused: no coarse candidate
  verified.

**Final runs compared with the development numbers in `CODEX_HANDOFF.md`.**
OHRC 3.84 (dev 3.0–3.3), TMC-2 2.34 (dev 2.27), IIRS 2025 1.87 (dev 1.62) and
IIRS 2024 1.31 (dev 1.49). The development "patch 61" runs did not measure with
a 61 px patch (see the previous section). The better OHRC and IIRS 2025
development figures therefore cannot be attributed to patch size, and they are
not substituted here.

**Synthetic controls with exact truth.** 7 of 9 were accepted. Truth RMSE was
0.000–0.184 source px for the seven that were accepted.
- `scale_0p5_no_metadata`: truth RMSE was 0.40 px, but 89.8% ≤ 1 px, more than
  25% extrapolation and too little independent-control coverage. Corner error
  reached 2.95 / 4.03 reference px, where the model extrapolates.
- `rotation_120_scale_1p5`: truth RMSE 0.76 px, with a 0.76 px bias that fails
  the < 0.5 px bias gate.

**LROC illumination controls** (known warp on real NAC mosaic imagery).
- **Exported:** 9 of 12 positive pairs, all at Sun Δ ≤ 90°. Grid truth RMSE was
  0.19–0.55 source px and corner RMSE 0.36–1.19 reference px.
- **Accepted:** none. Some internal gates fail: p90, ≤1 px or extrapolation.
  Independent certification is also impossible, because these controls declare
  0.8 px reference uncertainty and certification allows at most 0.25 px.
- **Refusals:** all three 180° pairs were refused. Candidates were rejected by
  NFA significance, local scale/anisotropy or minimum-point checks, so no wrong
  transform was exported. Both different-ground negatives were also refused.
- **Compared with 23 September:** that run exported 5 of 12 positives, and two of
  those had corner errors of 8.2 and 171.8 px.

**Harness fix during this validation.** The first illumination run reported
both negative controls as `crashed` in 0.0 s. `validate_registration.py`
passed the manifest's null `gt_homography` to the truth-grid builder. The
harness now builds a truth manifest only when a transform exists. The whole
illumination suite was rerun; the tables below come from that rerun.

**Resources.**
- **Real suite:** peak RSS was 4.24 GB. The 5.5 GB cgroup limit was reached
  (46,072 `max` reclaim events), mostly page cache from the 1.4–1.8 GB IIRS
  raster writes. There were zero OOM kills.
- **Synthetic and illumination suites:** peak RSS was 1.09 GB and 2.88 GB, with
  no limit events.
- **Tests:** all passed after the final code change: `test_tool` 35/35,
  `test_ohrc_dataset` 28/28, plus the six focused unittest modules.

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

## Limits and interpretation

- Sub-source-pixel acceptance is not established for the real pairs. One NAC
  pixel spans approximately 4.17 OHRC pixels, which makes precise source-pixel
  registration difficult. Sampling scale is not a proven mathematical error floor.
- Real pairs have no independently surveyed controls in this run. A low matcher
  residual cannot certify geodetic accuracy or accuracy outside tie-point support.
- Polar LOLA DEMs enable the OHRC terrain term. The equatorial TMC-2 and IIRS
  scenes have no DEM in this archive. Reapplying a parallax transform requires
  the referenced DEM; outside its domain the height correction is zero.
- Large illumination differences remain difficult. Known-transform results,
  refusal counts and corner errors are stratified by Sun-angle difference above;
  successful cases alone must not be presented as the whole suite.
- The September 23 baselines (106.5, 12.3, 12.7 and 24.4 source pixels for OHRC,
  TMC-2, IIRS 2025 and IIRS 2024) used different scoring: all held-out symmetric
  residuals with nominal unit conversion. Comparing them directly to screened,
  native-source backward errors would overstate a controlled improvement factor.
  The final table includes unscreened native-source errors for context.

## Reproduction and artifacts

Run one heavy job at a time. `/tmp` is tmpfs on this machine; even temporary
test rasters belong on disk. The validation commands used were:

```bash
mkdir -p outputs/validation_20260925/tmp
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export TMPDIR="$PWD/outputs/validation_20260925/tmp"
for suite in real synthetic illumination; do
  systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0 \
    .venv/bin/python -u scripts/validate_registration.py --suite "$suite" \
    --out reports/validation_20260925 --artifacts outputs/validation_20260925
done
systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 \
  .venv/bin/python scripts/summarize_validation.py --out reports/validation_20260925
```

Nonzero validation exit codes indicate failed fixed acceptance gates, not
necessarily failed export or a crash. Reports record the reason for each case.
OHRC/TMC-2 use a 2048-side request and 12×12 grid; IIRS uses sensor defaults,
subject to the cgroup-aware working-grid cap. Actual frames and settings are
preserved in the evidence.

Review files: [real.json](real.json), [synthetic.json](synthetic.json),
[illumination.json](illumination.json), [summary.json](summary.json),
[report_tables.md](report_tables.md), and `evidence/<case>/` model/metrics/evaluation
files. Full-resolution rasters, all match points and `tiepoints.npz` remain under
`outputs/validation_20260925/<case>/<job_id>/`. Earlier September 25 report files
were copied to `outputs/validation_20260925/previous_report/` before replacement.
Raw sweep and suite logs are under `outputs/validation_20260925/`.

To supply independent real-image controls, use
`scripts/validate_registration.py --ground-truth-directory <directory>` with
image-bound `<case>.json` manifests. Tests and documentation remain part of the
uncommitted review; no commit has been made.
