# Corrected registration validation — 2026-09-23

The seven requested fixes are committed separately. The current deck table is
[deck_summary.md](deck_summary.md), with [CSV](deck_summary.csv) for import.
It is generated from fresh post-fix runs of the four requested products, not
transcribed from the audit or earlier phase reports. All pre-fix accuracy
figures are superseded for presentation purposes. All four products exported
successfully with coverage/extrapolation warnings; none meets the source-pixel
subpixel threshold. The corrected RMSE includes held-out mismatches, which
earlier inlier-filtered evaluation concealed.

## What the table measures

RMSE is symmetric transfer error on **all sealed held-out matcher
correspondences**, in working, original reference, original source and metre
units. No held-out point is removed because its error is large. Incorrect
matches therefore increase this number. These are correspondence measurements,
not independent geodetic control measurements. Pixel-to-metre and source-pixel
conversions use the product metadata's nominal sampling scale.

`pass` means model verification and adequate fit-point coverage. It does not
mean sub-pixel accuracy. `subpixel` compares the unrounded error with one source
pixel. The reference sampling scale in `accuracy_statement` is the size of one
reference pixel in source pixels; it is not a proven accuracy lower bound.
Coverage is over eligible overlap cells; extrapolation is the fraction of those
cells outside the fit-inlier hull. A low RMSE cannot establish accuracy outside
that hull.

OHRC and TMC-2 use `max_side=2048`, a 12 × 12 coverage grid and six segments.
Both IIRS runs use bare sensor defaults (no registration option overrides).
The IIRS defaults are `max_side=6144` and a 12 × 12 grid; the runtime memory cap
uses the actual target-window aspect ratio. Run-specific settings and artifact
paths are in [acceptance.json](acceptance.json).

## Fixes and evidence

1. `470165c` — freeze spatial fit/validation/test cells before method selection.
   RANSAC, model/method selection and fine-window placement use only fit points;
   ECC adoption uses a separate validation fold. The final test set is scored
   once after serialization, without inlier filtering. `evaluation.json`
   records the exact points and model fingerprint. Regression tests poison
   test-only correspondences and verify that the selected/exported model does
   not change (including ECC adoption and segment fits), and recompute RMSE
   from the exported transform.
2. `697201a` — zero-based pixel centres across match CSVs and transforms;
   GDAL affine conversion uses centre + 0.5. Continuous back-map interpolation
   preserves fractional source coordinates. Identical PNG and GeoTIFF pairs
   agree within 1e-6 pixel; fractional decimation offsets are tested separately.
3. `bf4de92` — the raster and preview apply segment transforms. Smoothstep
   blending of inverse coordinate maps between segment centres gives continuous
   transitions and one source sample per output pixel. Insufficient local fits
   explicitly fall back to the global model. Final RMSE evaluates this same
   composite field, identified by `accuracy.applied_model` and a full JSON hash.
   Regression checks use opposite segment translations to distinguish the
   actual export from a global-only warp, inspect the transition, and recompute
   the saved composite model's error.
4. `2da8c3e` — exclude finite nodata before calibration at both native and
   reduced resolution. Compose the full rotated/sheared GDAL affine with the
   output-grid translation and scale. Separate regression tests reproduce both
   audit cases, including a finite -9999 hole and nonzero affine cross terms.
5. `072ecc8` — separate status, source-pixel subpixel flag and the one-line
   `accuracy_statement`. The UI and generated report show that distinction.
6. `874ed4e` — put IIRS defaults in its sensor YAML, respect explicit overrides,
   and propagate defaults through Python, CLI and API. A bare real IIRS→WAC
   run exported successfully before this commit; both strips are rerun here.
7. `cd5b97e` — rerun the complete original Phase 2 illumination sweep with one
   criterion: **mean four-corner error ≤ 3 evaluation-grid pixels**. All 120
   cases, including failed fits and large azimuth differences, are retained.
   [Phase 2 §6](../PHASE2.md#reconciled-illumination-benchmark--2026-09-23)
   contains the replacement table and the browse-grid resolution. Earlier
   translation-consensus figures are retired. This benchmark and the audit's
   native NAC fixtures have different populations; their rates are not pooled.

For each successful real run the acceptance harness reloads `transform.json`,
checks its hash and independently recomputes held-out RMSE, asserting exact
float equality with `metrics.json`. The small JSON artifacts are committed under
[evidence/](evidence/); the exported rasters and complete product packages are
in the output paths recorded in `acceptance.json`.

## Checks

- 12 audit regression tests passed.
- 35 existing registration-tool tests passed.
- 28 OHRC reader/dataset tests passed.
- Frontend production build passed; its existing bundle-size notice remains.
- Illumination sweep: all 120 cases completed without execution errors.
- Real-product acceptance results and export read-back checks are in the linked
  table and `acceptance.json`.

Reproduce the product runs:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u \
  reports/validation_fixes_20260923/run_acceptance.py
```

## Known limits

- Large illumination changes beyond approximately 90° Sun azimuth remain a
  declared limitation; the matcher was not changed to address them.
- Initial pose depends on available label geometry. Blind rotation/scale
  recovery is not established by these results.
- Partial-overlap strips have limited geometric coverage. Coverage and
  extrapolation warnings are reported, not suppressed.
- IIRS exports one pseudo-pan band, not a warped full spectral cube.
