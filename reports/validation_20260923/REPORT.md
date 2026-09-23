# Functional audit of the lunar registration requirements

Audit date: 2026-09-23. Verdict: **partially implemented; the complete stated requirement is not met.**

The software performs real image correspondence and registration. It reads real OHRC, TMC-2 and IIRS products, estimates source-to-reference alignment, writes imagery and match points, and reports metrics. However, the tested implementation does **not** establish general multimodal/sun-angle/scale invariance, sub-pixel source accuracy, or uniform correspondence across the overlap. There are also concrete coordinate, export and validation defects.

This audit evaluates the current working tree, including the pre-existing uncommitted locate changes, at base commit `0e3c128027b9220e1ba876cc4e6a333c2ec3b64d`. Application code was not modified. Scripts, logs and results created for this audit are in this directory; registration products are under `outputs/validation_20260923/`. `environment.json` records versions and source hashes.

## Requirement-by-requirement verdict

| Requirement | Verdict | Evidence and qualification |
|---|---|---|
| Source is moving; reference is fixed | Implemented | The source is placed in a common frame, matched, geometrically warped and exported using the reference CRS when available. |
| Find corresponding points | Implemented | Dense local NCC and sparse DISK+LightGlue/SIFT/AKAZE/ORB candidates, followed by robust geometric estimation. Real OHRC/NAC and TMC-2/SELENE runs produce hundreds or thousands of inliers. |
| OHRC, TMC-2 and IIRS input | Implemented for tested products | PDS4 labels and raw raster/cube readers; geometry CSV or IIRS ENVI location backplane; sensor profiles. This is not exhaustive support for every archive processing level or format. |
| Cross-sensor/multimodal correspondence | Partial | OHRC→NAC and TMC-2→SELENE worked. IIRS→WAC worked with a larger working grid. IIRS is reduced to a single reflected-light pseudo-pan for matching. These few pairs do not establish arbitrary sensor-pair generality. |
| Sun-angle invariance | Not demonstrated; fails tested difficult cases | Current tool solved 3/12 positive real-illumination NAC benchmark pairs within the existing 3 px corner-error criterion. None of the 90° or 180° cases met that criterion. |
| Viewpoint variation | Partial | Similarity, affine and homography models exist. A controlled modest perspective case works. Arbitrary rotation, terrain parallax and pushbroom line-dependent geometry are not solved generally. |
| Scale invariance | Partial | A known georeferenced 4:1 sampling difference works. A 1.5× same-canvas scale change without metadata is explicitly rejected. |
| Sub-pixel accuracy of the source image | Not met on tested real cross-sensor registrations | Reported residuals: OHRC→NAC 6.05 source px; TMC-2→SELENE 2.18; IIRS→WAC 2.11. These are internal correspondence residuals, not independent absolute accuracy. |
| Uniform distribution across images | Not met generally | Reference eligible-cell coverage is 16.18%, 60.00%, and 24.64% for those runs. Dense seeds are gridded; sparse inliers are not uniformly selected in the active tool. |
| Registered product and corresponding match points | Present, with defects/limitations | TIFF/CSV/JSON/PNG/report files are written. Export is reduced-resolution and single-band; coordinate conventions and rotated-grid export have reproducible defects. |
| RMSE, inlier count, inlier ratio | Present, with interpretation defects | Metrics are calculated, but selection uses the purported holdout set; an alternate fitted matrix may be scored rather than the exported matrix; `pass` does not require sub-pixel error. |

## Tests actually executed

- `tests/test_tool.py`: **35 passed, 0 failed, 0 skipped**.
- `tests/test_ohrc_dataset.py`: **28 passed, 0 failed, 0 skipped**, using the local OHRC archive.
- Frontend production build: **passed**, with a bundle-size advisory.
- API handler smoke test through ASGI: **passed** for file listing, starting a real registration, polling the job, exact agreement between job metrics and the metrics download, TIFF delivery and a ZIP containing all ten artifacts. See `api.json` and `api_heartbeat.log`. This does not constitute a browser interaction or live network-server test.
- Current generic registration tool: 14 NAC illumination cases, nine controlled registration cases, three successful real cross-sensor runs plus one IIRS resolution failure and one real disjoint-pair rejection.
- Additional probes: native same-image registration, exported-vs-holdout matrix, reference nodata handling, geographic pixel units and rotated GeoTIFF export.

Existing tests passing is useful but does not establish the science requirement. For example, `tests/test_tool.py:239` explicitly allows a rotated source to fail cleanly. The distribution test checks that a metric is reported, and one test asserts the scientifically unjustified reference-pixel floor discussed below. Archive tests mostly exercise loading, labels, geometry, and safe data access.

Tests ran on CPU with two compute threads. The tool dynamically caps the requested grid against available memory; exact working shapes and caps are in the run logs. Results can consequently vary with resource availability as well as parameters. Completed runs were preserved when two interrupted execution sessions had to be resumed; no termination command was sent.

Two preliminary ASGI probes stalled at file listing with idle worker threads. Adding a periodic event-loop timer in the audit harness allowed the complete API test to finish; the cause of the wakeup behavior was not established as an application defect. The preliminary sessions were left open in accordance with the request not to terminate test terminals. The registration benchmark processes themselves completed normally.

## Real product results from this audit

These are **new runs**, not numbers copied from previous reports. RMSE below means the tool's reported held-out correspondence residual, subject to the validation defects later in this report.

| Pair | Tool status | Inliers / candidates | Inlier ratio | RMSE, source px | RMSE, reference px | Reported RMSE, m | Eligible-cell coverage | Area outside tie-point hull |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OHRC 2024-11-15 13:26 → controlled NAC polar crop | warning | 7,764 / 23,477 | 33.07% | 6.0533 | 1.4528 | 1.453 | 16.18% | 88.24% |
| TMC-2 2026-08-13 06:27 → SELENE evening TC tile | pass | 628 / 1,635 | 38.41% | 2.1845 | 1.6170 | 11.971 | 60.00% | 27.27% |
| IIRS 2025-07-29 → WAC global mosaic, larger grid | warning | 81 / 379 | 21.37% | 2.1085 | 1.7960 | 179.600 | 24.64% | 86.96% |

The IIRS run with requested `max_side=2048` failed with `insufficient_matches`: at a 1,791×95 working grid the best methods supplied only seven candidates. Requesting the earlier report's `max_side=6144` was capped to 2,195 on this machine and produced a 2,150×114 grid, enabling registration. This is real resolution sensitivity, not a claim that IIRS ingestion fails.

IIRS 2024-01-20 → OHRC 2024-11-15 was correctly rejected as `no_overlap`. This is a negative case, not evidence of successful IIRS/OHRC correspondence.

IIRS was tested against **WAC**, which is a supplemental reference. This does not establish IIRS→NAC or IIRS→SELENE performance. No independently surveyed control points were supplied for the real cross-sensor pairs; therefore absolute source-sub-pixel accuracy cannot be certified from them.

Exact input filenames, options and output directories: `real.json` and `extra_iirs.json`. The IIRS reader selected **47 of 256 bands, 0.81–1.59 µm**, and used its location backplane.

### Real illumination benchmark

`data/lroc/manifest.json` contains 12 positive and two negative NAC pairs. The imagery and changes in illumination are real; the additional geometric warp is synthetic and known. This is stronger evidence than changing brightness in a single image, but it is not an independently surveyed cross-sensor control network. The manifest describes approximately 0.8 px (2σ) underlying control residual; results below that scale cannot certify absolute accuracy.

The evaluator uses the repository's existing **mean four-corner error ≤3 reference px** criterion, not the stricter user requirement of <1 source px. It compares the delivered transform with the known transform, separately from the tool's own RMSE.

| Change in sub-solar longitude | Positive cases | Within 3 px | Tool passes | Tool warnings | Tool failures |
|---|---:|---:|---:|---:|---:|
| 50° | 4 | 3 | 3 | 1 | 0 |
| 90° | 5 | 0 | 0 | 1 | 4 |
| 180° | 3 | 0 | 0 | 0 | 3 |

Both negatives were rejected. The three successful positives had corner errors **0.6660, 0.8637, and 0.9733 px**. Those are promising results on selected pairs, not proof of general invariance.

Two warning results demonstrate the distinction between internal consistency and true alignment:

| Pair | Reported internal RMSE, ref px | Known-transform mean corner error, px |
|---|---:|---:|
| `lroc_095_145_r9216_c43008` | 3.7571 | 171.8393 |
| `lroc_055_145_r5120_c43008` | 1.0487 | 8.2410 |

The software did warn on these cases, so this is not evidence of a wrong *clean pass*. It does show that a returned registered product, particularly with a warning, can be substantially wrong outside its tie-point support. The active tool does not evaluate against the manifest's ground truth; this audit does that externally. See `lroc.json` and the per-pair logs.

### Controlled transformation tests

The synthetic images contain crater-like texture with known mappings. These isolate geometric and reporting behavior; gamma adjustment and contrast inversion do not simulate all effects of a changing lunar Sun angle.

| Case | Tool status | Known-transform mean corner error, reference px |
|---|---|---:|
| Fractional translation | pass | 0.1062 |
| Translation plus nonlinear brightness change | pass | 0.1011 |
| Translation plus contrast inversion | warning | 0.1193 |
| 20° rotation | pass | 0.2049 |
| 60° rotation | failed: degenerate_transform | — |
| 1.5× scale change without metadata | failed: degenerate_transform | — |
| Modest perspective distortion | pass | 0.0718 |
| Georeferenced 4:1 resolution difference | pass | 0.1220 |
| Same georeferenced 1024² image, decimated processing | pass | 0.4434 |

For the 4:1 case the prealignment mapping is composed with the exported correction before measuring truth error. Applying `matrix_reference_px` directly to original source pixels is incorrect: it is a correction in the already prealigned reference frame. The audit records the direct-application error separately to expose this contract ambiguity; it is **not** scored as a failed registration.

## Specific implementation findings

### 1. Accuracy validation is not independent enough to certify the requirement

`backend/seleno/tool/register.py:765` estimates/verifies on all candidates and selects methods using all resulting inliers. The random holdout split happens only at line 884, after coarse method selection, RANSAC inlier selection, fine-window placement and fine verification. The held-out points have therefore already influenced the solution and support region.

At lines 920–937, ECC is adopted by whichever result scores better on that same holdout. That makes it a validation/model-selection set, not a final untouched test set. Adjacent and overlapping native windows also make a random point split weaker than spatially separated evaluation.

There is a second concrete issue: when ECC is disabled or not adopted, `Hf` is scored but `Hm` remains the earlier all-inlier model used for export. The contract probe with ECC disabled found **639 held-out points** and a delivered matrix different from the evaluated fit (maximum coefficient difference **0.05654**). The score is a useful cross-validation diagnostic, but not a held-out measurement of the exact delivered model. Exporting an all-inlier refit can be a legitimate design; its metric must be described accurately.

Required correction: split independent evaluation before fitting/selection, preferably by spatial block; keep a final test set untouched by ECC selection; distinguish validation error from final-product control-point error and persist the split IDs.

### 2. `pass` does not mean the source-sub-pixel requirement passed

`_quality_gates` (`register.py:1534`) tests inlier count, ratio, coverage and extrapolation. `_write_metrics` does not require source RMSE <1 or even a native-resolution refinement to return `pass`. The TMC-2/SELENE run is an observed example: **pass, 2.1845 source px**.

The flags claiming a hard one-reference-pixel accuracy floor (`register.py:1583–1623`) are scientifically unjustified. Coarser sampling makes localization harder, but reference pixel size alone is not a mathematical lower bound on estimated displacement; SNR, texture, blur/MTF, radiometric mismatch and reference/control accuracy matter. Subpixel estimators can estimate fractions of a reference pixel. The [scikit-image registration example](https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html) demonstrates fractional-pixel shift recovery; it does not guarantee such performance on these lunar pairs.

The native same-image GeoTIFF probe makes the implementation contradiction explicit: **`rmse_source_px=0`, `subpixel=true`, `subpixel_attainable=false`**, because the two inputs have equal pixel size. Remove the absolute impossibility claim and replace it with empirically justified uncertainty. Missing the measured <1 source-pixel target remains a failure of the stated acceptance criterion, regardless of whether the reference is a limiting factor.

### 3. Exported match coordinates mix pixel conventions and lose precision

`_sample` stores source positions in pixel-edge coordinates. `_write_matches` (`register.py:1437`) retrieves these using rounded match locations while reference points remain in pixel-center/index coordinates. A native same-image GeoTIFF should produce identical source/reference coordinates. Instead, the probe exported **+0.5 px on each source axis**, a **0.7071 px** discrepancy for every retained point, despite reported RMSE zero.

The coarse lookup also rounds subpixel keypoints. The native stage uses rounded lookup at lines 1184–1187, so the problem is not confined to the coarse branch. On the decimated identity case the exported CSV had **1.9798 px RMS** discrepancy from the exact identity mapping, compared with reported 1.2464 source px.

Required correction: define one coordinate convention throughout; evaluate/invert the continuous mapping at fractional keypoint coordinates; include the decimation sampling-center offset; validate the delivered CSV independently.

### 4. Registered products are reduced grids; segment corrections are not applied

The warp at `register.py:949` uses `A_raw` and the coarse `B_raw.shape`. Fine-stage correspondences can improve the transform, but the full-resolution source is not resampled anew for the scientific product.

Observed exports: OHRC/NAC **1574×1574 at 4 m/px** against a **6296×6296 at 1 m/px** reference; IIRS/WAC **2150×114 at 500 m/px** against the reference's nominal 100 m grid. Cropping itself can be valid; these are also explicitly decimated outputs.

`_segment_fit` computes per-band matrices and fit-set residuals, but the image is warped with one `Hm` before those are used anywhere. `segments=6` adds JSON diagnostics; it does **not** deliver a piecewise-registered pushbroom product. Its per-band RMSE is also an in-sample residual, not the main held-out metric.

IIRS export contains **one standardized pseudo-pan band**, not the registered 256-band reflectance cube. If the intended registered product must preserve spectral measurements, that functionality is missing.

Required correction: a tiled native-reference-grid export using the complete source mapping; actually apply/blend segment transforms when requested; offer propagation of the recovered geometry to every IIRS band.

### 5. Rotated/sheared reference GeoTIFF export is geometrically incorrect

`_write_registered` (`register.py:1482`) scales `transform.a` and `transform.e` by decimation but leaves `b` and `d` unchanged. Correct affine composition scales both basis vectors.

The isolated writer probe with a 4× decimation and reference affine `(2, 0.3, 100, 0.2, -2, 200)` produced a **137.37 m** corner displacement from the expected transform. This does not affect an axis-aligned grid with `b=d=0`, but it violates generic reference-grid registration.

### 6. Native-reference nodata masking is inconsistent

`_boxcar_decimate` returns immediately for `step<=1`; `_target_grid` (`register.py:179`) then checks finiteness and a very low sentinel threshold, omitting `ref.nodata`. Source masking does explicitly check nodata.

With a valid raster containing **16,384 pixels of finite nodata = −9999**, the native reference grid marked **all 16,384 as valid**. This contaminates normalization, overlap, coverage denominators and potentially matching. Decimated and native runs therefore use different validity semantics.

### 7. A fine stage can be labeled native when it is still decimated

For the 1024² identity case with `max_side=256`, the report says both **`native=true`** and **“windows still decimated 2x”**, and adopts that fine set. `_fine_stage` records the warning at line 1143 but unconditionally sets `native=True` near line 1222. This is a reporting defect directly relevant to trusting native-resolution accuracy claims.

### 8. Scale, rotation and spatial coverage remain constrained

At `register.py:814`, fitted working-frame scale outside 1±0.25 or rotation beyond 30° is rejected. That is reasonable for a successfully prealigned map frame; it excludes valid pairs when metadata is absent and localization is skipped.

`locate.wanted` (`locate.py:441`) normally skips no-geometry images with comparable footprints, and its footprint-size check requires GSDs. Generic differently sized PNGs with unknown GSD are not automatically searched merely because their dimensions differ. `--locate force` can request a search, but is not exposed in the current UI and does not make unknown-scale search general.

The dense matcher starts from a translation and grids its patch centers, but verification may discard many of them. Sparse matches pass directly to fitting/exports; the active tool measures coverage but never invokes `spatial.select`, although that selection exists in older pipelines. A 25% eligible-cell coverage gate is not a uniformity guarantee, and coverage is measured only in the reference grid.

Required correction: separate metadata-prealigned plausibility checks from blind matching; search unknown rotation/scale explicitly; ensure spatially distributed support on both images and report unsupported regions without claiming accuracy there.

### 9. Additional limits on units and Sun-angle interpretation

The GDAL reader estimates a scalar GSD from `abs(transform.a)` and converts geographic longitude degrees using the lunar radius, without latitude-dependent x distance, rotated basis lengths or anisotropy. A geographic test at 60°S reports **30.32 m** per pixel where the approximate east-west ground spacing is **15.16 m** and north-south spacing is **30.32 m**. Real metre errors should be measured through a local projection/geodesic mapping, particularly on angular or distorted map grids.

The active generic matcher plan uses sensor identity and observed correlation, not actual Sun angles. Its gradient representation, normalization and pretrained learned features can help with illumination variation. Existing DEM, phase-congruency and Sun-series modules elsewhere in the repository are not an active physical sun-angle normalization stage in this tool. Rolled OHRC labels have their Sun angles nulled; NAC GeoTIFF illumination metadata is not populated by this reader. Consequently, many produced metrics cannot themselves stratify results by actual Sun azimuth/elevation.

These observations do not mean a physical lighting model is mandatory for good correspondence. They mean the code's presence and its filenames cannot substitute for a measured illumination-robustness result.

## Data provenance and remaining validation gap

The local archive contains real OHRC, TMC-2, IIRS, SELENE and LROC/WAC data, so no new large download or account login was necessary for these tests. Earlier `data/pairs` cases labeled “TMC-2 class” are downsampled OHRC-derived fixtures; they are not actual OHRC→TMC-2 sensor tests. Likewise, `reports/LOCATE.md` explicitly describes OHRC patches pasted into an IIRS strip, which tests location search, not real overlapping OHRC/IIRS spectral correspondence.

The supplied [ISRO browser](https://chmapbrowse.issdc.gov.in/) requires registration/login for Chandrayaan-2 downloads. The supplied LROC download URL contains an extra dot: the accessible page is [LROC downloads](https://lroc.im-ldi.com/images/downloads/). The [controlled NAC mosaic product page](https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_POLE_SOUTH_CM_AVG) distinguishes map products from browse images and documents its 1 m/px grid. Arbitrary browse screenshots are not substitutes for science products plus labels and geometry.

To establish the full requirement, the remaining evidence must include multiple genuine overlapping OHRC/TMC-2/IIRS-to-reference pairs, independent spatially distributed control points, source-pixel error measured on the delivered product/CSV, negative pairs, and results stratified by sensor pair, scale ratio, Sun azimuth/elevation and viewpoint. A successful sample or many self-consistent matches is insufficient to establish invariance.

## Reproduction

From the repository root:

```bash
.venv/bin/python tests/test_tool.py
.venv/bin/python tests/test_ohrc_dataset.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u reports/validation_20260923/run_audit.py synthetic
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u reports/validation_20260923/run_audit.py lroc
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u reports/validation_20260923/run_audit.py real
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u reports/validation_20260923/extra_iirs.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u reports/validation_20260923/probe_contracts.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u reports/validation_20260923/api_smoke.py
npm --prefix frontend run build
```

Add `--resume` to `run_audit.py` to preserve previously completed rows after an interruption. The synthetic run generates the inputs used by `probe_contracts.py`. All correctness findings above are backed by current source inspection and saved results, with external absolute-accuracy limits stated explicitly.
