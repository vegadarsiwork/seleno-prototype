# The registration tool

`seleno.tool.register` takes **two files** and produces a registered product,
the match points behind it, and the evidence for whether to believe it. It
assumes nothing about where those files came from: no ISRO geometry, no `.spm`,
no refined corners. Whatever is missing is recorded rather than guessed.

```bash
systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python \
  -m seleno register --source A --reference B --out outputs/
```

```python
from seleno.tool import register
res = register("A.xml", "B.lbl", "outputs/")
print(res.status, res.metrics["accuracy"]["rmse_px"])
```

---

## 1. What it writes

Every run creates `outputs/<job_id>/`, **including a run that fails**. A failure
that leaves no artifact is indistinguishable from a crash, so there is always a
`metrics.json` saying what happened.

| file | contents |
|---|---|
| `matches.csv` | verified fit points with a strict per-cell quota; `src_x, src_y, ref_x, ref_y, confidence, inlier` in original source and full-resolution reference pixel centres |
| `matches_all.csv` | every fit candidate before the delivery quota, including outliers with `inlier=0`; excludes validation/test points |
| `tiepoints.npz` | when the native fit is adopted: all measured native-stage points, working coordinates, native source coordinates, confidence and fit/validation/test labels |
| `registered.tif` | all original source bands on the native reference grid, cropped to the registered footprint; adaptive 1–4× supersampling per axis and NaN nodata. When the reference is finer than the source, every N-th reference pixel is written (N = reference px per source px, e.g. 7 for IIRS on 7.4 m TC) so the source is never upsampled. An export needing more than half the free disk is skipped and reported |
| `transform.json` | working-grid residual matrix, `matrix_reference_px`, frame, decomposition, optional `local_field` B-spline coefficients and `parallax` DEM sampler/coefficient, optional segment matrices and unit conversions |
| `evaluation.json` | sealed test coordinates, split, exact matrix and complete-model hashes for recomputation; directional reference and surface east/north error vectors |
| `quality.json` | raw/screened native error distributions, surface east/north metres, threshold rates, per-point residuals, along/across-track profiles, block-bootstrap intervals, final-warp validity, agency-style statistics, support and acceptance; also included as `metrics.json:quality` |
| `metrics.json` | accuracy, match counts, distribution, illumination, pair character, every method tried, status and reason |
| `overlay.png` | source and reference with tie lines, over a checkerboard of reference and registered |
| `report.md` | the same numbers as prose |
| `source.png`, `reference.png`, `registered.png`, `preview.json` | separate layers and display-space tie points, for the UI |

On failure, `status` is `failed` and `reason` is exactly one of:

`unreadable_input` · `no_overlap` · `insufficient_matches` ·
`verification_failed` · `degenerate_transform`

**Coordinate convention.** CSV and JSON coordinates are zero-based pixel
centres: `(0, 0)` is the centre of the first pixel. Convert to world coordinates
with the GDAL geotransform at `(x + 0.5, y + 0.5)`. Fractional keypoints use a
continuous source mapping, with no integer rounding. `reference_sample_offset`
records the actual centre of averaged samples on a decimated grid; matrices,
CSV points, previews and the exported GeoTIFF use that same offset.
`matrix_reference_px` is the residual correction after initial prealignment,
expressed in full-resolution reference centres, not a direct raw-source matrix.

---

## 2. Method selection

The tool runs ordered matcher candidates and chooses a geometrically verified
model using fit-fold evidence. It records which one won in
`metrics.json:method_used`. The ordering is not a preference — it encodes what
was measured in Phase 2:

- **Same sensor, large Sun-azimuth difference.** On NAC↔NAC across
  sub-solar-longitude bins, classical descriptors scored **0 of 90** past 45° of
  azimuth change; DISK+LightGlue scored **7 of 12**. Learned matching goes first.
- **Anti-correlated pairs.** TMC-2 against SELENE TC measures a cross-correlation
  of **−0.41 to −0.51**: the scenes are contrast-inverted with respect to each
  other. Every sparse matcher failed on that pair; dense masked NCC locked. When
  the tool measures a negative correlation it puts dense NCC first.
- Otherwise it tries dense, then learned, then classical.

**The ranking also moves with working resolution**, which is why no fixed choice
would do. Measured on OHRC→NAC: DISK+LightGlue 113 inliers against AKAZE's **0**
at 25 m/px, and AKAZE 3223 against DISK's 537 at 1 m/px. Downsampling averages
away the fine shadow structure that breaks gradient descriptors, so the learned
matcher owns the coarse end and ordinary detectors come back once real texture
is resolved. The fine stage re-runs the selection at native resolution for
exactly this reason. Full tables in `reports/RESOLUTION_RANKING.md`.

`metrics.json:attempts` lists every candidate that ran with its own counts and
timing, so the choice is auditable rather than asserted.

DISK and LightGlue are Apache-2.0. SuperPoint/SuperGlue are deliberately not
used — their weights are non-commercial only.

---

## 3. Accuracy, and what the numbers are allowed to say

**Sealed folds and check points.** Spatial cells are frozen before candidate
selection, aligned to an elongated overlap's principal axes and sized roughly
square. Only fit cells influence model/method choice, patch selection and outward
growth. ECC and the coarse balanced refit are adopted on the validation fold.
The test fold is scored after `transform.json` is finalized and reloaded.

Every test correspondence is reported under `held_out_*`. Headline RMSE and
acceptance use `check_point_*`: test correspondences that agree with their ten
nearest held-out neighbours' displacements relative to the **coarse** model.
The screen uses a fixed 3-reference-pixel / robust-spread threshold and never
consults the exported model. It retains shared model errors and counts isolated
mismatches separately; at least 80% of the held-out measurements must survive.
This is matcher consistency, not surveyed accuracy. `evaluation.json` records
the coordinates, screen flags, rule, partition and complete-model fingerprint.
Both raw and screened statistics remain visible. Invalid predictions retain
`invalid_n` and fail acceptance, while valid predictions still have statistics.

**Quality scorecard.** The result opens with all-held-out native source RMSE,
p95, fraction strictly below one source pixel, and invalid count. Tables place
all held-out observations beside the neighbour-screened subset, in separate
source/backward and reference/forward grids. They include median, p90, p95,
maximum, x/y bias and component NMAD (`1.4826 × median absolute deviation from
the component median`). NMAD describes scatter; it does not remove or replace bias.
The interactive strip plot shows error magnitude or signed sample/line errors;
screened-out observations can be toggled, and invalid predictions remain marked.

`thresholds` records strict `<0.25`, `<0.5`, `<1` and `<2` pixel counts. Its
`fraction_all` includes invalid predictions as unsuccessful; `fraction_valid`
uses only finite observations. Empty sets have null fractions. Distance
statistics use finite observations and retain both valid and invalid counts.
The older `within_1_px` field keeps its valid-observation denominator for
compatibility; acceptance already fails on any invalid held-out prediction.

`distribution.valid_pixel_support` measures the share of valid overlap pixel
centres inside/outside the fit-point hull. Up to 100,000 uniformly ranked valid
pixels are sampled in bounded blocks; smaller overlaps are evaluated entirely.
This avoids the thin-strip bias of using coarse grid-cell centres. The original
grid-based coverage/extrapolation gates remain unchanged: the new support
estimate is an additional diagnostic, not a relaxation of acceptance.

It also reports the distance from each sampled valid pixel to its closest fit
point (`distance_to_fit_point`, native reference px): the maximum is the radius
of the largest unsupported gap, which being inside the hull does not rule out.

**Surface metres.** Nominal metres (`rmse_m`) multiply a reference-pixel error by
the reference's declared pixel size. On a cylindrical map that overstates
east-west distance by 1/cos(latitude): about 1.25x at 37 S and 3.2x at 72 S on
the WAC mosaic. When the reference has a CRS, every held-out forward error is
therefore also mapped through the local Jacobian of that CRS (native pixel
centre -> geodetic longitude/latitude -> east/north metres on the CRS's own
ellipsoid, `seleno/tool/geodesy.py`). `held_out_ground_*`, `check_point_ground_*`
and the headline `rmse_ground_m` hold the result, with component mean and
1-sigma (`bias_en_m`, `std_en_m`), and `evaluation.json` keeps every vector.
They are relative to the reference, not absolute lunar positions.

**Where the error is.** `quality.json:profiles` bins all held-out native source
errors along the strip's long axis (3-12 bins) and across it (2-5 bins) over the
whole image, so unmeasured stretches appear as empty bins; a bin's p95 is only
quoted with 8+ observations. `drift_xy_px_per_1000` is the least-squares slope of
each component along the strip. `intervals` gives 95% spatial-block bootstrap
intervals (2000 replicates, seed 0) that resample whole held-out split cells,
because neighbouring matches are not independent; they cover sampling only, not
bias shared with the reference or matcher.

**Final-warp validity.** `quality.json:warp` evaluates the complete exported
inverse map on a lattice over its footprint and compares its Jacobian with the
base (global/segment) model's: `folded_fraction` (orientation flips),
`relative_area` and `relative_shear` of the local field/terrain terms, and the
size of the non-global correction. Flags are diagnostic, not acceptance gates.

**Agency-style statistics.** `quality.json:conventions` restates the same
errors in the form each source publishes, checked against the primary sources
on 2026-09-26: NASA ASP `bundle_adjust` (mean and median reprojection error per
image, "under 1 pixel, ideally under 0.5 pixels", at least a dozen points);
USGS ISIS `jigsaw` (sample, line and overall residual in pixels); JAXA Kaguya TC
(Haruyama et al., LPSC 2012 #1200: mean/1-sigma longitude 5.4/8.0 m and latitude
3.6/7.2 m over nine repeated sites at 10 m GSD), with ours also per source GSD;
empirical CE90/CE95; and SLDEM2015's practice of reporting spatial variation.
Every row states how the measurement differs. None is a pass mark: an agency
residual comes from a camera model and triangulated ground points, ours from
held-out matcher correspondences through a 2-D warp.

The UI separates export completion, quality acceptance and independent checks.
The endpoint `GET /api/tool/jobs/{id}/quality` reconstructs diagnostics from
saved JSON evidence without reading the rasters or changing files. It checks
the complete transform hash against metrics and evaluation. Older runs can
therefore gain source p95, threshold rates and residual plots without a rerun;
missing reference vectors/support remain unavailable rather than inferred.

**Significance and model terms.** Each coarse candidate must pass an a-contrario
number-of-false-alarms test (`log10_nfa`) and geometry checks for folds, scale,
anisotropy and area-scale variation across the overlap. Dense fitting uses loose
RANSAC, neighbour consistency and cell-balanced least squares. Optional terrain
parallax and a smooth B-spline inverse correction are selected by spatial
cross-validation over fit cells only. Their CV tables are in `fine_stage.model_fit`.
Raster export, previews and scoring all apply the complete model through
`warp_model.inverse_points`; the matrix alone is insufficient when terms are adopted.

**Four units, always.** The solver runs on a working grid that exists only
inside the run and is usually much coarser than either input, so a residual
quoted only in its pixels says nothing about either image. Every run reports
working/reference transfer error in several units, plus backward transfer measured
directly in native source coordinates (`check_point_source_*`). The source error
is not inferred by multiplying a symmetric transfer distance by a nominal GSD ratio.
`check_point_reference_*` separately measures forward transfer in native reference
coordinates. The headline `rmse_reference_px` now uses that directional error,
not a converted symmetric working-grid residual. Metres use the reference's
declared sampling and are explicitly nominal reference-scale distances, not
absolute geolocation accuracy or a correction for map distortion. The legacy
`subpixel_attainable` is null: sampling ratio alone cannot establish attainability.
For example:

```json
"rmse_source_px": 2.285, "rmse_reference_px": 1.6914,
"rmse_m": 12.522, "rmse_working_px": 0.2819,
"units": {"metres_per_reference_px": 7.40, "source_px_per_reference_px": 1.351,
          "reference_decimation": 6}
```

**The fine stage.** A coarse solve cannot report a residual finer than its own
decimation. A TMC-2 strip against a 3° SELENE tile decimates 6x: every "pixel"
is 44 m wide, and 1.3 px is 59 m — **10.8 TMC-2 pixels**. So after the coarse
solve the tool lays roughly 6,000 **lattice seeds over the predicted overlap**,
places padded native-resolution tiles and measures the seeds using local NCC,
ECC and a forward/backward check. Patches shrink near valid-data boundaries.
The probe retains at most three tiles; subsequent processing streams tiles.
Weak tiles are retried using an affine fitted only to previously measured fit points.

Two things make it work, and it produces nothing without either:

1. *The windows are positioned through the coarse transform.* An unrefined OHRC
   strip's geometry is kilometres out, so a window placed from raw geometry is
   filled with ground from kilometres away. Measured in one 1948² window: 13
   AKAZE candidates from raw geometry, **8584** through the coarse transform.
2. *The method is re-selected at native resolution.* The coarse winner has no
   claim on a different grid — see `reports/RESOLUTION_RANKING.md`. The stage
   probes the plan once and records the winner in `fine_stage.method`.

Patch sizes 41 and 61 are compared on common fit-fold probe seeds by whole-cell
cross-validation. The larger patch must improve median prediction by 5%, avoid
a worse tail and retain at least 80% of the default's fit measurements. Insufficient
evidence keeps 41. `fine_stage.patch_cv` and `patch_selection` record the decision.
An explicit Python `fine_patch` or sensor `registration.fine_patch` overrides it.
Peak correlation and forward/backward thresholds stay at 0.55 and 0.35.

Pre-audit accuracy figures are superseded. Use only the corrected
[real-product summary](../reports/validation_20260925/deck_summary.md).

**Sub-pixel.** The flag is on **source** pixels, which is what the problem
statement asks for. Within the fine stage two refinements run:

1. *Parabolic peak fit* on the correlation surface. Dense NCC returns an integer
   peak — before this, a TMC-2 pair returned offsets of exactly 8, 8, 8, 8, 7, 9
   px — and the fit interpolates the true peak between samples.
2. *ECC polish* (Evangelidis & Psarakis) on **gradient magnitude**, not
   intensity. ECC maximises a correlation coefficient, which survives a linear
   brightness change but not an inversion — and inversion is exactly what a Sun
   azimuth reversal does here. Gradient magnitude is polarity-blind. The polish
   is warm-started from the **fit-subset** model and adopted **only if it lowers
   the separate validation RMSE**; the guard against a runaway warp is proportional to the
   frame, not a fixed pixel count.

**Reference sampling.** One reference pixel can span several source pixels.
`accuracy_statement` reports that scale next to the measured source-pixel RMSE.
It is contextual information, not an accuracy gate or a mathematically established
lower bound. A verified, adequately covered run can pass with `subpixel: false`.
The legacy floor fields represent a nominal one-reference-pixel sampling assumption.


**Metres.** `rmse_m` is `null` unless the reference genuinely carries a scale.
A bare PNG has an identity geotransform; treating that as 1 m pixels would turn
a pixel error into an invented metre error. Native source-pixel errors can still
be measured without a physical GSD. `subpixel` is null only when that source
error is unavailable; lack of metre scale alone does not hide pixel statistics.

**Resampling.** Reference decimation is a boxcar average, not striding: keeping
one pixel in `step` is aliasing. The source is averaged too, but only when the
affordable taps actually cover the footprint — spreading four taps across a
16.7 px footprint is a sparse comb that band-limits nothing, and measurably cost
AKAZE 442 inliers → 70. Past that point the grid is sampled nearest and
`degraded` records that it is aliased.

## 4. Uniform distribution

The fine stage starts from uniform lattice seeds throughout predicted overlap.
Coarse dense matching also uses lattice seeds around the locked translation,
falling back to the older grid matcher if needed. Successful measurements can
still cluster in textured regions, so distribution is measured explicitly.
`matches.csv` applies a strict quota to verified fit points; `matches_all.csv`
preserves all fit candidates. `tiepoints.npz` retains the complete measured
native set with fold labels for reproducible diagnostics. Delivery thinning
does not change the fitted model or held-out evaluation.

- **`coverage_fraction`** — fraction of *eligible* reference grid cells holding at
  least one inlier. Eligible means both images actually have data there. Against
  a whole-frame denominator, a good registration of a polar scene that is two
  thirds shadow scores badly for the wrong reason.
- **`dispersion`** — mean distance of inliers from their centroid, over the image
  half-diagonal. Coverage saturates once every cell is touched; dispersion keeps
  responding.
- **`extrapolation_fraction`** — share of eligible cells lying **outside the convex
  hull of the inliers**. This is the one that catches the dangerous case:
  tie points crowded into a single lit strip can clear the coverage bar while
  most of the frame is still extrapolated, and extrapolation beyond the hull is
  where a fitted transform's error grows fastest. Over 25% downgrades the run to
  `warning`.

---

## 5. Long strips

A pushbroom strip can need more than one global transform. When spatial CV
adopts a local field or terrain term, that continuous model replaces segment
fitting. Otherwise `--segments N`
splits the overlap into N along-track bands and fits each independently;
`transform.json:segments` carries the fitted per-band matrices and fit-fold residuals.
The exporter blends inverse coordinate maps with a smoothstep between segment centres
and samples the source once. Bands without enough fit points use the global matrix.
The preview and final held-out evaluation use this same serialized composite model;
`accuracy.applied_model` identifies it and `transform_sha256` fingerprints the complete
model. The global matrix alone does not reproduce a segmented export.

The geometry lattice is inverted by Newton iteration on its bilinear forward
map. Inverse Delaunay interpolation previously filled curved-edge hull slivers
with the edge detector column. Queries outside the image now return NaN;
limited rim extrapolation seeds the solve without clamping a whole band to an
edge. Lattice subsampling keeps the final row and column.

## 5b. Terrain parallax

`terrain.HeightSampler` samples LOLA DEM heights at reference pixel centres.
The inverse affine may add `(height - height_origin_m) * coefficient_px_per_m`:
a fitted two-component source displacement per metre of relief. It is adopted
only when fit-cell CV improves over the inverse affine without height. A local
field can then model remaining smooth error. The fitted coefficients describe
an effective image displacement, not an independently calibrated viewing angle.

`transform.json:parallax` stores the DEM path, reference CRS and affine,
working-to-reference map, height origin and coefficient. Reapplying the model
requires that DEM; the JSON does not embed raster heights. Grid conjugation
transforms both sample positions and displacement vectors. Heights outside the
DEM contribute zero correction. The current archive contains polar LOLA DEMs;
equatorial TMC-2/IIRS pairs have no available terrain term.

---

## 6. Memory

The working grid is **planned from the memory limit**: the tightest cgroup v2
`memory.max` on the process or its ancestors, else physical memory, less a 1 GiB
reserve for the process itself, budgeting 50%. It is not planned from what
happens to be free. The working grid moves the placement window, the folds and
the chosen model. When it followed free memory, the same pair and settings gave
1.3 px on one run and 3.0 px on another. Under the same cap it is now the same
on every run: an app run and a CLI run of IIRS 20240119 → WAC produced
bit-identical transforms. Free memory is still checked, as the smaller of
`MemAvailable` and cgroup headroom net of reclaimable page cache. When it cannot
hold the planned grid, the grid shrinks further and the note says the result is
not comparable. `metrics.json:working_grid` records requested, planned and used
sizes, the limit and its source, and the basis (`requested`, `memory limit` or
`memory pressure`).

The web app runs each registration in its own spawned child process, so one
job's leftover heap cannot shrink the next job's headroom. It also runs one
registration at a time; later submissions show the stage `queued`. A child
killed at the cap is reported as a crash naming the signal, and a pipe carries
its log so the last line before the kill still arrives.

The
dominant cost is not the imagery — it is masked NCC, which correlates in *full*
mode and allocates FFT buffers of (2H−1)×(2W−1) in float64, several at once.
Measured at roughly 900 bytes per target-grid pixel, so a 2048² grid peaks near
3.9 GB. `_cap_max_side` shrinks an over-large request and records that it did.

Rasters are opened as memmaps and sampled lazily; validity comes from a decimated
read. Loading a TMC-2 strip as float32 up front costs 2.37 GB before a single
pixel is needed.

`LazyRaster` copies strided samples out of each read block so a tiny slice does
not keep a full block alive. Large exports stream all bands in native tiles;
tiles without finite Jacobians use one sample without an all-NaN median warning.
On the 15 GB development machine every registration/test suite must run in a
systemd scope with `MemoryMax=5G` (3 GB for unit tests), `MemorySwapMax=0`,
`OPENBLAS_NUM_THREADS=2` and `OMP_NUM_THREADS=2`. Run one heavy job at a time.
Use `outputs/` for rasters and set `TMPDIR` to a directory there: `/tmp` is RAM.

---

## 6b. Hyperspectral input (IIRS)

A spectrometer cube is collapsed to one raster before anything else looks at it.
Band centres are read from the label and only the **reflected-light window,
0.8–1.6 µm**, is averaged in: past roughly 2.5 µm a lunar daytime spectrum is
the surface's own thermal emission, which tracks temperature rather than albedo
or topography. Bands are standardised before averaging, because a spectrometer's
response varies by an order of magnitude across its range and a straight mean
would be whichever band is brightest. The window lives in
`config/sensors/iirs.yaml`.

Validated against real Chandrayaan-2 IIRS derived-reflectance products
(`d_rfl`, processing level **Derived**). What those exposed:

- **The data file is a `.qub`, and the label names it.** The reader now uses the
  label's `file_name` rather than assuming `.img` beside the `.xml`.
- **Band centres are `<center_wavelength unit="nm">` inside `Band_Bin`.** Real
  centres were found — 256 of them, 0.7123 to 5.0097 µm — so the middle-third
  fallback did **not** fire. 47 bands fell in the window.
- **Geometry is an ENVI `_loc_` backplane, not a CSV.** Bands 1 and 2 are
  Longitude and Latitude at the full detector grid;
  `GeometryGrid.from_envi_loc` reads it, decimated on the way in.
- **No scaling.** `data_type` is `IEEE754LSBSingle` with no `scaling_factor` or
  `value_offset`; values are physical reflectance (median 0.09–0.15 in the
  reflected-light bands) and `0.0` is fill. Confirmed from the label rather than
  assumed, since this is a different processing level from the calibrated
  framing-camera products.

If a label carries no band centres the tool falls back to the middle third of
the cube and says so in `degraded` — that result is not a calibrated pseudo-pan
and is labelled as such.

## 6c. Very large references

Two things make a global mosaic usable as a reference.

**Lazy reading.** `_try_gdal` used to do `ds.read(1).astype(np.float32)`. For the
LROC WAC basemap (109164 × 54582) that asks for 23.8 GB, raises, and was
swallowed by the reader's `except Exception: return None` — the tool reported
that nothing could read the file. Rasters over 512 MB now go through
`LazyRaster`, which serves the two access patterns the pipeline uses — strided
2D slicing and paired fancy indexing — from windowed reads.

`LazyRaster` strides **exactly** as numpy would, reading in bounded row blocks.
GDAL's own `out_shape` decimation is the obvious shortcut and is wrong here: it
resamples at output cell centres rather than taking every `step`-th pixel, so
`a[0::4]` and `a[1::4]` come back identical — which would silently turn the
boxcar averaging in §3 into a no-op.

**Footprint windowing.** Before anything is read, the source's own geometry is
projected into reference pixels and the working grid is confined to that box,
padded generously because the source geometry is the thing being corrected.
Without it a 734 km IIRS strip lands about three pixels wide on a 54× decimated
global grid, and there is nothing left to match.

## 6d. Finding one image inside another

The stages above refine a placement; they do not search for one. When the
reference has a map projection, geometry places the source. When it does not —
two Chandrayaan-2 products carry per-pixel lon/lat lattices and no CRS, and a
plain PNG carries nothing — the tool used to assume both images start at the
same corner. For an OHRC frame (2.6 × 22 km) inside an IIRS strip (14 × 700 km)
that put the frame at the top of the strip, and a pair 500 km apart was reported
as overlapping 100%.

`seleno.tool.locate` now runs first whenever no CRS route applies and either
both inputs carry geometry or their footprints differ by more than 2× in area:

- **Geometry**, when both inputs carry a lattice or a CRS: the smaller footprint
  is projected into the larger image (in a stereographic plane centred on it, so
  poles and the 0/360 seam are harmless) and fitted with a similarity. If the
  footprints are further apart than 1.5 × the profiles' expected geolocation
  error, the run fails as `no_overlap` and says how far apart they are.
- **Image search**: the smaller footprint, area-averaged to the coarser ground
  sampling, is correlated over the larger one at a set of rotations (all of
  them without geometry, ±12° around the geometry's with it), on intensity and
  on gradient magnitude, coarse-to-fine. A position is accepted only if it
  stands clearly above the best position *elsewhere* (normalised margin ≥ 0.3,
  robust z ≥ 4) and both cues agree on it — or one cue is overwhelming. Otherwise
  the run fails as `insufficient_matches` with the search statistics, rather than
  registering against a corner chosen by assumption.

When the reduction onto the working grid is 16× or more, the source is now
area-averaged (from the viewer's cached block-mean overview) instead of
point-sampled; at OHRC → IIRS scale a point sample is noise. Calibration and
results: `reports/LOCATE.md`. `--locate off` restores the old behaviour.

## 7. Sensor profiles

`config/sensors/*.yaml` holds per-instrument behaviour — nominal GSD, bit depth,
normalisation percentiles, shadow and texture thresholds, no-data, expected
geolocation error. Adding a sensor is a new YAML file, not a patch to the
pipeline. A file matching no profile gets `_default` and the run reports reduced
capability instead of guessing.

---

## 8. CLI

```
python -m seleno register --source X --reference Y --out DIR
    --model {auto,similarity,affine,homography}
    --max-side N        working grid ceiling (capped against free memory)
    --grid N            N x N coverage grid
    --segments N        per-band transforms for long strips
    --no-fine           skip the native-resolution fine stage
    --fine-tiles N      cap on native windows (0 = tile the overlap, max 400)
    --no-subpixel       skip the ECC polish
    --locate {auto,force,off}   find where the source lies in the reference (§6d)
    --json              print metrics.json to stdout
    --quiet

python -m seleno profiles          # what sensors are configured
```

## 9. HTTP API

Mounted at `/api/tool` by `backend/app.py`:

| endpoint | purpose |
|---|---|
| `GET /files` | selectable inputs, each labelled `product` or `fixture` |
| `POST /register` | start a run, returns a job id |
| `GET /jobs/{id}` | state, the pipeline's own log, metrics when finished |
| `GET /jobs/{id}/file/{name}` | any artifact |
| `GET /jobs/{id}/download` | the whole artifact set as a zip |

The UI at `/` drives exactly these. Every number it shows is read back out of
the job's `metrics.json`; nothing is precomputed. Inputs that are not archive
products are labelled **FIXTURE** in the picker and a banner appears above the
results.

---

## 10. Tests

`python tests/test_tool.py` — 35 tests, run in CI by
`.github/workflows/tests.yml`. It builds its own rasters, so it needs no archive
data:

identical pair → near-identity · disjoint footprints → `no_overlap` ·
unrelated images → a declared failure code · truncated, empty and missing files
→ `unreadable_input` without a crash · 8-bit and 16-bit of the same scene agree ·
rotated and flipped sources either register or fail cleanly · a mostly-shadow
crop is not reported as a clean pass · a bare PNG runs but refuses to report
metres · the full artifact set is written · the memory ceiling shrinks an
over-large grid · RMSE agrees across all four units · the sub-pixel flag is on
source pixels · an unreachable floor is a note, not a status downgrade · the
fine stage measurably beats the working grid · a spectrometer cube collapses
to reflected light · thermal bands are excluded · a cube with no band centres
declares that its selection was a guess · a `.qub` named only in the label is
found · nanometre band centres convert · geometry is built from a `_loc_`
backplane · a large raster reads lazily and its strided and fancy reads match an
eager read exactly.

The focused unittest modules cover native export, matcher refinement, evaluation
acceptance, validation isolation, dense models, and geometry/terrain. Run
`tests.test_geometry_terrain` for curved-edge inversion, cache invalidation, rim
extrapolation, DEM centres/conjugation, coarse lattice measurements and patch-CV
isolation. `tests/test_ohrc_dataset.py` additionally checks 28 archive contracts.
Apply the caps and thread settings in §6 to every test command.

### Status and accuracy

`pass` requires source check-point RMSE and p90 below 1 px, at least 95% below
1 px, bias below 0.5 px, at least 12 check points, at least 80% check-point retention,
at least 50% fit-cell coverage and at most 25% extrapolation. No invalid held-out
predictions are allowed. This establishes internal consistency and support;
independent controls are required for verified accuracy. `accuracy.subpixel` compares
the unrounded check-point source RMSE with **one original source pixel**, independently
of status (null when source-pixel evaluation is unavailable).
`accuracy_statement` gives the source-pixel error and reference sampling scale
in one line. The scale is the size of one reference pixel in source pixels;
it is not an independently established mathematical lower bound on localization.
Legacy `subpixel_floor_source_px` / `subpixel_attainable` describe that nominal
one-reference-pixel sampling assumption and should not be used as quality gates.

### IIRS defaults

A bare CLI/Python/API registration reads `registration.max_side: 6144` and
`registration.grid: [12, 12]` from `config/sensors/iirs.yaml`. Explicit options
still override these values. The memory cap accounts for the target window's
aspect ratio, so a narrow strip is not budgeted as a square. The UI offers
“Sensor default” for working-grid size.
