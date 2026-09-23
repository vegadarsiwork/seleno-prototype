# The registration tool

`seleno.tool.register` takes **two files** and produces a registered product,
the match points behind it, and the evidence for whether to believe it. It
assumes nothing about where those files came from: no ISRO geometry, no `.spm`,
no refined corners. Whatever is missing is recorded rather than guessed.

```bash
python -m seleno register --source A --reference B --out outputs/
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
| `matches.csv` | `src_x, src_y, ref_x, ref_y, confidence, inlier` — original source pixels, full-resolution reference pixels |
| `registered.tif` | the source resampled onto the reference grid, georeferenced when the reference is, no-data preserved as NaN |
| `transform.json` | model, the working-grid matrix, the same transform in **full-resolution reference pixels** (`matrix_reference_px`, the one to apply to the product), decomposition, per-segment parameters, unit conversions |
| `metrics.json` | accuracy, match counts, distribution, illumination, pair character, every method tried, status and reason |
| `overlay.png` | source and reference with tie lines, over a checkerboard of reference and registered |
| `report.md` | the same numbers as prose |
| `source.png`, `reference.png`, `registered.png`, `preview.json` | separate layers and display-space tie points, for the UI |

On failure, `status` is `failed` and `reason` is exactly one of:

`unreadable_input` · `no_overlap` · `insufficient_matches` ·
`verification_failed` · `degenerate_transform`

---

## 2. Method selection

The tool does **not** have one matcher. It runs candidates and keeps the first
that a geometric verification stage accepts, and it records which one won in
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

**Held-out RMSE.** A seeded spatial partition is frozen before method selection.
Only the fit fold enters geometric verification, method/model selection and
native-window selection. ECC optimizes fit-region pixels and is adopted using
a separate validation fold. The test fold is scored once, at the end, without
model-based filtering. All retained test correspondences count, including wrong
matches. This is correspondence error, not independently surveyed ground truth.
`evaluation.json` stores the exact test coordinates, partition and matrix hash.
Scoring reloads `transform.json`; the raster warp uses the same matrix bytes.

**Four units, always.** The solver runs on a working grid that exists only
inside the run and is usually much coarser than either input, so a residual
quoted only in its pixels says nothing about either image. Every run reports the
same error four ways:

```json
"rmse_source_px": 2.285, "rmse_reference_px": 1.6914,
"rmse_m": 12.522, "rmse_working_px": 0.2819,
"units": {"metres_per_reference_px": 7.40, "source_px_per_reference_px": 1.351,
          "reference_decimation": 6}
```

**The fine stage.** A coarse solve cannot report a residual finer than its own
decimation. A TMC-2 strip against a 3° SELENE tile decimates 6x: every "pixel"
is 44 m wide, and 1.3 px is 59 m — **10.8 TMC-2 pixels**. So after the coarse
solve the tool re-places the source over native-resolution **windows** tiled
across the surviving tie points, re-correlates inside each, and refits from the
refined points. Peak memory is one window, the same as one coarse pass.

Two things make it work, and it produces nothing without either:

1. *The windows are positioned through the coarse transform.* An unrefined OHRC
   strip's geometry is kilometres out, so a window placed from raw geometry is
   filled with ground from kilometres away. Measured in one 1948² window: 13
   AKAZE candidates from raw geometry, **8584** through the coarse transform.
2. *The method is re-selected at native resolution.* The coarse winner has no
   claim on a different grid — see `reports/RESOLUTION_RANKING.md`. The stage
   probes the plan once and records the winner in `fine_stage.method`.

Measured: OHRC→NAC **6.27 m → 1.45 m**, TMC-2→SELENE **59.4 m → 12.52 m**.

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

**The floor.** Nothing can localise a source pixel against a reference to better
than about one reference pixel. When the reference is coarser than the source,
sub-source-pixel is unreachable *by any method*, and the tool says so:

```json
"subpixel": false, "subpixel_basis": "source pixels",
"subpixel_floor_source_px": 1.351, "subpixel_attainable": false
```

That statement goes in `notes`, not in `reason` — a limit of the reference is
not a defect in the registration and does not downgrade the run's status.

**Metres.** `rmse_m` is `null` unless the reference genuinely carries a scale.
A bare PNG has an identity geotransform; treating that as 1 m pixels would turn
a pixel error into an invented metre error. `subpixel` is `null`, not `false`,
in that case — without a scale the question cannot be answered either way.

**Resampling.** Reference decimation is a boxcar average, not striding: keeping
one pixel in `step` is aliasing. The source is averaged too, but only when the
affordable taps actually cover the footprint — spreading four taps across a
16.7 px footprint is a sparse comb that band-limits nothing, and measurably cost
AKAZE 442 inliers → 70. Past that point the grid is sampled nearest and
`degraded` records that it is aliased.

## 4. Uniform distribution

The problem statement asks for match points spread across the image, so the tool
measures it rather than claiming it.

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
  where a fitted transform's error grows fastest. Over 50% downgrades the run to
  `warning`.

---

## 5. Long strips

A pushbroom strip is not well described by one global transform. `--segments N`
splits the overlap into N along-track bands and fits each independently;
`transform.json:segments` carries the per-band parameters and RMSE.

Measured on TMC-2 → SELENE: global 1.4655 px against a per-band median of
**1.103 px**, i.e. 65.1 m → 49.0 m at 44.4 m/px.

---

## 6. Memory

The working grid is capped against `MemAvailable` before anything is read. The
dominant cost is not the imagery — it is masked NCC, which correlates in *full*
mode and allocates FFT buffers of (2H−1)×(2W−1) in float64, several at once.
Measured at roughly 900 bytes per target-grid pixel, so a 2048² grid peaks near
3.9 GB. `_cap_max_side` shrinks an over-large request and records that it did.

Rasters are opened as memmaps and sampled lazily; validity comes from a decimated
read. Loading a TMC-2 strip as float32 up front costs 2.37 GB before a single
pixel is needed.

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
    --fine-tiles N      cap on native windows (0 = tile the overlap, max 24)
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

`python tests/test_tool.py` — 26 tests, run in CI by
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
