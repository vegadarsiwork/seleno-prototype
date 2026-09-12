# Scientific assumptions and limitations

The purpose of this document is to stop the demo from being visually impressive
and scientifically misleading. Read it before quoting any number from the UI.

**Superseded scope.** The earlier version of this prototype worked on six
synthetic pairs cut from two OHRC strips. It now works on four real
Chandrayaan-2 OHRC Level-2 products and their genuine repeat-pass overlaps. The
six legacy pairs are retained as a regression fixture; where a statement below
applies only to them it says so.

---

## 1. What the dataset actually is

Four OHRC Level-2 (calibrated) south-polar strips, 0.96–1.21 GB each,
79 796–101 075 lines × 12 000 samples, uint8, headerless. **No labels, masks,
crater catalogues, boxes, DEMs or ground-truth annotations of any kind.** It is
4 scenes, not thousands of images.

| | 20211228T2209 | 20241115T1326 | 20241115T1525 | 20251010T0942 |
|---|--:|--:|--:|--:|
| Lines | 79 796 | 101 074 | 101 074 | 101 075 |
| `isda:pixel_resolution` (m/px) | 0.28 | 0.24 | 0.24 | 0.22 |
| Imaging orbit | 10445 | 23328 | 23329 | 27338 |
| Roll (deg) | −7.86 | 14.57 | 15.19 | 26.67 |
| Sun elevation, mid-strip (deg) | −0.043 | **−0.169** | **+0.786** | −0.771 |
| Solar incidence (deg) | 90.04 | 90.17 | 89.21 | 90.77 |
| Pixels ≤ DN 10 | 80.2 % | 71.4 % | 68.6 % | 57.6 % |
| 512 px tiles with usable texture | 22 % | 31 % | 33 % | 42 % |

The defining property: **solar incidence is 89–91° in every product.** The Sun
sits at or below the local horizon. Two thirds to four fifths of every strip is
at or below DN 10 out of 255, and a 256 × 2048 patch from mid-strip can contain
as few as five distinct DN values. Only 22–42 % of 512 px tiles carry enough
gradient information for a detector to work on at all.

---

## 2. Is there ground truth?

**For the OHRC pairs: no. None. There is no correct transform to be right about,
and the UI says so on every such run.**

| Pair source | Ground truth | What that means |
|---|---|---|
| OHRC window pairs (all of them) | **None** | Two real acquisitions of the same ground. No one knows the true transform. |
| Legacy `ohrc_raw_vs_calibrated` | Exact translation | Two delivered products of the *same* acquisition, cropped at a known offset. |
| Legacy `ohrc_crater_field`, `_tmc2_*`, `_low_illumination` | Synthetic | We warped real imagery by a transform we chose, so the answer is known. |
| Legacy `ohrc_disjoint_orbits` | None | The two windows do not overlap. |

For the five legacy pairs with a "ground truth", that truth is a transform *we
chose*, applied to imagery then matched back against itself: same regolith
texture, same noise realisation, same shadows modulo a gamma change. Sub-pixel
errors there demonstrate the implementation is arithmetically correct. They are
**not** evidence about registering two independent acquisitions — and the real
OHRC pairs, where we now have honest numbers, are two orders of magnitude worse.

---

## 3. The delivered geometry is a prior, not truth — and it is measurably wrong

Every product reports `isda:reference_data_used = System`, and in all four the
`Refined_Corner_Coordinates` block is **byte-identical** to
`System_Level_Coordinates`. No photogrammetric refinement has been applied.

We measured how wrong it is. Reprojecting both 2024-11-15 strips onto a common
south polar stereographic grid and correlating gradient magnitude over the
mutually lit ground gives a disagreement of **538 m** between the two products'
geolocation — about **2 240 pixels** at 0.24 m/px (peak NCC 0.180 against a
runner-up of 0.070).

**Consequences, and they are large:**

* A 1024 px window and its geometry-predicted counterpart in the other strip
  share almost no ground. Before correcting this, the baseline matcher returned
  33–58 candidates and 2–3 inliers per window (4–8 % inlier ratio) and every
  window was refused. **It would have been easy, and completely wrong, to
  publish that as "illumination change defeats SIFT".** After applying the
  measured offset the same windows give 200–3 400 candidates and 22–30 % inlier
  ratios.
* The correction is a **bulk translation only**. The two strips were acquired at
  different roll angles (14.57° and 15.19°), so the residual varies along the
  strip. Per-window disagreement with the corrected prior is still
  160–290 m.
* Longitude is degenerate here: one footprint spans longitude 222° → 110° → 22°
  while covering 25 km of ground, because the strips pass within 0.05° of the
  pole. **All footprint, overlap and offset arithmetic is done in projected
  metres**, never in lon/lat. A spherical Moon of radius 1 737 400 m is assumed;
  stereographic scale error at 25 km from the pole is about 1 m, far below the
  geolocation error.

---

## 4. Image geometry and the transform model

**Assumed:** over a window a few hundred metres across, viewed near nadir, the
mapping between two OHRC views of the same ground is well approximated by a
single global 2D transform.

**Actually true:** OHRC is a **pushbroom (TDI) line scanner**. Every image line
has its own exterior orientation. A rigorous mapping needs the sensor model,
the spacecraft ephemeris and attitude (the `.oat`/`.spm`/`.lbr` files and SPICE
kernels), plus a terrain model. A global 2D transform cannot represent
along-track attitude jitter, relief displacement, or the fact that a homography
is exact only for a planar scene or a pure camera rotation.

**Default model for OHRC pairs: affine (6 dof), not similarity.** The two
strips differ in roll by ~0.6°, which introduces a small shear and anisotropic
scale between two pushbroom views of the same ground. Measured on one window
from identical correspondences: similarity gave 52 inliers at 22.5 %, affine
gave 133 at 57.6 %. Homography fits more inliers still but its held-out error
was worse (4.93 px vs 1.70 px), i.e. it absorbs relief into perspective terms
and over-fits.

`verify.plausibility` warns when the fitted scale departs from what the two
products' GSDs imply, when scale is strongly anisotropic, when the determinant
flips handedness, or when a homography's perspective terms grow large. **These
are advisory. A model can pass all of them and be wrong.**

---

## 5. Illumination: what is and is not implemented

**Available and used:** per-line solar azimuth, elevation, phase and aspect,
interpolated from the `.spm` ancillary series at 40 ms cadence, plus spacecraft
position and the acquisition timing from the label.

Sun azimuth **sweeps tens of degrees along a single 16 s strip** (20241115T1525
runs 203.1° → 275.9°), because azimuth is referenced to local north and these
strips pass within 0.05° of the pole. The label's single scalar is only the
mid-strip sample; we interpolate per window instead.

For the primary pair, at mid-strip: **Δazimuth 67.3°, Δelevation 0.955°.**

### The number that constrains everything

At a solar elevation of 0.79°, **one metre of relief casts a 73 m shadow.** At
0.2° it casts 286 m. So a DEM with ±2 m of vertical error locates a shadow
boundary only to within 150–600 m — 600–2 500 pixels at 0.24 m/px. **A predicted
shadow mask built on any presently available lunar DEM cannot have a crisp
boundary at OHRC resolution.** This is why:

* masks are probabilistic, carrying `boundary_uncertainty_m = vertical_error /
  tan(elevation)`, and the boolean is eroded by that uncertainty in pixels;
* the prediction is computed on a grid spaced at the DEM's own resolution and
  upsampled — evaluating it per full-resolution pixel would manufacture detail
  the input cannot support.

### What is implemented

* **Illumination flattening** — divide by a heavily blurred copy of the image.
  A retinex-style *image operation*, not a photometric correction.
* **CLAHE** — local contrast equalisation.
* **Observed signal mask** — local standard deviation, local mean, and local DN
  level diversity, measured from the image. **No terrain model involved.** This
  is the DEM-free control.
* **Predicted shadow mask** — horizon ray-cast against a pluggable
  `TerrainModel`, with uncertainty feathering.
* **Broad shading removal** — Lambertian cos(incidence) from the terrain normal,
  low-pass filtered before division.

### What is NOT implemented, and the honest failure we found

* **No DEM ships with this repository.** `NullTerrain.available` is `False` and
  every call raises. The only working terrain implementation is
  `MockTerrain`, which is **band-limited value noise with synthetic craters**.
  It is not science. Every product derived from it is stamped
  `is_synthetic_terrain=True`, the UI shows a "SYNTHETIC TERRAIN" chip, and the
  decision layer adds a warning.
* **A local horizon ray-cast is inapplicable to most of this archive.** Three of
  the four products have a mid-strip solar elevation at or below zero, yet
  plainly contain lit terrain: near the pole, high ground is lit over a
  *depressed* horizon while the sub-satellite point is not. A local flat-horizon
  test declares the whole window shadowed, which contradicts the imagery.
  Rather than emit that mask, `predicted_shadow_mask` **declares itself
  inapplicable below 0.05° elevation** and the pipeline falls back to no masking
  with the reason stated. Doing this properly needs horizon angles computed over
  tens of kilometres of DEM — which is a different algorithm, not a tuning
  change.
* **No photometric model** (Hapke, Lommel-Seeliger). No opposition or roughness
  terms.
* Nothing recovers information from a pixel that is genuinely unlit.

---

## 6. What each error figure actually represents

Four different numbers. They are never merged, and the UI labels each one.

### Reprojection RMSE — always reported
Root-mean-square symmetric transfer error of the retained correspondences under
the fitted transform, in reference-frame pixels. Measures **self-consistency**.
A degenerate set clustered on one crater can fit a badly wrong transform with a
tiny reprojection RMSE.

### Held-out RMSE — reported whenever ≥ 8 correspondences survive
The transform is fitted on 70 % of the retained correspondences and evaluated on
the other 30 %, split deterministically from a seed. This tests
**generalisation** off the fitted points and is the strongest honest figure
available when there is no ground truth. It is still self-consistency: it says
nothing about absolute position.

### Ground-truth error — only where a true transform exists
Image corners projected through the estimated and the true transform, mean of
the four distances. This is the **accuracy** figure. It does not exist for any
OHRC pair.

### Disagreement with the delivered geometry
Mean corner displacement between the estimate and the geometry prior, in metres.
An **independent** check, not an accuracy figure: the prior is system-level and
unrefined. A large disagreement means one of the two is wrong; it does not say
which.

### Metres
Pixel errors are multiplied by the **reference** image's GSD.

### Overlap NCC
Normalised cross-correlation inside the overlap, before and after registration,
using no correspondences — an independent witness that alignment improved. A
similarity measure, not a geodetic figure.

---

## 6a. Frame overlap, and why self-consistency is not enough

A concrete case from this repository, worth stating because it is the clearest
demonstration that every self-consistency figure can look good on a result that
should not be trusted. On window (5888, 88832) of the primary pair the pipeline
produced a 73 % inlier ratio, 1.22 px reprojection RMSE, 1.33 px held-out RMSE
and an overlap NCC of 0.89 — and the warped source covered only **31 % of the
reference frame**, with a fitted rotation of 12.5° against the 7.6° implied by
the delivered geometry.

Both can be true at once: within the 31 % that overlaps, the alignment really is
good (hence the NCC). Outside it, the transform is extrapolated. And because the
surviving correspondences sit in a diagonal band rather than across the frame,
rotation and translation trade off against each other, so a several-degree
rotation error costs almost nothing in residual.

No self-consistency metric can detect this. `overlap_fraction` can, and it is now
a soft check: below 50 % the result is downgraded to a warning with that reason
stated. The threshold is judgement, like the others.

---

## 7. Spatial coverage, and why its denominator changed

Coverage is the fraction of N×N grid cells on the source window holding at least
one retained correspondence. Two thirds of a typical OHRC window is shadow, so
**scoring coverage against every cell refuses a perfectly good registration
whose matches necessarily sit in the lit minority.** Cells containing
essentially no matchable pixels are therefore excluded from the denominator when
a usability mask is active. Both figures are reported: `spatial_coverage` (over
matchable cells) and `spatial_coverage_whole_window`.

Coverage is also **sensitive to the grid size** — the same run scored 11 % on a
6×6 grid and 19 % on 4×4. Quote the grid with the number.

The comparison shown is against **top-K by confidence at the same K**, not
against all inliers, since any selection loses coverage when it drops points.

Coverage is a proxy for how well the transform is conditioned. The better metric
would be the covariance of the estimate, which is not implemented.

---

## 8. Selenographic coordinates

Latitude and longitude come from bilinear interpolation of the `*_g_grd_*.csv`
lattice ISRO ships with each product: 121 cross-track nodes × one node per 100
lines, ~22–28 m spacing on the ground. Verified against the label: grid node
(0, 0) reproduces `upper_left_*` and the last node reproduces `lower_right_*` to
six decimals, and the lattice's last scan node equals the last image line
exactly, so the grid indexes the raster 1:1 with no crop or offset.

Limits: the lattice is a **system-level** product; interpolation error between
nodes is not characterised; ground distances assume a spherical Moon.

---

## 9. Label and documentation defects found in this archive

All handled explicitly in code, all surfaced in the UI rather than silently
patched:

1. **`isda:line_exposure_duration` has the wrong unit.** Tagged `unit="ms"`
   with value 162.100; 101 074 lines × 162.1 ms would be 4.5 hours against a
   16.383 s acquisition. The value is **microseconds**. We derive the line period
   from the acquisition dwell instead.
2. **`Refined_Corner_Coordinates` == `System_Level_Coordinates`** in all four
   products (§3).
3. **`20241115T1525` writes `lower_right_longitude` as `" 4.686608"`** with a
   leading space. A parser anchored on the digits silently drops that corner —
   ours matches surrounding whitespace and additionally reports any incomplete
   corner block.
4. **`.spm` record timestamps must be read at fixed byte offsets.** The I4
   block-length field abuts the 4-digit year, so whitespace splitting yields a
   merged token (`2492024`) and shifts the whole time field. The trailing float
   columns, conversely, do *not* match the declared widths and must be split on
   whitespace. We do both, and self-check that the "phase angle" column equals
   90 − elevation before trusting the column order.
5. **`readme.txt` states the ancillary cadence is 512 ms; it is 40 ms.** Derived
   from the records, never assumed.
6. **`readme.txt` mislabels the `.spm` section** as "Liberation angle Data File"
   and misnumbers a field.
7. **Browse labels declare the wrong dimensions** (4 211 × 500 against actual
   10 107 × 1 200). We do not rely on browse labels.
8. The dataset report shipped with the archive states that
   `20241115T1326`'s browse PNG was not extracted. **It is present now** — that
   item is stale. Trust the filesystem.

---

## 10. Known weaknesses of this implementation

1. **The proposed contribution is unvalidated.** Predicted-shadow masking is
   implemented end to end but has no real DEM to run on, and is inapplicable at
   the negative solar elevations that dominate this archive (§5). No claim about
   its value is supported by anything here.
2. **Refusal thresholds are hand-picked**, not calibrated: 12 inliers, 15 %
   inlier ratio, 15 % coverage, 4× the RANSAC threshold, 6 px held-out RMSE, and
   a 50 % frame-overlap warning. The
   intended research step — fitting them on labelled overlapping and
   non-overlapping pairs and reporting precision/recall for the decision itself
   — has not been done. `thresholds_calibrated` is `false` in the API and the UI
   repeats it.
3. **`confidence` is a bounded monotone combination of the metrics already
   shown, not a probability.** It is not calibrated against outcomes.
4. **Coarse alignment is a bulk translation** measured from one template window,
   so it cannot represent the along-strip variation from differing roll.
5. **Single global transform, no piecewise or non-rigid refinement**, so relief
   displacement has nowhere to go but the residuals.
6. **Grid thinning is off by default and skipped below 60 inliers**, because on
   real windows it makes held-out error worse — 1.71 px against 1.41 px when it
   cuts ~1300 inliers to ~44. Sub-pixel refinement is off for the same reason
   (2.36 px against 1.41 px). Both thresholds are judgement, not calibration.
7. **No learned matcher has been run on this data by us.** The LoFTR interface
   exists and reports its import error rather than falling back to SIFT.
8. **Runtimes are single-run, single-thread, one machine**, with no warm-up.
9. **n = 4 scenes.** Windows from one strip are not independent samples, and the
   four strips mutually overlap 58–95 %, so even a per-strip split leaks
   terrain. Any train/test split would have to be geographic.
10. **MD5 checksums in the labels have not been verified** against the 4.6 GB of
    imagery.
