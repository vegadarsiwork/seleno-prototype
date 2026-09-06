# Scientific assumptions and limitations

The purpose of this document is to stop the demo from being visually impressive
and scientifically misleading. Read it before quoting any number from the UI.

---

## 1. Image geometry

**Assumed:** over a window a few hundred metres across, viewed near nadir from
~90 km, the mapping between two OHRC views of the same ground is well
approximated by a single global 2D transform.

**Actually true:** OHRC is a **pushbroom (TDI) line scanner**, not a frame
camera. Each image line has its own exterior orientation. A rigorous mapping
needs the sensor model, spacecraft ephemeris and attitude (the SPICE kernels and
the `.oat`/`.spm` files shipped with the product) plus a terrain model. A global
2D transform cannot represent:

- along-track attitude jitter between lines,
- relief displacement, which grows with terrain height and view-angle difference,
- the fact that a homography is exact only for a **planar** scene or a pure
  camera rotation, and lunar craters are neither.

**Why it is still defensible here:** the demo windows are 130–950 m across, the
pairs are near-nadir, and for the synthetic pairs the true transform *is* a
global 2D transform by construction. The approximation is a property of this
prototype's scope, not a claim about lunar imagery in general.

**Consequence:** residuals reported here are *not* a measure of how well a
sensor-model-based registration would do, in either direction.

---

## 2. Transformation model

Three models are selectable. None is "the correct one" in an absolute sense.

| Model | DoF | When it is defensible |
|---|---|---|
| Similarity | 4 | Two near-nadir passes at similar altitude, flat-ish terrain. Fewest parameters, least over-fitting. |
| Affine | 6 | Adds shear/anisotropic scale — absorbs mild pushbroom rate differences. |
| Homography | 8 | Only for a genuinely planar patch or a pure rotation. On relief it will **absorb terrain parallax into perspective terms** and look better than it is. |

The default per pair is set in the manifest (`recommended_model`). The
`plausibility` check in `verify.py` warns when the fitted scale departs from the
ratio the two products' GSDs already imply, when the scale is strongly
anisotropic, when the determinant flips handedness, or when a homography's
perspective terms grow large. **These warnings are advisory. A model can pass all
of them and still be wrong.**

---

## 3. Is there ground truth?

**It depends on the pair, and the UI says which.**

| Pair | Ground truth | What it means |
|---|---|---|
| `ohrc_crater_field` | Yes — synthetic | We built the reference from the source with a known transform, so the answer is known exactly. |
| `ohrc_raw_vs_calibrated` | Yes — exact translation | Two *different real products* of the same acquisition, cropped at a known pixel offset. Nothing was resampled. |
| `ohrc_tmc2_moderate` / `_extreme` | Yes — synthetic | Known resample + rotation. |
| `ohrc_low_illumination` | Yes — synthetic | Known transform on real shadowed terrain. |
| `ohrc_disjoint_orbits` | **No** | The scenes do not overlap. No transform exists to be right about. |

**The critical caveat.** For five of six pairs the "ground truth" is a transform
*we chose*, applied to imagery that then gets matched back to itself. That makes
the appearance statistics far more favourable than any real repeat-pass pair:
the same regolith texture, the same noise realisation, the same shadows modulo a
gamma change. Sub-pixel errors on these pairs are **evidence the implementation
is arithmetically correct**, not evidence it would register two genuine
independent acquisitions to sub-pixel accuracy.

`ohrc_raw_vs_calibrated` is the only pair whose two sides are independently
delivered products, and even there the underlying acquisition is identical — only
the processing level differs.

**No pair in this prototype consists of two independent acquisitions of the same
ground.** We looked: the three OHRC scenes available without credentials cover
disjoint ground tracks (see `data/README.md`).

---

## 4. Are the images from the same region? The same sensor?

| Pair | Same region | Same sensor | Same acquisition |
|---|---|---|---|
| `ohrc_crater_field` | Yes (derived) | OHRC / OHRC | Yes |
| `ohrc_raw_vs_calibrated` | Yes | OHRC / OHRC | Yes — raw vs calibrated |
| `ohrc_tmc2_moderate` | Yes (derived) | OHRC / **simulated** TMC-2 GSD | Yes |
| `ohrc_tmc2_extreme` | Yes (derived) | OHRC / **simulated** TMC-2 GSD | Yes |
| `ohrc_low_illumination` | Yes (derived) | OHRC / OHRC | Yes |
| `ohrc_disjoint_orbits` | **No — tile centres 10.95 km apart** | OHRC / OHRC | No — orbits 2297 and 2298 |

**There is no real cross-sensor pair.** The `tmc2` pairs resample OHRC to 5 m/px.
That reproduces TMC-2's ground sampling distance and nothing else — not its
optics, MTF, noise, stereo geometry or illumination. Results on those pairs are
evidence about **scale-ratio robustness only**.

---

## 5. Illumination

**Available:** acquisition timestamps, orbit numbers, spacecraft altitude, and
the geometry grid, all from the PDS4 labels.

**Not used:** the products' incidence/emission/phase angles were not extracted,
and no photometric model (Lommel-Seeliger, Hapke) is applied.

What the pipeline actually does about illumination:

- **Illumination flattening** — divide by a heavily Gaussian-blurred copy of the
  image, removing the low-frequency brightness ramp. This is a
  *retinex-style image operation*, not a physically-based photometric correction.
- **CLAHE** — local contrast equalisation so shadowed regions retain gradient
  structure a detector can use.

Neither recovers information from a pixel that is genuinely unlit. In the
`ohrc_low_illumination` pair roughly 35 % of pixels sit below DN 20; no amount of
stretching creates texture there. The UI labels the stage honestly, and the
pipeline diagram marks physically-based photometric correction as **planned**,
not implemented.

The reference in that pair has a gamma of 1.6 and a gain of 0.8 applied. That is
an *intensity* change. It does **not** move shadows, which is what a genuine
repeat pass at a different sun azimuth would do — the single hardest thing about
polar lunar matching, and the thing this prototype does not test.

---

## 6. What the RMSE figures actually represent

Three different numbers, never to be conflated:

### Reprojection RMSE (always reported)
Root-mean-square **symmetric transfer error** of the retained correspondences
under the fitted transform, in reference-frame pixels.

This measures **self-consistency**, not accuracy. A degenerate set of matches
clustered on one crater can fit a badly wrong transform with a tiny reprojection
RMSE. It is reported because it is the only figure available without ground
truth, and it is always labelled as such in the UI.

### Ground-truth corner error (reported only where a true transform exists)
The four image corners projected through both the estimated and the true
transform; the mean of the four distances. This is the **accuracy** figure. It
cannot be made small by a self-consistent wrong answer.

### Ground-truth RMSE
The same comparison evaluated at the retained correspondence locations. Always
smaller than the corner error, because those points are in the interior where the
fit is best constrained. Quote the corner error, not this one, if you quote one.

### Metres
Pixel errors are multiplied by the **reference** image's GSD. On
`ohrc_tmc2_extreme` the reference is 5 m/px, so 0.04 px is 0.2 m — a small
*pixel* error on a coarse grid, not a 20 cm measurement.

### Overlap NCC
Normalised cross-correlation inside the overlap, before and after registration.
It uses no correspondences, so it is an independent witness that alignment
improved. It is a *similarity* measure and says nothing about geodetic accuracy.

---

## 7. Spatial coverage

Two numbers are reported, both over the same N×N grid on the source image:

- **cell coverage** — fraction of grid cells holding at least one retained match,
- **dispersion** — mean distance from the centroid, over the image half-diagonal.

The comparison shown is **spatially selected vs the top-K by confidence at the
same K**, not vs all inliers. Comparing against all inliers would be meaningless,
since any selection loses coverage when it drops points. Holding the budget fixed
isolates the effect of the spatial constraint itself.

Coverage is a proxy for how well the transform is conditioned. It is not a direct
measurement of the estimate's covariance, which would be the better metric and is
not implemented.

---

## 8. Selenographic coordinates

Latitude/longitude come from bilinear interpolation of the `*_g_grd_*.csv`
lattice ISRO ships with each product, sampled every 100 pixels. Limits:

- The lattice is a **system-level** geometry product, not a bundle-adjusted
  geodetic solution.
- Interpolation error between lattice nodes is not characterised.
- Ground distances use a **spherical** Moon, R = 1 737 400 m.

The 10.95 km separation quoted for `ohrc_disjoint_orbits` is the great-circle
distance between the centres of the two demo windows, not between the strips
themselves. It is good enough to establish that the two windows image different
ground. It is not a survey-grade figure.

---

## 9. Known weaknesses of this implementation

1. **`harmonise_gsd` throws away resolution.** For cross-resolution pairs the
   finer image is decimated to the coarser GSD before matching, so the extreme
   pair does not exploit OHRC's native 0.23 m/px at all. Matching fine to coarse
   directly, or in a coarse-to-fine cascade, would be the better design.
2. **The verdict thresholds** (12 inliers, 15 % inlier ratio, 15 % coverage, 4×
   the RANSAC threshold) are hand-picked, not calibrated against a labelled set.
3. **The learned matcher is an interface, not a result.** LoFTR is wired in and
   selectable; this environment has no `torch`/`kornia`, so it raises an error.
   No learned matcher has been run on this data by us.
4. **Single global model only.** No piecewise or non-rigid refinement, so relief
   displacement has nowhere to go but into the residuals.
5. **No cross-validation of the transform.** All retained matches are used in the
   fit; none are held out. A held-out subset would give an honest generalisation
   error where ground truth is absent.
6. **Runtimes are single-run, on one machine, single-threaded**, with no warm-up
   or repetition. Treat them as order-of-magnitude.
