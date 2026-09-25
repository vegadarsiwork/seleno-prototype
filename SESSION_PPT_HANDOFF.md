# SELENO — PPT handoff

**What this is.** Material for the presentation: the problem, how we solve it, why
our approach stands out, the numbers we can defend, a slide-by-slide outline and
answers to likely judge questions. Everything here is true as of 2026-09-25.
All numbers come from the final validation run
(`reports/validation_20260925/REPORT.md`, deck table `deck_summary.md`).

---

## 0. The one-liner

> **SELENO registers Chandrayaan-2 images onto lunar reference maps. It uses the
> physics of the orbit, the terrain and the Sun. It measures its own accuracy on
> points it never trained on, and it refuses to answer when it can't be sure.**

Taglines, pick one:

- *"We don't match pictures. We match the Moon."*
- *"Geometry first, pixels second, honesty always."*
- *"Tie points everywhere, trust earned everywhere."*
- *"The best registration is the one that tells you when it's wrong."*

---

## 1. The problem (slide: "What we were asked to solve")

Image registration puts a **source (moving)** image into the coordinate system
of a **reference (fixed)** image. Here:

| Source (Chandrayaan-2) | Reference (lunar maps) |
|---|---|
| **OHRC**: 0.24 m/px, the sharpest camera ever flown to the Moon | **LRO NAC** controlled mosaic, 1 m/px |
| **TMC-2**: 5 m/px terrain mapping camera | **SELENE (Kaguya) TC** ortho map, 7.4 m/px |
| **IIRS**: 256-band hyperspectral, ~80 m/px | **LRO WAC** global mosaic, 100 m/px (109,164 × 54,582 px) |

**Required deliverables:**
- Correspondence at **sub-pixel accuracy** (of the source image).
- Match points **uniformly distributed** across the image.
- A **registered product** plus its **match points**.
- **Evaluation metrics**: RMSE, inlier count, inlier ratio.

**The three named challenges:**
- **Illumination**: sun azimuth and elevation change what the surface looks like.
- **Viewpoint**: different camera positions and orientations distort the scene.
- **Scale**: missions fly at different altitudes with different resolutions.

---

## 2. Why this is harder on the Moon than it sounds (slide: "The Moon fights back")

These are facts measured on our own data. They make great "shock" slides.

- **The Sun is on the horizon.** At the south pole the solar incidence is
  **89–91°**. **57–80% of every OHRC strip is at or below DN 10 out of 255**,
  i.e. black. Only **22–42%** of image tiles have enough texture to match at all.
- **Shadows move bodily.** Two OHRC passes **2 hours apart** differ by **67° of
  sun azimuth**. A shadow that fills one image is absent from the other.
- **Published geolocation is off by hundreds of metres.** Two OHRC products of
  the same ground disagree by **538 m = 2,244 pixels**. A naive matcher
  "failed on illumination" when the real cause was a 2 km offset. *This is our
  "geometry first" origin story.*
- **Scale gaps are large.** One NAC pixel = **4.17 OHRC pixels**. One SELENE
  pixel = 1.35 TMC-2 pixels.
- **Terrain relief moves pixels.** The south pole has **km-scale relief**, and
  OHRC looked at it **~15–19° off-nadir**, so a 100 m hill shifts by ~30 m in
  the image. No flat (affine/homography) model can remove that.
- **Illumination can be opposite.** TMC-2 against the SELENE evening map has a
  raw correlation of **−0.42**: bright is dark and dark is bright.
- **The data is big.** Single products run 1.2 GB (OHRC) and 3.3 GB (IIRS
  cube), and the WAC reference is a 6 GB global mosaic, on a 15 GB laptop.

---

## 3. Our approach in one picture (slide: "The SELENO pipeline")

```
 Chandrayaan-2 product            Lunar reference map
 (PDS4 + geometry lattice)        (GeoTIFF / PDS3, CRS)
            \                        /
             v                      v
  1. GEOMETRY-FIRST PLACEMENT  -- put the source where its own orbit geometry says
             |
  2. COARSE CONSENSUS          -- 7 matchers compete (dense NCC, DISK+LightGlue, SIFT
             |                    variants, AKAZE, ORB); a-contrario significance test
             |                    rejects coincidences; geometry sanity checks
             |
  3. DENSE NATIVE TIE POINTS   -- thousands of seeds on an even LATTICE over the whole
             |                    overlap, measured at native resolution (NCC -> ECC
             |                    sub-pixel -> forward-backward check); grows outward
             |
  4. PHYSICS-AWARE MODEL       -- global affine + TERRAIN PARALLAX (LOLA DEM height x
             |                    fitted view angle) + smooth B-spline correction field;
             |                    each term must win SPATIAL CROSS-VALIDATION to be used
             |
  5. HONEST EVALUATION         -- sealed held-out test cells never seen by any decision;
             |                    RMSE in 4 units; check points; coverage & extrapolation
             |
  6. PRODUCTS                  -- registered GeoTIFF at native reference resolution (all
                                  256 IIRS bands), match points CSV, transform JSON,
                                  metrics, overlay, report, web viewer
```

---

## 4. What makes it stand out (slide: "Why SELENO is different")

Use these as the core "USP" slide or split over two slides.

1. **Geometry first, not pixels first.** We start from the spacecraft's own
   geometry (PDS4 lattice, CRS). Matching only corrects what geometry got
   wrong. That's why a 2 km offset doesn't break us.
2. **Physics in the model.** We model **terrain parallax** with NASA's LOLA
   elevation model. One physically meaningful coefficient (the view angle)
   explains most of OHRC's error. On OHRC→NAC it cut the cross-validated
   error **5.6×** (fit-cell CV median 1.45 → 0.26 working px) in the final run.
   The fitted coefficient corresponds to ~18° of effective view angle,
   comparable to the spacecraft roll recorded in the label (14.6°).
   *Matching finds points; physics makes them agree.*
3. **Uniform by construction.** Most pipelines match where a detector fires,
   which clusters points on a few bright craters. We lay seeds on an **even
   lattice over the whole overlap** and measure each one. Uniform distribution
   is **designed in, not hoped for**. We also measure it: grid coverage and
   the share of area outside the tie-point hull.
4. **Sub-pixel measurement, three ways.** Correlation peak interpolation, then
   **ECC** (enhanced correlation coefficient) optimisation, then a
   **forward-backward check** where a point must match back to where it came
   from. Polarity-aware, so inverted illumination still matches.
5. **Model complexity earns its place.** The terrain term and the smooth
   correction field are adopted **only if they predict held-out regions
   better** (spatial cross-validation). No hand-tuned "magic" model.
6. **Statistically sound acceptance.** An **a-contrario significance test
   (number of false alarms)** separates real consensus from coincidence. Two
   unrelated images are rejected, and a mirrored image is refused by the
   matchers.
7. **Honest by design.** A spatial test fold is **sealed before matching
   starts**; nothing ever tunes on it. We report RMSE in **source px, reference
   px, working px and metres**. The tool says "warning" or "failed" with
   reasons, never a silent wrong answer. *Judges can trust our numbers because
   our pipeline doesn't grade its own homework.*
8. **Sensor-agnostic and scale-proof.** One tool handles OHRC, TMC-2, IIRS
   (256-band hyperspectral), SELENE, NAC, WAC and any GeoTIFF. It reads 6 GB
   mosaics lazily and respects container memory limits. Loading the 6 GB WAC
   mosaic was cut from **>3 GB RAM to ~290 MB**.
9. **Science-grade product.** The registered GeoTIFF is written at **native
   reference resolution** and cropped to the actual footprint. For a hyperspectral
   source it keeps **all bands** (IIRS: 256 bands, with wavelengths and
   radiometric scale/offset). A 4×-finer source is averaged, not aliased.

---

## 5. Results (slides: "Numbers" + "Before → after")

The **metric** is RMSE on sealed held-out check points, in **source pixels**.
These are points no fitting or selection step ever saw. A "check point" is a
held-out correspondence that agrees with its held-out neighbours. That screen
never consults the model, and every excluded point is counted and reported.

### Final numbers (four real pairs)

| Pair | Check-point RMSE, source px (median) | In reference px (median) | Model terms | Inliers |
|---|---:|---:|---|---:|
| **OHRC → NAC** | **3.84** (3.09) | **1.06 NAC px ≈ 1.06 m** (0.85 m) | affine + terrain | 97.8% (1,259 / 1,287) |
| **TMC-2 → SELENE** | **2.34** (1.39) | 1.75 (1.01) = 12.9 m | affine + field | 96.4% |
| **IIRS 2025 → WAC** | **1.87** (1.24) | 1.62 (1.06) | affine + field | 93.9% |
| **IIRS 2024 → WAC** | **1.31** (**0.74**, 67% ≤ 1 px) | 1.19 (0.70) | affine + field | 99.3% |

None passes the full acceptance gate, which requires sub-source-pixel RMSE and p90,
95% of points ≤ 1 px and ≤ 25% extrapolation. The tool reports this as a `warning`
with reasons. Zero invalid held-out predictions on all four.

**Before → after, with the caveat.** On 23 September the same pairs scored 106.5,
12.3, 12.7 and 24.4 source px (OHRC, TMC-2, IIRS 2025, IIRS 2024). That run
scored all held-out points by symmetric working-grid residual with nominal unit
conversion. Today's numbers are native backward errors on screened check points.
The comparison is indicative, not a like-for-like factor. The unscreened
held-out numbers today are 7.28, 2.59, 2.04 and 1.49 source px. OHRC's 7.28 comes
from a few isolated mismatches; its median is 3.09 either way.

**Headline sentences for the slide:**
- "**OHRC → NAC to ~1 NAC pixel (1.06 m RMSE, 0.85 m median)**, from 106 source px
  on 23 September."
- "**IIRS 2024: median 0.74 source px**, two thirds of check points within one pixel."
- "**Thousands of tie points**, evenly spread: 1,986–2,302 native points per
  OHRC, TMC-2 and IIRS 2024 scene; 858 for IIRS 2025."

**Synthetic controls with exact known truth** (final run; 7 of 9 accepted):

| Case | Error vs truth (source px RMSE) |
|---|---:|
| Pure translation | 0.011 |
| 60° rotation, no metadata | 0.008 |
| 1.5× scale, no metadata | 0.184 |
| 0.5× scale, no metadata | 0.399 (warning: tail and extrapolation) |
| Gamma change | 0.008 |
| Contrast inversion (simulated opposite lighting) | 0.007 |
| 120° rotation + 1.5× scale | 0.764 (warning: 0.76 px bias) |
| 4× scale, georeferenced | 0.000 |
| Decimated identity | 0.008 |

**Real illumination change with known truth (LROC NAC, 12 pairs + 2 negatives):**
- Sun Δ 50–90°: **9 of 9 registered**, truth error **0.19–0.55 source px**.
  The worst corner is 1.19 reference px.
- Sun Δ 180°: **all 3 refused**, with no wrong answer exported.
- Different-ground negatives: **both refused**.
- On 23 September: 5 of 12 exported, two with corner errors of 8 px and 172 px.

**Robustness (35 adversarial tests):**
- Unrelated images: **rejected** (`degenerate_transform`).
- Disjoint footprints: `no_overlap`.
- Truncated, empty or missing files: clean, declared failure codes.
- 90° rotation: **registered**.
- Mirrored image: refused by the matchers, or flagged as a warning.

---

## 6. How we cover the three challenges (slide: "Challenge → our answer")

| Challenge | Our answer |
|---|---|
| **Illumination** | Polarity-aware local NCC (handles inverted shading) + gradient representation + learned matcher (DISK+LightGlue) at coarse scale + phase-congruency SIFT variant; illumination-matched reference selection (lesson: **sun elevation matters more than azimuth**) |
| **Viewpoint** | Any rotation and large scale when there is no metadata; homography/affine; **terrain parallax from LOLA DEM**; cross-validated smooth field for attitude jitter along the strip |
| **Scale** | Resolution-aware placement from ground sampling distance and CRS; area-averaged resampling (no aliasing); native-resolution fine stage; errors reported in source AND reference pixels |

---

## 7. Evaluation metrics we report (slide: "Metrics that can't lie")

Asked for: RMSE, inlier count, inlier ratio. We deliver those **plus**:

- **RMSE in 4 units**: source px, reference px, working px, metres.
- **Check-point RMSE, median, p90, % within 1 px, bias vector.** A bias would
  expose a systematic shift that a single RMSE figure can hide.
- **Sealed spatial held-out fold**, laid along the overlap, fixed before matching.
- **Coverage**: share of usable grid cells holding tie points.
- **Extrapolation fraction**: share of area outside the tie-point hull.
- **Significance (log10 number of false alarms, NFA)** for every candidate matcher.
- **Cross-validation table** for every model term: why it was, or wasn't, used.
- **Evidence files** (`evaluation.json`, `transform.json` hashes). Anyone can
  recompute our RMSE exactly from the exported model.

---

## 8. Honest limits (slide: "What we know we don't know")

Judges reward teams that know their limits. Say these confidently:

- **Sub-*source*-pixel is not reached on any real pair.** On OHRC→NAC the
  reference is the limit: one NAC pixel = 4.17 OHRC pixels. We are at **1.06 NAC
  px RMSE (1.06 m), 0.85 m median**. Going further needs a finer reference
  (e.g. OHRC↔OHRC or DEM-orthorectified NAC). IIRS 2024 is closest in source
  pixels (median 0.74, RMSE 1.31).
- **Extrapolation.** 32–57% of each real overlap lies outside the fit-point
  support. Errors quoted there are extrapolated, and the tool says so.
- **Very large sun-azimuth changes (180°)** remain unsolved. Shadows invert
  and move. All three 180° LROC pairs were refused instead of faked.
- **Terrain parallax correction needs a DEM.** We use LOLA at the poles
  (5 m/px). Equatorial scenes would use SLDEM/Kaguya DTM, which isn't in our
  current archive.
- **Independent ground truth.** The real pairs are scored on held-out
  correspondences, not surveyed control points. The tool accepts an external
  control-point file for independent certification.

---

## 9. Suggested deck (12 slides)

| # | Slide title | Key message | Visual |
|---|---|---|---|
| 1 | SELENO | One-liner + team | Moon south-pole OHRC crop |
| 2 | The problem | Source→reference, sub-pixel, uniform, metrics | Source/reference/registered triptych |
| 3 | The Moon fights back | 89–91° incidence, 80% black, 538 m offset, 4× scale | Dark OHRC strip + "2,244 px off" arrow |
| 4 | Our pipeline | 6 stages, geometry first | Pipeline diagram (§3) |
| 5 | Geometry first | Placement from spacecraft geometry; matching fixes the residual | Before/after overlay of the 538 m offset |
| 6 | Uniform by construction | Lattice seeds over the whole overlap vs detector clusters | Tie-point scatter: ours vs SIFT cluster |
| 7 | Physics in the model | Terrain parallax: 1 coefficient, 5.6× CV error cut on OHRC | DEM height map + residual arrows before/after |
| 8 | Earned complexity | Cross-validation decides terrain term & field | CV table / bar chart |
| 9 | Results | OHRC ~1 NAC px (1.06 m); TMC-2 2.3, IIRS 1.3–1.9 source px; LROC illumination 0.2–0.6 px | Final-numbers table + before/after (with scoring caveat) |
| 10 | Trust | Sealed test fold, NFA, refusals, 35 adversarial tests | "Refused" examples (unrelated / flipped) |
| 11 | Products | GeoTIFF (all 256 IIRS bands), CSV, JSON, report, web viewer | Screenshot of the web UI (`python run.py`) |
| 12 | Limits & next | Honest limits + roadmap | Roadmap timeline |

**Demo:** `python run.py` → http://127.0.0.1:8000. Upload two files, watch the
stages stream live, inspect the deep-zoom compare view, download the
product ZIP.

---

## 10. Judge Q&A prep

**"Did you hit sub-pixel?"**
Not on the real pairs, and we say so. On OHRC→NAC we're at **1.06 NAC px RMSE,
0.85 m median**, about one reference pixel. One NAC pixel is 4.17 OHRC pixels, so
sub-*source*-pixel needs a finer reference. IIRS 2024→WAC has a **median of 0.74
source px** (RMSE 1.31). With known truth, we are sub-pixel: 0.01–0.18 px on
synthetic controls and 0.19–0.55 px on real LROC pairs up to 90° of sun change.
We report on held-out points, not on the points we fitted.

**"How do you know your RMSE isn't cheating?"**
A spatial test fold is frozen before matching. No method choice, model choice or
refinement ever sees it. The final model is serialized, reloaded and scored once;
`evaluation.json` lets anyone recompute the number.

**"What about illumination differences?"**
We use polarity-aware local correlation, gradient and phase representations, and
a learned matcher at coarse scale. We also learned something useful: sun
*elevation* matters more than azimuth. TMC-2 at 51° sun elevation matches the
SELENE map with the right shading regime better, even with opposite azimuth.

**"Why not just deep learning end to end?"**
The learned matcher (DISK+LightGlue, Apache-2.0) is one of 7 competitors and
wins where it should, on coarse, shadowed scenes. But geometry, physics
(parallax) and statistics (significance, cross-validation) are what make the
answer *trustworthy*, and they need no training data we don't have.

**"How are points uniformly distributed?"**
The seeds are on a lattice by construction, and we measure it: coverage of
usable grid cells, dispersion, and the share of area outside the tie-point hull.
The delivered `matches.csv` also applies a per-cell quota.

**"Will it scale?"**
Everything is streamed or memory-mapped. It has run a 3.3 GB IIRS cube against a
6 GB WAC mosaic inside a 5 GB memory cap, and it adapts its working grid to the
memory it is actually given.

---

## 11. Where we left off (engineering, for the next session)

Done (all uncommitted in the working tree):

- Coarse significance (NFA) and degeneracy checks.
- Native lattice fine stage.
- Dense model with local field and terrain parallax.
- Check-point evaluation.
- Export: crop, supersampling and all bands.
- Memory fixes and geometry-lattice fixes.
- UI panel and docs (`docs/TOOL.md`).
- Correlation patch 41 vs 61 chosen per pair by fit-cell CV; peak and
  forward-backward thresholds stay at 0.55 / 0.35. Relaxing them was harmful in
  the sweep.
- All tests pass: `test_tool` 35/35, `test_ohrc_dataset` 28/28, six focused
  unittest modules.
- Full real, synthetic and illumination validation, capped at 5.5 GB with zero
  OOM kills. Report: `reports/validation_20260925/REPORT.md`.

Next steps:

1. Commit, after the user approves. The suggested 10-commit split is in
   `CODEX_HANDOFF.md`.
2. Reduce extrapolation (32–57%). It fails the 25% gate on every real pair.
3. Find a DEM for the equatorial TMC-2/IIRS scenes (SLDEM/Kaguya DTM) so they can
   use the terrain term.
4. Obtain independent control points for a real pair
   (`--ground-truth-directory`).
