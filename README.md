# SELENO

**Geometry-first lunar image registration with predicted unmatchable regions and
calibrated refusal — prototype (SIH26166)**

Seleno registers real Chandrayaan-2 OHRC imagery. It reads the delivered PDS4
geometry and Sun series, works out which parts of an image can be matched at all,
finds and verifies correspondences on what is left, and then **decides whether the
result should be trusted** — returning a refusal with reasons rather than a
confident transform it cannot support.

```
python run.py        →  http://127.0.0.1:8000
```

---

## 1. The headline result

The most important number in this repository is not a matcher score.

> **The delivered geolocation of the two 2024-11-15 OHRC products disagrees by
> 538 m — about 2 244 pixels at 0.24 m/px.**

Before correcting it, a baseline SIFT + MAGSAC++ pipeline on 1024 px window pairs
returned 2–3 inliers at a 4–8 % inlier ratio and refused every window. It would
have been easy to publish that as *"a 67° solar azimuth change defeats SIFT"*. It
was not illumination. The two windows were 2 200 pixels apart.

| window | without the coarse-offset correction | with it |
|---|--:|--:|
| s10240 l52224 | 46 cand · 2 inliers · 4.3 % | 366 cand · 79 inliers · 21.6 % |
| s9216 l52224 | 58 cand · 3 inliers · 5.2 % | 73 cand · 22 inliers · 30.1 % |
| s10976 l41984 | 33 cand · 2 inliers · 6.1 % | 206 cand · 59 inliers · 28.6 % |

That is what "geometry-first" means here, and it is measurable:
`python scripts/illumination_experiment.py`.

---

## 2. The data

Four **real Chandrayaan-2 OHRC Level-2 (calibrated) PDS4 products** of south-polar
terrain, ~4.6 GB of unique pixels. **No labels, masks, crater catalogues, DEMs or
ground truth of any kind.** Four scenes, not thousands of images.

| | 20211228T2209 | 20241115T1326 | 20241115T1525 | 20251010T0942 |
|---|--:|--:|--:|--:|
| Lines × samples | 79 796 × 12 000 | 101 074 × 12 000 | 101 074 × 12 000 | 101 075 × 12 000 |
| GSD (m/px) | 0.28 | 0.24 | 0.24 | 0.22 |
| Orbit | 10445 | 23328 | 23329 | 27338 |
| Sun elevation, mid-strip | −0.043° | **−0.169°** | **+0.786°** | −0.771° |
| Solar incidence | 90.04° | 90.17° | 89.21° | 90.77° |
| Pixels ≤ DN 10 | 80.2 % | 71.4 % | 68.6 % | 57.6 % |
| Usable 512 px tiles | 22 % | 31 % | 33 % | 42 % |

The defining property: **solar incidence is 89–91° everywhere** — the Sun is on
or below the local horizon. Two thirds to four fifths of every strip is at or
below DN 10 out of 255, and only 22–42 % of tiles carry enough gradient
information for a detector to work on. Deciding what *can* be matched is not a
refinement here; it is the problem.

**Primary illumination-stress pair:** `20241115T1326` → `20241115T1525`.
Consecutive orbits two hours apart, **95.3 %** mutual footprint, and at mid-strip
**Δazimuth 67.3°, Δelevation 0.955°**. Cast shadows move bodily: a shadow that
fills one window is absent from the other over the same ground.

Every `.img` is accessed through `numpy.memmap(mode="r")` and never loaded whole.
The archive is never written to. Provenance in `data/README.md`.

---

## 3. Architecture

```
              Strip A (memmap)                Strip B (memmap)
                     |                              |
                     +--------------+---------------+
                                    v
  ohrc/label,geometry,sun    METADATA / GEOMETRY      PDS4 label, geometry
                                    v                 lattice, per-line Sun
  ohrc/reproject             REFERENCE / OVERLAP      footprints in projected
  alignment                  ESTIMATION               metres; coarse offset
                                    v                 between products measured
  illumination               ILLUMINATION-AWARE       observed usability mask
  preprocess                 PREPROCESSING            (no DEM) or predicted
                                    v                 shadow mask (terrain model)
  matchers                   CORRESPONDENCE           SIFT / AKAZE / ORB /
                                    v                 pyramid; LoFTR slot
  verify                     ROBUST VERIFICATION      MAGSAC++ / RANSAC,
                                    v                 similarity|affine|homography
  spatial                    CORRESPONDENCE           grid quota | top-K | all
                             SELECTION                coverage over MATCHABLE area
                                    v
  subpixel                   SUB-PIXEL REFINEMENT     local NCC + paraboloid
                                    v
  registration               TRUST / REFUSAL          accept | warning | refuse,
                                    v                 with every reason listed
                                 RESULT
```

```
prototype/
├── run.py                        single entry point
├── README.md  RESEARCH.md  ASSUMPTIONS.md  HANDOFF.md
├── backend/
│   ├── app.py                    FastAPI: /api/ohrc/*, /api/register, /api/compare
│   └── seleno/
│       ├── ohrc/                 THE DATASET LAYER (new)
│       │   ├── label.py          PDS4 parsing + defect detection
│       │   ├── product.py        discovery, memmap tiles, thumbnails
│       │   ├── geometry.py       geometry lattice + polar stereographic plane
│       │   ├── sun.py            .spm / .oat ancillary Sun series
│       │   ├── tiles.py          tile index, illumination classes, DN profiles
│       │   └── reproject.py      common-grid reprojection + coarse alignment
│       ├── alignment.py          cached inter-product offset
│       ├── pairs.py              PairSpec: OHRC windows or legacy manifest pairs
│       ├── illumination.py       TerrainModel, masks — the proposed contribution
│       ├── subpixel.py           sub-pixel refinement
│       ├── registration.py       the staged pipeline + RegistrationResult
│       ├── experiments.py        presets and the ablation arms
│       ├── theme.py              colour tokens shared with the frontend
│       ├── preprocess.py verify.py spatial.py register.py metrics.py viz.py
│       ├── pipeline.py store.py  (retained: the earlier prototype's pipeline)
│       └── geodesy.py            (retained)
├── tests/test_ohrc_dataset.py    28 tests, read-only against the archive
├── frontend/                     Vite + React (dist/ prebuilt and committed)
├── data/                         legacy manifest pairs + provenance
├── results/
│   ├── dataset/                  Phase-2 inspection figures + tile indices
│   └── illumination/             the experiment tables, figures, alignment cache
└── scripts/
    ├── inspect_dataset.py        dataset visualisation and statistics
    ├── run_registration.py       one pair, one configuration
    ├── illumination_experiment.py the primary experiment + ablation
    ├── benchmark.py              legacy-pair regression table
    └── fetch_products.py prepare_data.py   (legacy data build)
```

---

## 4. Installation

```bash
pip install -r requirements.txt
```

Built and measured on Python 3.14.3, OpenCV 4.13, NumPy 2.4.4, SciPy 1.17.1 on
Windows 11. The OHRC archive is expected at `../datasettesting/dataset`; override
with `SELENO_OHRC_ROOT`. Nothing else needs downloading — the legacy demo pairs
and the built frontend are committed.

To rebuild the frontend (only if you change it):

```bash
cd frontend && npm install && npm run build
```

---

## 5. Running

```bash
python run.py                       # UI + API on :8000
python tests/test_ohrc_dataset.py   # 28 dataset-layer tests
```

**Reproduce the baseline (Phase 4)** — minimal preprocessing, SIFT, ratio test,
MAGSAC++, no masking:

```bash
python scripts/run_registration.py \
    --a 20241115T1326 --b 20241115T1525 --window 9984 51968 --size 1024 \
    --preset baseline_verified
```

**Reproduce the illumination experiment and the full ablation (Phases 5 and 7):**

```bash
python scripts/illumination_experiment.py --windows 5 --size 1024
# writes results/illumination/ILLUMINATION.md + illumination_raw.json + figures
```

**Show what happens without the geometry-first correction** — the finding in §1:

```bash
python scripts/illumination_experiment.py --windows 3 --no-offset
```

**Inspect the dataset (Phase 2):**

```bash
python scripts/inspect_dataset.py            # full pass
python scripts/inspect_dataset.py --quick    # coarser, faster
```

**Legacy regression** — the earlier prototype's six pairs, numbers preserved:

```bash
python scripts/run_registration.py --legacy ohrc_crater_field --preset seleno
python scripts/benchmark.py
```

In the UI: pick source and reference products, choose a candidate window, press
**RUN REGISTRATION**; **RUN ABLATION** runs every arm on that window. The
*Apply measured coarse offset* switch reproduces §1 interactively.

---

## 6. What each stage does

**Metadata and geometry.** Dimensions come from the PDS4 label, never a constant
— the four products have three different line counts. Sun geometry is
interpolated per image line from the `.spm` series, because **solar azimuth
sweeps tens of degrees along a single 16 s strip** near the pole and the label's
scalar is only the mid-strip sample.

**Reference and overlap estimation.** Footprints are reduced to a **south polar
stereographic plane in metres**. This is not fussiness: one footprint spans
longitude 222° → 110° → 22° while covering 25 km of ground, so any overlap
arithmetic in lon/lat is wrong. Both strips are then reprojected onto a common
grid and the bulk offset between their delivered geolocation is measured by
correlating gradient magnitude over the mutually lit ground — gradient because it
survives an illumination-direction change far better than raw DN. The result
carries a peak, a runner-up and a margin, so "no alignment found" is a reportable
outcome rather than a silent zero.

**Illumination-aware preprocessing.** Two mask sources, kept strictly apart:
the **observed** mask (local standard deviation, local mean, local DN level
diversity — measured from the image, no DEM) and the **predicted** shadow mask
(horizon ray-cast against a pluggable `TerrainModel`, uncertainty-feathered). The
observed mask is the DEM-free **control**: excluding dark pixels helps a matcher
whether or not a prediction was any good.

**Correspondence.** SIFT/AKAZE/ORB with Lowe's ratio test and a mutual-best
check. Correspondences whose source pixel falls in an unmatchable region are
dropped *after* detection — blanking the region first would manufacture an edge
the detector then fires on.

**Robust verification.** MAGSAC++ or RANSAC over similarity / affine /
homography. Default for OHRC pairs is **affine**: the two strips differ in roll
by ~0.6°, which a 4-dof similarity cannot absorb.

**Correspondence selection.** Grid quota, top-K by confidence at the same budget,
or all inliers. The usability mask also fixes the coverage *metric*: cells with
nothing matchable in them are excluded from the denominator, otherwise a good
registration over mostly-shadowed terrain is refused for the wrong reason.

**Sub-pixel refinement.** Local NCC with a paraboloid peak fit, rejected where a
patch is too flat to hold a peak.

**Trust and refusal.** ≥12 inliers, ≥15 % inlier ratio, ≥15 % coverage,
reprojection RMSE within 4× the RANSAC threshold, held-out RMSE under 6 px, and a
warning below 50 % frame overlap. That last one exists because of a measured
case: a 73 % inlier ratio with 1.22 px reprojection RMSE and 0.89 overlap NCC,
where the warped source covered only 31 % of the reference frame and the fitted
rotation disagreed with the delivered geometry by 4.9°. **No self-consistency
metric can detect that.** Every failing check is listed. **The thresholds are
hand-picked, not calibrated**, and the API and UI both say so.

---

## 7. Metrics — four error concepts, never merged

| Metric | Meaning |
|---|---|
| **Reprojection RMSE** | Retained correspondences agree with the fitted model. **Self-consistency, not accuracy** — a wrong model can score well. |
| **Held-out RMSE** | Fitted on 70 % of correspondences, evaluated on the other 30 %. Tests generalisation. The strongest honest figure when there is no ground truth. |
| **Ground-truth error** | Corners projected through estimated vs *true* transform. **Does not exist for any OHRC pair.** |
| **Disagreement with delivered geometry** | Corner displacement against ISRO's geolocation, in metres. An independent check against a prior that is itself only metre-to-decametre accurate. |

Also reported: inlier counts and ratio, matchable fraction, coverage over
matchable cells *and* over the whole window, overlap NCC before/after, per-stage
runtime.

---

## 8. Results

From `results/illumination/ILLUMINATION.md`. Five 1024 px windows on the primary
pair, affine model, mean over windows. **No ground truth exists**, so held-out
RMSE is the figure to read.

| arm | configuration | cand. | inliers | ratio | reproj RMSE | held-out RMSE | coverage | accepted |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| A | no verification | 2554 | 2554 | 100.0 % | 81.47 px | 85.55 px | 80 % | 0/5 |
| B | + robust verification | 2554 | 1347 | 55.8 % | 1.46 px | 1.46 px | 44 % | 5/5 |
| C | + GSD harmonisation | 2554 | 1347 | 55.8 % | 1.46 px | 1.46 px | 44 % | 5/5 |
| D | + flattening and CLAHE | 2364 | 1345 | 59.7 % | 1.50 px | 1.53 px | 46 % | 5/5 |
| E | + observed usability mask | 2362 | 1303 | 58.2 % | 1.40 px | **1.41 px** | 50 % | 5/5 |
| F | + predicted shadow mask *(SYNTHETIC terrain)* | 986 | 487 | 21.0 % | 1.61 px | 1.57 px | 16 % | 2/5 |
| G | + shading removal *(SYNTHETIC terrain)* | 934 | 453 | 40.8 % | 0.96 px | 1.49 px | 20 % | 2/5 |

Four things to read out of that:

1. **Arm A is the cleanest possible warning about inlier ratio.** 100 % "inliers"
   with an 81 px reprojection RMSE. Without geometric verification the ratio
   means nothing.
2. **Arm C is a no-op** — both products are 0.24 m/px, so there is nothing to
   harmonise. Reported as such.
3. **Arms F and G are worse on every figure that matters.** The synthetic mask
   cuts candidates 2362 → 986 and coverage 50 % → 16 %, and acceptance falls from
   5/5 to 2/5. Arm G's low reprojection RMSE (0.96 px) comes from fitting a much
   smaller, better-behaved region — a selection effect, not an improvement.
   **These rows are not evidence about the real surface.**
4. **Arm E is the best defensible configuration** and it needs no DEM at all.

### Two negative results about our own stages

The same five windows, arm E as the parent (rows H, I, J of the generated table):

| arm | configuration | retained | reproj RMSE | held-out RMSE |
|---|---|--:|--:|--:|
| E | all inliers, no refinement | 1303 | 1.40 px | **1.41 px** |
| H | E + spatial grid selection | 44 | 1.43 px | 1.71 px |
| I | E + sub-pixel refinement | 1303 | 2.36 px | 2.36 px |
| J | E + both | 44 | 2.41 px | 2.92 px |

**Spatially uniform selection — prior art we adopted — hurts in this regime.**
With ~1300 inliers already confined to the lit minority, thinning to ~44 removes
information without improving conditioning — note reprojection RMSE barely moves
(1.40 → 1.43 px) while held-out error rises by a fifth (1.41 → 1.71 px), which is
the signature of a worse-conditioned fit. On the legacy *synthetic* pair the
same stage helps (ground-truth corner error 0.163 → 0.035 px), so it is a regime
difference, not a bug.

**Sub-pixel refinement hurts, and tightening it does not rescue it.** On a
separate three-window sweep the held-out RMSE went 1.346 px unrefined → 2.438 px
(NCC ≥ 0.35, shift ≤ 4 px) → 2.087 px (≥ 0.60, ≤ 2 px) → 1.521 px (≥ 0.80,
≤ 1 px): monotone toward *not refining*. Patch correlation assumes the two
patches are one scene under a photometric transform; under a 67° azimuth change
at grazing incidence they are two different light fields, so the correlation peak
is biased toward whatever the changed shading favours.

Both stages remain implemented and selectable, and both are **off** in the
`seleno` preset. A preset named after the project should be the best
configuration we can defend.

---

## 9. Current limitations

1. **The proposed contribution is not validated.** Predicted-shadow masking is
   implemented end to end but there is **no DEM in this repository**; the only
   working terrain model is synthetic and is labelled as such everywhere.
2. **A local horizon ray-cast is inapplicable to most of this archive.** Three of
   four products have a mid-strip solar elevation ≤ 0 yet contain lit terrain —
   near the pole, high ground is lit over a *depressed* horizon. The prediction
   declares itself inapplicable below 0.05° rather than emitting a mask that
   says "everything is shadowed". Doing it properly needs wide-area horizon
   angles, a different algorithm.
3. **At these Sun angles shadow prediction is intrinsically imprecise.** At 0.79°
   elevation one metre of relief casts a 73 m shadow, so a DEM good to ±2 m
   locates a shadow boundary only to ±150 m — about 600 pixels.
4. **Refusal thresholds are not calibrated.** No precision/recall for the
   decision has been measured.
5. **No ground truth for any OHRC pair**, so no absolute accuracy is reported.
6. **Coarse alignment is one bulk translation**; the residual is still 160–290 m
   and varies along the strip because of the roll difference.
7. **A global 2D transform is an approximation** — OHRC is a pushbroom scanner
   where every line has its own exterior orientation.
8. **No learned matcher has been run on this data by us**; LoFTR reports its
   import error rather than falling back to SIFT.
9. **No real TMC-2 or IIRS data**; IIRS cross-modal matching is not implemented.
10. **n = 4 scenes**, mutually overlapping 58–95 %, so any train/test split would
    have to be geographic.

`ASSUMPTIONS.md` states all of this in full, along with eight label and
documentation defects found in the archive and handled explicitly in code.

---

## 10. Future work

1. **Wide-area horizon angles from a real polar DEM** — the correct formulation
   of the shadow-prediction idea, and the one thing that would let the central
   claim be tested at all.
2. **Calibrate the refusal decision** on labelled overlapping and
   non-overlapping window pairs and report its precision and recall. The
   disjoint-orbit geometry supplies negatives for free.
3. **Along-strip alignment** instead of a single bulk translation.
4. **Install and evaluate LoFTR / SuperGlue / LightGlue** through the existing
   interface — given §8, the remaining gap is exactly where they are reported to
   help.
5. **Sensor-model registration** with SPICE and the ancillary series, replacing
   the global 2D transform.
6. **Multi-frame use of all four strips** over the 53–80 km² they share.
7. **Verify the label MD5 checksums** against the 4.6 GB of imagery.
8. **IIRS cross-modal matching** — a genuinely different problem. Not started.
