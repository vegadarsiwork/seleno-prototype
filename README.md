# SELENO

**Robust Lunar Image Correspondence & Registration — prototype**

Seleno takes two images of lunar terrain, finds corresponding physical
locations, throws out the correspondences that cannot be geometrically
explained, keeps a spatially distributed subset of what survives, registers the
pair, and reports what it actually measured.

Every pixel it runs on comes from a delivered ISRO Chandrayaan-2 OHRC product.
Every number in the interface is computed by the run you just triggered.

```
python run.py        →  http://127.0.0.1:8000
```

---

## 1. What this demonstrates

The problem is lunar image correspondence and registration across Chandrayaan-2
imagery: matching views of the same ground when resolution, illumination and
viewing geometry differ.

The prototype shows six things end to end:

1. **Preprocessing** — illumination flattening, CLAHE, and ground-sampling-distance harmonisation.
2. **Correspondence extraction** — SIFT / AKAZE / ORB, ratio test, mutual-consistency check, optional multi-scale pyramid.
3. **Geometric verification** — MAGSAC++/RANSAC splits candidates into a self-consistent set and outliers.
4. **Spatially distributed selection** — a grid quota with minimum separation, compared against the fair top-K-by-confidence baseline.
5. **Registration** — least-squares refit, warp, two-colour overlay, checkerboard, difference image.
6. **Evaluation** — counts, ratios, RMSE, coverage, runtime, and *where a true transform exists*, ground-truth accuracy reported separately.

It also demonstrates **refusing to register**. Two real OHRC strips from
consecutive orbits whose demo tiles are 10.95 km apart are rejected under every
configuration tested — see the ablation table in `results/BENCHMARK.md`.

### What it is not

Feature matching on Chandrayaan-2 data is established work. **Makharia et al.
(2025)**, [arXiv:2509.04775](https://arxiv.org/abs/2509.04775), already benchmark
SIFT, ASIFT, AKAZE, RIFT2 and SuperGlue on exactly this data and report that
preprocessing and learned matching both matter. **We do not present feature
matching as novel.** No claim is made here of sub-pixel geodetic accuracy,
state-of-the-art performance, a novel architecture, superiority to SuperGlue,
cross-modal IIRS matching, production readiness, or any ISRO validation.
See `RESEARCH.md` §5 and `ASSUMPTIONS.md`.

---

## 2. Architecture

```
                    Image A (source)        Image B (reference)
                            |                       |
                            +-----------+-----------+
                                        v
   preprocess.py    PREPROCESSING          illumination flattening, CLAHE,
                                           anti-aliased GSD harmonisation
                                        v
   matchers.py      CORRESPONDENCE         SIFT / AKAZE / ORB / pyramid
                                           ratio test + mutual check
                                           (LoFTR slot, probed not faked)
                                        v
                    candidate correspondences
                                        v
   verify.py        GEOMETRIC              MAGSAC++ / RANSAC
                    VERIFICATION           similarity | affine | homography
                                           + physical plausibility checks
                                        v
   spatial.py       SPATIAL SELECTION      grid quota + minimum separation
                                           vs top-K by confidence
                                        v
   register.py      REGISTRATION           least-squares refit, warp,
                                           anaglyph / checkerboard / difference
                                        v
   metrics.py       EVALUATION             counts, inlier ratio, reprojection
                                           RMSE, ground-truth error, coverage,
                                           NCC, runtime
                                        v
   pipeline.py      VERDICT                accept / reject, with reasons
```

```
prototype/
├── run.py                     single entry point
├── README.md  RESEARCH.md  ASSUMPTIONS.md
├── requirements.txt
├── backend/
│   ├── app.py                 FastAPI: /api/config, /api/run, /api/compare, images
│   └── seleno/
│       ├── pipeline.py        stage orchestration, timing, verdict
│       ├── preprocess.py      stage 1
│       ├── matchers.py        stage 2 + pluggable learned-matcher interface
│       ├── verify.py          stage 3 + transform decomposition, plausibility
│       ├── spatial.py         stage 4
│       ├── register.py        stage 5 + visual products
│       ├── metrics.py         stage 6
│       ├── geodesy.py         pixel → selenographic, from ISRO's geometry grid
│       ├── store.py           manifest and image access
│       └── viz.py             all drawing
├── frontend/                  Vite + React (dist/ is prebuilt and committed)
├── data/
│   ├── README.md              full provenance
│   ├── manifest.json          pair definitions + ground-truth transforms
│   └── pairs/                 the six demo pairs
├── results/BENCHMARK.md       generated, never hand-edited
└── scripts/
    ├── fetch_products.py      download ISRO products
    ├── prepare_data.py        build the six pairs
    ├── run_pipeline.py        CLI harness
    └── benchmark.py           regenerate results/BENCHMARK.md
```

---

## 3. Data

Three real OHRC products, from the public Internet Archive mirror of ISRO's
release (no credentials needed). Read from the PDS4 labels, not from brochures:

| | Orbit 2297 | Orbit 2298 |
|---|---|---|
| Array | 93693 × 12000 uint8 | 93693 × 12000 uint8 |
| `isda:pixel_resolution` | **0.22977 m/px** | **0.23022 m/px** |
| Altitude | 90.41 km | 90.58 km |
| Latitude | −74.37° … −73.52° | −73.92° … −73.07° |

Both are south polar. Their longitude ranges do not overlap.

Six demo pairs. "Synthetic" never means synthetic terrain — it means a known
transform applied to real imagery so ground truth exists.

| Pair | Difficulty | Ground truth |
|---|---|---|
| `ohrc_crater_field` | baseline | synthetic transform |
| `ohrc_raw_vs_calibrated` | **fully real**, calibrated vs raw of the same acquisition | exact translation, nothing resampled |
| `ohrc_tmc2_moderate` | cross-resolution 2.3 → 5.0 m/px | synthetic |
| `ohrc_tmc2_extreme` | cross-resolution 0.23 → 5.0 m/px (21.8×) | synthetic |
| `ohrc_low_illumination` | real shadowed polar terrain, ~35 % of pixels below DN 20 | synthetic |
| `ohrc_disjoint_orbits` | **true negative**, tile centres 10.95 km apart | none exists |

**No real TMC-2 or IIRS imagery was obtainable without credentials.** The
`tmc2` pairs resample OHRC to TMC-2's 5 m/px GSD; TMC-2's optics, MTF, noise and
stereo geometry are *not* simulated. Full detail in `data/README.md`.

---

## 4. Installation

```bash
pip install -r requirements.txt
```

Python 3.14.3, OpenCV 4.13, NumPy 2.4.4 on Windows 11 is what this was built and
benchmarked on. The demo pairs are committed, so nothing needs downloading to run
the demo.

To rebuild the data from the ISRO products:

```bash
pip install remotezip
python scripts/fetch_products.py --out data/raw     # ~1 GB streamed, ~180 MB kept
python scripts/prepare_data.py --raw data/raw
```

To rebuild the frontend (only if you change it — `dist/` is committed):

```bash
cd frontend && npm install && npm run build
```

---

## 5. Running

```bash
python run.py                                   # UI + API on :8000
```

CLI, no browser needed:

```bash
python scripts/run_pipeline.py --pair ohrc_crater_field
python scripts/run_pipeline.py --all
python scripts/run_pipeline.py --pair ohrc_low_illumination \
       --ratio 0.95 --no-mutual --no-preprocess --no-spatial   # the naive baseline
python scripts/run_pipeline.py --pair ohrc_crater_field --save results/
python scripts/benchmark.py                     # regenerates results/BENCHMARK.md
```

In the UI: pick a pair in the left rail, adjust the experiment controls, press
**RUN CORRESPONDENCE**. **RUN MATCHER COMPARISON** runs every available matcher
plus the naive configuration on the current pair and tabulates them.

---

## 6. What each stage does

**Preprocessing.** Illumination flattening divides the image by a heavily blurred
copy of itself, removing the large-scale brightness ramp while keeping
crater-scale texture. CLAHE equalises contrast locally so shadowed regions keep
usable gradients. GSD harmonisation anti-aliases before decimating — a naive
`resize` on a 21.8× reduction turns crater rims into noise. This is *image
processing*, not photometric correction; no Hapke or Lommel-Seeliger model is
applied and the pipeline diagram says so.

**Correspondence.** A detector/descriptor pass, Lowe's ratio test, and a
mutual-best check. The pyramid variant matches the source at several decimation
levels and pools the results, which is what recovers matches across a large scale
ratio.

**Geometric verification.** MAGSAC++ (homography) or RANSAC (affine, similarity)
finds the largest subset explainable by one transform. Repeated crater rims and
uniform regolith generate many individually plausible, collectively impossible
matches; this is the stage that removes them. The fitted transform is then
decomposed into scale, rotation, shear and perspective and checked against the
scale the two products' GSDs already imply.

**Spatial selection.** Ranking by descriptor confidence concentrates matches on
the few high-contrast structures in a scene. A transform fitted from a cluster is
well constrained locally and badly extrapolated at the corners. The selector
imposes a per-cell quota on an N×N grid plus a minimum separation, and is
compared against **top-K by confidence at the same K** — comparing against all
inliers would be meaningless, since any selection loses coverage when it drops
points.

**Registration.** Least-squares refit on the retained correspondences, then warp
into the reference frame. Three views: a green/magenta anaglyph (grey where the
two agree, coloured fringes where they do not), a checkerboard (structures must
run straight across seams), and a contrast-stretched difference.

**Evaluation and verdict.** See below. The verdict rejects unless there are ≥12
inliers, ≥15 % inlier ratio, ≥15 % grid coverage, and reprojection RMSE within 4×
the RANSAC threshold — with every failing reason listed.

---

## 7. Metrics

| Metric | What it means |
|---|---|
| Candidate matches | Survivors of descriptor matching, before geometry |
| Inliers / outliers / ratio | The robust estimator's split |
| Retained matches | After spatial selection |
| **Reprojection RMSE** | Symmetric transfer error of retained matches under the fitted model. **Self-consistency, not accuracy** — a wrong model can score well |
| **GT corner error** | Image corners projected through estimated vs true transform. **This is the accuracy figure.** Only where a true transform exists |
| GT RMSE | Same comparison at the match locations; always flattering relative to corner error |
| Spatial coverage | Fraction of grid cells occupied, vs the same-budget confidence baseline |
| Overlap NCC | Photometric agreement before vs after registration; uses no correspondences, so it is an independent witness |
| Runtime | Per stage and total, single run, single thread |

Pixel errors are converted to metres using the **reference** image's GSD.

---

## 8. Results

From `results/BENCHMARK.md`, regenerated by `python scripts/benchmark.py`.

Default pipeline, all six pairs:

| Pair | Cand. | Inliers | Inlier ratio | Reproj RMSE | GT corner err | Coverage | NCC | Verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `ohrc_crater_field` | 4394 | 4373 | 99.5 % | 0.221 px | 0.163 px | 100 % | 0.31 → 0.99 | accepted |
| `ohrc_raw_vs_calibrated` | 6358 | 6356 | 100.0 % | 0.034 px | 0.005 px | 100 % | 0.23 → 1.00 | accepted |
| `ohrc_tmc2_moderate` | 3385 | 3381 | 99.9 % | 0.148 px | 0.032 px | 100 % | −0.04 → 0.89 | accepted |
| `ohrc_tmc2_extreme` | 232 | 231 | 99.6 % | 2.237 px | 0.039 px | 94 % | −0.04 → 0.77 | accepted |
| `ohrc_low_illumination` | 2277 | 2203 | 96.8 % | 0.651 px | 0.288 px | 94 % | 0.63 → 0.96 | accepted |
| `ohrc_disjoint_orbits` | 22 | 2 | 9.1 % | — | none | 6 % | −0.00 → 0.08 | **REJECTED** |

### The stages that are load-bearing

Ablation on `ohrc_tmc2_extreme` (the 21.8× scale gap):

| Configuration | Candidates | Inliers | GT corner err | Verdict |
|---|--:|--:|--:|---|
| full pipeline | 232 | 231 | 0.039 px | accepted |
| **no GSD harmonisation** | **9** | **6** | 0.516 px | **REJECTED** |
| no geometric verification | 232 | 232 | — (RMSE 85.7 px) | **REJECTED** |

Ablation on `ohrc_low_illumination` (real shadowed polar terrain):

| Configuration | Inlier ratio | Reproj RMSE | **GT corner err** | Coverage |
|---|--:|--:|--:|--:|
| full pipeline | 96.8 % | 0.651 px | **0.288 px** | 94 % |
| naive (ratio 0.95, no mutual check, no preprocessing, no spatial selection) | 37.3 % | 0.413 px | **1.008 px** | 44 % |
| no spatial selection (top-K by confidence) | 96.8 % | 0.368 px | **0.452 px** | 44 % |
| no preprocessing | 98.8 % | 0.502 px | **0.137 px** | 92 % |

Three things worth saying out loud about that table:

1. **The naive configuration is 3.5× less accurate** (1.008 px vs 0.288 px) while
   its inlier ratio collapses to 37 %.
2. **Dropping spatial selection halves coverage (94 % → 44 %), *improves*
   reprojection RMSE (0.651 → 0.368) and *worsens* real accuracy (0.288 →
   0.452).** This is the clearest evidence in the prototype that reprojection
   RMSE must not be read as accuracy — and it is precisely what the spatial stage
   exists to fix. The same effect is visible in the `3a`/`3b` panels of the UI:
   the confidence-ranked matches pile into the right-hand third of the frame.
3. **Our preprocessing makes this pair worse.** Turning it off yields fewer
   candidates (1076 vs 2277) but better accuracy (0.137 px vs 0.288 px). CLAHE
   and illumination flattening amplify noise in the genuinely dark regions and
   buy matches of lower positional quality. That is a real negative result about
   our own preprocessing and it is not hidden; it is the first thing to
   investigate next.

`ohrc_disjoint_orbits` is rejected under **all seven** configurations tested.

---

## 9. Current limitations

1. **No real cross-sensor pair.** No TMC-2 or IIRS imagery was obtainable without
   credentials. The `tmc2` pairs simulate a GSD, nothing more.
2. **No pair of two independent acquisitions of the same ground.** The three
   available OHRC scenes cover disjoint tracks. Five of six pairs match real
   imagery against a transformed copy of itself, which is substantially easier
   than a genuine repeat pass — sub-pixel results must be read in that light.
3. **A global 2D transform is an approximation.** OHRC is a pushbroom scanner
   where every line has its own exterior orientation; rigorous registration needs
   the sensor model, SPICE kernels and a DEM.
4. **Illumination handling is image processing, not physics.** No photometric
   model. Nothing recovers detail from unlit pixels, and no pair tests a genuine
   change of sun azimuth — the hardest part of polar matching.
5. **The learned matcher is an interface, not a result.** LoFTR is wired in and
   selectable; `torch`/`kornia` are not installed here, so it reports the import
   error instead of silently falling back to SIFT. We have run no learned matcher
   on this data.
6. **GSD harmonisation discards resolution** — the finer image is decimated
   rather than matched coarse-to-fine.
7. **Verdict thresholds are hand-picked**, not calibrated against a labelled set.
8. **No held-out validation** of the fitted transform, and single-run untuned
   runtimes.

`ASSUMPTIONS.md` states all of this in full, including exactly what each RMSE
figure does and does not represent.

---

## 10. Future work

Ranked by what would most change the result:

1. **Get real TMC-2 and OHRC data over the same ground** via PRADAN. Everything
   above is bounded by not having a genuine repeat-pass or cross-sensor pair.
2. **Chase the negative preprocessing result** in §8 — establish where CLAHE and
   flattening help and where they cost accuracy, per illumination regime.
3. **Physically-based photometric normalisation** (Lommel-Seeliger / Hapke) using
   the incidence and emission angles already present in the PDS4 labels, tested
   against a genuine sun-azimuth change.
4. **Install and evaluate LoFTR / SuperGlue / LightGlue** through the existing
   interface, and reproduce Makharia et al.'s polar finding on our pairs.
5. **Coarse-to-fine cross-resolution matching** instead of decimating the finer
   image, so OHRC's native 0.23 m/px is actually used.
6. **Select correspondences to minimise the transform's covariance** rather than
   filtering spatially after the fact.
7. **Calibrate the accept/reject rule** on a labelled set of overlapping and
   non-overlapping pairs, and report precision/recall for the decision itself.
8. **Sensor-model registration** with SPICE and a TMC-2 DEM, replacing the global
   2D transform.
9. **IIRS cross-modal matching** — a genuinely different problem needing
   band selection and a modality-invariant descriptor. Not started.
