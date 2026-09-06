# SELENO — handoff brief for slide-deck drafting

**Purpose of this document.** You are helping build a Smart India Hackathon
presentation about a prototype called **Seleno**. This brief is the single source
of truth for what Seleno is, what it measured, and — critically — what it must
not be said to do. Everything below was produced by running the code; no figure
is aspirational.

**Hard rule for anything you draft: do not upgrade a claim.** The prototype's
credibility rests on being precise about its limits. If a slide would be more
exciting with a stronger claim, the slide is wrong. Section 9 lists the exact
phrases that are banned.

---

## 1. One-paragraph summary

Seleno takes two images of lunar terrain, finds corresponding physical
locations, discards the correspondences that no single geometry can explain,
keeps a spatially distributed subset of what survives, registers the pair, and
reports quantitative results that distinguish *self-consistency* from *accuracy*.
It runs on real Chandrayaan-2 OHRC imagery. It also knows when to refuse: given
two real OHRC strips that do not overlap, it rejects the registration rather than
returning a confident wrong answer.

**Tagline:** Robust Lunar Image Correspondence & Registration.

---

## 2. The problem

Lunar missions need imagery from different sensors and different passes aligned
to the same ground frame — for surface mapping, hazard assessment, resource
localisation and mission planning. Chandrayaan-2 carries three relevant imagers
at wildly different scales:

| Payload | Spatial resolution | Type |
|---|---|---|
| **OHRC** | 0.25 m/px nominal (0.32 m for the 12×3 km two-orbit product) | Panchromatic, very high resolution |
| **TMC-2** | ~5 m/px | Panchromatic, stereo triplet for DEM/morphometry |
| **IIRS** | ~80 m, 0.8–5.0 µm, ~20 nm spectral resolution | Hyperspectral |

Aligning them is hard because resolution, illumination and viewing geometry all
differ at once. Near the poles — where OHRC's most important targets are — low
sun angles produce long shadows that dominate the scene and move between passes,
and classical descriptors degrade badly.

---

## 3. Data — real, and traceable

Three real ISRO Chandrayaan-2 OHRC products, obtained from a **public Internet
Archive mirror of ISRO's release** (PRADAN, the authoritative archive, is
login-gated). Every specification below was **parsed from the PDS4 `.xml` labels
shipped with the products**, not taken from brochures.

| | Orbit 2297 | Orbit 2298 |
|---|---|---|
| Array | 93 693 × 12 000, uint8 | 93 693 × 12 000, uint8 |
| `isda:pixel_resolution` | **0.22977 m/px** | **0.23022 m/px** |
| Spacecraft altitude | 90.41 km | 90.58 km |
| Focal length / detector pitch | 2080 mm / 5.2 µm | same |
| TDI stages | TDI64 | TDI64 |
| Acquired | 2020-02-29 07:39:31 Z | 2020-02-29 09:38:00 Z |
| Latitude range | −74.37° … −73.52° | −73.92° … −73.07° |
| Longitude range | 43.36° … 43.96° | 42.46° … 43.03° |

Both are **south polar**. Their longitude ranges do not overlap — they image
different ground, which is what makes the true-negative pair possible.

*Nice detail for a slide:* the products are **finer than OHRC's quoted 0.25 m**
because they were taken from ~90 km rather than the nominal 100 km. Reading the
per-product label instead of the spec sheet is itself a small point about rigour.

**Selenographic coordinates** in the UI are interpolated from the `*_g_grd_*.csv`
geometry lattice ISRO ships with each product, not from a projection we invented.

**Engineering note worth one line:** the full-resolution image is a 1.12 GB
deflated member inside a ~780 MB ZIP, so random access is impossible. We stream
the compressed member and inflate only as far as the rows we need.

---

## 4. The six demo pairs

Every pixel is real ISRO OHRC data. "Synthetic" **never** means synthetic
terrain — it means a *known transform applied to real imagery* so that a ground
truth exists to score against.

| Pair | What it is | Ground truth |
|---|---|---|
| `ohrc_crater_field` | Well-lit crater field, native 0.2298 m/px, 294 m across. Reference = 11° rotation, 1.12× scale, small perspective, gamma 1.28 | exact, by construction |
| `ohrc_raw_vs_calibrated` | **Fully real both sides**: ISRO's calibrated (`ncp`) vs raw (`nrp`) product of the *same* acquisition, cropped at a known offset. Nothing resampled | exact translation |
| `ohrc_tmc2_moderate` | 2.30 m/px → simulated 5.0 m/px (2.2× ratio) + 6° rotation | exact |
| `ohrc_tmc2_extreme` | 0.2298 m/px → simulated 5.0 m/px — **the real 21.8× OHRC↔TMC-2 scale gap** | exact |
| `ohrc_low_illumination` | Genuinely shadowed south-polar terrain (~35 % of pixels below DN 20) + 9° rotation, gamma 1.6, added noise | exact |
| `ohrc_disjoint_orbits` | **True negative.** Two real strips, orbits 2297 and 2298; the two demo windows' centres are **10.95 km apart** | **none exists** |

### Two data caveats that MUST appear on the limitations slide

1. **There is no real TMC-2 or IIRS imagery in this prototype.** No such product
   was obtainable without credentials in the time available. The `tmc2` pairs
   resample OHRC to TMC-2's *ground sampling distance* and nothing else —
   TMC-2's optics, MTF, noise, stereo geometry and illumination are absent.
   Results there are evidence about **scale-ratio robustness only**.
2. **No pair consists of two independent acquisitions of the same ground.** The
   three available strips cover disjoint tracks. Five of six pairs match real
   imagery against a *transformed copy of itself*, which is substantially easier
   than a genuine repeat pass. Sub-pixel results demonstrate the implementation
   is arithmetically correct; they are **not** evidence of sub-pixel accuracy on
   real repeat-pass data.

---

## 5. The pipeline

```
INPUT (two OHRC products)
  ↓
PREPROCESSING            illumination flattening (divide-by-blur),
                         CLAHE, anti-aliased GSD harmonisation
  ↓
MULTI-SCALE              SIFT / AKAZE / ORB / SIFT-pyramid,
CORRESPONDENCE           Lowe ratio test + mutual-best check
  ↓
GEOMETRIC VERIFICATION   MAGSAC++ / RANSAC,
                         similarity | affine | homography,
                         + physical plausibility checks
  ↓
SPATIAL MATCH SELECTION  N×N grid quota + minimum separation
  ↓
REGISTRATION             least-squares refit, warp,
                         anaglyph / checkerboard / difference
  ↓
METRICS + VERDICT        accept or reject, with reasons
```

Two stages are marked **EXPERIMENTAL / PLANNED** in the UI and are not claimed to
work: a learned matcher (LoFTR — interface implemented, weights not installed)
and IIRS cross-modal matching (not started).

### Why each stage exists — one line each, good for speaker notes

- **Illumination flattening / CLAHE.** Lunar strips span deep shadow and bright
  regolith in one frame; a global stretch leaves the shadowed part featureless.
- **GSD harmonisation.** A naive `resize` on a 21.8× reduction aliases crater
  rims into noise and destroys the features you want to match.
- **Ratio test + mutual check.** Decides how clean the candidate set is *before*
  the robust estimator ever sees it.
- **Geometric verification.** Repeated crater rims and uniform regolith generate
  many individually plausible, collectively impossible matches.
- **Spatial selection.** A transform fitted from a cluster is well constrained
  locally and badly extrapolated at the corners.
- **Verdict.** Deciding when *not* to register is operationally important and
  under-reported.

---

## 6. Results — all measured, all reproducible

Regenerate with `python scripts/benchmark.py`; full tables in
`results/BENCHMARK.md`.

### 6.1 Default pipeline, every pair

| Pair | Cand. | Inliers | Inlier ratio | Reproj RMSE | **GT corner err** | Coverage | NCC before→after | Runtime | Verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `ohrc_crater_field` | 4394 | 4373 | 99.5 % | 0.221 px | **0.163 px** (0.037 m) | 100 % | 0.31 → 0.99 | 1.07 s | accepted |
| `ohrc_raw_vs_calibrated` | 6358 | 6356 | 100.0 % | 0.034 px | **0.005 px** (0.011 m) | 100 % | 0.23 → 1.00 | 0.72 s | accepted |
| `ohrc_tmc2_moderate` | 3385 | 3381 | 99.9 % | 0.148 px | **0.032 px** (0.161 m) | 100 % | −0.04 → 0.89 | 0.24 s | accepted |
| `ohrc_tmc2_extreme` | 232 | 231 | 99.6 % | 2.237 px | **0.039 px** (0.194 m) | 94 % | −0.04 → 0.77 | 0.04 s | accepted |
| `ohrc_low_illumination` | 2277 | 2203 | 96.8 % | 0.651 px | **0.288 px** (0.066 m) | 94 % | 0.63 → 0.96 | 1.00 s | accepted |
| `ohrc_disjoint_orbits` | 22 | 2 | 9.1 % | — | none exists | 6 % | −0.00 → 0.08 | 0.76 s | **REJECTED** |

### 6.2 THE headline finding — put this on its own slide

Ablation on `ohrc_low_illumination` (real shadowed polar terrain):

| Configuration | Inlier ratio | Reproj RMSE | **GT corner err** | Coverage |
|---|--:|--:|--:|--:|
| Full Seleno pipeline | 96.8 % | 0.651 px | **0.288 px** | 94 % |
| Naive (ratio 0.95, no mutual check, no preprocessing, no spatial selection) | 37.3 % | 0.413 px | **1.008 px** | 44 % |
| No spatial selection (top-K by confidence) | 96.8 % | 0.368 px | **0.452 px** | 44 % |
| No preprocessing | 98.8 % | 0.502 px | **0.137 px** | 92 % |

Three things to say, in this order:

1. **Naive is 3.5× less accurate** — 1.008 px vs 0.288 px — while its inlier
   ratio collapses from 96.8 % to 37.3 %.
2. **Dropping spatial selection halves coverage (94 % → 44 %), *improves*
   reprojection RMSE (0.651 → 0.368) and *worsens* real accuracy (0.288 →
   0.452).** This is the single most valuable result in the prototype: it is
   direct evidence that **reprojection RMSE is not accuracy**, and it is exactly
   the failure the spatial stage exists to prevent. Most teams will quote
   reprojection RMSE as if it were accuracy. We can show why that is wrong, with
   our own numbers.
3. **Our own preprocessing makes this pair worse** — turning it off gives fewer
   candidates (1076 vs 2277) but better accuracy (0.137 px vs 0.288 px). CLAHE
   and illumination flattening amplify noise in genuinely dark regions and buy
   matches of lower positional quality. **Do not hide this.** Presenting a
   negative result about our own design is a credibility asset, and it is the
   first item on the future-work list.

### 6.3 Stages that are load-bearing

Ablation on `ohrc_tmc2_extreme` (the 21.8× scale gap):

| Configuration | Candidates | Inliers | GT corner err | Verdict |
|---|--:|--:|--:|---|
| Full pipeline | 232 | 231 | 0.039 px | accepted |
| **No GSD harmonisation** | **9** | **6** | 0.516 px | **REJECTED** |
| No geometric verification | 232 | 232 | reproj RMSE 85.7 px | **REJECTED** |

Without preprocessing the extreme cross-resolution pair collapses from 232
candidates to 9 and fails outright. Without geometric verification, reprojection
RMSE explodes to 85.7 px. Both stages are doing real work.

### 6.4 Matcher comparison (real runs, `ohrc_crater_field`)

| Matcher | Cand. | Inliers | Inlier ratio | Reproj RMSE | GT corner err | Runtime |
|---|--:|--:|--:|--:|--:|--:|
| SIFT | 4394 | 4373 | 99.5 % | 0.221 px | **0.163 px** | 1.25 s |
| SIFT multi-scale pyramid | 8827 | 8746 | 99.1 % | 0.471 px | 0.531 px | 2.19 s |
| AKAZE | 7148 | 7029 | 98.3 % | 0.690 px | 0.255 px | 1.49 s |
| ORB | 2019 | 1833 | 90.8 % | 1.038 px | 0.520 px | 1.01 s |
| LoFTR | — | — | — | — | — | reports `torch/kornia not importable` |

On `ohrc_tmc2_extreme`, AKAZE gets the best accuracy (0.019 px) from only 44
inliers but reaches just 39 % coverage — a nice illustration that accuracy and
conditioning are different things.

**Note on the LoFTR row:** when a matcher is unavailable, Seleno **raises an
error and shows it**, rather than silently falling back to SIFT and mislabelling
the result. That behaviour is deliberate and is worth one sentence on a slide.

### 6.5 The true negative

`ohrc_disjoint_orbits` is **rejected under all seven ablation configurations
tested**. Rejection reasons are reported explicitly:

- only 2 geometrically consistent matches (need ≥ 12)
- inlier ratio 9.1 % is below the 15 % floor
- matches cover only 6 % of the grid

---

## 7. Metrics vocabulary — get this right on the slide

| Metric | What it actually means |
|---|---|
| Candidate matches | Survivors of descriptor matching, before any geometry |
| Inliers / outliers / ratio | The robust estimator's split |
| **Reprojection RMSE** | Symmetric transfer error of retained matches under the fitted model. **Self-consistency, NOT accuracy** — a wrong model can score well |
| **GT corner error** | Image corners projected through estimated vs true transform. **This is the accuracy figure.** Only exists where a true transform exists |
| GT RMSE | Same comparison at the match locations; always flatters relative to corner error |
| Spatial coverage | Fraction of grid cells occupied, compared against top-K by confidence **at the same match budget** |
| Overlap NCC | Photometric agreement before vs after; uses no correspondences, so it is an independent witness |

Pixel errors are converted to metres using the **reference** image's GSD. On
`ohrc_tmc2_extreme` the reference is 5 m/px, so 0.039 px = 0.194 m — a small
*pixel* error on a coarse grid, not a 19 cm measurement.

The coverage comparison is deliberately **same-budget**: comparing the spatial
selection against *all* inliers would be meaningless, because any selection loses
coverage when it drops points.

---

## 8. Prior art — what is explicitly NOT novel

**Makharia, R., Singla, J. G., Amitabh, Dube, N., Sharma, H. (2025). "Comparative
Evaluation of Traditional and Deep Learning Feature Matching Algorithms using
Chandrayaan-2 Lunar Data." arXiv:2509.04775** — <https://arxiv.org/abs/2509.04775>

They benchmark SIFT, ASIFT, AKAZE, RIFT2 and SuperGlue on Chandrayaan-2 data and
report: SuperGlue gives the lowest RMSE and fastest runtimes; SIFT and AKAZE do
well near the equator but degrade under polar lighting; on SAR (SELENE) sets all
classical methods failed while SuperGlue succeeded; preprocessing and learned
matching both matter.

**Therefore: comparing feature matchers on lunar data is done work, and Seleno
does not claim it as a contribution.** Our comparison table characterises our own
pipeline on our own pairs. Our polar degradation result being consistent with
theirs is a *sanity check on our implementation*, not a discovery.

Other related work is listed in `RESEARCH.md` (MoonMetaSync, Yutu-2 PCAM deep
local features, triangulated-network full-Moon registration, remote-sensing
registration review).

---

## 9. Claims discipline — banned phrases

Never say, in any slide, caption or speaker note:

- "sub-pixel accuracy" (unqualified) · "state-of-the-art" · "novel architecture"
- "better than SuperGlue" (SuperGlue was never run here)
- "solves cross-modal matching" · "works on IIRS" (IIRS is not implemented)
- "production-ready" · "ISRO validated" / "ISRO approved"
- "real TMC-2 data" (there is none)
- "AI-powered" as branding

Safe framings that are true:

- "Runs on real Chandrayaan-2 OHRC products, with every specification read from
  the delivered PDS4 labels."
- "Every number is computed by the run on screen; nothing is precomputed."
- "We are not claiming a new model. We are measuring how these approaches behave
  under lunar imaging conditions, and being explicit about which numbers are
  accuracy and which are only self-consistency."
- "It knows when to say no."

---

## 10. Suggested deck structure

Roughly 12 slides; adapt to the time limit.

1. **Title** — SELENO · Robust Lunar Image Correspondence & Registration.
2. **The problem** — §2 table of three payloads at 0.25 m / 5 m / 80 m; the
   resolution, illumination and geometry gap; why polar is worst.
3. **What already exists** — Makharia et al. 2025 and its findings; state plainly
   that feature matching is not our contribution. *Putting this early is a
   strength: it shows we did the literature review.*
4. **What Seleno actually is** — the §5 pipeline diagram, one line per stage.
5. **The data** — §3 table; real OHRC, specs from PDS4 labels, finer than the
   quoted 0.25 m because of the 90 km altitude; the streaming-inflate trick.
6. **Live demo / screenshots** — the four-panel story: candidates → verified
   (inliers green, outliers red) → spatial before/after → registered overlay.
7. **Spatial selection** — the before/after point plots side by side.
   Coverage 44 % → 94 % at the same match budget.
8. **Results** — the §6.1 table.
9. **The headline** — §6.2. Reprojection RMSE improves while real accuracy gets
   worse. Reprojection RMSE is not accuracy.
10. **Knowing when to say no** — §6.5, the true negative and its three reasons.
11. **Honesty slide** — what we do not claim (§9) and the two data caveats (§4).
    *Judges reward this.*
12. **Future work** — §11, ranked.

If there is room for one more, add the ablation slide (§6.3): "which stages are
actually load-bearing".

---

## 11. Future work, ranked by what would most change the result

1. **Obtain real TMC-2 and OHRC data over the same ground** via PRADAN.
   Everything is currently bounded by having no genuine repeat-pass or
   cross-sensor pair.
2. **Chase the negative preprocessing result** (§6.2 item 3) — establish where
   CLAHE and flattening help and where they cost accuracy, per illumination regime.
3. **Physically-based photometric normalisation** (Lommel-Seeliger / Hapke) using
   the incidence and emission angles already present in the PDS4 labels, tested
   against a genuine sun-azimuth change.
4. **Install and evaluate LoFTR / SuperGlue / LightGlue** through the existing
   interface; reproduce the polar finding on our pairs.
5. **Coarse-to-fine cross-resolution matching** instead of decimating the finer
   image, so OHRC's native 0.23 m/px is actually used.
6. **Select correspondences to minimise the transform's covariance** rather than
   filtering spatially after the fact.
7. **Calibrate the accept/reject rule** on a labelled set; report precision and
   recall for the decision itself.
8. **Sensor-model registration** with SPICE and a TMC-2 DEM, replacing the global
   2D transform.
9. **IIRS cross-modal matching** — a genuinely different problem needing band
   selection and a modality-invariant descriptor. Not started.

---

## 12. Known limitations (condense for slide 11)

1. No real cross-sensor pair; `tmc2` pairs simulate a GSD only.
2. No pair of two independent acquisitions of the same ground.
3. A global 2D transform is an approximation — OHRC is a **pushbroom TDI
   scanner** where every image line has its own exterior orientation; rigorous
   registration needs the sensor model, SPICE kernels and a DEM.
4. Illumination handling is image processing, not physics. No photometric model.
   Nothing recovers detail from unlit pixels, and **no pair tests a real change
   of sun azimuth** — the hardest part of polar matching.
5. The learned matcher is an interface, not a result.
6. GSD harmonisation discards resolution (decimates rather than matching
   coarse-to-fine).
7. Verdict thresholds are hand-picked, not calibrated.
8. No held-out validation of the fitted transform; runtimes are single-run,
   single-thread, untuned.

Full detail in `ASSUMPTIONS.md`, which also states exactly what each RMSE figure
does and does not represent.

---

## 13. Practical facts for the demo

- Run: `cd prototype && python run.py` → <http://127.0.0.1:8000>. No setup —
  data pairs and the built frontend are both committed.
- Stack: Python 3.14 · OpenCV 4.13 · NumPy 2.4 · FastAPI · React/Vite.
  No GPU, no model weights, no network access at demo time.
- Runtimes are ~0.04–1.3 s per run, so the demo is live, not pre-rendered.
- UI is a dark, restrained scientific-instrument style — no gradients, no
  dashboard chrome, imagery is the focus. Match the deck to it: dark background,
  monospace for numbers, one accent colour (amber), green for inliers, red for
  outliers.
- Saved screenshots for slides: `results/showcase/<pair>/*.png` —
  `candidates`, `verified`, `spatial_before`, `spatial_after`, `overlay`,
  `checker`, `difference`, `footprint`.
- Suggested live demo order:
  1. `ohrc_crater_field` → **RUN CORRESPONDENCE**, walk the four result tabs.
  2. `ohrc_low_illumination` → run, then untick **Spatial distribution** and run
     again to show coverage 94 % → 44 % and accuracy degrading while
     reprojection RMSE improves.
  3. `ohrc_disjoint_orbits` → rejected, with its three reasons.
  4. **RUN MATCHER COMPARISON** for the live table.

---

## 14. What to ask me for if you need more

- `results/BENCHMARK.md` — the complete generated tables (all pairs × all
  matchers × all ablations).
- `ASSUMPTIONS.md` — the full scientific-caveats document.
- `RESEARCH.md` — sources with URLs, prior art, and the claims/non-claims list.
- `data/README.md` — full provenance, PDS4 label fields, licensing.
- Screenshots from `results/showcase/`.
