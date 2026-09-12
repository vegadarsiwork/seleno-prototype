# SELENO — handoff brief (SIH26166)

**Purpose.** This is the single source of truth for what Seleno is, what it
measured, and what it must not be said to do. Written to be pasted whole into
another tool (an LLM drafting slides, a teammate catching up) without further
context.

**Hard rule for anything drafted from this: do not upgrade a claim.** The
project's credibility rests on being precise about its limits, and its two most
valuable results are a *correction of our own initial conclusion* and a *pair of
negative results about our own stages*. §8 lists the phrases that are banned.

---

## 1. One paragraph

Seleno registers real Chandrayaan-2 OHRC imagery of the lunar south pole. It
reads the delivered PDS4 geometry and per-line Sun series, measures and corrects
the disagreement between two products' geolocation, decides which parts of an
image can be matched at all, finds and robustly verifies correspondences on what
remains, and then decides whether the result should be trusted — returning a
refusal with reasons rather than a confident transform it cannot support.

**Framing:** *geometry-first lunar image registration with predicted unmatchable
regions and calibrated refusal.*

---

## 2. The one number that matters most

> **The delivered geolocation of the two 2024-11-15 OHRC products disagrees by
> 538 m — about 2 244 pixels at 0.24 m/px.**

Before we measured and corrected that, the baseline matcher on 1024 px window
pairs returned 2–3 inliers at a 4–8 % inlier ratio and refused every window.

| window | without the correction | with it |
|---|--:|--:|
| s10240 l52224 | 46 cand · 2 inliers · 4.3 % | 366 cand · 79 inliers · 21.6 % |
| s9216 l52224 | 58 cand · 3 inliers · 5.2 % | 73 cand · 22 inliers · 30.1 % |
| s10976 l41984 | 33 cand · 2 inliers · 6.1 % | 206 cand · 59 inliers · 28.6 % |

**Why this is the strongest slide in the deck.** The obvious, publishable and
completely wrong conclusion was sitting right there: *"a 67° solar azimuth change
collapses SIFT to a 4 % inlier ratio."* It collapsed because the two windows were
2 200 pixels apart. We diagnosed it instead of publishing it. Framed correctly:
**the geometry-first stage is worth more than every other stage in the pipeline
combined**, which is exactly the project's thesis, arrived at by measurement
rather than assertion.

How it was measured: reproject both strips onto a common south polar
stereographic grid, restrict to ground that is *valid and lit in both*, and
correlate **gradient magnitude** (which survives an illumination-direction change
far better than raw DN) over an explicit search range. Peak NCC 0.180 against a
runner-up of 0.070, margin 0.110 → confident. Reproduce:
`python scripts/illumination_experiment.py`.

---

## 3. The data

Four **real Chandrayaan-2 OHRC Level-2 (calibrated) PDS4 products**, ~4.6 GB of
unique pixels, south-polar. **No labels, masks, crater catalogues, DEMs or ground
truth of any kind.** Four scenes, not thousands of images.

| | 20211228T2209 | 20241115T1326 | 20241115T1525 | 20251010T0942 |
|---|--:|--:|--:|--:|
| Lines × samples | 79 796 × 12 000 | 101 074 × 12 000 | 101 074 × 12 000 | 101 075 × 12 000 |
| GSD (m/px) | 0.28 | 0.24 | 0.24 | 0.22 |
| Orbit | 10445 | 23328 | 23329 | 27338 |
| Roll | −7.86° | 14.57° | 15.19° | 26.67° |
| Sun elevation, mid-strip | −0.043° | **−0.169°** | **+0.786°** | −0.771° |
| Solar incidence | 90.04° | 90.17° | 89.21° | 90.77° |
| Pixels ≤ DN 10 | 80.2 % | 71.4 % | 68.6 % | 57.6 % |
| Usable 512 px tiles | 22 % | 31 % | 33 % | 42 % |

**The defining property: solar incidence is 89–91° in every product** — the Sun
sits on or below the local horizon. Two thirds to four fifths of every strip is
at or below DN 10 out of 255; a mid-strip patch can hold as few as five distinct
DN values; only 22–42 % of tiles carry usable gradient information. Deciding what
*can* be matched is not a refinement on this data — it is the problem.

**Primary illumination-stress pair:** `20241115T1326` → `20241115T1525`.
Consecutive orbits (23328, 23329) two hours apart, **95.3 %** mutual footprint,
and at mid-strip **Δazimuth 67.3°, Δelevation 0.955°**.

Good detail for a slide: **sun azimuth sweeps up to 72.8° along a single 16 s
strip**, because azimuth is referenced to local north and these strips pass
within 0.05° of the pole. The label's single scalar is only the mid-strip sample;
we interpolate the `.spm` ancillary series (40 ms cadence) per image line.

Another: **longitude is degenerate here.** One footprint spans longitude
222° → 110° → 22° while covering 25 km of ground. All footprint, overlap and
offset arithmetic is done in projected metres.

Engineering notes worth one line each: every `.img` is read through
`numpy.memmap(mode="r")` and never loaded whole; the archive is never written to;
28 tests assert exactly that, plus byte-exact tile reads and that the geometry
lattice indexes the raster 1:1.

---

## 4. The pipeline

```
INPUT (two OHRC products, memory-mapped)
  ↓
METADATA / GEOMETRY            PDS4 label, geometry lattice, per-line Sun series
  ↓
REFERENCE / OVERLAP            footprints in polar stereographic metres;
ESTIMATION                     coarse offset between products MEASURED
  ↓
ILLUMINATION-AWARE             observed usability mask (no DEM) or predicted
PREPROCESSING                  shadow mask (terrain model); flatten + CLAHE
  ↓
CORRESPONDENCE                 SIFT / AKAZE / ORB / pyramid; LoFTR slot
  ↓
ROBUST VERIFICATION            MAGSAC++ / RANSAC; similarity | affine | homography
  ↓
CORRESPONDENCE SELECTION       grid quota | top-K | all; coverage measured over
                               the MATCHABLE area, not the whole window
  ↓
SUB-PIXEL REFINEMENT           local NCC + paraboloid peak fit
  ↓
TRUST / REFUSAL DECISION       accept | warning | refuse, every reason listed
  ↓
RESULT
```

Two design decisions worth mentioning:

* **Masked correspondences are dropped after detection, not before.** Blanking a
  region first manufactures a hard edge the detector then fires on.
* **The usability mask also fixes the coverage metric.** Two thirds of a window
  is shadow, so scoring coverage against every grid cell refuses a good
  registration whose matches necessarily sit in the lit minority. Cells with
  nothing matchable in them are excluded from the denominator; both figures are
  reported.

---

## 5. Results

Regenerate with `python scripts/illumination_experiment.py`; the generated table
is `results/illumination/ILLUMINATION.md`.

Five 1024 px windows on the primary pair, affine model, mean over windows.
**No ground truth exists for any of them**, so held-out RMSE is the figure to
read.

| arm | configuration | cand. | inliers | ratio | reproj RMSE | held-out RMSE | coverage | accepted |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| A | no verification | 2554 | 2554 | 100.0 % | 81.47 px | 85.55 px | 80 % | 0/5 |
| B | + robust verification | 2554 | 1347 | 55.8 % | 1.46 px | 1.46 px | 44 % | 5/5 |
| C | + GSD harmonisation | 2554 | 1347 | 55.8 % | 1.46 px | 1.46 px | 44 % | 5/5 |
| D | + flattening and CLAHE | 2364 | 1345 | 59.7 % | 1.50 px | 1.53 px | 46 % | 5/5 |
| E | + observed usability mask | 2362 | 1303 | 58.2 % | 1.40 px | **1.41 px** | 50 % | 5/5 |
| F | + predicted shadow mask *(SYNTHETIC terrain)* | 986 | 487 | 21.0 % | 1.61 px | 1.57 px | 16 % | 2/5 |
| G | + shading removal *(SYNTHETIC terrain)* | 934 | 453 | 40.8 % | 0.96 px | 1.49 px | 20 % | 2/5 |
| H | E + spatial grid selection | 2362 | 1303 | 58.2 % | 1.43 px | 1.71 px | 47 % | 5/5 |
| I | E + sub-pixel refinement | 2362 | 1303 | 58.2 % | 2.36 px | 2.36 px | 50 % | 5/5 |
| J | E + both | 2362 | 1303 | 58.2 % | 2.41 px | 2.92 px | 47 % | 5/5 |

*(These are the committed figures from `results/illumination/ILLUMINATION.md`.
The window sampler is deterministic, so re-running reproduces them; changing
`--windows` or `--size` will not.)*

### What to say about each row

1. **Arm A is the cleanest warning about inlier ratio you will ever get:** a
   100 % "inlier ratio" alongside an 81 px reprojection RMSE and 0/5
   acceptances. Without geometric verification the ratio is meaningless. Good
   slide on its own.
2. **Arm C is a no-op** — both products are 0.24 m/px, so there is nothing to
   harmonise. Reported as a no-op rather than dressed up as a contribution.
3. **Arms F and G are worse on every figure that matters.** The synthetic mask
   cuts candidates 2362 → 986 and coverage 50 % → 16 %, and acceptance falls from
   5/5 to 2/5. Arm G's low reprojection RMSE (0.96 px) comes from fitting a much
   smaller, better-behaved region — a selection effect, not an improvement.
   **These rows are not evidence about the real surface** — the terrain model is
   synthetic.
4. **Arm E is the best configuration we can defend, and it needs no DEM at all.**

### A third finding, and the best argument for the refusal layer

On one window the pipeline produced a **73 % inlier ratio, 1.22 px reprojection
RMSE, 1.33 px held-out RMSE and an overlap NCC of 0.89** — and the warped source
covered only **31 % of the reference frame**, with a fitted rotation of 12.5°
against the 7.6° the delivered geometry implies. Both facts are true: inside the
31 % the alignment is genuinely good; outside it the transform is extrapolated,
and because the correspondences sit in a diagonal band rather than across the
frame, rotation and translation trade off almost freely.

**No self-consistency metric can detect this.** It is the sharpest available
answer to "why do you need a refusal layer at all", and it is why frame overlap
is now one of the checks.

### The two negative results about our own stages

**Spatially uniform selection — prior art we adopted — hurts in this regime.**
Grid thinning cuts ~1300 inliers to ~44; reprojection RMSE barely moves
(1.40 → 1.43 px) while held-out error rises by a fifth (1.41 → 1.71 px). That
divergence is the signature of a worse-conditioned fit, and it is a second
independent demonstration that reprojection RMSE must not be read as accuracy.
On the legacy *synthetic* pair the same stage helps (ground-truth corner error
0.163 → 0.035 px), so it is a regime difference, not a bug. The selector now
skips thinning below 60 inliers.

**Patch-correlation sub-pixel refinement hurts, and tightening it does not
rescue it.** In the table above it takes held-out RMSE from 1.41 px to 2.36 px.
On a separate three-window sweep: 1.346 px unrefined → 2.438 px (NCC ≥ 0.35) →
2.087 px (≥ 0.60) → 1.521 px (≥ 0.80). Monotone toward *not refining*. Clean
physical reason: patch correlation assumes the two patches are one scene under a
photometric transform; under a 67° azimuth change at grazing incidence they are
two different light fields, so the peak is biased toward whatever the changed
shading favours.

Both stages stay implemented and selectable, and both are **off** in the `seleno`
preset. A preset named after the project should be the best configuration we can
defend, not an aspirational one.

### Legacy regression (synthetic pairs, ground truth exists)

The new architecture is equal or better on all six retained pairs:

| pair | ground-truth corner error | status |
|---|--:|---|
| `ohrc_crater_field` | 0.071 px (was 0.163) | accepted |
| `ohrc_raw_vs_calibrated` | 0.004 px | accepted |
| `ohrc_tmc2_moderate` | 0.034 px | accepted |
| `ohrc_tmc2_extreme` | 0.035 px | **warning** (held-out 4.14 px) |
| `ohrc_low_illumination` | 0.138 px (was 0.288) | accepted |
| `ohrc_disjoint_orbits` | none exists | **refused** |

`tmc2_extreme` being downgraded to a warning is an improvement in honesty: the
old pipeline accepted it silently despite a poor held-out error.

---

## 6. The second important finding: the contribution is not yet testable

**A local horizon ray-cast is inapplicable to most of this archive.** Three of
the four products have a mid-strip solar elevation at or below zero yet plainly
contain lit terrain — near the pole, high ground is lit over a *depressed*
horizon while the sub-satellite point is not. A local flat-horizon test declares
the entire window shadowed, which contradicts the imagery.

Rather than emit that mask, `predicted_shadow_mask` **declares itself
inapplicable below 0.05° elevation** and the pipeline falls back to no masking
with the reason stated in the UI.

Doing it properly requires **horizon angles computed over tens of kilometres of
DEM** — a different algorithm, not a tuning change. That is the single most
valuable next step, and it is the honest position: the proposed contribution is
**implemented, plumbed end to end, and not yet validated**, and the reason is
specific and fixable.

Supporting number worth quoting: **at 0.79° solar elevation, one metre of relief
casts a 73 m shadow** (at 0.2°, 286 m). So a DEM good to ±2 m locates a shadow
boundary only to ±150 m — about 600 pixels at 0.24 m/px. Any predicted shadow
mask on this data is intrinsically coarse, which is why masks here are
probabilistic, carry a `boundary_uncertainty_m`, and are computed at the DEM's
own spacing rather than per pixel.

---

## 7. Metrics vocabulary — get this right on the slide

| Metric | What it actually means |
|---|---|
| **Reprojection RMSE** | Retained correspondences agree with the fitted model. **Self-consistency, NOT accuracy** — a wrong model can score well. |
| **Held-out RMSE** | Fitted on 70 % of correspondences, evaluated on the other 30 %. Tests generalisation. **The strongest honest figure when there is no ground truth.** |
| **Ground-truth error** | Corners projected through estimated vs *true* transform. **Does not exist for any OHRC pair.** |
| **Disagreement with delivered geometry** | Corner displacement against ISRO's geolocation, in metres. An independent check against a prior itself only metre-to-decametre accurate — not truth. |
| Inlier ratio | Meaningless without geometric verification (see arm A). |
| Coverage | Fraction of **matchable** grid cells occupied. Sensitive to grid size: the same run scored 11 % on 6×6 and 19 % on 4×4 — quote the grid. |
| **Frame overlap** | Fraction of the reference frame the warped source actually covers. Below 50 % the result is downgraded to a warning: outside the overlap the transform is extrapolated. |
| Confidence | A bounded monotone combination of the above. **Not a probability, not calibrated.** |

---

## 8. Claims discipline — banned phrases

Never say, in any slide, caption or speaker note:

- "sub-pixel accuracy" unqualified — the sub-pixel figures belong to *synthetic*
  legacy pairs; on real repeat-pass windows the honest figure is ~1.2–1.3 px
  held-out RMSE with **no ground truth at all**
- "validated" / "proven" / "demonstrates that shadow prediction works" — it does
  not; see §6
- "calibrated refusal" as a *result* — the mechanism exists, the calibration does
  not
- "state-of-the-art" · "novel architecture" · "better than SuperGlue" (never run)
- "solves cross-modal matching" · "works on IIRS" (not implemented)
- "real TMC-2 data" (there is none)
- "ISRO validated" / "ISRO approved" · "production-ready"
- "AI-powered" as branding

Safe framings that are true:

- "Runs on four real Chandrayaan-2 OHRC Level-2 products, with every
  specification read from the delivered PDS4 labels."
- "We measured that the delivered geolocation of two products disagrees by 538 m,
  and correcting it is worth more than every other stage in our pipeline
  combined."
- "It refuses. On a real non-overlapping pair, under every configuration tested."
- "We are not claiming a new matcher. Matcher comparisons on Chandrayaan-2 data
  are already published (Makharia et al. 2025, arXiv:2509.04775)."
- "Two of our own stages made things worse on this data, and we report that."

---

## 9. Suggested deck structure

1. **Title** — SELENO · geometry-first lunar image registration with predicted
   unmatchable regions and calibrated refusal.
2. **The problem** — south-polar OHRC at 89–91° solar incidence; two thirds of
   every strip below DN 10; only 22–42 % of tiles matchable. Show the §3 table.
3. **The data** — four real Level-2 products, no labels, 95.3 % overlap pair with
   Δazimuth 67.3°. Show `results/dataset/pair_2024_regions.png`: the same ground
   under two illuminations, with a cast shadow that dominates one and is absent
   from the other.
4. **What already exists** — Makharia et al. 2025; spatially uniform selection is
   prior art. State plainly that matcher choice is not our contribution.
   *Putting this early is a strength.*
5. **Architecture** — the §4 diagram.
6. **THE FINDING** — §2. The 538 m geolocation error, the 4 % → 22–30 % table,
   and the wrong conclusion we avoided. This is the best slide in the deck.
7. **Live demo / screenshots** — source · reference · registered overlay, then
   the usability mask and inliers/outliers panels.
8. **Ablation** — the §5 table. Lead with arm A (100 % inliers, 81 px RMSE,
    0/5 accepted).
9. **Negative results** — grid selection and sub-pixel refinement both hurt here,
   with the physical explanation. *Judges reward this.*
10. **Knowing when to say no** — the refusal decision and its reasons; the
    disjoint-orbit pair refused under every configuration.
11. **Honesty slide** — §8 banned claims + §6: the contribution is implemented,
    plumbed, and not yet validated, and here is exactly why.
12. **Future work** — §10.

---

## 10. Future work, ranked by what would most change the answer

1. **Wide-area horizon angles from a real polar DEM (LOLA/LDEM).** The correct
   formulation of the shadow-prediction idea, and the only way the central claim
   becomes testable. See §6.
2. **Calibrate the refusal decision** on labelled overlapping and
   non-overlapping window pairs; report precision and recall. The disjoint-orbit
   geometry supplies negatives for free.
3. **Along-strip alignment** instead of one bulk translation — the residual
   disagreement is still 160–290 m and varies along the strip because of the
   roll difference.
4. **Install and evaluate LoFTR / SuperGlue / LightGlue** through the existing
   interface. Given the negative results in §5, the remaining appearance gap is
   exactly where they are reported to help.
5. **Sensor-model registration** with SPICE and the `.oat`/`.lbr`/`.spm` series,
   replacing the global 2D transform, which is what a rigorous pushbroom
   solution requires.
6. **Multi-frame use of all four strips** over the 53–80 km² they share.
7. **Verify the label MD5 checksums** against the 4.6 GB of imagery.
8. **IIRS cross-modal matching** — a genuinely different problem. Not started.

---

## 11. Practical facts for the demo

- Run: `cd prototype && python run.py` → <http://127.0.0.1:8000>. The OHRC
  archive is expected at `../datasettesting/dataset` (override with
  `SELENO_OHRC_ROOT`).
- Stack: Python 3.14 · OpenCV 4.13 · NumPy 2.4 · SciPy 1.17 · FastAPI ·
  React/Vite. **CPU only, no GPU, no model weights, no network at demo time.**
- A 1024 px registration runs in ~2–4 s. The coarse alignment takes ~50 s the
  first time per product pair and is then cached under
  `results/illumination/alignment/`.
- Suggested live order:
  1. Pick `20241115T1326` → `20241115T1525`, take the top candidate window,
     **RUN REGISTRATION**. Walk the mask → candidates → verified → overlay tabs.
  2. **Untick "Apply measured coarse offset"** and run again. This is §2, live:
     the inlier ratio collapses and the result is refused.
  3. **RUN ABLATION** for the live arm table.
  4. Switch to *Legacy pairs* → `ohrc_disjoint_orbits` → refused with three
     reasons.
- Reproduce anything: `python scripts/illumination_experiment.py`,
  `python scripts/inspect_dataset.py`, `python tests/test_ohrc_dataset.py`.
- Figures already on disk: `results/dataset/` (strip thumbnails, tile montages,
  DN histograms, cross/along-track profiles, the 2024 pair regions) and
  `results/illumination/` (ablation chart, per-window diagnostics).

---

## 12. Ask me for these if you need more

- `results/illumination/ILLUMINATION.md` — the generated experiment tables.
- `RESEARCH.md` — sources with URLs, prior art, the full claims/non-claims list.
- `ASSUMPTIONS.md` — every scientific caveat, plus eight label and documentation
  defects found in the archive and handled explicitly in code.
- `data/README.md` — full provenance for both data sources.
- `results/dataset/dataset_summary.json` — every number behind the figures.
