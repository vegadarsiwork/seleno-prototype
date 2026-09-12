# Planner handoff — Seleno / SIH26166

Written 2026-09-10. Self-contained: assume no prior context.

Companion docs, in the order a planner should read them:
`ROADMAP.md` (the plan) → this file (what changed, what to decide) →
`ASSUMPTIONS.md` (what is measured vs assumed) → `HANDOFF.md` (presentation
brief) → `SESSION_HANDOFF.md` (earlier build session).

---

## 1. The project in one paragraph

**Seleno** registers Chandrayaan-2 optical imagery for **SIH26166 — "Multi-modal,
Sun angle and scale invariant image correspondence using Chandrayaan-2 optical
images (OHRC, TMC and IIRS)"**. It runs on four real OHRC Level-2 south-polar
products (4.6 GB, no annotations of any kind). The pipeline is nine stages:
metadata/geometry → reference & overlap estimation → illumination-aware
preprocessing → correspondence → robust verification → correspondence selection
→ sub-pixel refinement → metrics → **trust/refusal decision**. It is classical
computer vision: **no neural network, no training, no model weights.** Python,
CPU-only, FastAPI + React demo at `python run.py`.

---

## 2. Current state: what works, what does not

**Working and verified:**
- Memory-safe OHRC dataset layer (`numpy.memmap`, archive never written to),
  28/28 tests passing.
- Full registration pipeline on real CH-2↔CH-2 repeat-pass windows.
- Honest metrics: reprojection RMSE, held-out RMSE, ground-truth error, and
  disagreement-with-delivered-geometry are **four separate fields, never merged**.
- Refusal layer with enumerated reasons.
- Six legacy synthetic pairs retained as a regression fixture.

**Not working / not built:**
- **Multi-modal (TMC-2, IIRS): absent.** Everything is OHRC↔OHRC.
- **LRO NAC / SELENE reference: access solved, correlation not yet achieved.**
- Sub-pixel accuracy: unmeasurable (no external reference yet).
- Refusal thresholds: hand-picked, **not calibrated**.
- Predicted-shadow masking: implemented but runs on a **synthetic** terrain
  model; also inapplicable at the ≤0° solar elevations dominating this archive.

---

## 3. What changed this session

### 3.1 We read the actual problem statement for the first time — and it reframes the project

Until now the team worked from a brief, not the PS. The PS says:

> *"correspondence between Chandrayaan-2 acquired optical images **and Lunar
> reference images**"* — LRO NAC, SELENE.

**The reference is meant to be an external, controlled basemap — not a second
Chandrayaan-2 strip.** The entire existing demo is CH-2↔CH-2. This is the single
biggest gap and it drives the whole roadmap.

The PS also explicitly requires match points *"maintaining uniform distribution
across the images"* — a stage that exists in the code (`spatial.py`) but was
**turned off** because it worsened held-out RMSE (1.41 → 1.71 px). See §5.1.

### 3.2 `ROADMAP.md` was created

P0 (PS compliance) → P1 (multi-modal) → P2 (differentiators), with a cut-order
for time pressure and a risk table. **That is the plan to continue with.**

### 3.3 The P0.1 spike was run: LRO NAC access GREEN, correlation RED

Reproduce: `python scripts/nac_spike.py`. Full detail in `ROADMAP.md` §P0.1a.

Access — all verified by fetching real bytes:

| | |
|---|---|
| Product | `NAC_POLE_SOUTH_CM_AVG` (LROC South Pole Controlled NAC Average Mosaic) |
| Access | public, no credentials |
| Control | ISIS jigsaw, 18 323 NAC images tied to LOLA, **0.8 px at 2-sigma** |
| Grid | 1 m/px polar stereographic, R = 1 737 400 m — **identical to our `ohrc/geometry.py`** |
| Format | attached PDS3 label, 45 488², 32-bit float, `RECORD_BYTES` 181 952 |
| Range reads | HTTP 206, ~4 MB/s — no 7.71 GB downloads needed |

Correlation — three escalating attempts, all failed:

| test | result |
|---|---|
| 384 m template | margin 0.036, below threshold |
| sweep 384/768/1536/2304 m | offsets disagree by **1 658 m**, y-sign flips |
| rotation removed (both on the same stereographic grid) | peaks 0.12 / 0.07 / 0.01 / **−0.03**, offsets pinned at the search boundary |

**Most likely cause:** `CM_AVG` is an *average* mosaic; its own README says it
*"does not accurately represent any realistic lighting condition."* We asked a
grazing-incidence OHRC window (**49.9 % of covered pixels ≤ DN 10**) to correlate
against a shadow-free average. Worst-case pairing.

**This is NOT evidence that OHRC cannot register to NAC.** It is evidence it
cannot correlate against a *shadow-free average* of NAC. Statement about the
reference product, not the method.

### 3.4 The 538 m geolocation finding was stress-tested

Previously n=1. Now measured across all six CH-2 product pairs: **3 of 6 give a
confident lock, all in the 498–896 m band.** The other three all involve the 2021
strip (darkest, different roll and TDI) and correctly report *not confident*
rather than returning garbage — the system degrades to refusing, not to being
wrong.

### 3.5 Four methodological lessons worth keeping

1. Per-row HTTP range requests earn a **429**; use one contiguous request.
2. **The pole is the corner where all four NAC tiles meet.** A naive box there
   goes to negative rows, which curl reads as a *suffix* range and starts pulling
   the whole 7.7 GB file. Bounds-check everything.
3. **OHRC strips are only 12 000 samples = 2.88 km wide.** Windows near the
   cross-track edge cannot host a large correlation template at all. Rank
   candidates by the template they can physically support, not by texture.
4. **A single correlation's margin is a weak confidence signal** — it produced
   two false "LOCK" verdicts. **Agreement across independent template sizes is
   far stronger.** Fold this into `coarse_align` as a second gate (~20 lines).

---

## 4. The roadmap we are continuing with

From `ROADMAP.md`. Status updated for this session.

| ID | Item | Effort | Status |
|---|---|---|---|
| **P0.1** | LRO NAC reference ingestion — **the unblocker** | 2–3 d | ⚠️ access green, correlation red. Blocked on the §5.2 decision |
| **P0.2** | Uniform match-point distribution as an **output** | ½ d | 🟢 ready, independent of everything |
| **P0.3** | Scale invariance on the real 4.2× OHRC→NAC ratio | 1 d | ⛔ needs P0.1 |
| **P1.1** | Real TMC-2 from chmapbrowse (registration required) | 2 d | 🟡 **register today — lead-time item, blocks nothing** |
| **P1.2** | Resolution-appropriate reference routing (avoid 160× jumps) | 1 d | ⛔ needs P1.1 |
| **P1.3** | IIRS, routed via TMC-2, **last** | 2 d | ⛔ needs P1.2 |
| **P2.1** | Calibrate the refusal decision (labels are free, see §6) | 1 d | 🟢 ready |
| **P2.2** | Pretrained LoFTR/SuperGlue via the existing interface | ½ d | 🟢 ready |
| **P2.3** | Predicted shadow with a real DEM (wide-area horizon angles) | 3–5 d | ⛔ high risk, do last |

**Cut order under time pressure:** IIRS → predicted shadow → real TMC-2.
**Never cut P0.1** — without an external reference the submission does not
address the problem statement.

---

## 5. Decisions the planner needs to make

### 5.1 (Decided, needs implementing) Uniform distribution: output vs estimator

The PS requires uniformly distributed match points. Our measurement says grid
selection *hurts* accuracy (held-out 1.41 → 1.71 px). **Resolution: decouple
them.** Fit the transform on all verified inliers; export a uniformly distributed
subset as the delivered match-point product. Compliance and accuracy, with the
trade-off documented. That is P0.2.

### 5.2 (Open) Which NAC reference product?

Ordered options:

1. **`NAC_POLE_SOUTH_CM_235`** — single-illumination controlled mosaic, same
   family, same projection, same access path. Real coherent shadows, which is
   what OHRC has. **~5 minutes to test; one constant change in
   `scripts/nac_spike.py`.** Do this first.
2. Widen the search to ±5 km (the absolute CH-2 error may exceed what we searched).
3. Low-Sun controlled ROI mosaics (Haworth, Malapert, Nobile, Shackleton) if our
   footprint intersects one.
4. Feature-based matching (SIFT/LoFTR) against NAC instead of intensity
   correlation — intensity correlation may simply be the wrong tool across this
   appearance gap.
5. SELENE TC (10 m/px) as a coarser fallback.

**Do not conclude "NAC is unusable" until at least options 1 and 2 are tried.**

### 5.3 (Open, inherited) `.gitignore` vs cited results

`results/*/` currently ignores `results/dataset/` and `results/illumination/`,
but `README.md` and `HANDOFF.md` both cite
`results/illumination/ILLUMINATION.md` as "the committed figures". Either track
those results or reword the docs. A proposed `.gitignore` edit was rejected by
the user; needs a decision, not a silent change.

---

## 6. Facts the planner must not violate

**Claims discipline.** `HANDOFF.md` §8 has the full banned-phrase list. The
critical ones:

- Do not say **"multi-modal"** until a genuinely different sensor is registered.
  Resampling OHRC to 5 m/px is **not** TMC-2.
- Do not say **"sub-pixel accuracy"** until it is measured against an external
  reference. Self-consistency at 1.2 px is not sub-pixel accuracy.
- Do not say **"validated"** about shadow prediction — it runs on synthetic
  terrain and is inapplicable at this archive's solar elevations.
- Do not say **"calibrated refusal"** as a *result*; the mechanism exists, the
  calibration does not.
- Never: "state-of-the-art", "novel architecture", "better than SuperGlue"
  (never run), "ISRO validated", "production-ready".

**Measured numbers that must not drift** (regenerate, never hand-edit —
`scripts/illumination_experiment.py` writes the tables and
`sync_tables.py`/`sync_prose.py` in the session scratchpad push them into the
docs):

- CH-2 inter-product geolocation disagreement: **498–896 m**, confident on 3/6 pairs
- Best ablation arm (E): 58.2 % inlier ratio, **1.41 px held-out RMSE**, 50 % coverage, 5/5 accepted
- Arm A (no verification): 100 % "inlier ratio" with **81 px** reprojection RMSE, 0/5 accepted
- Grid selection: held-out 1.41 → **1.71** px (worse). Sub-pixel: 1.41 → **2.36** px (worse). Both off by default.
- Only **22–42 %** of 512 px OHRC tiles carry usable texture
- Solar incidence **89–91°** across all four products

**Two negative results about our own stages are deliberate and must stay
reported** (grid selection and sub-pixel refinement both make held-out error
worse on this data). They are a credibility asset, not a bug to hide.

---

## 7. Free training labels — relevant if P2.1 is scheduled

There is no ground truth and no way to hand-label lunar correspondences. But the
refusal decision can be calibrated without any annotation:

- **Certain negatives are free.** The geolocation error is 500–900 m, so any two
  windows whose geometry places them **>2 km apart cannot overlap**. Measured:
  **91.9 %** of sampled cross-strip window pairs qualify. Scaled across 4 081
  windows and 6 strip pairs that is ~**4 × 10⁶** labelled negatives.
- **Positives** from cross-matcher consensus (SIFT/AKAZE/ORB agreeing on one
  transform) and A→B→C cycle consistency across three strips.
- Discard the ambiguous 300 m–2 km band rather than guessing.
- Train on the ~10 **geometric scalars** the pipeline already emits (inlier
  ratio, coverage, held-out RMSE, frame overlap, prior disagreement, …). Those
  transfer across sensors because they are not pixel statistics. sklearn 1.8.0 is
  already installed.
- **Do not fine-tune a matcher** on four mutually-overlapping polar scenes — it
  would memorise this terrain and not generalise. Use pretrained weights.

---

## 8. Recommended next three actions

1. **Test `NAC_POLE_SOUTH_CM_235`** (~5 min). Decides P0.1 green vs rethink.
2. **Register for chmapbrowse** (~5 min of form-filling, days of lead time).
   Blocks nothing now, unblocks all multi-modal work later.
3. **Implement P0.2** (½ day). PS-required, independent of everything above.

Items 2 and 3 should proceed regardless of how item 1 turns out.
