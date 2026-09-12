# Seleno — roadmap

Target: **SIH26166 — Multi-modal, Sun angle and scale invariant image
correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)**

Written 2026-09-10. Effort is in working-days for one person. Read
`SESSION_HANDOFF.md` for what exists today and `ASSUMPTIONS.md` for what is
measured versus assumed.

---

## 0. Where we actually are

A working registration pipeline on **real Chandrayaan-2 OHRC Level-2 data**,
with honest evaluation and a refusal layer. Strong on illumination. **Absent on
the two words in the PS title: "multi-modal" and, in the PS's own sense,
"reference".**

| PS requirement | State | Gap |
|---|---|---|
| Source (moving) / Reference (fixed) | ✅ `PairSpec` | — |
| **Illumination variation** | ✅ measured on a real 67° azimuth / 0.96° elevation change at 89–91° incidence | none |
| Viewpoint variation | ✅ affine/homography + MAGSAC++ | — |
| **Scale variation** | ⚠️ GSD harmonisation + multi-scale pyramid implemented | only tested on *simulated* ratios |
| **Multi-modal OHRC/TMC/IIRS** | ❌ | everything is OHRC↔OHRC |
| **Reference = LRO NAC / SELENE** | ❌ | not fetched at all |
| **Sub-pixel accuracy** | ⚠️ implemented | unmeasurable without a reference; measured *harmful* on CH-2↔CH-2 |
| **Uniform match-point distribution** | ⚠️ implemented, currently off | see P0.2 |
| Registered product + match points | ✅ | — |
| Metrics (RMSE, inlier count, inlier ratio) | ✅ exceeds the ask | — |

### The one structural mistake to fix

The PS says *"correspondence between Chandrayaan-2 acquired optical images **and
Lunar reference images**"*. The reference is meant to be an **external basemap**
— LRO NAC or SELENE — not a second Chandrayaan-2 strip. Our whole demo is
CH-2↔CH-2.

Consequence: **LRO NAC is the critical path.** Until a reference basemap exists:

- "multi-modal" is impossible,
- "sub-pixel accuracy" is unmeasurable (no ground truth anywhere in the system),
- the deliverable "registered product" is registered to the wrong thing.

Everything else is secondary to that.

### What the CH-2↔CH-2 work is still worth

Not wasted — reframe it. The measured finding that **CH-2 inter-product
geolocation disagrees by 500–900 m** (3 of 6 product pairs, confidently) is a
*diagnostic* that directly motivates the reference-based approach: you cannot
trust delivered geolocation to place a window, so you need image-based
correspondence against a controlled basemap. That is the PS's premise, arrived
at by measurement. Keep it as the motivating slide, not the deliverable.

---

## P0 — PS compliance. Do these first.

### P0.1 LRO NAC reference ingestion — **the unblocker** · 2–3 days

> **SPIKE RESULT (2026-09-10): access is GREEN, correlation is RED.**
> Reproduce with `python scripts/nac_spike.py`. Detail in §P0.1a below.

Everything downstream depends on this.

1. **Spike first (½ day).** Confirm NAC products covering our footprint
   (lat −89.2° … −89.95°, the four OHRC strips) are fetchable without
   credentials. Links in the PS: `lroc.im-ldi.com`, `quickmap.lroc.im-ldi.com`.
   NAC EDR/CDR live in the PDS Geosciences node. **Do not assume — verify, and
   record the exact product IDs.** If polar NAC coverage is thin or the products
   are unusable, fall back to SELENE TC (10 m/px) and say so.
2. **`backend/seleno/reference/` package**, mirroring `ohrc/`: label parse,
   memmap or GDAL read, geometry → the same south polar stereographic plane.
   The projected plane already exists (`ohrc/geometry.py`) and is sensor-agnostic
   — reuse it, do not write a second one.
3. **Extend `PairSpec`** to accept a reference from a different sensor. The
   fields are already there (`gsd_reference_m`, `sensor_reference`); the pipeline
   already harmonises GSD. This should be a small change if the abstraction holds
   — if it isn't, that's a signal the abstraction is wrong and worth fixing now.

**Acceptance:** `run_registration.py --a <ohrc> --ref-nac <product>` produces a
registered product and match points, with real NAC pixels on the reference side.

**Why it also fixes sub-pixel:** NAC is geodetically controlled. Registering
OHRC → NAC gives an external frame, so RMSE against NAC-derived control points
becomes a *real* accuracy figure rather than self-consistency. That is the first
time "sub-pixel accuracy" in this project means anything.

### P0.1a Spike findings — read before starting P0.1

**Access: solved.** All of this was verified by fetching real bytes, not by
reading documentation.

| | |
|---|---|
| Product | `NAC_POLE_SOUTH_CM_AVG`, LROC South Pole Controlled NAC Average Mosaic |
| Access | public, no credentials; `pds.lroc.asu.edu` 302s to `pds.mcp.nasa.gov` — resolve once and reuse |
| Control | ISIS jigsaw, 18 323 NAC images, 1.68 M points tied to LOLA; **0.8 px at 2-sigma** |
| Grid | 1 m/px, polar stereographic, centre 90 S, **radius 1 737 400 m — identical to `ohrc/geometry.py`** |
| Format | attached PDS3 label, 45 488 x 45 488, 32-bit `PC_REAL`, `RECORD_BYTES` 181 952, `^IMAGE` 2 |
| Mapping | `col = x + 45487.5`, `row = -y - 0.5` (0-based; round-trips exactly) |
| Range reads | **yes**, HTTP 206 at ~4 MB/s — no 7.71 GB downloads needed |
| Tiles | band 1 (88.5-90 S) is four tiles at 45/135/225/315 E; the bulk of our strips is in `P892S2250` |

**Correlation: failed.** OHRC could not be locked onto this mosaic.

| test | result |
|---|---|
| 384 m template, raw pixel window | margin 0.036 — below threshold |
| template sweep 384/768/1536/2304 m | offsets disagree by **1 658 m**, y-component flips sign |
| rotation removed (both reprojected to the same stereographic grid at 1 m/px) | peaks 0.12 / 0.07 / 0.01 / **-0.03**; several offsets pinned at the search boundary |

**Three traps this spike hit, all now guarded in `scripts/nac_spike.py`:**

1. **Per-row range requests earn an HTTP 429.** Use one contiguous request
   covering whole records, then slice columns.
2. **The pole is the corner where all four tiles meet.** Our strips pass within
   2 km of it, so a naive box runs off the tile into negative rows — which curl
   reads as a *suffix* range and happily starts downloading the whole 7.7 GB
   file. Bounds-check every window.
3. **An OHRC strip is only 12 000 samples = 2.88 km wide.** A window near the
   cross-track edge cannot host a large template at all. Rank candidate windows
   by the template they can physically support, not by texture.

**And one methodological lesson worth keeping:** a single correlation's margin is
a weak confidence signal — it produced two false "LOCK" verdicts here.
**Agreement across independent template sizes is far stronger.** Fold that into
`ohrc/reproject.coarse_align` as a second confidence gate.

**Why it most likely failed, in order:**

1. `CM_AVG` is an **average** mosaic. Its own README: it *"does not accurately
   represent any realistic lighting condition."* Correlating a grazing-incidence,
   half-shadowed OHRC window against a shadow-free average is close to a worst
   case. In the test box, **49.9 % of the OHRC pixels that had coverage were at
   or below DN 10.**
2. The **absolute** CH-2 geolocation error may exceed the searched range. We
   searched +/-0.9 to +/-2.9 km; the *relative* CH-2 inter-product error alone is
   500-900 m, so the absolute error against a controlled frame could be larger.
3. Only ~59 % of the 6 km test box had OHRC coverage at all (thin ribbon).

**Next action, cheap and specific:** switch the reference to
**`NAC_POLE_SOUTH_CM_235`** — a *single-illumination* controlled mosaic in the
same family, same projection, same access path. It has real, coherent shadows,
which is what OHRC has. Everything in `scripts/nac_spike.py` carries over by
changing one product name. If that also fails, widen the search to +/-5 km before
concluding anything about matchability.

**Do not conclude from this spike that OHRC cannot be registered to NAC.** What
was shown is that it cannot be correlated against a *shadow-free average* of NAC.
That is a statement about the reference product, not about the method.

---

### P0.2 Uniform match-point distribution as an **output** · ½ day

The PS explicitly requires match points *"maintaining uniform distribution across
the images"*. We implemented that (`spatial.py` grid selection) and I turned it
**off**, because on real CH-2 pairs it made held-out RMSE worse (1.41 → 1.71 px).

Both are true. Resolve by **decoupling the delivered points from the fitting
set**:

- fit the transform on **all verified inliers** — better conditioned,
- export a **uniformly distributed subset** as the match-point product,
- report both counts and the measured trade-off.

The PS asks for uniform distribution in the *output*, not in the estimator. This
gives compliance and accuracy, with the tension documented instead of hidden.

**Acceptance:** the result object carries `match_points` (uniform, with grid
coverage stated) separately from `fit_correspondences`.

### P0.3 Scale-invariance on real ratios · 1 day

Once NAC exists, OHRC (0.25 m/px) ↔ NAC (0.5–2 m/px) is a genuine 2–8× ratio.
Run the existing pyramid matcher and GSD harmonisation against it and report.
Retire the simulated `tmc2_*` legacy pairs from the headline results — keep them
as regression fixtures only, clearly marked.

---

## P1 — the rest of "multi-modal" · 3–5 days

### P1.1 Real TMC-2 · 2 days
From `chmapbrowse.issdc.gov.in` (registration required — **start the account
today**, it is a lead-time item). Gives a real OHRC↔TMC-2 pair at ~20×, and
TMC-2↔NAC at ~2–5×.

### P1.2 Resolution-appropriate reference routing · 1 day
The router your original brief called for, and the answer to "don't do IIRS→NAC
at 160×": choose the reference whose GSD is closest to the source, and *chain*
when the gap is too large — IIRS → TMC-2 → OHRC/NAC rather than one absurd jump.
Small piece of logic, strong story, directly addresses the PS's scale challenge.

### P1.3 IIRS · 2 days, **last**
~80 m/px hyperspectral. Pick a band, treat it as panchromatic, route through
TMC-2. Do not attempt IIRS→OHRC directly. Expect this to be hard and be prepared
to report it as a negative result — that is a legitimate outcome and better than
a fabricated one.

---

## P2 — differentiators (only after P0 lands)

### P2.1 Calibrate the refusal decision · 1 day
The thresholds are hand-picked and the API says so. Labels are **free**: any two
windows whose geometry puts them >2 km apart cannot overlap (the geolocation
error is 500–900 m), giving ~4×10⁶ certain negatives with zero annotation.
Positives from cross-matcher consensus and A→B→C cycle consistency. Fit on the
~10 geometric scalars the pipeline already emits — those transfer across sensors
because they are not pixel statistics. Report precision/recall. sklearn is
already installed.

### P2.2 Learned matcher · ½ day
`pip install torch kornia` → LoFTR pretrained. The interface exists and reports
its own unavailability today. **Use pretrained weights; do not fine-tune** — four
mutually-overlapping polar scenes would overfit and not generalise. Makharia et
al. (2025) reports learned matching wins under polar lighting; this tests that on
our data in an afternoon.

### P2.3 Predicted shadow with a real DEM · 3–5 days
The original research idea. Currently blocked twice over: no DEM ships here, and
a *local* horizon ray-cast is inapplicable at the ≤0° solar elevations that
dominate this archive. Correct formulation is **wide-area horizon angles** over
LOLA/LDEM. High risk, high novelty. Only worth it if P0 and P1 are done.

---

## Sequencing

```
P0.1 LRO NAC ────────────────┬──> P0.3 real scale ratios
     (unblocks everything)   ├──> P1.1 TMC-2 ──> P1.2 routing ──> P1.3 IIRS
                             └──> sub-pixel becomes measurable
P0.2 uniform output  (independent, do any time, ½ day)
P2.*                 (only after P0)
```

**Start today, in parallel:** the NAC fetch spike, and the chmapbrowse account
registration (lead time).

---

## If time runs out — cut in this order

Cut from the bottom. A smaller honest deliverable beats a broad shaky one.

1. Cut IIRS (P1.3). Say plainly it is not implemented and why.
2. Cut predicted shadow (P2.3). It is already labelled experimental everywhere.
3. Cut real TMC-2 (P1.1) if chmapbrowse access does not come through.
4. **Never cut P0.1.** Without a reference basemap the submission does not
   address the problem statement.

Minimum defensible submission: **OHRC → LRO NAC registration, sun-angle robust,
uniformly distributed match points, RMSE/inlier metrics, with the multi-modal
extension shown as architecture plus an honest statement of what was not
reached.**

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Polar NAC coverage is thin or unusable | Critical — P0.1 fails | Spike it in the first half-day. Fall back to SELENE TC. |
| chmapbrowse registration is slow | Blocks TMC-2/IIRS | Register today; the demo does not depend on it |
| OHRC↔NAC fails at 89–91° incidence | Weakens the headline | Expected to be hard. Report honestly — the refusal layer exists precisely for this |
| Over-claiming sub-pixel accuracy | Credibility | Only claim it against NAC control, never on CH-2↔CH-2 |
| Scope creep into IIRS | Everything half-done | It is P1.3 for a reason |

---

## Demo narrative — how it should evolve

**Now:** "we found CH-2 geolocation is 538 m out." True and interesting, but it
answers a question the PS did not ask.

**After P0:** "we register Chandrayaan-2 OHRC to an LRO NAC reference basemap
across a 67° change in solar azimuth at grazing incidence, deliver uniformly
distributed match points, and report RMSE, inlier count and inlier ratio — plus
the system tells you when it should not be trusted."

The 500–900 m geolocation finding becomes slide 2, the *motivation*: delivered
geolocation cannot place a window, so image-based correspondence against a
controlled reference is required. That is the PS's own premise, reached by
measurement.

---

## Claims discipline (unchanged)

`HANDOFF.md` §8 lists the banned phrases. Two additions for this phase:

- Do not say "multi-modal" until a genuinely different sensor is registered.
  Resampling OHRC to 5 m/px is **not** TMC-2.
- Do not say "sub-pixel accuracy" until it is measured against an external
  reference. Self-consistency at 1.2 px is not sub-pixel accuracy.
