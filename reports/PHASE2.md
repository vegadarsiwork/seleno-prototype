# Phase 2 — report

Date: 2026-09-18. Follows `docs/AUDIT.md` (Phase 0) and `docs/DATA_INVENTORY.md`
(Phase 1). Addresses the eight decisions given at Phase 2 approval.

Claims are tagged **[measured]** when a script in this repository produced the
number on this machine, and **[archive]** when it is read from a product's own
label or README.

---

## 0. Status against the eight decisions

| # | decision | status |
|---|---|---|
| 1 | Δ sub-solar longitude as primary stratifier, keep Δ incidence reported | **done** — both columns in `data/index/pairs.csv`, benchmark stratifies by Δaz |
| 2 | Re-establish and verify the OHRC↔NAC offset, one command | **done, partially locked** — §2. 2 of 4 strips confirmed, reproducing the previous session independently; 1 provisional, 1 honest failure |
| 3 | DEM-mediated illumination normalisation vs direct matching | **done, answer is negative** — §3. Does not lift the >50° bins; §6 shows what does |
| 4 | Coarse-to-fine mandatory; report search radius needed | **done** — §4. ±8 km coarse needed; offsets are 3.9–4.6 km |
| 5 | Add a non-polar site with real incidence variation | **done, via TMC-2** — 3 TMC-2 products acquired 2026-09-18 at lat −3.7…−41.1, incidence 39–46°. See `reports/TMC2_SELENE.md`. OHRC↔NAC is still not constructible off-pole (§5) |
| 6 | Wire RIFT2 and a pretrained learned matcher into the seam | **reconciled** — §6 uses the four-corner error criterion on the complete original sweep |
| 7 | Keep the SELENE morning/evening experiment | **partly done** — TMC-2 ↔ SELENE TC evening registers at 60 m (`reports/TMC2_SELENE.md`); the morning/evening contrast still needs a site where both maps carry data |
| 8 | Apply the `SELENO_OHRC_ROOT` backend fix | **done** — 28/28 dataset tests pass with no environment variable set |
| — | T2 (SPICE/ISIS/ASP) feasibility | **infeasible, three independent reasons** — §8 |

---

## 1. New code

| module | what it is |
|---|---|
| `backend/seleno/dem.py` | LOLA DEM windowed access; per-pixel solar geometry from the body-fixed Sun vector and local ENU frame; Lambertian relighting with ray-marched cast shadow |
| `backend/seleno/nac.py` | LROC NAC mosaic access at two levels (browse pyramid, full-resolution GeoTIFF over `/vsicurl`), with the north-up→+y flip handled and asserted |
| `backend/seleno/align.py` | masked NCC and feature-vote translation estimators on a shared grid, with a sign contract pinned by a self-test, and cluster-based cross-family agreement |
| `backend/seleno/phasecong.py` | Kovesi phase congruency + RIFT-style max-index-map descriptor |
| `scripts/measure_geolocation_offset.py` | the one-command item-2 measurement |
| `scripts/benchmark_illumination.py` | matcher benchmark against real Sun-azimuth change and controlled common-grid geometry |

`backend/seleno/matchers.py` gained `rift` and `disk_lightglue` behind the existing
`run_matcher` seam; nothing else in the pipeline changed.

---

## 2. The geolocation offset (decision 2)

### 2.1 Why the reference chain is not the explanation

The claim is that *someone else's* delivered geolocation is wrong by kilometres,
so the burden is on us to rule out our own projection, row order and sign
conventions first. Three guards, all **[measured]**:

**Guard A — the estimators are pinned to known truth.** `seleno.align._selftest`
builds B by rolling A a known amount; both masked NCC and the feature vote must
recover it. Literal output:

```
sign contract: (dx, dy) added to A's coordinates reaches B

   true (  +0,   +0)   ncc (   +0.0,    +0.0) err  0.00   vote (   -0.0,    +0.0) err  0.00
   true ( -23,  +17)   ncc (  -23.0,   +17.0) err  0.00   vote (  -23.0,   +17.0) err  0.00
   true ( +11,  -31)   ncc (  +11.0,   -31.0) err  0.00   vote (  +11.0,   -31.0) err  0.00
   true ( +40,  +40)   ncc (  +40.0,   +40.0) err  0.00   vote (  +40.0,   +40.0) err  0.00

worst masked-NCC error 0.00 px, worst feature-vote error 0.00 px  -> PASS
```

This guard earned its keep: the first version of `masked_ncc_shift` returned the
**negated** shift, and the self-test caught it. An inverted sign is
indistinguishable from a real offset by inspection.

**Guard B — the reference registers against independent terrain at zero.** Each
NAC illumination mosaic was registered against a LOLA relight of the same ground.
Both products are tied to LOLA, so a correct chain must return ~0:

| bin | dx (m) | dy (m) | \|d\| (m) | margin | shadow-mask corr |
|---|--:|--:|--:|--:|--:|
| CM_045 | −32.0 | 0.0 | **32.0** | 0.158 | +0.826 |
| CM_115 | 0.0 | 0.0 | **0.0** | 0.128 | +0.701 |
| CM_355 | +32.0 | 0.0 | **32.0** | 0.029 | +0.527 |

Worst residual **32 m = one browse pixel**, across three different Sun azimuths.
Our plane, row order, Sun geometry and NAC georeferencing are therefore not the
source of a kilometre-scale offset.

This also validates the Phase 1 sub-solar-longitude model: rendering the DEM at
bin `XXX` reproduces mosaic `CM_XXX`'s shadow field (correlation +0.53 to +0.83).
If the longitude convention were flipped or offset, it would not.

**Guard C — the conventions were perturbed on purpose.** **[measured]**

| perturbation | change in the answer | required | verdict |
|---|--:|---|---|
| longitude expressed in [−180, 180) instead of [0, 360) | **0.0 m** | zero | PASS |
| reference raster row order deliberately flipped | **9 183.5 m** | large | PASS |

The first confirms the result is not an artefact of a longitude wrapping choice.
The second is a positive control: a convention bug of exactly the kind being
ruled out *does* move the answer by kilometres, so the test has the power to
detect one.

**Guard D — a second, physical cross-check.** The relight is not a fit; it is the
terrain plus a Sun vector. That it predicts the real shadow field at all is an
independent confirmation that the DEM, the mosaic and our grid agree.

### 2.2 The measurement

`(dx, dy)` is the correction to be **added to the Chandrayaan-2 geometry** to
reach the LOLA-tied frame. Confirmed = at least two different algorithm families
agree within tolerance. **[measured]**

| product | families agreeing | coarse dx, dy | fine refinement | **final dx, dy** | **\|d\|** | spread | verdict |
|---|---|---|---|---|--:|--:|---|
| 20241115T1326 | akaze + ncc + sift | (−2526.5, −3798.4) | (−23.6, +36.3) | **(−2550.1, −3762.1)** | **4544.9 m** | 51.4 m | **confirmed** |
| 20241115T1525 | akaze + ncc + sift | (−1950.3, −3402.2) | (−99.3, +10.4) | **(−2049.6, −3391.8)** | **3963.0 m** | 37.1 m | **confirmed** |
| 20251010T0942 | ncc only | (−1409.0, −3906.7) | not attempted | — | 4153.0 m | — | provisional |
| 20211228T2209 | — | — | — | — | — | — | **no lock** |

The fine stage moved the answer by only **43 m** and **100 m** respectively, which
is itself a result: the coarse estimate from a sub-megabyte browse pyramid was
already good to about a fine-stage pixel.

### 2.3 Independent corroboration

`results/experiments/P0_FINDINGS.md` measured the same quantity in a previous
session with different code, a different reference product (`CM_AVG`, the
shadow-free average) and a different window. It is an independent measurement,
not a re-run:

| product | this run | previous session | difference |
|---|---|---|--:|
| 20241115T1326 | (−2550, −3762) after fine | (−2546, −3679) | **83 m** |
| 20241115T1525 | (−2050, −3392) after fine | (−2103, −3385) | **54 m** |
| 20251010T0942 | (−1409, −3907) provisional | (−1668, −3420) | 553 m |
| 20211228T2209 | no lock | no lock | — |

Two confirmed strips agree with an independent prior measurement to **54 m and
83 m** on a ~4 km quantity — that is agreement to better than 2%, between two
measurements sharing no code, no reference product and no window. The strip that fails here also failed there, for the same
stated reason (it is the darkest: 71–80% of its pixels are at or below DN 10).

### 2.4 Consistency, and what it implies

All four estimates — including the provisional one — point into the **same
quadrant** (dx and dy both negative), with magnitudes 3.9–4.6 km. A bug in our
own projection would apply the *same* correction to every product; these differ
per product by 0.4–0.9 km, which is the inter-product scatter measured
independently from OHRC↔OHRC alone in an earlier session (498–896 m).

Every product label carries `corners_refined = False` **[archive]** — ISRO does
not claim these are photogrammetrically refined. This measures how large that is.

### 2.5 What this does not claim

- No value is claimed for strips that did not lock. They are reported as
  failures, not omitted.
- The offset is modelled as a pure translation. Whether it varies along a
  101 000-line strip is **untested** and is the obvious next experiment.
- 20251010T0942 is single-family and is labelled provisional, not confirmed.

Reproduce: `python scripts/measure_geolocation_offset.py` → `reports/GEOLOCATION.md`,
`results/geolocation/offsets.json`.

---

## 3. DEM-mediated vs direct matching (decision 3) — reporting early, as instructed

**The DEM-mediated arm does not currently beat direct matching, and it does not
lift the high-Δaz cases.** Per the instruction to say so early: this is the
finding.

| product | direct NAC | DEM relight |
|---|---|---|
| 20241115T1326 | **confirmed**, 3 families, spread 51 m | single-family only |
| 20241115T1525 | **confirmed**, 3 families, spread 37 m | single-family only |
| 20251010T0942 | provisional | single-family, different answer |
| 20211228T2209 | no lock | no lock |

Two things were learned that are worth keeping:

1. **The relight itself is sound.** It reproduces the real NAC shadow field
   (mask correlation +0.53 to +0.83) and registers against it at 0–32 m. As a
   *reference-side* product it works.
2. **The failure is in the DEM's spatial content, not the physics.** The LOLA GDR
   is gridded at 5 m but interpolated from altimeter tracks, so it carries strong
   along-track striping. The gradient of the relight is dominated by those
   artefacts: matching on `grad` against the relight fails while matching on raw
   intensity succeeds. Smoothing the DEM to ~60 m before relighting fixed the
   reference control (CM_355 went from no-lock to lock) but did not rescue the
   OHRC arm.

Interpretation: at 0.24 m native GSD, OHRC resolves terrain roughly two orders of
magnitude finer than LOLA actually constrains. The relight predicts *where the
big shadows are*, which is a low-frequency constraint, and the OHRC strip is a
2.88 km-wide ribbon that often contains only one shadow boundary — a single
straight edge, which constrains one direction and leaves the other free.

**Recommendation:** keep the renderer — it earns its place as the reference-side
validator in §2.1 and as the item-7 tool — but **do not make DEM-mediated
matching the primary path**. Direct matching against an illumination-matched NAC
bin is the stronger arm today, and §6 shows that the thing which actually lifts
the high-Δaz band is a pretrained learned matcher, not DEM mediation. Revisit if
a higher-resolution DEM becomes available over the site; a NAC-derived DTM at
2–5 m would change this assessment.

Per the instruction "if this doesn't lift the >50° bins, say so early — it
changes the whole plan": **it does not, and the plan should pivot to the learned
matcher.**

---

## 4. Coarse-to-fine and the search radius actually needed (decision 4)

**[measured]**

| stage | sampling | search | substrate | cost |
|---|---|---|---|---|
| coarse | 32.0 m/px | **±8 km** | NAC browse pyramid, whole tile | ~0.3–0.9 MB per illumination bin |
| fine | 8 m/px on a 4 km box | ±600 m | full-resolution GeoTIFF window | ~180 MB per window |

The coarse radius **must** exceed the placement error: measured offsets are
3.9–4.6 km, so ±2 km — the radius used in the earliest spike of this project —
was structurally incapable of finding them, and the resulting "offsets disagreeing
by 1658 m" were runs pinned against different search walls.

**±8 km is the recommended coarse radius** for Chandrayaan-2 polar strips: it
covers the observed 4.6 km maximum with roughly 1.7× headroom.

The browse pyramid is what makes ±8 km affordable. The full-resolution GeoTIFF is
striped (one row per block), so a windowed read fetches whole 45 488-pixel
scanlines: a ±8 km search box at 1 m/px would cost about 1 GB, while the whole
tile at 32 m/px costs under a megabyte. Search coarse, refine fine.

---

## 5. A non-polar site (decision 5) — found, but not as an OHRC↔NAC pair

**[measured]** The public Internet Archive mirror carries **three** OHRC
acquisitions, none of them polar:

| product | orbit | latitude | longitude | GSD | incidence span over all sub-solar longitudes |
|---|--:|---|---|--:|--:|
| `…20200229T0739312111…` | 2297 | −74.35 … −73.50 | 43.31 … 43.91 E | 0.230 m | ~33° |
| `…20200229T0938004033…` | 2298 | −73.91 … −73.06 | 42.41 … 42.99 E | 0.230 m | ~33° |
| `…20200824T1003365280…` | 4455 | −62.49 … −61.66 | 56.58 … 56.90 E | 0.258 m | ~55° |

These carry **real incidence variation** — the thing the polar archive
structurally cannot provide — and they are public, needing no credentials. They
also carry a populated `corners_refined` block that differs from the system
corners, unlike the four polar products.

**But OHRC↔NAC is not constructible at any of them.** Searching all 21 551
map-projected NAC products in the LROC RDR cumulative index:

- products overlapping the orbit-2297 site: **232**, of which the finest is
  **100 m/px** — every one is a WAC product;
- map-projected **NAC** products covering the orbit-2297 site: **0**;
- covering the orbit-4455 site: **0**;
- nearest NAC product of any kind: `NAC_ROI_BGSLWSKYLOA_P731S0392` at 1.5 m/px,
  **0.5° away** — it overlaps in latitude but its longitude ends at 41.51°E while
  the OHRC strip starts at 42.40°E, missing by roughly 26 km.

The LROC RDR archive does not contain blanket map-projected NAC coverage; NAC
RDRs exist where someone made them. Producing one over an arbitrary site means
calibrating and projecting NAC EDRs through ISIS — the same wall as T2 (§8).

**What the non-polar site can be, today:**

| pairing | reference GSD | scale ratio | status |
|---|--:|--:|---|
| OHRC ↔ SELENE TC morning/evening | 7.403 m | ~31× | **available**, verified present over the orbit-2297 site |
| OHRC ↔ LROC WAC global mosaic | 100 m | ~435× | available, already on disk |
| OHRC ↔ NAC | — | — | **not constructible without ISIS** |

Recommendation: add the orbit-2297 site as a non-polar OHRC ↔ SELENE TC pair.
That gives real incidence variation *and* a genuine cross-mission, cross-scale
pairing, which is closer to the problem statement's "multi-modal" requirement
than anything currently in the repository. Flag plainly that it is not an
OHRC↔NAC pair and why.

---

## 6. Matchers (decision 6)

`run_matcher` now offers **[measured]**:

| name | kind | available | note |
|---|---|---|---|
| `sift`, `sift_pyramid`, `akaze`, `orb` | classical | yes | unchanged |
| `rift` | radiation-insensitive | yes | **flagged unvalidated** — see below |
| `loftr` | learned | yes | torch 2.14.0+cpu, kornia 0.8.3 now installed |
| `disk_lightglue` | learned | yes | DISK + LightGlue, both pretrained |

DISK + LightGlue was chosen over SuperPoint + SuperGlue deliberately: the
SuperPoint/SuperGlue weights are released for non-commercial research only, while
DISK and LightGlue are Apache-2.0. There is no CUDA on this machine
(`torch.cuda.is_available()` is `False`, 14 CPU threads), so it runs on CPU.

### RIFT: an honest negative, correctly attributed

Our RIFT implementation **does not work well enough to be quoted**. The
diagnosis is precise, and it matters which half failed **[measured]**, on
co-registered LROC bins 30° apart in Sun azimuth where the truth is identity:

| | phase-congruency detector | SIFT detector |
|---|--:|--:|
| keypoints found | 2 643 / 2 746 | 1 094 / 1 137 |
| repeat within 3 px across the illumination change | **54.6%** | 43.1% |

**The detector half works, and beats SIFT's repeatability.** The descriptor half
does not: a 0.85 ratio test returns zero matches and looser thresholds return
noise (median implied displacement (83, 82) px against a truth of (0, 0)).

This is a statement about **our descriptor**, not about RIFT2. It is registered
in `MATCHERS` with `validated: False` and a caveat string so it cannot be quoted
accidentally. Porting the authors' reference descriptor is the open task; the
phase-congruency front end is already worth keeping on the repeatability number
alone.

### Reconciled illumination benchmark — 2026-09-23

This replaces the earlier translation-consensus result. The sole solve criterion
is **mean Euclidean error at the four image corners ≤ 3 evaluation-grid pixels**.
Each matcher feeds a homography fitted with USAC_MAGSAC (3 px inlier threshold).
Identity ground truth is used only for final corner scoring; no translation-only
or identity prior constrains the fitted model. Insufficient matches and failed
fits remain failures in the denominator.

The complete original Phase 2 sweep was rerun: tile `P892S2250`, bins
`005, 065, 125, 185, 245, 305`, all 15 unordered pairs, two 512 × 512 patches
at **32.01099488134059 m per browse pixel**, four matchers, 120 runs on CPU.
No azimuth range or failed case was omitted. The common controlled map grid
provides the identity reference and inherits the mosaic's control uncertainty;
it is not an exact independent survey of every detector pixel.

| Matcher | Runs | Solved | Rate | Median corner error among solved (px) |
|---|---:|---:|---:|---:|
| sift | 30 | 0 | 0.0% | — |
| akaze | 30 | 0 | 0.0% | — |
| orb | 30 | 0 | 0.0% | — |
| disk_lightglue | 30 | 3 | 10.0% | 1.9 |

| Matcher | Δaz 45–90° | Δaz 90–135° | Δaz 135–180° |
|---|---:|---:|---:|
| sift | 0/12 | 0/12 | 0/6 |
| akaze | 0/12 | 0/12 | 0/6 |
| orb | 0/12 | 0/12 | 0/6 |
| disk_lightglue | 3/12 | 0/12 | 0/6 |

**The reconciled DISK+LightGlue result is 3/12 at Δaz 45–90° under the stated
corner criterion.** None of these six bins produces a <45° pair. No tested
method solved a >90° case in this sweep. This is one tile and two patches,
not a mission-wide success rate.

The audit's native NAC fixture result used a different population and sampling,
including larger azimuth differences, through the full registration tool. Its
numerically matching fraction does not make the datasets equivalent. The
reconciled table above is the illumination benchmark to quote; the earlier
translation-based number is superseded.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u \
  scripts/benchmark_illumination.py --bins 005 065 125 185 245 305 \
  --patches 2 --matchers sift akaze orb disk_lightglue
```

[All cases, matrices and corner errors](../results/illumination_reconciled_20260923/raw.json)
and [criterion, grid and summary](../results/illumination_reconciled_20260923/summary.json)
are committed. A regression with zero median translation but incorrect scale
ensures this criterion cannot silently regress to translation consensus.

---

## 7. SELENE morning/evening (decision 7)

Kept, and verified present over the non-polar OHRC site **[measured]**:

```
morning TCO_MAPm04_S72E042S75E045   HTTP 200
evening TCO_MAPe04_S72E042S75E045   HTTP 200
```

Label **[archive]**: 12 288², simple cylindrical, 4 096 px/deg = 7.403 m/px,
`STANDARD_GEOMETRY = (30.0, 0.0, 30.0)`, `PHOTO_CORR_ID = "USGS"`.

The value is exactly as stated at approval: both products are photometrically
normalised to the *same* standard geometry, so the brightness difference between
them is largely removed while the topographic shading and cast shadows from the
actual morning/evening acquisitions remain. That isolates the **geometric** part
of the illumination problem from the **radiometric** part — a separation nothing
else in the data inventory provides.

Not yet fetched (288 MB per tile); it is the next acquisition.

---

## 8. T2 feasibility (SPICE / ISIS / ASP) — infeasible, and not for the expected reason

Reported as a finding, as instructed. **Three independent blockers**, any one of
which is sufficient **[measured]**:

1. **NAIF publishes no Chandrayaan-2 kernels at all.** The full mission list at
   `naif.jpl.nasa.gov/pub/naif/` contains `LRO` and `SELENE`, and no
   Chandrayaan-1 or Chandrayaan-2. ISRO does not deposit CH-2 SPICE kernels
   there. This is the decisive one: no kernels, no rigorous geometry, regardless
   of software.
2. **No `usgscsm` wheel for CPython 3.14.** `spiceypy`, `ale`, `pysis`,
   `kalasiris` and `knoten` all resolve; `usgscsm` — the Community Sensor Model
   implementation that actually projects pixels through a camera model — does
   not.
3. **No ISIS and no way to get it here.** `isis3`, `spiceinit`, `conda`, `mamba`
   and `micromamba` are all absent; ISIS is a conda-forge distribution. Docker is
   present, so a container is possible in principle — but ISIS ships **no OHRC
   sensor model**, so it would not help for the source side even if installed.

**Conclusion: T2 is not reachable for Chandrayaan-2.** Proceeding on T1
(synthetic) + T3 (manual), as pre-authorised.

One qualification worth recording: T2 *is* reachable for the reference side —
LRO kernels, ISIS and ALE support all exist. But the reference is already
geodetically controlled to 0.8 px at 2σ **[archive]**, so there is nothing to
gain there. The gap is entirely on the Chandrayaan-2 side.

A fourth tier is available and better than T1 or T3 for this project: the LROC
controlled mosaics are themselves a LOLA-tied geodetic reference, so registering
against them **is** an external accuracy measurement, not a self-consistency one.
That is what §2 does.

---

## 8a. Addendum — TMC-2 arrived (2026-09-18)

Three TMC-2 products were downloaded from PRADAN after this report was first
written. Full write-up in `reports/TMC2_SELENE.md`; the parts that change
conclusions above:

- **TMC-2 ↔ SELENE TC registers at ~60 m** over six independent sub-windows
  (consensus (+7, +59) m, spread 82 m). PS pair type B is constructible.
- **The Phase 2 kilometre offset is specific to the unrefined polar OHRC
  products.** TMC-2 carries `reference_data_used = SELENE` and a populated
  refined-corner block, and lands 65× closer. §2 must not be generalised to
  Chandrayaan-2 as a whole.
- **§3's dense-beats-sparse finding reproduces at a completely different site.**
  On TMC-2 ↔ SELENE, every sparse matcher fails in all three representations
  tried while dense gradient NCC locks with margins up to 0.37. The cause is
  local shadow reversal: the images are anti-correlated at −0.410, and
  high-passing to −0.002 does **not** rescue the sparse matchers.
- **A real bug was fixed**: the reader hardcoded `uint8`, and TMC-2 is 16-bit.

---

## 9. Open items

1. 20211228T2209 and 20251010T0942 do not lock at coarse scale. The first is the
   darkest strip and fails for a stated reason; the second is provisional and
   should be re-attempted against more illumination bins than the single matched
   one.
2. Whether the offset varies **along** a strip is untested. A pure translation is
   an assumption, and a pushbroom strip 101 000 lines long is exactly where it
   would break.
3. The RIFT descriptor needs the authors' reference implementation.
4. SELENE morning/evening tiles not yet fetched.
5. The fine stage depends on `/vsicurl` reads that intermittently fail
   (`TIFFReadEncodedStrip`); retries are now in place but the failure mode is
   recorded rather than assumed fixed.
