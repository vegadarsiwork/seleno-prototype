# TMC-2 against SELENE TC — the first cross-mission pair

Date: 2026-09-18. Reproduce with `python scripts/test_tmc2_selene.py`.

Three TMC-2 products were downloaded from PRADAN this session (see
`docs/DATA_INVENTORY.md` §2.1a). This is the first registration in the project
that is genuinely cross-mission, cross-sensor, non-polar, and at a workable scale
ratio.

| | |
|---|---|
| source | ISRO Chandrayaan-2 TMC-2, `ch2_tmc_ncn_20260813T0627378557`, **5.48 m/px**, 16-bit |
| reference | JAXA SELENE TC map v4.0, `TCO_MAP{m,e}04_S15E141S18E144SC`, **7.403 m/px** |
| scale ratio | **1.35×** (against 31× for OHRC ↔ TC) |
| site | lon 141.6–141.9 E, lat −16.6 … −16.3 |
| TMC-2 solar incidence | **39.06°** — not the 90° the polar archive is pinned at |
| common grid | plate carrée at SELENE's own sampling, both north-up |

---

## 1. The headline: TMC-2 geolocation agrees with SELENE to ~60 m

Dense masked normalised cross-correlation on gradient magnitude, run over **six
independent sub-windows** spread across ~36 km of the strip. A single box could
be a fluke; six that agree cannot be. **[measured]**

| box (lon0, lat0) | TMC-2 coverage | dx (m) | dy (m) | \|d\| (m) | margin |
|---|--:|--:|--:|--:|--:|
| (141.62, −16.55) | 100% | −67 | +59 | 89 | 0.021 |
| (141.85, −16.55) | 100% | +30 | +59 | 66 | **0.321** |
| (141.62, −16.25) | 100% | −30 | +59 | 66 | 0.048 |
| (141.85, −16.25) | 100% | 0 | +59 | 59 | **0.367** |
| (141.70, −15.90) | 100% | +22 | +52 | 56 | **0.282** |
| (141.95, −17.10) | 100% | +89 | +67 | 111 | **0.217** |

**Consensus (+7, +59) m, \|d\| = 60 m, spread 82 m across six boxes.**

The along-track component is the striking part: **+59, +59, +59, +59, +52, +67 m**
— a systematic bias, not scatter. The cross-track component scatters ±80 m, which
is what a 22 km-wide swath with no strong cross-track constraint looks like.

### Why this number matters

| product family | geolocation residual | what the label says |
|---|--:|---|
| Polar OHRC (Phase 2) | **3 900 – 4 600 m** | `corners_refined = False` |
| TMC-2 (here) | **~60 m** | `reference_data_used = SELENE`, refined corners populated |

A factor of **65**. This is strong evidence that the Phase 2 offset is a property
of *those unrefined polar products*, not of Chandrayaan-2 geolocation in general,
and it vindicates the decision to report that offset per-strip rather than as a
mission-wide claim.

**Caveat, and it is a real one:** SELENE TC is the reference ISRO themselves used
to refine this product's corners. So 60 m is **agreement, not independent
accuracy** — it measures how well ISRO's refinement reproduces on our side, and
cannot be quoted as an absolute geolocation error. An independent check would
need LOLA or a controlled NAC product over the same ground, and §4 explains why
the latter does not exist.

---

## 2. Every sparse feature matcher fails on this pair — and why

Same box, same grid, TMC-2 covering 100% of it: **[measured]**

| representation | sift | akaze | orb | disk_lightglue |
|---|---|---|---|---|
| as-is | 5 matches, no consensus | 18, no consensus | 15, no consensus | insufficient |
| TMC-2 contrast-inverted | 10, no consensus | 16, no consensus | 12, no consensus | insufficient |
| high-pass both (σ=12 px) | 13, no consensus | 15, no consensus | 13, no consensus | insufficient |

Meanwhile dense gradient NCC locks at 60 m with margins up to 0.37 on the same
pixels.

### The diagnosis

The two images are **anti-correlated**: −0.410 over the common valid area.
TMC-2 was acquired with the Sun at azimuth 69.6°; the SELENE *evening* map is the
opposite terminator. Slopes lit in one are shadowed in the other.

`reports/gt_qc/phase2_tmc2_selene2.png` shows it plainly: a small bright crater
with a linear ejecta ray sits in the same place in both frames — the ground is
unambiguously the same and unambiguously aligned — while the broad brightness
field is reversed.

The three rows above rule out the easy explanation. If this were a *global*
contrast inversion, inverting one image or high-passing both would fix it.
High-passing takes the correlation from −0.410 to −0.002, i.e. it removes the
broad shading field entirely, and **the matchers still fail**. So the reversal is
local: each crater's own lit and shadowed rims have swapped, so every descriptor
patch is individually inconsistent.

That is why a dense method survives and a sparse one does not:

* gradient **magnitude** is polarity-blind, and averaging it over 1.5 M pixels
  beats the per-patch inconsistency;
* a keypoint descriptor has only its own patch, and that patch is reversed.

This is the same ordering found at the pole in Phase 2, where masked NCC was the
estimator that locked and the feature voters were the ones that wandered. Two
completely different sites, sensors and latitudes; the same conclusion.

---

## 3. What this does and does not establish

**Does:**

- PS pair type B (TMC-2 ↔ SELENE TC) is constructible and registers.
- TMC-2's delivered geolocation is good to ~60 m against its own reference.
- Registration across a real ISRO→JAXA sensor gap at a 1.35× scale ratio works
  with a dense estimator.
- The Phase 2 kilometre-scale offset is specific to the unrefined polar products.

**Does not:**

- Claim independent accuracy — SELENE is ISRO's own reference here.
- Claim a sparse matcher can do this. None tested could, in any of three
  representations.
- Say anything about the SELENE **morning** map at this site: it carries no data
  over this box (constant 0.010, std 0.0000 — 23% tile coverage against the
  evening map's 74%). Those "failures" in the first run were empty input, not
  matching failures, and are excluded rather than counted.

---

## 4. Loose ends

1. **Morning vs evening at one site** was the point of keeping this experiment
   (decision 7) and is still not done, because morning coverage is sparse here. A
   site where both maps carry data is needed; the tile index would find one.
2. **TMC-2 ↔ NAC is unconstructible.** Zero of 21 551 map-projected NAC RDR
   products overlap any of the three TMC-2 footprints.
3. **The 60 m residual is not explained.** It could be ISRO's refinement residual,
   our lattice resampling, or internal SELENE error. Separating those needs a
   third reference.
4. **Only one of three TMC-2 strips was tested.** The other two are at
   lon 187–189 E and 138.8–140.5 E and have not been checked for SELENE coverage.
