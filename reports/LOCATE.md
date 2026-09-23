# Locating one image inside another

Date: 2026-09-22. Code: `backend/seleno/tool/locate.py`, wired into
`seleno.tool.register` (see `docs/TOOL.md` §6d).

## 1. The problem

The registration stages refine a placement; they never searched for one. With no
map projection on the reference, the tool assumed both images start at the same
corner. Two measured consequences, both on real Chandrayaan-2 data:

| case | before |
|---|---|
| IIRS 2024-01-20 → OHRC 2024-11-15 13:26 (531.9 km apart) | refused as `insufficient_matches`, log claimed **100% overlap** |
| OHRC 2025-10-10 → IIRS-sampling strip holding it 6,400 rows down | `verification_failed`: only the top of the strip was ever compared |

## 2. Method

1. **Geometry prior** (both inputs carry a lattice or a CRS). The smaller
   footprint is projected into the larger image in a stereographic plane centred
   on it and fitted with a similarity. Footprints further apart than
   1.5 × max(expected geolocation error, 3 km) raise `no_overlap`.
2. **Image search.** The smaller footprint is area-averaged to the common (coarser)
   sampling and correlated over the larger image with masked normalised
   cross-correlation, on intensity and on gradient magnitude. All rotations in
   6° steps without a prior, ±12° in 3° steps around the prior's with one;
   searched at up to 4× reduction (template area kept ≥ 1,500 px), the winner
   refined at full level in 0.5° steps.
3. **Acceptance.** Raw score gaps are uninformative here: a mostly-shadowed OHRC
   frame scores ~0.87 against IIRS ground it is not on, because one bright ridge
   correlates with any ridge. Two scale-free statistics are used, both against
   the best peak at a *different place* (a peak counts as different only if it
   overlaps the winner's footprint by less than half along both of its axes):

   - **normalised margin** `(best − rival) / (1 − rival)` ≥ 0.30
   - **robust z** of the winner above all other places' peaks ≥ 4.0

   and the two cues (intensity, gradient) must agree on the place, unless one
   alone reaches margin ≥ 0.50 and z ≥ 6.0. When the template nearly fills the
   search area there are no other places and no z; then both cues must agree
   *and* both reach margin ≥ 0.50.
4. If nothing is accepted: with a geometry prior, the geometry placement is used;
   without one, the run fails as `insufficient_matches` and reports the search
   statistics.

Related change: when the reduction onto the working grid is ≥ 16×, the source is
area-averaged from its cached block-mean overview instead of point-sampled.

## 3. Calibration data

Real OHRC frames, reduced to 80 m by 333–364 px block means, pasted into a real
IIRS strip (2025-07-29, resampled to 80 m, 266 × 13,913 px) at a known place.
Negatives search the same strip with nothing pasted.

| case | intensity margin / z | gradient margin / z | cues agree | decision | position error |
|---|--:|--:|---|---|--:|
| A: OHRC 2025-10-10, pasted copy of itself | 0.96 / 7.6 | 0.42 / 5.5 | yes | **placed** | 0.2 px |
| B: OHRC 2024-11-15 15:25, strip holds **13:26** (Sun azimuth 67° different) | 0.56 / 5.0 | 0.03 / 2.5 | yes | **placed** | 5.9 px ¹ |
| C1: OHRC 15:25, not in strip | 0.12 / 3.2 | 0.01 / 2.6 | — | refused | — |
| C2: OHRC 13:26, not in strip | 0.08 / 2.8 | 0.02 / 2.8 | — | refused | — |
| C3: OHRC 2025-10-10, not in strip | 0.02 / 2.9 | 0.03 / 2.5 | — | refused | — |

¹ The expected position comes from the two products' own lattices, which disagree
with each other by ~538 m (≈ 7 px at 80 m), so 5.9 px is inside the truth's own
uncertainty.

**Caveat, stated plainly:** the thresholds rest on two positive and three negative
real cases, plus synthetic tests. They separate these cleanly (positives ≥ 0.42 /
5.0, negatives ≤ 0.12 / 3.2), but the sample is small. A false placement is still
checked downstream: registration against the wrong ground fails verification.

One synthetic negative (a 60 × 60 px frame of unrelated terrain) passed the
single-cue thresholds before the cue-agreement rule and the minimum search-level
area were added; both were added because of it, and it is now a test.

## 4. End-to-end results

| run | before | after |
|---|---|---|
| IIRS 2024 → OHRC 13:26 (disjoint) | `insufficient_matches`, "100% overlap" | **`no_overlap`**: "531.9 km apart at the closest", 3.2 s |
| OHRC 2025-10-10 → strip (A) | `verification_failed` | **warning**, 44 inliers (92%), RMSE 0.81 px; tie points median **0.27 px** (22 m) from truth |
| OHRC 15:25 → strip holding 13:26 (B) | `verification_failed` | **warning**, 44 inliers (77%), RMSE 1.47 px |
| Strip (A) → OHRC 2025-10-10 (reverse roles) | failed | **warning**, 687 inliers; median **0.80 px** (64 m) from truth |
| OHRC 15:25 → OHRC 13:26 (lattice pair) | 799 inliers, 21.9 m, 223 s | **823 inliers, 20.2 m, 55 s** |

The warnings are the tool's existing extrapolation check: these polar frames are
mostly shadow, so tie points cluster on the lit ground.

In the lattice pair the image search measured the products' own geometry to be
**660 m** out relative to each other, consistent with the 538 m disagreement
recorded for these two products earlier in the project.

## 5. Tests

`tests/test_tool.py`:

- `locate_finds_a_small_source_in_a_large_reference`: a 4× finer frame rotated 25°
  inside a 2.4 km strip; found at 25°, 130 inliers, median 0.18 px from truth.
- `locate_refuses_a_source_that_is_not_there`: unrelated ground fails as
  `insufficient_matches`, "could not be located".
- `locate_geometry_proves_disjoint_and_places_overlap`: lattices 314 km apart are
  disjoint; an overlapping frame is placed at the right scale and 30° rotation.

All 35 tests pass.

## 6. Limits

- No real overlapping IIRS/OHRC pair exists in the archive we hold; section 3
  uses real OHRC and real IIRS data combined synthetically.
- Lattice geometry is read through a south-polar stereographic interpolation,
  which degrades near the north pole.
- A similarity is fitted to the geometry prior; along a 22 km strip that misses
  by up to ~400 px at 0.24 m, which is why the image search, not the prior, sets
  the final placement whenever it is accepted.
- Area-averaged sampling treats a nodata value of 0 as black, which suits shadow
  but would darken genuine fill at the edge of a product.
