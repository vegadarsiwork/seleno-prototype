# P0 — OHRC against an external lunar reference

Status: **OHRC → LRO NAC registration works.** The previous "correlation red"
verdict was wrong, and wrong for reasons that had nothing to do with the
reference product.

Reproduce everything here with `scripts/exp_reference.py`. No new bytes were
downloaded for any of it — every region was cropped out of the 6144² CM_AVG
region already in `results/nac_cache/`.

---

## 1. The headline

Chandrayaan-2 OHRC **system-level** geolocation disagrees with the LOLA-tied
LRO NAC controlled basemap by **3.8–4.5 km**, with a further **0.4–0.9 km** of
product-to-product scatter on top of that common component.

| OHRC product | offset vs NAC (dx, dy) m | magnitude | SIFT↔AKAZE spread |
|---|---|---|---|
| 20251010T0942 | (−1668, −3420) | 3806 m | 90 m |
| 20241115T1525 | (−2103, −3385) | 3985 m | **6 m** |
| 20241115T1326 | (−2546, −3679) | 4474 m | **3 m** |
| 20211228T2209 | no lock | — | — (71.4 % of the window ≤ DN 10) |

Reference: `NAC_POLE_SOUTH_CM_AVG_P892S2250`, region col0 35062 row0 4491,
6144², coarse pass at 4 m/px, correlation search ±2000 m.

### Why this is believed

Four independent checks, each of which could have killed it:

1. **Applying the offset makes everything relock at zero.** Re-running
   `20251010T0942` with `--align -1668,-3420` at 2 m/px drops the residual from
   3806 m to **(−33, +92) m**, and matched features multiply: akaze-grad goes
   from 159 candidates / 102 votes to **1895 / 1192**.
2. **The offsets differ per product.** A bug in `stereo_to_nac` would give all
   products the *same* offset. They differ.
3. **The differences reproduce an independently measured number.** Differencing
   the products gives 531 m (1525↔1326), 436 m (1525↔0942), 915 m (1326↔0942).
   The CH-2 inter-product disagreement measured earlier from OHRC↔OHRC alone,
   with no NAC involved, was **498–896 m**. Two independent routes, same answer.
4. **Both georeferencing chains audited against their own labels.**
   - NAC PDS3 label confirms `SAMPLE_PROJECTION_OFFSET = 45487.5`,
     `LINE_PROJECTION_OFFSET = -0.5`, `MAP_SCALE = 1.0`, radius 1737.4 km,
     centre −90/0 — every constant the code assumes. Independently,
     `lonlat_to_south_stereo` at the tile's stated `MAXIMUM_LATITUDE = -88.5`
     gives rho = 45 486.6 m against a 45 488 px tile half-width: 1.4 m over 45 km.
   - OHRC PDS4 label corners vs the delivered geometry CSV agree to **0.0 m** on
     all four corners of all four products.

The common ~4 km component is consistent with ISRO's own declaration: every
product's label carries `corners_refined = False`, i.e. *no photogrammetric
refinement has been applied*. These are ephemeris-and-attitude geolocations.

---

## 2. What actually caused the earlier failure

Not the reference product. `CM_AVG` — the shadow-free average mosaic whose own
README says it *"does not accurately represent any realistic lighting
condition"* — is the very mosaic that produced every lock in the table above.
**The hypothesis recorded in `ROADMAP.md` §P0.1a is refuted.** Two real causes:

1. **Window selection.** Windows were chosen by distance from the pole, never by
   whether OHRC carried any signal there. A control run landed on a window that
   was **92.8 % shadow**; unsurprisingly nine methods returned garbage. There are
   **378** lit tiles (< 35 % dark, room for a ≥1600 m box) inside the same
   cached region.
2. **Search range.** The true offset is ~4 km. Earlier sweeps searched hundreds
   of metres. What looked like several template sizes "disagreeing by 1658 m"
   was several runs pinned against different search walls.

---

## 3. Intensity correlation is the wrong tool here — and it fails confidently

Same window, same grid, same everything; only the correspondence method varies.

| | correlation (NCC) | SIFT / AKAZE + translation vote |
|---|---|---|
| result at ±2000 m | (+465, +1915), score 0.61–0.72 | (−1668, −3420), 6 runs agree |
| verdict | **confident false lock** | correct |
| after alignment to ~98 m | still pinned at the ±300 m search edge, margins 0.004–0.009 | relocks at (−33, +92) m |

Correlation could not find the peak *even when the true residual was inside its
search range*. Feature descriptors survive the appearance gap between grazing-
incidence OHRC and a shadow-free average mosaic; raw intensity NCC does not.

**Caveat, stated so it is not forgotten:** the correlation template spans
nodata, which poisons NCC, while the feature path erodes 24 px off the validity
boundary first. Correlation was given gradient and CLAHE variants but not a
masked NCC. That comparison is not perfectly fair and should be redone before
correlation is deleted from the pipeline.

---

## 4. Two methodological bugs found and fixed

1. **`estimateAffinePartial2D` was fitting noise.** It has a 2-point minimal
   sample, so "2–3 inliers, rmse 0.00 px" is the noise floor, not a result — it
   produced scales of 0.157–2.68 and rotations to 157° on images that share a
   grid and must therefore be related by scale ≈ 1, rotation ≈ 0. Replaced with
   a **translation-only Hough vote** (1-point model) scored against the null of
   matches scattered uniformly over the span they occupy.
2. **"Three methods agree" was counting the same method three times.** corr-raw,
   corr-grad and corr-clahe are one algorithm on three renderings of the same
   pixels; they fail together and manufactured the false lock above. Cluster
   scoring now counts distinct **algorithm families** (corr / sift / akaze), and
   every cluster is printed rather than only the winner, so a false-lock cluster
   stays visible beside the true one.

---

## 5. What this does not yet show

- **n = 1 window per product.** All four runs sit inside one 6144 m box.
- **Residual is ~98 m**, measured at 2 m/px with no refinement. Nothing here
  supports any sub-pixel claim; P6 remains untouched.
- **20211228T2209 never locked.** Consistent with it being the darkest strip and
  the one that already failed 3/6 OHRC↔OHRC pairs — but it is a failure.
- **`CM_235` is still untested.** It is no longer on the critical path, since
  CM_AVG works, but illumination-matched reference remains the P2 question.
- **The offset is assumed to be a pure translation.** Whether it varies across a
  strip (a timing or attitude signature rather than a constant bias) is
  untested and is the obvious next experiment.

---

## 6. Unrelated finding: the `.spm` sun angles are not body-fixed solar geometry

While trying to pair each OHRC product to an illumination-matched NAC mosaic:

`20241115T1326` and `20241115T1525` are consecutive orbits ~2 h apart over
nearly the same ground, yet report solar elevations of **−0.455°** and
**+0.500°**. The sub-solar point moves ~1° in 2 h; a 0.955° swing is impossible.
Solving exactly for the sun vector from (lon, lat, azimuth, elevation) gives a
sub-solar longitude that varies by 67–149° *along a single strip*, which is also
impossible.

Two consequences:
- Sub-solar longitude **cannot currently be derived from CH-2 metadata**, so
  illumination-matched reference selection needs the direction measured from the
  pixels (shadow elongation) instead. A first attempt got only 1.11–1.27× axis
  contrast because it was fed lit tiles; it needs shadow-rich windows.
- Anything in the project that treats these as true solar incidence — including
  the "solar incidence 89–91°" figure — needs re-deriving. The fields are
  probably folded with the off-nadir viewing geometry, which would explain why
  `phase == 90 − elevation` holds despite roll angles of 8–27°.
