# Phase 7 — the tool, and what it produces on real data

Updated 2026-09-19. Every number below is read out of a `metrics.json` that a
run actually wrote; the job directories are named so they can be re-opened.

The deliverable asked for: *"software and registered product with corresponding
match points"*, *"evaluation metric (RMSE, inlier match count, inlier ratio)"*,
*"sub-pixel accuracy"*, *"maintaining uniform distribution across the images"*.
See `docs/TOOL.md` for the tool itself and `reports/RESOLUTION_RANKING.md` for
the matcher study.

> **On the commit history.** Commits `a32050e` and `bbed302` appeared without
> anyone running `git commit`; something in the working environment commits
> automatically. The history is therefore not fully hand-made, and commit
> boundaries should not be read as deliberate checkpoints.

---

## 1. Accuracy is now measured at native resolution

Earlier Phase 7 numbers were quoted in **working-grid pixels**, and the working
grid is far coarser than either input. That made the sub-pixel flag meaningless:

| pair | working grid | old RMSE | in source pixels |
|---|--:|--:|--:|
| TMC-2 -> SELENE | 44.42 m/px | 1.3365 px = 59.4 m | **10.8 source px** |
| OHRC -> NAC | 4.00 m/px | 1.568 px = 6.27 m | **26.1 source px** |

(IIRS had not been delivered when those numbers were taken; its run is §4.)

The tool now runs a **fine stage**: after the coarse solve it re-places the
source over native-resolution windows positioned on the surviving tie points,
re-correlates inside each, and refits from the refined points. RMSE is reported
in four units everywhere, and the sub-pixel flag is set on **source** pixels,
which is what the problem statement asks for.

| pair | old (working grid) | new (native fine stage) | improvement |
|---|--:|--:|--:|
| TMC-2 -> SELENE | 59.4 m / 10.8 src px | **11.54 m / 2.11 src px** | 5.1x |
| OHRC -> NAC | 6.27 m / 26.1 src px | **1.45 m / 6.05 src px** | 4.3x |

### Sub-source-pixel is not reachable on any of the three pairs

In every case the **reference is coarser than the source**, so one reference
pixel spans more than one source pixel, and nothing can localise a source pixel
against a reference to better than roughly one reference pixel:

| pair | source | reference | 1 reference px = | achieved |
|---|--:|--:|--:|--:|
| OHRC -> NAC | 0.24 m | 1.00 m | 4.167 source px | 6.05 source px |
| TMC-2 -> SELENE | 5.48 m | 7.40 m | 1.351 source px | 2.11 source px |
| IIRS -> WAC (2024) | 56.73 m | 100 m | 1.763 source px | 3.78 source px |
| IIRS -> WAC (2025) | 85.18 m | 100 m | **1.174 source px** | 2.20 source px |

`metrics.json` reports this as `subpixel_floor_source_px` with
`subpixel_attainable: false`, and states it in `notes` rather than counting it
against the run's status — a limit of the reference is not a defect in the
registration. **Measured against the reference**, where the comparison is fair,
the three runs sit at **1.45, 1.56 and 1.87 reference pixels**, i.e. within a
factor of two of the floor in every case.

Reaching sub-source-pixel needs a reference at or finer than the source: NAC at
its own ~0.5 m native resolution rather than a 1 m mosaic, or same-sensor pairs.
No such pairing exists in the data delivered to this project.

---

## 2. Run 1 — OHRC → LROC NAC

Chandrayaan-2 OHRC at 0.24 m/px onto a controlled NAC polar mosaic at 1 m/px.
A 4:1 scale ratio, contrast inversion, and a source whose own label says its
geolocation is unrefined ephemeris.

```
python -m seleno register \
  --source data/raw/ch2/ohrc/data/calibrated/20241115/ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml \
  --reference data/raw/nac/NAC_CM355_P892S2250_pole_6km.tif \
  --out outputs --max-side 2048 --grid 12
```

```
job            outputs/e06715134e70          runtime 19.7 s
frame          source geometry lattice -> reference projected plane
pair           xcorr -0.5733, anti_correlated, scale_ratio 0.24, overlap 0.364
coarse         working grid 1574 x 1574 at 4.00 m/px
                 dense-ncc 2/6   disk_lightglue 334/526   sift 325/444
                 akaze 442/697   orb 300/458          -> akaze
fine           8 windows tiling 3448 x 1788 reference px (4 x 2 grid)
  native probe akaze 5535/6744                        -> akaze
  result       23477 points -> 7764 verified inliers (33.1%)

HELD-OUT RMSE  (n = 2717)
  source px          6.0533
  reference px       1.4528
  metres             1.453
  working-grid px    0.3632
sub-pixel      False on source pixels; floor is 4.17 source px, so not attainable
distribution   coverage 0.162 of 68 eligible cells, extrapolated 88.2%
STATUS         WARNING
  matches cover 16% of the reference grid; 88% of the reference area lies
  outside the tie-point hull, so the transform is extrapolated there.
```

The warning is correct and unresolved: the OHRC strip crosses only part of the
NAC window and its matches concentrate where there is texture. The 1.45 m is
real for that region and is not claimed for the rest.

---

## 3. Run 2 — TMC-2 → SELENE TC

Chandrayaan-2 TMC-2 at 5.48 m/px onto a Kaguya Terrain Camera tile at 7.40 m/px,
**contrast-inverted** with respect to each other.

```
python -m seleno register \
  --source data/raw/ch2/tmc2/extracted/data/calibrated/20260813/ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml \
  --reference data/raw/selene/TCO_MAPe04_S15E141S18E144SC.lbl \
  --out outputs --max-side 2048 --grid 12 --segments 6
```

```
job            outputs/9656abc457e0          runtime 42.1 s
frame          source geometry lattice -> reference lon/lat
pair           xcorr -0.5045, anti_correlated, scale_ratio 0.7402, overlap 0.228
coarse         working grid 2048 x 2048 at 44.42 m/px
               plan reordered by the measured negative correlation:
               dense-ncc first -> 100 candidates, 70 inliers, stopped early
fine           12 windows tiling 2772 x 11268 reference px (2 x 6 grid)
  native probe dense-ncc 81/119                       -> dense-ncc
  result       1429 points -> 597 verified inliers (41.8%)

HELD-OUT RMSE  (n = 209)
  source px          2.105
  reference px       1.558
  metres             11.536
  working-grid px    0.2597
sub-pixel      False on source pixels; floor is 1.35 source px, so not attainable
distribution   coverage 0.648 of 39 eligible cells, extrapolated 46.2%
per-segment    6 bands, median 0.248 working px = 2.01 source px
STATUS         PASS
```

This is the pair the method-selection logic exists for. The tool measures
cross-correlation −0.50, flags the pair anti-correlated, reorders the plan to
put dense masked NCC first, and dense NCC locks — where every sparse matcher
fails at every resolution tried (see `reports/RESOLUTION_RANKING.md`).

---

## 4. Run 3 — IIRS → WAC

Chandrayaan-2 IIRS derived reflectance onto the LROC WAC global 100 m basemap.
IIRS is an imaging spectrometer, so the source is a **pseudo-pan** built from
the reflected-light bands; the reference is a **5.96 gigapixel global mosaic**.

Two strips were delivered and both were run. They do not have the same ground
sampling, which turns out to matter more than expected.

```
python -m seleno register \
  --source data/raw/ch2/iirs/extracted/ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.xml \
  --reference data/raw/wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif \
  --out outputs --max-side 6144 --grid 12 --segments 6
```

```
job            outputs/30a9d466deae          runtime 36.7 s
source         IIRS derived reflectance, 256 bands, 85.18 m/px (from the label)
               pseudo-pan from 47 of 256 bands, 0.81-1.59 um, centres from the label
               geometry from the _loc_ backplane
reference      WAC global mosaic 109164 x 54582 at 100 m/px, read lazily in windows
frame          source geometry lattice -> reference projected plane
               windowed to the source footprint before anything was read
pair           xcorr +0.1981, NOT anti-correlated, scale_ratio 0.8518, overlap 0.402
coarse         working grid 2146 x 111 at 500 m/px
               dense-ncc 17 candidates, 13 inliers -> dense-ncc
fine           4 windows tiling 171 x 7929 reference px (1 x 4 grid)
  native probe dense-ncc 54/93                        -> dense-ncc
  result       369 points -> 79 verified inliers (21.4%)

HELD-OUT RMSE  (n = 28)
  source px          2.199
  reference px       1.873
  metres             187.3
  working-grid px    0.3746
sub-pixel      False on source pixels
  floor        1 reference px = 1.174 source px -> not attainable
distribution   coverage 0.296 of 71 eligible cells, extrapolated 74.7%
STATUS         WARNING
  75% of the reference area lies outside the tie-point hull.
```

The second strip, `20240120T1432235872` (`outputs/659496c27a32`), was acquired
from a lower orbit and is therefore **finer**: 56.73 m/px against the other's
85.18. It registers with more inliers (141 of 304, 46.4%) and better coverage
(0.415) but a worse residual — 3.78 source px, 2.15 reference px, 214.6 m —
and `xcorr -0.0376`, essentially uncorrelated.

### This pair comes closest to sub-source-pixel, and still does not reach it

The reference is coarser than the source in all three pairs, but by the least
here:

| pair | source | reference | 1 reference px = | achieved |
|---|--:|--:|--:|--:|
| OHRC -> NAC | 0.24 m | 1.00 m | **4.167** source px | 6.05 source px |
| TMC-2 -> SELENE | 5.48 m | 7.40 m | **1.351** source px | 2.11 source px |
| IIRS -> WAC (2024) | 56.73 m | 100 m | **1.763** source px | 3.78 source px |
| **IIRS -> WAC (2025)** | **85.18 m** | 100 m | **1.174** source px | **2.20 source px** |

`subpixel_floor_source_px` for the 2025 strip is **1.174** — the closest any
pair in this project gets to a reference fine enough to support a
sub-source-pixel claim, and still above 1. The result is 2.20 source pixels,
so the claim is not available even here, and the tool says so in `notes`
rather than in `reason`.

Worth stating plainly: **the floor is a property of the product, not the
instrument.** IIRS's quoted ~80 m is nominal for the 100 km design orbit. These
two strips were flown at different altitudes, so the same instrument delivers
56.73 m/px in one product and 85.18 m/px in the other, and the floor moves from
1.763 to 1.174 with it. The tool reads the resolution from each label rather
than from the sensor profile, which is why the two runs report different floors
instead of one nominal figure.

### What the real product exposed that the synthetic test did not

The hyperspectral path was written against the PDS4 spec and tested on a
synthetic cube. Three things were wrong, and only real data showed them:

- **The data file is a `.qub`, named in the label.** The reader looked for an
  `.img` beside the `.xml`. It now uses the label's own `file_name`.
- **Band centres live in `<center_wavelength unit="nm">` inside `Band_Bin`, in
  nanometres.** The tag was in the reader's candidate list and the unit
  conversion was right, so this one held — **real band centres were found, 256
  of them, 0.7123 to 5.0097 um. There was no fall back to the middle-third
  guess.** 47 bands fell in the 0.81-1.59 um window.
- **Geometry arrives as an ENVI `_loc_` backplane, not a CSV.** IIRS ships a
  band-sequential cube whose first two bands are Longitude and Latitude at the
  full 250 x 12945 detector grid. `GeometryGrid.from_envi_loc` reads it,
  decimated on the way in.

Scaling and units were confirmed from the label rather than assumed, as the
processing level is **Derived** rather than Calibrated: `data_type` is
`IEEE754LSBSingle`, there is **no `scaling_factor` or `value_offset`**, and the
values are physical reflectance — median 0.09 to 0.15 across the
reflected-light bands, which is plausible lunar albedo. `0.0` is fill. The
long-wavelength end is exactly why the window exists: band 255 at 5010 nm is
50.3% zeros with a maximum of 1123.

A fourth defect was in the reference, not the source: **the WAC mosaic could
not be opened at all.** `_try_gdal` did `ds.read(1).astype(np.float32)`, which
for 109164 x 54582 asks for 23.8 GB, raises, and is swallowed by the reader's
`except Exception: return None` — the tool simply reported that nothing could
read the file. Rasters over 512 MB now go through a `LazyRaster` that reads in
windows, and a source footprint is projected into the reference before anything
is read, so a 734 km strip does not land three pixels wide on a global basemap.

## 5. The geolocation error, shown as a functional failure

Phase 2 measured Chandrayaan-2's polar geolocation error and reported it as a
distance: roughly **4.5 km** for the OHRC strips, confirmed by three independent
algorithm families. That is a number on a page. It is easy to read it as a
correctable offset — something a shift would absorb.

The fine stage turned it into a **functional** result, because the stage does
not work at all until the error is removed.

The fine stage re-places the source over native-resolution windows positioned on
tie points the coarse solve already found. In the first version it placed those
windows using the source's **own geometry**. Same window, same resolution, same
matchers, one difference — whether the source was sampled through the coarse
transform:

| placement | dense-ncc | DISK+LightGlue | SIFT | AKAZE | ORB |
|---|--:|--:|--:|--:|--:|
| from raw geometry | 3 / 5 | **0 / 1** | **0 / 1** | 3 / 13 | 3 / 11 |
| through the coarse transform | 12 / 28 | 537 / 1076 | 803 / 1751 | **3223 / 8100** | 607 / 1468 |

*(inliers / candidates, one 2048² native window on the OHRC/NAC pair, from
`reports/RESOLUTION_RANKING.md`)*

Read the top row again. DISK+LightGlue found **one candidate** in a four
megapixel window and verified **none** of it. SIFT found one. This is not
degraded matching, or a shifted answer, or a larger residual. It is **nothing**:
the window contains a patch of ground from several kilometres away, the two
images have no common feature in them at all, and every matcher correctly
returns that there is no correspondence to find.

That is what a 4 km geolocation error means operationally at native resolution.
At 4 m/px the coarse stage tolerates it, because a 4 km error is 1000 coarse
pixels and the search still overlaps the truth somewhere. At 1 m/px, inside a
2 km window, the truth is simply not in the frame. **Fine matching against
unrefined Chandrayaan-2 geometry does not return a worse answer — it returns no
answer**, and it will do so silently unless something has already solved the
coarse alignment and is used to place the window.

Two consequences worth carrying forward:

1. **A coarse solve is not an optimisation, it is a precondition.** Any pipeline
   that goes straight to full resolution on these products will find nothing and
   have no way to tell that from a genuinely unmatchable pair.
2. **The reported geolocation error is a lower bound on what must be corrected
   before fine work.** 4 km is the offset; the window half-width is what decides
   whether fine matching is possible, and here that is about 1 km.

---

## 6. Defects this phase found and fixed

- **`no_overlap` on images that plainly overlap.** OHRC's geometry lattice spans
  the full 0–360° longitude range; the NAC mosaic's projected grid converts back
  to −180..180. The lattice triangulation carried a seam at the antimeridian and
  the OHRC/NAC pair landed on it — source coverage came out at **0.0%**.
  Interpolating in the reference's own projected plane removes the seam and is
  better conditioned at the pole, where meridians converge into degenerate
  triangles. Coverage went **0.0% → 51.5%**.
- **The fine stage must re-select its method.** The coarse winner has no claim
  on the native grid. On OHRC/NAC the ranking changes between 4 m/px and 1 m/px;
  the stage now probes the plan once at native resolution and records what won.
- **Reference decimation was aliasing.** `arr[r0:r1:step]` keeps one pixel in
  `step`. It is now a boxcar average, which also fixes a half-block offset
  against `_grid_xy`, which had always addressed block centres.
- **The anti-aliasing estimator was 16x wrong.** The source reduction factor was
  measured by differencing every 16th row without dividing the stride back out,
  which silently kept the filter switched off. Source averaging is now applied
  only when the affordable taps actually cover the footprint — a 4-tap comb
  across a 16.7 px footprint is not a box filter, and measurably cost AKAZE
  442 inliers → 70 on the coarse pass.
- **A real NAC mosaic silently fell back to `_default`.** The instrument
  patterns used `\bnac\b`; `_` is a word character, so it never matched
  `NAC_CM355_...`, which is exactly how these products are named.

---

## 7. Reproducibility

The working grid is capped against free memory before anything is read, so the
same inputs on a loaded machine can land on a coarser grid. Both runs above pin
`--max-side` and record whether the cap fired in `metrics.json:degraded`.

## 8. Tests

`python tests/test_tool.py` — **26 passed, 0 failed**, run in CI by
`.github/workflows/tests.yml`. Four cover this phase's accuracy work directly: RMSE
agrees across all four units; the sub-pixel flag is on source pixels; an
unreachable floor is a note and not a status downgrade; and the fine stage
measurably beats the working grid (**2.670 → 1.234 reference px** on a
synthetic pair forced onto a 4x-decimated grid). Seven more cover the hyperspectral and
large-raster paths described in §4.

## 9. Matcher ranking

`reports/RESOLUTION_RANKING.md`, regenerated by
`python scripts/resolution_ranking.py`. Phase 2 and Phase 7 were read as
contradicting each other on which matcher wins; they do not. Phase 2 ran at
32 m/px, and at a comparable resolution Phase 7 agrees with it exactly — at
25 m/px on the OHRC pair, DISK+LightGlue scores 113 inliers and AKAZE scores
**0**. The ranking inverts as resolution gets finer (AKAZE 3223 vs DISK 537 at
1 m/px), and pair polarity overrides resolution entirely: on the
contrast-inverted TMC-2 pair no sparse method reaches double figures at any
resolution while dense NCC locks at all of them. That is the justification for
running candidates and taking the geometrically verified winner rather than
choosing a matcher in advance.
