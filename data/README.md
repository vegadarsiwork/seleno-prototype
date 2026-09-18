# Data provenance

Two distinct data sources, kept separate because they support very different
claims.

1. **`datasettesting/dataset/`** — the primary dataset: four real Chandrayaan-2
   OHRC Level-2 products. Everything the current architecture is measured on.
   **No ground truth of any kind.**
2. **`data/pairs/`** — six legacy demo pairs from the earlier prototype, five of
   which carry a *synthetic* ground-truth transform. Retained as a regression
   fixture so the earlier numbers stay reproducible. They are not evidence about
   real repeat-pass registration.

No file in either source is ever modified. Every `.img` is opened
`numpy.memmap(mode="r")`; `tests/test_ohrc_dataset.py` asserts the memmap is
read-only, that writes raise, and that mtime, size and a content hash are
unchanged after a full read pass.

---

## 1. Primary dataset — Chandrayaan-2 OHRC Level-2

Four PDS4 products of south-polar terrain, produced by ISRO/ISSDC.

```
datasettesting/dataset/
├── data/calibrated/{20211228,20241115,20251010}/     *.img  (the pixels) + *.xml
├── browse/calibrated/…/                              *.png  10x decimated previews
├── geometry/calibrated/…/                            *.csv  (Pixel,Scan)->(lon,lat)
├── miscellaneous/calibrated/…/                        *.oat .oath .lbr .spm
├── miscellaneous/readme.txt                          ancillary format spec
├── maindataset.tar + 4x *.zip                        the original archives (redundant)
└── DATASET_REPORT.md                                 an inventory report shipped with it
```

### Facts read from the PDS4 labels, not from documentation

| | 20211228T2209123959 | 20241115T1326321339 | 20241115T1525004388 | 20251010T0942085687 |
|---|---|---|---|---|
| Logical id | `urn:isro:isda:ch2_cho.ohr:data_calibrated:…` | idem | idem | idem |
| Lines × samples | 79 796 × 12 000 | 101 074 × 12 000 | 101 074 × 12 000 | 101 075 × 12 000 |
| Bytes | 957 552 000 | 1 212 888 000 | 1 212 888 000 | 1 212 900 000 |
| Sample type | `UnsignedByte`, `offset=0`, "Last Index Fastest" | idem | idem | idem |
| `isda:pixel_resolution` | 0.28 m/px | 0.24 m/px | 0.24 m/px | 0.22 m/px |
| Acquired (UTC) | 2021-12-28 22:09:12.396 | 2024-11-15 13:26:32.134 | 2024-11-15 15:25:00.439 | 2025-10-10 09:42:08.569 |
| Dwell | 16.384 s | 16.383 s | 16.383 s | 16.384 s |
| Imaging orbit | 10445 | 23328 | 23329 | 27338 |
| S/C altitude | 111.97 km | 93.67 km | 93.46 km | 85.71 km |
| TDI stages | TDI128 | TDI64 | TDI64 | TDI64 |
| Roll / pitch / yaw | −7.862 / 18.422 / 0.003 | 14.567 / 11.610 / 0.013 | 15.185 / −11.076 / 0.026 | 26.671 / 14.148 / 0.024 |
| Sun azimuth (label) | 152.400° | 180.300° | 242.981° | 149.774° |
| Sun elevation (label) | −0.044° | −0.170° | +0.786° | −0.771° |
| Solar incidence | 90.044° | 90.170° | 89.214° | 90.771° |
| Projection / area | Polar stereographic / South Pole | idem | idem | idem |
| `reference_data_used` | System | System | System | System |
| Corners genuinely refined? | **No** | **No** | **No** | **No** |

Declared array size equals bytes on disk exactly for all four products, and the
label `file_size` agrees — verified by
`test_declared_size_equals_lines_times_samples`.

### Radiometry, measured

Independently reproduced by `scripts/inspect_dataset.py`:

| product | mean DN | median | ≤ DN 10 | usable 512 px tiles |
|---|--:|--:|--:|--:|
| 20211228T2209 | 27.47 | 2 | 80.2 % | 22 % |
| 20241115T1326 | 26.01 | 4 | 71.4 % | 31 % |
| 20241115T1525 | 31.22 | 3 | 68.6 % | 33 % |
| 20251010T0942 | 40.28 | 7 | 57.6 % | 42 % |

Cross-track column means swing by 30–45 DN across the 12 000-sample swath
(e.g. 10.3 → 53.1 DN); along-track row means span 0.4 → 165 DN within one strip.
**Normalise per tile, never per strip.** Tile normalisation in this repository is
used for visualisation only and never before matching.

### Footprints and overlap

Computed in a south polar stereographic plane at 100 m cells — **not** in
lon/lat, which is degenerate here (one footprint spans longitude 222° → 110° →
22° while covering 25 km of ground, because the strips pass within 0.05° of the
pole).

| pair | overlap | % of smaller |
|---|--:|--:|
| 20241115T1326 ↔ 20241115T1525 | 79.76 km² | **95.3 %** |
| 20211228 ↔ 20241115T1326 | 69.85 km² | 83.5 % |
| 20211228 ↔ 20241115T1525 | 67.06 km² | 79.7 % |
| 20241115T1525 ↔ 20251010 | 64.97 km² | 77.2 % |
| 20241115T1326 ↔ 20251010 | 62.11 km² | 74.2 % |
| 20211228 ↔ 20251010 | 53.27 km² | 57.6 % |

### The geometry grid

Each product ships `geometry/**/*_g_grd_*.csv`: `Longitude,Latitude,Pixel,Scan`
on a lattice of 121 cross-track nodes (0, 100, …, 11 900, **11 999** — the last
step is 99) by one node per 100 lines. ~22–28 m spacing on the ground.

Verified in tests: node (0, 0) reproduces the label's `upper_left_*` to six
decimals, the final node reproduces `lower_right_*`, and the lattice's last scan
node equals `lines − 1` **exactly**, which is what proves the grid indexes the
raster 1:1 with no crop or offset.

**It is a prior, not truth.** All four products report
`reference_data_used = System` and their `Refined_Corner_Coordinates` block is
byte-identical to `System_Level_Coordinates` — no photogrammetric refinement has
been applied. Measured consequence: the delivered geolocation of the two 2024
products disagrees with each other by **538 m** (≈2 244 pixels at 0.24 m/px).
See README §1.

### Ancillary Sun and orbit series

`.spm` carries one record per **40 ms** (not the 512 ms `readme.txt` claims) with
phase angle, Sun aspect, Sun azimuth, Sun elevation and Moon-centred J2000
spacecraft position/velocity. `.oat` adds attitude quaternions and sub-satellite
point; `.lbr` adds libration angles.

We interpolate this per image line rather than using the label's scalar, because
**Sun azimuth sweeps tens of degrees along a single 16 s strip**:

| product | azimuth range | sweep | elevation range |
|---|---|--:|---|
| 20211228T2209 | 131.3° → 164.8° | 33.4° | −0.56° → +0.47° |
| 20241115T1326 | 165.2° → 205.1° | 39.9° | −0.67° → +0.33° |
| 20241115T1525 | 203.1° → 275.9° | **72.8°** | +0.28° → +1.29° |
| 20251010T0942 | 135.5° → 169.0° | 33.4° | −0.87° → −0.68° |

The label scalar matches the mid-strip interpolated elevation to four decimals in
every product, which is how we confirmed it is a scene-centre sample.

### What does NOT exist in this archive

No crater catalogue. No boulder or hazard labels. No segmentation masks. No DEM,
slope or roughness raster. No illumination or permanently-shadowed-region mask.
No train/val/test split. **No ground truth of any kind.** Any supervised task
would require manual annotation or externally co-registered labels.

### Label and documentation defects

Eight found, all handled explicitly in code and surfaced in the UI rather than
silently patched. Full list in `ASSUMPTIONS.md` §9. The ones that would break a
naive loader:

* `isda:line_exposure_duration` is tagged `unit="ms"` but the value is
  **microseconds** (162.100 µs, not ms — 101 074 × 162.1 ms would be 4.5 hours
  against a 16.383 s acquisition).
* `20241115T1525` writes `lower_right_longitude` as `" 4.686608"` with a
  **leading space**; a regex anchored on the digits silently drops that corner.
* `.spm` timestamps must be read at **fixed byte offsets** — the I4
  block-length field abuts the 4-digit year, so whitespace splitting yields a
  merged token (`2492024`) and shifts the whole time field. The trailing float
  columns, conversely, do not match the declared widths and must be split on
  whitespace.
* Browse labels declare the wrong dimensions (4 211 × 500 against an actual
  10 107 × 1 200). We do not rely on browse labels.

**One stale claim in the shipped report.** `DATASET_REPORT.md` §7.5 states that
`20241115T1326`'s browse PNG was not extracted. It **is present** on disk
(4 918 131 bytes). Trust the filesystem.

### Redundant archives

`maindataset.tar` and the four `.zip` files together account for ~3.67 GB and are
fully redundant with the extracted tree. Nothing in this repository reads them,
and nothing deletes them.

---

## 2. Legacy demo pairs (regression fixture)

Six pairs built by `scripts/prepare_data.py` from three OHRC products obtained
from the public Internet Archive mirror of an earlier ISRO OHRC release
(<https://archive.org/details/chandrayaan-2-high-resolution-images-of-the-moon>),
which needs no credentials. Those products are *different* from the four above:
orbits 2297/2298, acquired 2020-02-29, at 0.22977 and 0.23022 m/px.

| Pair | Real | Synthetic | Ground truth |
|---|---|---|---|
| `ohrc_crater_field` | OHRC full-res crater field | 11° rotation, 1.12× scale, small perspective, gamma 1.28 | exact, by construction |
| `ohrc_raw_vs_calibrated` | **Both sides real**: calibrated vs raw product of the same acquisition | none | exact translation |
| `ohrc_tmc2_moderate` | OHRC browse, 2.30 m/px | resampled to 5.0 m/px + 6° rotation | exact |
| `ohrc_tmc2_extreme` | OHRC full-res, 0.2298 m/px | resampled to 5.0 m/px (21.8×) + 4° rotation | exact |
| `ohrc_low_illumination` | genuinely shadowed polar terrain | 9° rotation, 1.08× scale, gamma 1.6 + noise | exact |
| `ohrc_disjoint_orbits` | **Both sides real**, orbits 2297 and 2298 | none | **none** — tile centres 10.95 km apart |

**Why they are kept and what they cannot show.** Five of the six match real
imagery against a *transformed copy of itself*: same regolith texture, same noise
realisation, same shadows modulo a gamma change. Sub-pixel results there
demonstrate the implementation is arithmetically correct — they say nothing about
registering two independent acquisitions, and the real OHRC pairs are two orders
of magnitude worse. The `tmc2` pairs resample OHRC to TMC-2's *ground sampling
distance* only; TMC-2's optics, MTF, noise and stereo geometry are absent, so
those rows are evidence about scale-ratio robustness and nothing else.

They remain useful for exactly two things: a regression check that the pipeline
still produces the numbers it used to, and a controlled case where ground-truth
error genuinely exists so the distinction between it and reprojection RMSE can be
demonstrated.

---

## 3. Reproducing

The primary dataset is expected at `../datasettesting/dataset`; set
`SELENO_OHRC_ROOT` to point elsewhere. Nothing needs downloading for it — it is
already on disk and is never written to.

```bash
python tests/test_ohrc_dataset.py            # 28 read-only checks
python scripts/inspect_dataset.py            # statistics and figures
```

The legacy pairs are committed. To rebuild them from the Internet Archive mirror:

```bash
pip install remotezip
python scripts/fetch_products.py --out data/raw     # ~1 GB streamed
python scripts/prepare_data.py --raw data/raw
```

---

## 4. Licence and attribution

Chandrayaan-2 data is produced by the **Indian Space Research Organisation**.
Use is governed by ISRO/ISSDC's terms — see
<https://pradan.issdc.gov.in/ch2/ack.xhtml>. Users are asked to acknowledge
ISRO. This prototype is a non-commercial academic demonstration for the Smart
India Hackathon (SIH26166) and claims no rights over the imagery.

---

## 3. `data/lroc/` — LROC real-illumination pairs (ground truth by control network)

Built by `scripts/fetch_lroc_pairs.py`; evaluated by `scripts/evaluate_lroc.py`
→ `results/lroc/LROC.md`.

- **Source:** LROC `NAC_POLE_SOUTH_CM_XXX` controlled south-polar mosaics, tile
  `P892S2250` (1 m/px, polar stereographic, 1 737 400 m radius), one mosaic per
  10° bin of sub-solar longitude. Public NASA PDS, no credentials; read by HTTP
  range requests. Citation: Archinal et al. (2023), LPSC 54, #2333.
- **Why the truth is real:** every bin comes from one ISIS jigsaw network
  (18 323 NAC images, LOLA-tied, 0.8 px at 2σ) on one grid, so the same pixel
  in two bins is the same ground. Checked: gradient phase correlation between
  unwarped bins gives shifts ≤ 0.4 px.
- **Real:** all pixels, and the illumination change (Δ sub-solar longitude 50°,
  90°, 180°; different NAC images, different cast shadows).
- **Synthetic:** a known similarity + mild perspective warp on the reference
  window (±20°, ×0.9–1.1, ±60 px), stored as `gt_homography`.
- **Negatives:** two pairs of different ground (`*_neg`); the right answer is a
  refusal.
- Per-window percentile stretch to 8 bit; windows with > 2 % nodata dropped.
