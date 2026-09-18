# Phase 1 — Data inventory

Date: 2026-09-18. Companion to `docs/AUDIT.md`.

Every number here was produced on this machine by a script in this repository,
or read out of bytes fetched in this session. Reproduce with:

```
python scripts/build_footprint_index.py     # -> data/index/footprints.gpkg
python scripts/discover_pairs.py            # -> data/index/pairs.csv
python scripts/verify_reference_data.py     # -> reports/gt_qc/phase1b_reference_terrain.png
```

Claims taken from a product's own README or label, rather than measured here,
are marked **[archive]**.

---

## 1. The headline, first

**1. The source data is not where the brief assumed.** All four Chandrayaan-2
OHRC products are on this machine and now extracted to `data/raw/ch2/ohrc`
(4.4 GB). The dataset layer went from 4 passing / 24 skipped tests to
**27 passing / 1 skipped** the moment they were mounted.

**2. Incidence angle is the wrong stratification variable for this archive.**
The problem statement asks for results binned by incidence-angle difference at
0–10° / 10–30° / >30°. Every Chandrayaan-2 product we hold sits between
**−89.19° and −90.00°** latitude. At that latitude the solar incidence angle is
**89.9°–90.4° no matter where the Sun is**, so the largest incidence difference
between any two of our products is **0.25°**. One hundred percent of candidate
pairs fall in the first bin, and the stratification carries no information.

The variable that does move is the **sub-solar longitude**, i.e. the Sun
*azimuth*. Across our four products it spans **1.1° to 118.4°**, and against the
LROC reference it spans the full **0–180°**. Cast-shadow direction — the thing
that actually breaks a descriptor — is governed by that, not by incidence.

Modelled incidence range on a sphere, sub-solar latitude 0:

| latitude | incidence span over all sub-solar longitudes |
|---|---|
| −89.9° | 89.90° – 90.10°  (0.20°) |
| −89.5° | 89.50° – 90.50°  (1.00°) |
| −89.0° | 89.00° – 91.00°  (2.00°) |
| −85.0° | 85.00° – 95.00°  (10.00°) |
| −80.0° | 80.00° – 100.00° (20.00°) |

**Recommendation:** keep `d_incidence_deg` as a reported column so the PS bins can
still be filled in, and make `d_sub_solar_lon_deg` the primary stratifier, binned
0–10 / 10–45 / 45–90 / >90°. This is written up for approval, not done unilaterally.

**3. The reference side is solved, and better than expected.** The LROC NAC
controlled south-polar mosaics exist in **36 separate sub-solar-longitude bins**
over the same controlled ground, and the one tile that carries the bulk of every
OHRC strip — `P892S2250` — **is present in all 36 of them**. That is a complete
10°-resolution illumination series, with geometry tied to LOLA at 0.8 px
(2σ) **[archive]**, available without credentials.

**4. The brief's suggested ROI does not match the data.** The brief names
approximately 20–21°E, −70 to −67° as a known-good starting ROI. That string
appears nowhere in this repository, and no Chandrayaan-2 product we hold is
within 19° of it. The proposed site is in §7.

---

## 2. Source products (Chandrayaan-2)

### 2.1 What exists

| | |
|---|---|
| OHRC | **4 products**, calibrated L2, 4.4 GB extracted |
| TMC-2 | **3 products**, calibrated, 1.3 GB — added 2026-09-18, see §2.1a |
| IIRS | **none** |

The four OHRC products arrived as an ISSDC/PRADAN SFTP bundle
(`~/Downloads/ch2_ohr_ncp_20211228T2209123959_d_img_d18_Bundle.tar`, 1.8 GB) and
were extracted this session. Each carries the full tree: PDS4 label, `.img`
raster, geometry CSV lattice, `.oat`/`.oath`/`.lbr`/`.spm` ancillary files and a
browse PNG.

**There is no public script in this repository that fetches these.**
`scripts/fetch_products.py` targets a *different*, older OHRC set (orbits
2297/2298 of 2020-02-29) from an Internet Archive mirror. Re-acquiring the four
products used here requires ISSDC/PRADAN credentials.

### 2.1a TMC-2 — acquired 2026-09-18 via PRADAN

Three calibrated TMC-2 products were downloaded from PRADAN with a
session-scoped script the user generated from their logged-in browser. **[measured]**

| product | orbit | GSD (m) | lines x samples | latitude | longitude | solar incidence |
|---|--:|--:|---|---|---|--:|
| `…20260809T1606017180…` | 31029 | 5.13 | 106 962 x 4 000 | −41.14 … −23.59 | 187.00 … 188.86 E | **45.68°** |
| `…20260813T0627378557…` | 31073 | 5.48 | 148 108 x 4 000 | −27.87 … −3.73 | 140.96 … 142.71 E | **39.06°** |
| `…20260813T1023298745…` | 31075 | 5.48 | 147 741 x 4 000 | −27.89 … −3.81 | 138.79 … 140.54 E | **39.22°** |

1.3 GB of zips; each unpacks to the same tree as an OHRC product (PDS4 label,
`.img`, geometry CSV, `.oat`/`.oath`/`.lbr`/`.spm`, browse PNG).

**This changes several Phase 1 conclusions:**

1. **Solar incidence is 39–46°, not 90°.** The polar archive pinned incidence at
   89.9–90.4° and made the problem statement's incidence stratification
   degenerate (§5.3). These products are at 3.7–41.1° south, where incidence is
   a real variable again.
2. **They are 16-bit** (`UnsignedLSB2`), where OHRC is 8-bit (`UnsignedByte`).
   The reader hardcoded `uint8`; reading TMC-2 through it would have returned an
   image of half the declared line count made of interleaved high and low bytes.
   It is now driven by the label's declared type, and `verify_raster_size`
   refuses any type not in an explicit table rather than guessing.
3. **The strips are 730 km long and 21.9 km wide.** A single global 3x3
   homography over that is not slightly wrong, it is the wrong model class. The
   piecewise-transform requirement stops being hypothetical.
4. **`reference_data_used = SELENE`, and the refined-corner block is populated**
   and differs from the system corners. Every polar OHRC product carries
   `corners_refined = False`. So the 3.9–4.6 km offset measured in Phase 2 is a
   statement about *those* products, and must not be generalised to TMC-2
   without measuring it separately.
5. **The label Sun geometry looks trustworthy here.** Roll/pitch/yaw are all
   under 0.03°, so the values are not confounded by off-nadir viewing the way
   `results/experiments/P0_FINDINGS.md` §6 found for the polar OHRC products, and
   `incidence == 90 − elevation` holds to 0.01°. The Phase 1 ephemeris model
   disagrees by 10–14° at these latitudes, which is outside its stated ±10°;
   **prefer the label here and treat the model as a cross-check, not a
   substitute.**

**Reference coverage over these footprints [measured]:**

| reference | available? |
|---|---|
| SELENE TC morning map v4.0 | **yes** — all sampled tiles HTTP 200 |
| SELENE TC evening map v4.0 | **yes** — all sampled tiles HTTP 200 |
| LROC WAC (100 m) | yes, 237 products overlap |
| map-projected LROC NAC | **none** — 0 of 21 551 NAC RDR products overlap |

So **PS pair type B (TMC-2 ↔ SELENE TC) is now constructible**: 5.48 m against
7.403 m is a scale ratio of **1.35×**, both map-projected, both public, with
morning *and* evening illumination over the same ground. This is the first
genuinely cross-mission, cross-sensor pairing available to the project at a
workable scale ratio, and it is what the "multi-modal" requirement needs.

TMC-2 ↔ NAC remains unconstructible for the same reason as §5: the LROC RDR
archive has no blanket map-projected NAC coverage.

### 2.2 Measured footprints and illumination

Footprints traced from the delivered `(Pixel, Scan) → (lon, lat)` lattice and
projected into the south polar stereographic plane.

| product | GSD (m) | latitude range | area (km²) | sub-solar lon | modelled incidence |
|---|--:|---|--:|--:|--:|
| `…20211228T2209123959…` | 0.280 | −89.998 … −89.253 | 92.05 | 115.57° | 90.10° |
| `…20241115T1326321339…` | 0.240 | −89.977 … −89.200 | 80.00 | 357.19° | 90.26° |
| `…20241115T1525004388…` | 0.240 | −89.967 … −89.193 | 80.49 | 358.31° | 90.27° |
| `…20251010T0942085687…` | 0.220 | −89.978 … −89.220 | 88.76 | 42.88° | 90.35° |

Sub-solar longitude is **modelled from the acquisition UTC**, not read from the
label — see §2.3. Accuracy is about ±10° once optical libration in longitude is
ignored; validated against four known lunar phase epochs:

| epoch | expected | modelled | error |
|---|--:|--:|--:|
| 2024-11-15 21:28 full moon | 0° | 1.73° | +1.73° |
| 2024-12-01 06:21 new moon | 180° | 178.06° | −1.94° |
| 2025-10-07 03:47 full moon | 0° | 358.40° | −1.60° |
| 2021-12-27 02:24 last quarter | 90° | 92.43° | +2.43° |

### 2.3 The label Sun angles are not usable, and now there is a workaround

`results/experiments/P0_FINDINGS.md` §6 established that the OHRC `.spm`/label Sun
angles are not body-fixed solar geometry: two consecutive orbits two hours apart
report solar elevations 0.955° apart, which is physically impossible, and solving
for the sub-solar point gives 67–149° of drift along a single strip. The repo's
conclusion was that *"sub-solar longitude cannot currently be derived from CH-2
metadata"*.

That is still true of the **metadata**. It is not true of the **product**: the
acquisition UTC in the label is sound, and a standard ephemeris turns it into a
sub-solar longitude good to ±10°. The two 2024 products, two hours apart, come
out **1.1°** apart — which is what physics requires and what the `.spm` values
contradicted.

`sun_geometry_trusted` is `False` for every OHRC row in the index, and the label
values are preserved in `label_sun_azimuth_deg` / `label_sun_elevation_deg` so the
defect stays visible rather than being quietly replaced.

### 2.4 Source-to-source pairs

All six combinations, overlaps computed in the stereographic plane:

| pair | Δ sub-solar lon | Δ incidence | overlap (km²) | % of smaller |
|---|--:|--:|--:|--:|
| 20241115T1326 ↔ 20241115T1525 | **1.1°** | 0.01° | 76.13 | 95.2% |
| 20241115T1525 ↔ 20251010T0942 | 44.6° | 0.08° | 61.53 | 76.4% |
| 20241115T1326 ↔ 20251010T0942 | 45.7° | 0.09° | 58.65 | 73.3% |
| 20211228T2209 ↔ 20251010T0942 | 72.7° | 0.25° | 50.06 | 56.4% |
| 20211228T2209 ↔ 20241115T1525 | 117.3° | 0.17° | 63.83 | 79.3% |
| 20211228T2209 ↔ 20241115T1326 | **118.4°** | 0.16° | 66.56 | 83.2% |

These overlap percentages reproduce the independently-computed table in
`data/README.md` (95.3 / 83.5 / 79.7 / 77.2) to within 0.8 points, which is a
check on the whole geometry chain, not a coincidence.

**This table also explains the repository's existing results.** The pair the old
pipeline handles best (95.2% overlap) is the pair with a **1.1°** illumination
change. The strip that never locks — 20211228T2209 — is the one **118°** away in
Sun azimuth. The archive's difficulty ordering is an illumination ordering.

---

## 3. Reference products

### 3.1 LROC NAC controlled south-polar mosaics — the primary reference

| | |
|---|---|
| Access | public, **no credentials** |
| Discovery | S3 REST listing of the PDS Imaging archive bucket |
| Bulk endpoint | `https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/` |
| Product sets | **39** — 36 single sub-solar-longitude bins (`CM_005` … `CM_355`), plus `CM_AVG`, `CM_MAX`, and one uncontrolled `NAC_POLE_SOUTH` |
| Products | **982** `.IMG` files over **76 distinct tiles** |
| Size if fully mirrored | **15.60 TB** as 32-bit `.IMG`, **3.90 TB** as the 8-bit GeoTIFF |
| Grid | polar stereographic, 1 m/px, centre 90°S / 0°E, R = 1 737 400 m — **identical to `backend/seleno/ohrc/geometry.py`** |
| Control | ISIS `jigsaw` over 18 323 NAC images, 1 677 811 points tied to LOLA; σ₀ 0.38, 0.8 px at 2σ **[archive]** |
| Tiling | four latitude bands radiating from the pole: 4 tiles in 88.5–90°, 8 in 87–88.5°, 12 in 85.5–87°, 16 in 84–85.5° **[archive]** |

**Use the GeoTIFF mirror, not the `.IMG`.** Under
`EXTRAS/BROWSE/NAC_POLE/<set>/<product>.TIF` there is a full-resolution 8-bit
GeoTIFF of every tile. Verified on `CM_235_P892S2250`:

```
driver       GTiff           size  45488 x 45488, 1 band, uint8
crs          PolarStereographic Moon, R=1737400, lat_0=-90, lon_0=0
transform    | 1.00, 0.00, -45488.00 | 0.00, -1.00, 0.00 |
blockshape   [(1, 45488)]    overviews  []
```

It is **4× smaller** than the 32-bit `.IMG` (2.0 GB vs 8.3 GB) and it carries the
CRS and transform, so `rasterio` reads it directly over `/vsicurl` with no PDS3
label arithmetic. The catch: it is **striped, not tiled**, and has no overviews,
so a 1024×1024 window costs 1024 full scanlines ≈ 47 MB and took 49.5 s over
`/vsicurl` here. That is still 4× better than the `.IMG` route the repository
currently uses.

**Band-1 availability — the only band our data touches:**

| tile | longitude quadrant | present in | size (IMG / TIF) |
|---|---|--:|---|
| `P892S2250` | 180–270°E | **39 of 39 sets** (all 36 illumination bins) | 8.28 / ~2.07 GB |
| `P892S1350` | 90–180°E | 36 of 39 | 8.28 / ~2.07 GB |
| `P892S3150` | 270–360°E | 33 of 39 | 8.28 / ~2.07 GB |
| `P892S0450` | 0–90°E | 30 of 39 | 8.28 / ~2.07 GB |

Band 1 in total: 138 products, 1 142 GB as `.IMG`, ~286 GB as GeoTIFF.

**The pole is the corner where all four band-1 tiles meet, and every OHRC strip
crosses it.** Measured extents in the stereographic plane:

| product | x range (m) | y range (m) | quadrants touched |
|---|---|---|--:|
| 20211228T2209 | −18 155 … +4 247 | −16 202 … +1 318 | all 4 |
| 20241115T1326 | −18 300 … +2 593 | −17 957 … +1 489 | all 4 |
| 20241115T1525 | −18 551 … +2 400 | −18 124 … +1 493 | all 4 |
| 20251010T0942 | −16 975 … +1 822 | −18 627 … +3 681 | all 4 |

The bulk of each strip is in the x<0, y<0 quadrant (`P892S2250`), but **a
reference mosaic for any of them must stitch across up to four tiles**. A naive
window near the pole runs off a tile into negative row indices, which `curl`
interprets as a *suffix* range and begins downloading the whole 8 GB file — a
trap already documented in `ROADMAP.md` §P0.1a and still live.

### 3.2 LROC WAC global mosaic — fetched and verified

| | |
|---|---|
| Source | `https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` |
| Local | `data/raw/wac/…` , **5 959 263 751 bytes**, sha256 `87159ead1888ae31…` |
| Raster | 109 164 × 54 582, uint8, 1 band |
| CRS | SimpleCylindrical MOON (equirectangular), R = 1 737 400 m, 100 m pixels |
| Structure | striped `(1, 109164)`, **no overviews** |

Reprojects cleanly onto the project grid (DN 1–255, mean 32.4 over the polar box).

**Caveat that matters:** an equirectangular mosaic is a poor polar product. At
−89.5° each 100 m pixel spans 100·cos(89.5°) ≈ 0.87 m of ground in longitude, so
the polar rows are hugely oversampled in one axis and carry no more real
information than ~100 m. Combined with grazing WAC illumination, most of the
polar cap is shadow or nodata — visible in the right-hand panel of
`reports/gt_qc/phase1b_reference_terrain.png`, where Shackleton's interior is
solid black. **WAC is the right reference for an IIRS-class (~80 m) source at
low latitude. It is not a useful reference for polar OHRC.**

### 3.3 SELENE / Kaguya TC — access is green

| | |
|---|---|
| Access | public, **no credentials**, plain Apache directory listings |
| Host | `https://darts.isas.jaxa.jp/pub/pds3/` → redirects to `data.darts.isas.jaxa.jp` |
| Relevant sets | `sln-l-tc-5-morning-map-v4.0`, `sln-l-tc-5-evening-map-v4.0`, `sln-l-tc-5-ortho-map(-seamless)-v2.0`, `sln-l-tc-4-dtm-ortho-v3.0`, `sln-l-tc-5-sldem2013-v1.0` |
| Coverage | **global, pole to pole** — 120 longitude volumes × 60 tiles = 7 200 tiles per map |
| Tile | 12 288 × 12 288, 16-bit MSB, scale 0.01, reflectance, ~288 MB |
| Grid | simple cylindrical, R = 1 737.4 km, 4 096 px/deg = **7.403 m/px** |
| Naming | `TCO_MAP{m,e}04_<Nlat><Wlon><Slat><Elon>SC.img`, 3°×3°; the wrap is spelled `E360`, not `E000` |

Verified by resolving real URLs, including the polar tile
`TCO_MAPm04_S87E021S90E024SC.img` (HTTP 200) and the evening equivalent.

Two things make this more useful than expected:

* **Morning and evening maps are the same ground under opposite Sun azimuths**,
  both map-projected, both free. That is a real, global, zero-cost
  illumination-change dataset at a second scale.
* Each tile's label carries `STANDARD_GEOMETRY = (30.0, 0.0, 30.0)` and
  `PHOTO_CORR_ID = "USGS"` — the products are photometrically normalised to a
  fixed geometry. So the brightness difference between morning and evening is
  largely removed while the **topographic shading and cast shadows remain**.
  That isolates the geometric part of the illumination problem from the
  radiometric part, which is exactly the split a matcher has to survive.

Limitation: near the pole, simple cylindrical is heavily distorted, and TC's own
polar coverage is weak. SELENE TC is indexed as available; it is **not**
recommended as the polar reference.

**No TMC-2 exists in this project, so PS pair type B (TMC-2 ↔ SELENE TC) cannot
be built today.** SELENE TC is usable now only as an OHRC ↔ TC cross-sensor pair
at a ~31× scale ratio.

### 3.4 LOLA polar DEMs — fetched and verified

| product | size | grid | coverage | sha256 |
|---|--:|---|---|---|
| `ldem_875s_5m` | 1 840 545 792 B | 30 336², 5 m/px | −87.5 … −90° | `f47a8bf8ca98de00…` |
| `ldem_80s_20m` | 1 848 320 000 B | 30 400², 20 m/px | −80 … −90° | `db44c3b2444acff1…` |

Source: `https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/`.
16-bit LSB integers, scale 0.5 m, offset 1 737 400 m, polar stereographic.

Every check in `scripts/verify_reference_data.py` passes on both:

```
PASS  size matches the label exactly
PASS  covers -90 .. -87.5 deg latitude at 5 m/px
PASS  half-width 75840.0 m agrees with the -87.5 deg circle at 75820.4 m (0.026%)
PASS  GDAL opens the detached label with driver 'PDS', CRS present
PASS  GDAL and raw memmap agree on all 1 048 576 pixels
PASS  elevations are plausible lunar radii
PASS  relief 4830 m over the 50 km box
PASS  WAC reprojects onto the same grid and carries signal
```

`reports/gt_qc/phase1b_reference_terrain.png` renders the DEM, a hillshade and the
WAC crop on one grid. Shackleton is recognisable in all three, its floor ~2.9 km
below the rim and permanently shadowed in WAC — the rendering is not merely
"not-crashing", it is the right mountain.

**This closes the biggest gap in the old design.** `illumination.TerrainModel`
has existed with `NullTerrain.available = False` and a `MockTerrain` stand-in
since the last session. There is now a real DEM at 5 m/px covering every
Chandrayaan-2 footprint we hold.

### 3.5 Other

`~/Downloads/M1294287273LE.IMG` (218 MB) is a genuine LRO NAC **EDR**, PDS3
attached label, `START_TIME = 2018-10-16T00:00:06.423`, orbit 41939. Uncalibrated
and unprojected; without ISIS/SPICE it is not a usable reference, but it is a
real PDS3 EDR to test a PDS3 reader against in Phase 2.

The LROC RDR cumulative index (`INDEX/CUMINDEX.TAB`, 105 MB, 38 620 rows,
27 columns) carries `MAXIMUM_LATITUDE`, `MINIMUM_LATITUDE`,
`EASTERNMOST_LONGITUDE`, `WESTERNMOST_LONGITUDE`, `MAP_SCALE` and
`MAP_PROJECTION_TYPE` for every RDR product — enough for a footprint index of the
non-polar NAC archive. It carries **no incidence angle**. It was not downloaded:
our entire source archive is polar, where the controlled mosaics are strictly
better. It is the right next step if a non-polar Chandrayaan-2 product appears.

---

## 4. The index

`data/index/footprints.gpkg`, two layers:

* `footprints` — lunar geographic degrees (`+proj=longlat +R=1737400`)
* `footprints_stereo` — `+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs`

**All area and intersection arithmetic is done in the stereographic layer.** The
geographic layer exists for display only. This is not pedantry: a single OHRC
strip spans longitude 222° → 110° → 22° while covering 25 km of ground, so a
lon/lat intersection of two of our footprints is meaningless.

Columns: `product_id, mission, instrument, level, role, acquired_utc,
sub_solar_lon_deg, sun_incidence_deg, sun_azimuth_deg, label_sun_azimuth_deg,
label_sun_elevation_deg, sun_geometry_trusted, sun_provenance, gsd_m, bytes,
source_url, local_path, lat_min, lat_max, area_km2, notes`.

One thing that looks wrong and is not: GDAL writes the stereographic CRS into the
GeoPackage as `lat_ts=-90` rather than `k=1`, so `pyproj.CRS.equals()` reports the
two as different. On a sphere they are the same projection — true scale at the
pole is scale factor 1 at the pole. Checked numerically over 20 000 random points
between −90° and −60°: **maximum difference 0.000e+00 m**, round-trip latitude
error 2.6e-10°.

`sun_azimuth_deg` is **null everywhere**. Azimuth from a point near the pole
depends on that point's longitude relative to the sub-solar longitude, and no
product in the index states it. Leaving it null is the honest answer; it is not
inferred.

---

## 5. Candidate pairs

`data/index/pairs.csv`, from `scripts/discover_pairs.py`. 4 sources against 3 383
reference footprints, every intersection computed in the stereographic plane.

### 5.1 Counts

**1 363 candidate pairs.**

| pair type | pairs | median coverage of source | max |
|---|--:|--:|--:|
| OHRC ↔ NAC (controlled polar mosaic) | **519** | 6.4% | **92.0%** |
| OHRC ↔ SELENE TC | 840 | 0.2% | 19.3% |
| OHRC ↔ WAC global mosaic | 4 | 100% | 100% |

Coverage of the source footprint, all pair types:

| coverage | pairs |
|---|--:|
| 0–10% | 1 135 |
| 10–50% | 68 |
| 50–90% | 78 |
| 90–100% | 82 |

The long tail at 0–10% is the SELENE TC grid: near the pole a 3°×3° simple
cylindrical tile is a narrow longitude wedge, so a single tile clips only a
sliver of a strip that circles the pole. Registering OHRC against TC would need
a mosaic of ~120 tiles, which is why TC is indexed but not recommended for the
polar site.

### 5.2 The evaluation matrix this produces

Restricting to references that cover **≥50% of the source footprint**: 156 OHRC↔NAC
pairs, and **every one of them is tile `P892S2250`** (median coverage 89%). Of
those, 144 are single-illumination-bin products and 12 are the `CM_AVG` / `CM_MAX`
/ uncontrolled composites.

| source | reference bins available | Δ Sun azimuth range | coverage |
|---|--:|---|--:|
| `…20211228T2209123959…` | **36** | 0.6° … 179.4° | 85% |
| `…20241115T1326321339…` | **36** | 2.2° … 177.8° | 91% |
| `…20241115T1525004388…` | **36** | 3.3° … 176.7° | 92% |
| `…20251010T0942085687…` | **36** | 2.1° … 177.9° | 87% |

That is a **4 × 36 = 144-pair matrix**: real cross-mission sensors (OHRC 0.22–0.28 m
against NAC 1 m), real illumination change sampled every 10° over the full circle,
the same controlled ground each time, on one 89% common footprint, with the
reference tied to LOLA at 0.8 px (2σ). It is available at a cache cost of a few GB.

### 5.3 The stratification result

| Δ incidence (the PS bins) | pairs | share |
|---|--:|--:|
| 0–10° | **474** | **100.0%** |
| 10–30° | 0 | 0.0% |
| >30° | 0 | 0.0% |

Observed range: **0.042° … 1.251°**.

| Δ sub-solar longitude (Sun azimuth) | pairs | share |
|---|--:|--:|
| 0–10° | 30 | 6.3% |
| 10–45° | 105 | 22.2% |
| 45–90° | 120 | 25.3% |
| >90° | 219 | 46.2% |

Observed range: **0.6° … 179.4°**.

The first table is the problem statement's requested stratification, and it is
empty in two of three bins. The second is the same pairs binned by the variable
that actually moves. Both columns are in `pairs.csv`; nothing is discarded.

---

## 6. Storage

### On disk now

| | GB |
|---|--:|
| `data/raw/ch2/` — 4 OHRC products | 4.4 |
| `data/raw/lola/` — 2 polar DEMs | 3.5 |
| `data/raw/wac/` — global mosaic | 5.6 |
| **total** | **13.5** |

Free space on the volume: 415 GB at the start of the session.

### Reference imagery — do not mirror it

| option | cost |
|---|--:|
| Entire LROC south-polar mosaic archive, `.IMG` | 15.60 TB |
| Same, 8-bit GeoTIFF | 3.90 TB |
| Band 1 only (88.5–90°S), GeoTIFF | ~286 GB |
| `P892S2250` in all 39 product sets, GeoTIFF | ~81 GB |
| `P892S2250` in the 6 bins matched to our sources, GeoTIFF | ~12 GB |
| **Range reads of the windows actually used** | **~2–5 GB of cache** |

The last line is the recommendation. `scripts/fetch_lroc_pairs.py` already
demonstrates chunked, retried HTTP range reads against this archive, and
`/vsicurl` + `rasterio` does the same through the GeoTIFF mirror with the CRS
attached. The existing `results/lroc_cache/` (107 MB, 26 crops) is what this
looks like in practice.

SELENE TC, if used: 288 MB per tile. The 87–90°S band is 120 tiles per map, so a
full polar morning+evening set is ~69 GB. Fetch individual tiles on demand.

---

## 7. Proposed first development site

**Site S1 — the south pole, a 50 km box centred on it.**

| | |
|---|---|
| Bounds (stereographic) | x ∈ [−25 000, +25 000] m, y ∈ [−25 000, +25 000] m |
| Approximate lat/lon | −89.2° to −90°, all longitudes |
| Source | all four OHRC products — every one of them crosses this box |
| Reference | `NAC_POLE_SOUTH_CM_*_P892S2250` (+ the three adjacent band-1 tiles at the seams) |
| Terrain | `ldem_875s_5m`, already local, 5 m/px, covers the whole box |
| Illumination available | 36 reference bins at 10° spacing, the full 0–360° |
| Source-to-source Δ Sun azimuth | 1.1°, 44.6°, 45.7°, 72.7°, 117.3°, 118.4° |
| Storage | ~2–5 GB of cached reference crops |
| Already verified here | `reports/gt_qc/phase1b_reference_terrain.png` |

Why this site and not the brief's 20–21°E / −70 to −67°: **we hold no
Chandrayaan-2 data anywhere near there.** Every product is within 0.8° of the
south pole.

Two caveats to carry into Phase 2:

1. **Texture is scarce.** `scripts/inspect_dataset.py` measured that only
   **22–42%** of 512 px OHRC tiles carry usable texture, the rest being shadow.
   Windows must be chosen by measured signal, not by position — a control run in
   the previous session landed on a window that was 92.8% shadow and nine methods
   returned garbage.
2. **Geolocation is off by kilometres.** `results/experiments/P0_FINDINGS.md`
   measured the OHRC-to-NAC offset at **3.8–4.5 km**, with a further 0.4–0.9 km
   of product-to-product scatter. Consistent with every label carrying
   `corners_refined = False`. Any search radius, any overlap test and any
   `no_overlap` decision must be sized for that, and a 384 m window placed by
   CH-2 geometry shows **different ground**.

An illumination-matched reference bin for each source, from the modelled
sub-solar longitudes:

| source | sub-solar lon | nearest LROC bin | Δ |
|---|--:|---|--:|
| 20241115T1326 | 357.19° | `CM_355` | 2.2° |
| 20241115T1525 | 358.31° | `CM_355` | 3.3° |
| 20251010T0942 | 42.88° | `CM_045` | 2.1° |
| 20211228T2209 | 115.57° | `CM_115` | 0.6° |

Every one of those four tiles exists (`P892S2250` is in all 36 bins). This is a
concrete, testable prediction: registering each OHRC product against its
illumination-matched bin should outperform registering it against `CM_AVG`, which
is the shadow-free average the earlier spike used.

---

## 8. Data-access limitations, stated plainly

1. **Chandrayaan-2 acquisition is not automatable from this repository.** The four
   products came from a credentialed ISSDC/PRADAN delivery. No script here can
   re-fetch them. TMC-2 and IIRS would need the same route.
2. **The PDS Imaging archive rate-limits.** A burst of directory listings earned
   HTTP 429 and locked us out for minutes. The catalogue builder now spaces
   requests ≥3 s, honours `Retry-After`, and checkpoints every directory.
3. **That archive also ignores `continuation-token`.** A paginated recursive S3
   listing silently loops: 50 000 keys collected, **1 000 distinct**. Use one
   delimited listing per directory, or `start-after`. This is a real trap and
   would have produced a plausible-looking but wrong catalogue.
4. **`results/nac_cache/` is gone.** The ~209 MB of `CM_AVG` crops that
   `results/experiments/P0_FINDINGS.md` was computed from is not on this machine,
   so that report's headline 3.8–4.5 km offset is **not currently reproducible**
   without re-fetching. Re-deriving it is a Phase 2 task.
5. **No SPICE, ISIS, ASP or ALE.** None publish cp314 wheels; all would need a
   separate conda environment. Whether GT tier T2 (rigorous camera geometry) is
   reachable at all is still open and will be reported, not assumed.
6. **No CUDA.** Integrated Intel Iris Xe only. Any learned matcher runs CPU-only;
   training on this machine is not viable.

---

## 9. What changed on disk this session

| path | what |
|---|---|
| `data/raw/ch2/ohrc/` | 4 OHRC products extracted from the local bundle, 4.4 GB |
| `data/raw/lola/` | 2 LOLA polar DEMs fetched and verified, 3.5 GB |
| `data/raw/wac/` | LROC WAC global mosaic fetched and verified, 5.6 GB |
| `data/manifest.jsonl` | provenance + SHA-256 for every fetched file |
| `data/index/nac_south_catalog.json` | 982 LROC products over 39 product sets |
| `data/index/nac_tile_geometry.json` | 524 measured GeoTIFF tile bounds, used to validate the analytic tile geometry |
| `data/index/footprints.gpkg` | the unified footprint index, 3 389 features, 2 layers |
| `data/index/pairs.csv` | 1 363 candidate source→reference pairs |
| `data/processed/testcrops/` | 3 verified crops for unit tests |
| `reports/gt_qc/phase1b_reference_terrain.png` | the visual check |
| `scripts/build_footprint_index.py` | new |
| `scripts/discover_pairs.py` | new |
| `scripts/verify_reference_data.py` | new |
| `docs/AUDIT.md`, `docs/DATA_INVENTORY.md` | new |

Everything under `data/raw/` is already covered by the `.gitignore` rule
`data/raw/`. `data/index/` and `data/processed/` were added to `.gitignore` this session;
`data/manifest.jsonl` is deliberately **not** ignored, because it is the provenance
record.

No file under `backend/` was modified in Phase 0 or Phase 1.
