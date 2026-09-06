# Data provenance

Every pixel in this prototype comes from a delivered ISRO Chandrayaan-2 product.
No lunar imagery has been generated, painted, or sourced from an image search.

## Source products

All three products are Chandrayaan-2 Orbiter High Resolution Camera (OHRC) PDS4
archives released by ISRO/ISSDC and redistributed in the Internet Archive item
["India's Chandrayaan 2 — High Resolution Images Of The Moon"](https://archive.org/details/chandrayaan-2-high-resolution-images-of-the-moon).
That mirror is publicly downloadable without credentials, which is why it was
used. The authoritative source is ISRO's own PRADAN portal
(<https://pradan.issdc.gov.in/ch2/>) and the Chandrayaan Map Browse service
(<https://chmapbrowse.issdc.gov.in/>), which require registration.

| Product | Logical identifier | Level | Acquired | Orbit |
|---|---|---|---|---|
| `ch2_ohr_ncp_20200229T0739312111_d_img_d18` | `urn:isro:isda:ch2_cho.ohr:data_calibrated:...` | Calibrated | 2020-02-29T07:39:31.2111Z | 2297 |
| `ch2_ohr_nrp_20200229T0739312111_d_img_d18` | `urn:isro:isda:ch2_cho.ohr:data_raw:...` | Raw | 2020-02-29T07:39:31.2111Z | 2297 |
| `ch2_ohr_ncp_20200229T0938004033_d_img_d32` | `urn:isro:isda:ch2_cho.ohr:data_calibrated:...` | Calibrated | 2020-02-29T09:38:00.4033Z | 2298 |

### Facts read from the delivered PDS4 labels

These are not quoted from press material; they were parsed out of the `.xml`
labels that ship with each product.

| Field | Orbit 2297 | Orbit 2298 |
|---|---|---|
| Array size (lines × samples) | 93693 × 12000 | 93693 × 12000 |
| Sample type | `UnsignedByte`, offset 0 | `UnsignedByte`, offset 0 |
| `isda:pixel_resolution` | **0.22977 m/px** | **0.23022 m/px** |
| `isda:spacecraft_altitude` | 90.4057 km | 90.5839 km |
| `isda:focal_length` | 2080 mm | 2080 mm |
| `isda:detector_pixel_width` | 5.2 µm | 5.2 µm |
| `isda:tdi_stages` | TDI64 | TDI64 |
| Refined corners (lat) | −74.366° … −73.517° | −73.920° … −73.071° |
| Refined corners (lon) | 43.359° … 43.958° | 42.457° … 43.032° |

Both strips lie in the lunar **south polar region**. Their longitude ranges do
not overlap, so the two strips image different ground. The two windows cut for
the demo pair sit 10.95 km apart (great-circle distance between their centres,
computed from the delivered geometry grids), which is exactly why they make a
usable true-negative pair.

### Files actually used

| File | Size | What it is |
|---|---|---|
| `*_d_img_*.img` | 1.12 GB each | Raw uint8 image array. Four row bands were extracted by streaming the archive's ZIP and inflating only as far as needed. |
| `*_b_brw_*.png` | ~9.6 MB | ISRO's own browse product: the full strip decimated 10×, i.e. ~2.30 m/px. |
| `*_g_grd_*.csv` | ~3.9 MB | Longitude/latitude on a (pixel, scan) lattice at 100-pixel steps. Used for all selenographic coordinates reported in the UI. |
| `*_d_img_*.xml` | ~8 KB | PDS4 label; the table above comes from these. |

## Sensor context

| Sensor | Spatial resolution | Band | Used here? |
|---|---|---|---|
| **OHRC** | 0.25–0.32 m/px nominal; **0.2298 m/px** in the products used | Panchromatic | Yes — every pair |
| **TMC-2** | ~5 m/px | Panchromatic | **No real data.** Its 5 m/px GSD is simulated by resampling OHRC. |
| **IIRS** | ~80 m, 0.8–5.0 µm hyperspectral | Hyperspectral | **No.** Not implemented. |

ISRO's *Payloads Data & Science Handbook* is the authority for the nominal
figures: <https://www.isro.gov.in/media_isro/pdf/science/hand_book_payloads_data_and_science.pdf>

## The six demo pairs

Built by `scripts/prepare_data.py`. "Synthetic" below never means synthetic
*terrain* — it means a known geometric or radiometric transform applied to real
imagery so that a ground truth exists.

| Pair | Real | Synthetic | Ground truth |
|---|---|---|---|
| `ohrc_crater_field` | OHRC full-res crater field, 0.2298 m/px | 11° rotation, 1.12× scale, small perspective, gamma 1.28 | Exact, by construction |
| `ohrc_raw_vs_calibrated` | Both sides real: calibrated vs raw browse of the *same* acquisition | **None** | Exact translation between two crops |
| `ohrc_tmc2_moderate` | OHRC browse, 2.30 m/px | Resampled to 5.0 m/px + 6° rotation | Exact |
| `ohrc_tmc2_extreme` | OHRC full-res, 0.2298 m/px | Resampled to 5.0 m/px (21.8×) + 4° rotation | Exact |
| `ohrc_low_illumination` | Genuinely shadowed south-polar terrain (~35 % of pixels below DN 20) | 9° rotation, 1.08× scale, gamma 1.6 + noise | Exact |
| `ohrc_disjoint_orbits` | Both sides real, orbits 2297 and 2298 | **None** | **None** — the scenes do not overlap |

### What the TMC-2 simulation is not

Resampling OHRC to 5 m/px reproduces TMC-2's *ground sampling distance* and
nothing else. TMC-2's optics, modulation transfer function, noise
characteristics, stereo viewing geometry and its own illumination conditions are
all absent. A result on `ohrc_tmc2_*` is evidence about scale-ratio robustness
only, and must not be reported as an OHRC↔TMC-2 result.

## Reproducing the data

`scripts/prepare_data.py` needs these files in one directory:

```
ohrcA_rows78000_82096.npy                            # full-res band, 4096 x 12000
ohrcA_rows48000_52096.npy
ohrcA_rows2000_6096.npy
ch2_ohr_ncp_20200229T0739312111_b_brw_d18.png
ch2_ohr_nrp_20200229T0739312111_b_brw_d18.png
ch2_ohr_ncp_20200229T0938004033_b_brw_d32.png
ch2_ohr_ncp_20200229T0739312111_g_grd_d18.csv
ch2_ohr_ncp_20200229T0938004033_g_grd_d32.csv
```

`scripts/fetch_products.py` downloads all of them from the Internet Archive
mirror. The browse PNGs and geometry CSVs are pulled as individual ZIP members
(a few MB each); the full-resolution bands are obtained by streaming the ~780 MB
compressed member and inflating only as far as the requested rows, so nothing is
written to disk beyond the bands themselves.

```
python scripts/fetch_products.py --out data/raw
python scripts/prepare_data.py --raw data/raw
```

## Licence and attribution

Chandrayaan-2 data is produced by the **Indian Space Research Organisation**.
Use is governed by ISRO/ISSDC's terms — see
<https://pradan.issdc.gov.in/ch2/ack.xhtml>. Users of the data are asked to
acknowledge ISRO. This prototype is a non-commercial academic demonstration for
the Smart India Hackathon and claims no rights over the imagery.
