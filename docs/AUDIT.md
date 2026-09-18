# Phase 0 — Read-only audit

Date: 2026-09-18. Branch `main`, HEAD `a29a1df`.
Nothing in `backend/`, `scripts/`, `tests/` or `frontend/` was modified to produce
this document. The only write is this file.

Every claim below is tagged:

- **[verified]** — I ran the command or read the bytes in this session.
- **[reported]** — taken from a document already in the repository; not re-run here.

---

## 1. Repository map

### 1.1 Tree (depth 3, generated content collapsed) **[verified]**

```
seleno-prototype/
├── run.py                       single entry point: serves API + built UI on :8000
├── requirements.txt             8 runtime deps; torch/kornia commented out
├── backend/
│   ├── app.py                   FastAPI, 17 routes
│   └── seleno/
│       ├── matchers.py          SIFT / AKAZE / ORB / sift_pyramid / LoFTR slot
│       ├── registration.py      the 9-stage OHRC pipeline  (686 L, largest module)
│       ├── pipeline.py          the older legacy-pair pipeline (332 L)
│       ├── preprocess.py verify.py register.py metrics.py spatial.py
│       ├── subpixel.py illumination.py pairs.py experiments.py
│       ├── alignment.py geodesy.py store.py theme.py viz.py
│       └── ohrc/                label.py product.py geometry.py sun.py
│                                tiles.py reproject.py
├── scripts/                     11 scripts (fetch, prepare, inspect, benchmark,
│                                experiments, nac_spike, fetch_lroc_pairs,
│                                evaluate_lroc, run_pipeline, run_registration,
│                                exp_reference)
├── tests/test_ohrc_dataset.py   28 tests, single file, hand-rolled runner
├── frontend/                    React 19 + Vite 8; src/ 1 193 L; dist/ committed
├── data/                        manifest.json, pairs/ (6 legacy PNG pairs),
│                                lroc/ (14 LROC pairs + manifest)
├── results/                     246 MB: BENCHMARK.md, dataset/, experiments/,
│                                illumination/, lroc/, lroc_cache/, showcase/
├── Dataset/                     **empty directory**
└── 8 top-level .md docs         README, HANDOFF, PLANNER_HANDOFF, ROADMAP,
                                 SESSION_HANDOFF, ASSUMPTIONS, RESEARCH,
                                 research_by_space (untracked, 1 240 L)
```

There is **no `docs/`, no `config/`, no `src/`** directory. This file creates `docs/`.

### 1.2 Code size **[verified]**

| Area | Lines |
|---|--:|
| `backend/` (Python) | 4 733 |
| `scripts/` (Python) | 2 605 |
| `tests/` (Python) | 430 |
| `run.py` | 48 |
| **Python total** | **9 319** |
| `frontend/src` | 1 193 |
| Markdown (13 files) | 4 251 |

### 1.3 Entrypoints **[verified]**

| Entrypoint | What it does | Runs today? |
|---|---|---|
| `python run.py` | serves `frontend/dist` + API on 127.0.0.1:8000 | yes — `data/manifest.json` exists |
| `python -m uvicorn app:app` (`backend/app.py`) | API only | yes |
| `scripts/fetch_products.py` | pulls OHRC from an Internet Archive mirror | untested this session |
| `scripts/prepare_data.py` | builds the 6 legacy PNG pairs from raw OHRC | needs `data/raw/` — absent |
| `scripts/fetch_lroc_pairs.py` | builds `data/lroc/` from LROC polar mosaics | output already present |
| `scripts/evaluate_lroc.py` | scores `data/lroc/` → `results/lroc/LROC.md` | yes (pairs are on disk) |
| `scripts/benchmark.py` | scores the 6 legacy pairs → `results/BENCHMARK.md` | yes |
| `scripts/nac_spike.py`, `scripts/exp_reference.py` | OHRC ↔ LRO NAC experiments | **no** — need OHRC dataset + `results/nac_cache/`, both absent |
| `scripts/inspect_dataset.py`, `illumination_experiment.py`, `run_registration.py` | OHRC dataset work | **no** — need `SELENO_OHRC_ROOT` |

### 1.4 Configuration **[verified]**

There is **no configuration system**. Specifically:

- No `config/`, no YAML anywhere in the repo, no `.env`, no `.env.example`.
- The only environment variables read by the whole codebase are
  `SELENO_OHRC_ROOT` (`backend/seleno/ohrc/product.py:39`) and `PORT`
  (`backend/app.py:474`).
- All tuning lives in Python dataclass defaults (`registration.Options`,
  `pipeline.Options`) and in a preset table (`backend/seleno/experiments.py`).
- Sensor constants are hardcoded per call site — GSD, shadow thresholds,
  percentile clips and CLAHE parameters are arguments with literal defaults in
  `preprocess.py` and `ohrc/tiles.py`.
- `.claude/settings.local.json` disables the `code-review-graph` MCP server, which
  is why the knowledge graph named in `CLAUDE.md` reports 0 nodes. Grep/Read is
  the only available route, and is what this audit used.

### 1.5 Dependencies **[verified]**

`requirements.txt` declares 8 runtime packages. Installed in `.venv` (Python 3.14.7):

| Declared | Installed | Note |
|---|---|---|
| `opencv-python-headless>=4.8,<5` | **4.14.0.94** | the `<5` pin is uncommitted; 5.0 drops `AKAZE_create` |
| `numpy>=1.24` | 2.5.3 | |
| `scipy>=1.10` | 1.18.1 | uncommitted addition |
| `matplotlib>=3.7` | 3.11.2 | uncommitted addition |
| `fastapi>=0.110` | 0.141.1 | |
| `uvicorn>=0.27` | 0.52.4 | |
| `pydantic>=2.0` | 2.13.5 | |
| `remotezip>=0.12` | 0.12.6 | |
| `torch`, `kornia` | **absent** | commented out in the file |

Absent and needed for the revised scope: `rasterio`, `pyproj`, `shapely`,
`geopandas`/`pyogrio`, `pds4_tools`, `pvl`, `scikit-image`, `torch`, `kornia`.

`frontend/package.json`: React 19.2, Vite 8.2, oxlint. **`node_modules/` is not
installed**; `frontend/dist/` is committed, so `run.py` serves a stale build.

### 1.6 Tests **[verified]**

One test file, `tests/test_ohrc_dataset.py`, with a hand-rolled runner (no
pytest). Literal output on this machine:

```
$ .venv/bin/python tests/test_ohrc_dataset.py
...
4 passed, 0 failed, 24 skipped
```

The 4 that pass are pure-geometry (stereographic round-trip, pole→origin,
longitude degeneracy, shadow-length scaling). **The other 24 skip because the
OHRC dataset is not mounted.** There is no CI configuration anywhere in the repo,
and no test at all for `matchers.py`, `verify.py`, `spatial.py`, `register.py`,
`metrics.py`, `registration.py`, `app.py` or the frontend.

---

## 2. Existing acquisition architecture

There are **three** distinct acquisition paths, plus one manual delivery. None of
them share code; there is no common fetcher, no manifest schema, and no checksum
verification anywhere.

### 2.1 OHRC — Internet Archive mirror (`scripts/fetch_products.py`) **[verified by reading]**

- **Discovery**: none. Three ZIP filenames are hardcoded (`ZIPS` dict), all from
  orbits 2297/2298 of 2020-02-29 — a *different* set from the four products the
  rest of the repo is built on.
- **Auth**: none. `https://archive.org/download/chandrayaan-2-high-resolution-images-of-the-moon/Optical High Resolution Camera (OHRC)/<name>.zip`
- **Request sequence**:
  1. `RemoteZip(url)` — HTTP range reads of the ZIP end-of-central-directory, then
     the central directory, to enumerate members.
  2. For each small member (`.png`, `.csv`, `.xml` under `/data/`): `z.read(name)`,
     one ranged GET per member.
  3. For the 1.12 GB `.img` member: a 30-byte ranged GET of the local file header
     to learn `name_len`/`extra_len`, then **one** ranged GET over the whole
     compressed stream, fed to `zlib.decompressobj(-15)` and abandoned once the
     deepest requested row band has been produced.
- **Output layout**: flat — `data/raw/<basename>` plus `ohrcA_rows<lo>_<hi>.npy`
  for three hardcoded 4 096-row bands.
- **Manifest**: none written by this script. `data/manifest.json` is written later
  by `scripts/prepare_data.py` and describes *pairs*, not downloads.
- **Politeness**: none — no rate limit, no backoff, no retry. Resumability is
  file-existence only ("have %s"), with no size or hash check.

### 2.2 LROC polar mosaics — PDS store (`scripts/fetch_lroc_pairs.py`) **[verified by reading]**

This is the newest and best of the three.

- **Discovery**: none; the tile `P892S2250` and seven sub-solar-longitude bin
  pairs are hardcoded.
- **Auth**: none.
- **Base**: `https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/`
  - image: `DATA/BDR/NAC_POLE/NAC_POLE_SOUTH_CM_<bin>/NAC_POLE_SOUTH_CM_<bin>_P892S2250.IMG`
  - browse: `EXTRAS/BROWSE/NAC_POLE/.../<...>.BROWSE.PNG`
- **Request sequence**: `curl -sfL --retry 3 -r <lo>-<hi>`, chunked at 32 MB, with
  6 attempts and linear backoff per chunk, writing `.part` then appending. Row
  bands are computed from the PDS4 geometry (`RECORD = 45488*4`, header 181 952 B).
- **Output layout**: `data/lroc/pairs/*.png` + `data/lroc/manifest.json`;
  full-resolution reads cached as `.npy` in `results/lroc_cache/` (gitignored).
- **Manifest**: `data/lroc/manifest.json` **does** record provenance, GSD, window,
  the synthetic warp and the ground-truth homography per pair. It is the only
  manifest in the repo worth generalising.
- **Politeness**: chunked + retried + backed off. No concurrency cap declared, but
  a `ThreadPoolExecutor` is imported and used for browse fetches.
- **Verification**: chunk length only. No SHA-256, no end-to-end hash.

### 2.3 LROC `CM_AVG` — ASU redirect (`scripts/nac_spike.py`, `scripts/exp_reference.py`) **[verified by reading]**

- Base `https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/NAC_POLE`,
  which **302s to `pds.mcp.nasa.gov`**. The resolved URL was cached in a
  `.nac_url` file — **that file is absent on this machine**, so the redirect will
  be re-resolved on first use.
- Same PDS3 attached-label arithmetic: 45 488², 32-bit `PC_REAL`,
  `RECORD_BYTES = 181 952`, `^IMAGE = 2`, `SAMPLE_PROJECTION_OFFSET = 45 487.5`,
  `LINE_PROJECTION_OFFSET = -0.5`, `MAP_SCALE = 1.0`, radius 1 737 400 m. **[reported, and re-derived in `results/experiments/P0_FINDINGS.md` §1]**
- Two traps are documented and guarded **[reported]**: per-row range requests earn
  HTTP 429 (use one contiguous request); and a negative row index makes `curl`
  issue a *suffix* range, which starts downloading the whole 7.71 GB file.

### 2.4 OHRC — ISSDC/PRADAN manual delivery **[verified]**

The four products the entire OHRC layer is built on were not fetched by any script
in this repo. They arrived as an SFTP bundle and are sitting **outside the repo**:

```
/home/paardhu/Downloads/ch2_ohr_ncp_20211228T2209123959_d_img_d18_Bundle.tar   1.8 GB
```

`tar -tvf` shows four ZIPs, owner `sftpusr/sac_poc` — exactly the four products in
`data/README.md`:

| ZIP | Bytes |
|---|--:|
| `ch2_ohr_ncp_20211228T2209123959_d_img_d18.zip` | 303 543 249 |
| `ch2_ohr_ncp_20241115T1326321339_d_img_d18.zip` | 465 728 092 |
| `ch2_ohr_ncp_20241115T1525004388_d_img_d18.zip` | 486 003 676 |
| `ch2_ohr_ncp_20251010T0942085687_d_img_d18.zip` | 577 658 020 |

I extracted and listed the 2021 ZIP. It contains the **complete** product tree the
code expects — nothing is missing:

```
data/calibrated/20211228/….xml          8 899 B   PDS4 label
data/calibrated/20211228/….img    957 552 000 B   UnsignedByte, 79 796 × 12 000
geometry/calibrated/20211228/….xml      6 206 B
geometry/calibrated/20211228/….csv  3 355 951 B   (Pixel,Scan) → (lon,lat)
miscellaneous/calibrated/20211228/….oat   325 304 B   orbit/attitude
miscellaneous/calibrated/20211228/….oath      201 B
miscellaneous/calibrated/20211228/….lbr    133 644 B
miscellaneous/calibrated/20211228/….spm    128 982 B   Sun parameters
miscellaneous/readme.txt                 11 119 B   ancillary format spec
browse/calibrated/20211228/….xml        4 770 B
browse/calibrated/20211228/….png    5 913 851 B   10× decimated preview
```

Uncompressed, the four products are ≈ 4.6 GB. **The OHRC data is on this machine,
one `tar -x` away from working.** The brief's assumption that "OHRC is NOT on this
machine" is incorrect.

Also present and unaccounted for by any script:

```
/home/paardhu/Downloads/M1294287273LE.IMG    218 MB
```

A real **LRO NAC EDR**, PDS3 attached label, `DATA_SET_ID = LRO-L-LROC-2-EDR-V1.0`,
`PRODUCT_ID = M1294287273LE`, `START_TIME = 2018-10-16T00:00:06.423`, orbit 41939,
`RECORD_BYTES = 5064`, `FILE_RECORDS = 45057`. Uncalibrated, unprojected, no
useful geometry without ISIS/SPICE — of limited value as a reference product, but
it is a genuine PDS3 EDR to test a PDS3 reader against.

---

## 3. Current model availability

**There is no model.** Verified by exhaustive search:

- No `.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors`, `.h5` anywhere in the repo.
- No training script, no dataset class, no loss, no optimiser, no config.
- No `torch` or `kornia` in the environment; `matchers.py:_probe_learned()` returns
  `False` and the UI reports `loftr` as unavailable rather than substituting SIFT.
  That honesty mechanism is already correct and should be kept.

The only "learned" surface is a **slot**: `matchers.MATCHERS["loftr"]`, a lambda
behind the same `MatchResult` interface as the classical matchers, that raises
`RuntimeError` when called. `run_matcher(name, ...)` refuses unavailable matchers
by name. This is the correct seam for adding RIFT2 / SuperPoint+LightGlue.

---

## 4. Current data availability

### 4.1 In the repository **[verified]**

| Path | Size | What it really is | Real or synthetic |
|---|--:|---|---|
| `data/pairs/` | 30 MB | 6 legacy PNG pairs, 1024², from OHRC orbits 2297/2298 | **5 of 6 are an image matched to a warped copy of itself.** Ground truth is the warp, exact by construction. Illumination change is a gamma/gain edit, not real |
| `data/manifest.json` | — | describes those 6 pairs + 3 product LIDs | — |
| `data/lroc/pairs/` | included in the 30 MB | **14 pairs** (12 positive, 2 negative), 1024², cut from LROC `NAC_POLE_SOUTH_CM_<bin>` controlled mosaics at 1 m/px | **Illumination change is real** (different sub-solar longitude, different NAC source images, real shadows). The extra geometric warp is synthetic and recorded. Truth carries the network's ~0.8 px 2σ residual |
| `data/lroc/manifest.json` | — | per-pair provenance, Δ sub-solar longitude, warp, GT homography | — |
| `data/raw/` | — | **does not exist** (gitignored) | — |
| `Dataset/` | 0 | **empty directory** | — |
| `results/lroc_cache/` | 107 MB | 26 `.npy` full-resolution LROC mosaic crops, bins 005/055/095/145/235/325 | real, gitignored cache |
| `results/nac_cache/` | — | **absent.** The ~209 MB of `CM_AVG` crops that `results/experiments/P0_FINDINGS.md` was computed from | — |
| `results/showcase/`, `results/dataset/`, `results/experiments/`, `results/illumination/` | 140 MB | committed figures and JSON from previous runs | evidence, not inputs |

### 4.2 Outside the repository **[verified]**

| Path | Size | Status |
|---|--:|---|
| `~/Downloads/ch2_ohr_..._Bundle.tar` | 1.8 GB | 4 complete OHRC L2 products, **not extracted**, `SELENO_OHRC_ROOT` unset |
| `~/Downloads/M1294287273LE.IMG` | 218 MB | one LRO NAC EDR, PDS3 |

### 4.3 What is therefore NOT present

- **No DEM of any kind.** No LOLA, no TMC-2 DEM, no SLDEM2015.
  `illumination.NullTerrain.available` is `False` and every shadow-prediction
  claim in the repo runs on `MockTerrain`.
- **No TMC-2 imagery** (the `ohrc_tmc2_*` pairs are OHRC decimated to ~5 m/px, not
  TMC-2 — `PLANNER_HANDOFF.md` §6 flags this explicitly).
- **No IIRS data.**
- **No SELENE / Kaguya TC data.**
- **No LROC WAC global mosaic.**
- **No footprint index** — `data/index/` does not exist.
- **No training data, no ground-truth tie-point set, no held-out manual GT.**

### 4.4 Honest state of the evidence **[reported, from the repo's own reports]**

The one result that matters for the revised scope is
`results/lroc/LROC.md` — the only evaluation in the repo on **real illumination
change with independent ground truth**:

| Preset | Matcher | Positives solved (≤3 px) | False accepts | Negatives refused |
|---|---|--:|--:|--:|
| baseline_verified | sift | **1/12** | 0 | 2/2 |
| baseline_verified | akaze | **1/12** | 0 | 2/2 |
| baseline_verified | orb | **1/12** | 0 | 2/2 |
| seleno | sift | **0/12** | 0 | 2/2 |
| seleno | akaze | **1/12** | 0 | 2/2 |
| seleno | orb | **1/12** | 0 | 2/2 |

Stratified by Sun-direction change: 5 solved out of 24 runs at Δaz = 50°, **0 of
30 at 90°, 0 of 18 at 180°.**

Read plainly: **the current pipeline does not solve the problem the PS names
first.** It fails safely — zero false accepts across 84 runs, and both negative
controls refused by every configuration — but "honest refusal" is the whole of the
current capability beyond Δaz ≈ 50°.

By contrast `results/BENCHMARK.md` reports 0.005–0.288 px ground-truth corner
error on the legacy pairs. Those numbers are real but measure a problem with the
Sun-angle challenge removed (self-matched imagery). They must not be quoted as
registration accuracy.

The separate OHRC ↔ LRO NAC result in `results/experiments/P0_FINDINGS.md`
**[reported]** — that OHRC locks onto `CM_AVG` with a ~3.8–4.5 km systematic
geolocation offset, corroborated by four independent checks — is the single most
valuable finding in the repository. **It is not currently reproducible on this
machine** (needs the OHRC tree extracted and `results/nac_cache/` refetched).

---

## 5. Compute **[verified]**

| | |
|---|---|
| `nvidia-smi` | `command not found` |
| GPU | `Intel Corporation Raptor Lake-P [Iris Xe Graphics]`, Mesa 26.1.7 — **integrated, no CUDA, no ROCm** |
| VRAM | none dedicated; shared system memory |
| `torch.cuda.is_available()` | not answerable — **torch is not installed** |
| CPU | 20 logical cores |
| RAM | 15 GiB total, 9.1 GiB available, 4 GiB swap |
| Disk | 691 GB volume, **415 GB free** |
| Python | 3.14.7 (venv at `.venv`) |

**Consequence: training is not feasible on this machine.** Any learned matcher
must run CPU-only inference. Python 3.14 wheel availability was checked with
`pip install --dry-run` **[verified]**:

| Package | Resolves on cp314? |
|---|---|
| `torch`, `kornia`, `onnxruntime` | **yes** (CPU) |
| `rasterio`, `pyproj`, `shapely`, `pyogrio`, `geopandas` | **yes** |
| `scikit-image`, `pds4_tools`, `pvl`, `planetaryimage` | **yes** |
| `fiona` | **no** — builds from source, needs `gdal-config`. Use `pyogrio` for GPKG I/O instead |

ISIS / ASP / ALE / SpiceQL were not probed; they are conda-forge distributions that
would need a separate environment, and none of them publish cp314 wheels.

---

## 6. Gap list

Ranked by how much each blocks the revised scope.

| # | Gap | Impact | Evidence |
|---|---|---|---|
| 1 | **No illumination-invariant matcher.** Only SIFT/AKAZE/ORB. No phase congruency, no RIFT2, no learned matcher installed | Fails the PS's headline requirement. 0/48 solved beyond Δaz 50° | `results/lroc/LROC.md` |
| 2 | **No `register(source, reference)` callable and no output contract.** No `matches.csv`, `registered.tif`, `transform.json`, `metrics.json`, `overlay.png`, `report.md`; no `outputs/<job_id>/`; no CLI; no reason codes | The core deliverable does not exist in the required shape | grep: no such writers |
| 3 | **No sensor abstraction.** No `config/sensors/*.yaml`; GSD, clip percentiles, shadow thresholds are literals at call sites; `pairs.py` knows about OHRC specifically | Blocks "sensor-agnostic", blocks TMC-2/IIRS/SELENE | §1.4 |
| 4 | **No reference data beyond one LROC tile.** No WAC mosaic, no SELENE TC, no LOLA DEM, no footprint index, no polar intersection code outside `ohrc/geometry.py` | Blocks Phase 1 entirely | §4.3 |
| 5 | **Global 3×3 homography only.** `verify.py` fits homography/affine/similarity; no piecewise or per-segment model | Structurally wrong for 80–101 k-line pushbroom strips | `verify.py:_min_points`, `research_by_space.md` §2.3 |
| 6 | **No common reader layer.** PDS4 parsing is hand-rolled OHRC-only regex in `ohrc/label.py`; PDS3 is hand-rolled per script; no GeoTIFF reader at all | Blocks Phase 2 | §1.1 |
| 7 | **Metrics are not matcher-independent.** `metrics.reprojection_metrics` scores the transform on its own surviving inliers; the hold-out split in `registration._split_holdout` is drawn from the same matched set | Violates the non-negotiable | `metrics.py:28`, `registration.py:181` |
| 8 | **No synthetic GT generator, no manual GT set.** T1 needs a DEM (absent); T3 does not exist | Only T-tier available today is the LROC controlled-mosaic pairs | §4.3 |
| 9 | **Thresholds are chosen, not derived.** ≥12 inliers, 15 % ratio, 15 % coverage, 4× RANSAC | `thresholds_calibrated` is `false` in the API by the repo's own admission | `PLANNER_HANDOFF.md` §6 |
| 10 | **No adversarial / CI tests.** 24 of 28 tests skip; no CI file; none of the 9 Phase-8 cases exist | Phase 8 is greenfield | §1.6 |
| 11 | **No download integrity or provenance discipline.** No SHA-256 anywhere, no `data/manifest.jsonl`, three unrelated fetchers | Phase 2 requirement | §2 |
| 12 | **Sun geometry in the OHRC labels is not trustworthy.** Two consecutive orbits report solar elevations −0.455° / +0.500°; solving for the sub-solar point gives 67–149° of variation along one strip | Incidence-angle stratification cannot use CH-2 metadata as-is | `results/experiments/P0_FINDINGS.md` §6 **[reported]** |
| 13 | **Frontend is a stale committed `dist/`** with no `node_modules`; no job-submit/status/results endpoints | Phase 7 UI needs rework | §1.5 |
| 14 | **Two parallel pipelines** (`pipeline.py` for legacy pairs, `registration.py` for OHRC) with duplicated Options, Timer, verdict logic | Maintenance hazard; neither has the required contract | §1.1 |

---

## 7. Code that is now out of scope

Under the revised scope these are no longer on the critical path. **None should be
deleted during Phase 0**; this is the disposition list to act on later.

| Code | Why out of scope | Disposition |
|---|---|---|
| `data/pairs/*` + `scripts/prepare_data.py` + the `ohrc_tmc2_*` pairs | OHRC↔TMC-2 was the old headline task and is explicitly deleted from the plan. The pairs are also self-matched, so they cannot evidence anything about illumination | **Keep as a regression fixture only.** Relabel in the manifest as `synthetic`, never quote in results |
| `backend/seleno/pipeline.py` | Superseded by `registration.py`; exists only to serve the legacy pairs | Fold into the new pipeline or keep frozen behind the legacy endpoints |
| `scripts/nac_spike.py` | Its conclusion ("correlation red") was **refuted** by `P0_FINDINGS.md` | Keep for its documented traps (429s, suffix-range, template-size ranking); do not cite its verdict |
| `backend/seleno/illumination.py` `MockTerrain` + `predicted_shadow_mask` | Shadow prediction on synthetic terrain, inapplicable at 89–91° incidence | Keep the `TerrainModel` interface for a real LOLA DEM; do not present Mock results |
| `backend/seleno/subpixel.py` | Measured to make held-out error **worse** (1.41 → 2.36 px) | Keep, off by default, and keep reporting the negative result |
| IIRS→TMC-2→OHRC cascade | Deleted from the plan | Not implemented — nothing to remove |

---

## 8. Code that can be reused

This is the genuinely valuable inheritance, and it is substantial. Reuse and
generalise; do not rewrite.

| Module | Why it is worth keeping |
|---|---|
| **`ohrc/geometry.py`** | Lunar south polar stereographic plane at R = 1 737 400 m, `lonlat_to_south_stereo` / inverse, bilinear lattice interpolation, `footprint_overlap` on occupancy cells. **Already sensor-agnostic and already the same projection the LROC polar mosaics use** — this is the Phase 1 §5 requirement, present and tested (4/4 geometry tests pass) |
| **`ohrc/product.py`** | `numpy.memmap(mode="r")` discipline, bounded-window `read_tile`, per-product dimensions from the label, read-only archive invariants. The right pattern for every large raster in the project |
| **`ohrc/reproject.py`** | Both images onto one common stereographic grid so the residual is near-translation. This is the correct architecture for cross-sensor matching and generalises directly |
| **`matchers.py`** | The `MatchResult` + `run_matcher` seam, and especially `_probe_learned()` reporting unavailability instead of silently substituting. Add RIFT2 / LightGlue behind this interface unchanged |
| **`spatial.py`** | Coverage over *matchable* area, grid/topk/all selection, dispersion. Directly serves the PS's "uniform distribution" requirement and the Phase 4 spatial-coverage metric |
| **`verify.py`** | MAGSAC++ verification, `decompose`, `plausibility` checks on scale/rotation/shear |
| **`scripts/fetch_lroc_pairs.py`** | Chunked ranged GETs with per-chunk retry and backoff; the PDS store URL scheme; the PDS4/PDS3 record arithmetic. **Generalise this into `src/data/fetch.py`** — it is the best of the three fetchers |
| **`data/lroc/manifest.json`** | The only provenance-carrying manifest. Its schema (source, control, GSD, window, warp, GT homography, real vs synthetic components) is the model for `data/manifest.jsonl` |
| **`ohrc/label.py`, `ohrc/sun.py`, `ohrc/tiles.py`** | PDS4 field extraction, defect detection, `.spm` parsing, tile texture/shadow classification. Keep, but move the OHRC-specific parts behind a sensor profile |
| **`results/experiments/P0_FINDINGS.md`** | The ~4 km OHRC-vs-NAC offset, the four corroborating checks, and the two methodological bug fixes (2-point affine fitting noise; "three methods agree" counting one method thrice). **Do not lose this** |
| **`research_by_space.md`** | 1 240 lines of sourced agency methodology with explicit uncertainty flags. Directly informs Phases 3–6 |

---

## 9. Revised project architecture

Target layout, mapped onto what exists:

```
src/
  io/         readers.py       PDS4 / PDS3 / GeoTIFF behind one interface
              ← generalise ohrc/label.py; add pds4_tools + pvl + rasterio backends
  data/       fetch.py         one polite, resumable, SHA-256-verified fetcher
              ← generalise scripts/fetch_lroc_pairs.py
              index.py         footprints.gpkg, polar intersection, pair discovery
              ← reuse ohrc/geometry.py for every intersection
  preprocess/ per-profile normalisation, GSD harmonisation
              ← generalise preprocess.py; parameters move to YAML
  gt/         T1 synthetic renderer (needs LOLA), T3 manual tie-point store
              ← new; T2 (ISIS/ASP/ALE) to be reported feasible or not, not assumed
  match/      sift / orb / phase-congruency(RIFT2) / learned
              ← matchers.py interface unchanged; two new backends
  verify/     MAGSAC++ + plausibility + piecewise/per-segment models
              ← verify.py extended; segment support is new
  register/   warp, georeference, no-data handling
              ← register.py extended with CRS/transform propagation
  metrics/    matcher-independent evaluation, stratification, coverage
              ← metrics.py + spatial.py, with the hold-out fixed

config/sensors/  ohrc.yaml nac.yaml tmc2.yaml selene_tc.yaml iirs.yaml   [new]
api/             job submit / status / results                            [rework of app.py]
frontend/        source select → run → matches → overlay → metrics        [rework]
docs/            AUDIT.md (this) DATA_INVENTORY.md EVAL_PROTOCOL.md
reports/         baselines.md metrics.md gt_qc/
data/            raw/ processed/ index/          (all gitignored)
outputs/<job_id>/ matches.csv registered.tif transform.json metrics.json
                  overlay.png report.md
```

Three architectural decisions that follow from the audit:

1. **Everything meets on one south polar stereographic grid before matching.**
   `ohrc/geometry.py` and the LROC polar mosaics already agree on
   R = 1 737 400 m, centre −90°/0°. `ohrc/reproject.py` already implements the
   pattern. This is the backbone; keep it and make it sensor-agnostic.

2. **The matcher is a plug, and the harness is the product.** `run_matcher` +
   `MatchResult` already enforce this and already refuse to fake availability.
   Phase 5's five baselines all fit behind it without touching the pipeline.

3. **The ~4 km CH-2 geolocation offset is a design input, not a bug.** Any search
   radius, any "no_overlap" decision and any footprint intersection must be sized
   for it; a 384 m window placed by CH-2 geometry shows different ground.

---

## 10. Corrections to the brief's stated assumptions

| Brief said | Actually **[verified]** |
|---|---|
| "OHRC is NOT on this machine" | All four OHRC L2 products are in `~/Downloads/…Bundle.tar`, complete with labels, geometry CSVs, `.spm`/`.oat`/`.lbr` and browse PNGs |
| "TMC-2 may be partially downloaded" | No TMC-2 data of any kind is present |
| "There may be NO model — report what exists" | Correct: there is none |
| "There is currently NO TRAINING DATA" | Correct. But 14 LROC pairs with *real* illumination change and independent ground truth do exist, and are the strongest evaluation asset in the repo |
| "Existing OHRC acquisition path — reuse it" | The path that produced the working dataset was a **manual SFTP delivery**, not a script. `scripts/fetch_products.py` targets a *different*, older product set from an Internet Archive mirror |
| Phase 1 goal "reference side is missing" | Partly present: one LROC controlled polar mosaic tile (`P892S2250`) across 6 illumination bins is already cached locally (107 MB) |

---

## 11. Immediate next actions (Phase 1 entry)

1. Extract the OHRC bundle to `data/raw/ch2/ohrc/` and export `SELENO_OHRC_ROOT`;
   this alone turns 24 skipped tests into real ones. ≈ 4.6 GB, disk is fine.
2. Install the Phase-1 dependency set (`rasterio pyproj shapely pyogrio geopandas
   pds4_tools pvl scikit-image`) — all resolve on cp314; avoid `fiona`.
3. Acquire zero-auth reference terrain first (Phase 1b): LOLA south polar DEM and
   the LROC WAC global mosaic. Neither needs credentials and both unblock
   synthetic GT, which is the only route to a dense illumination benchmark.
4. Build `data/index/footprints.gpkg` with all intersections in the polar
   stereographic plane, reusing `ohrc/geometry.py`.
5. Report SELENE/Kaguya TC access honestly — investigate DARTS, and if it is
   impractical, say so rather than substituting a sensor.
