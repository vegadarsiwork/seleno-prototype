# Session handoff — 2026-09-09

State of the work at the end of the OHRC integration session. This is the
*continuation* doc; `HANDOFF.md` is the project/presentation brief.

Nothing is committed. `git status` in `prototype/` shows 15 new paths, 12
modified, 2 deleted (the two superseded `frontend/dist/assets/*` bundles).

---

## 1. What was done

Phases 1–10 of the brief were all touched. The existing prototype was extended,
not replaced: `pipeline.py`, `store.py`, `preprocess.py`, `verify.py`,
`register.py`, `metrics.py`, `viz.py`, `geodesy.py` and the six legacy manifest
pairs are all still in place and still work.

### New modules

| Path | Purpose |
|---|---|
| `backend/seleno/ohrc/label.py` | PDS4 label parsing + defect detection |
| `backend/seleno/ohrc/product.py` | product discovery, `numpy.memmap` tiles, thumbnails |
| `backend/seleno/ohrc/geometry.py` | geometry lattice + south polar stereographic plane |
| `backend/seleno/ohrc/sun.py` | `.spm` ancillary Sun series (per-line interpolation) |
| `backend/seleno/ohrc/tiles.py` | tile index, illumination classes, DN profiles |
| `backend/seleno/ohrc/reproject.py` | common-grid reprojection + `coarse_align` |
| `backend/seleno/alignment.py` | cached inter-product offset |
| `backend/seleno/pairs.py` | `PairSpec` — OHRC windows or legacy pairs, one interface |
| `backend/seleno/illumination.py` | `TerrainModel`, observed + predicted masks |
| `backend/seleno/subpixel.py` | local-NCC sub-pixel refinement |
| `backend/seleno/registration.py` | the 9-stage pipeline + `RegistrationResult` |
| `backend/seleno/experiments.py` | presets + the 10 ablation arms |
| `backend/seleno/theme.py` | colour tokens shared with the frontend |
| `tests/test_ohrc_dataset.py` | 28 read-only dataset-layer tests |
| `scripts/inspect_dataset.py` | Phase-2 dataset visualisation |
| `scripts/run_registration.py` | one pair, one configuration |
| `scripts/illumination_experiment.py` | the primary experiment + ablation |

### Modified

`backend/app.py` (new `/api/ohrc/*`, `/api/register`, `/api/compare`; legacy
endpoints preserved), `backend/seleno/spatial.py` (matchable-area coverage
denominator, `grid`/`topk`/`all` selection modes, sparse-set guard), all four
docs, and the whole frontend (`App.jsx`, `components.jsx`, `api.js`,
`index.css`).

---

## 2. Verified working

- `python tests/test_ohrc_dataset.py` → **28 passed, 0 failed, 0 skipped**
- `python run.py` → UI 200, `/api/health` reports 4 OHRC products
- All 15 API endpoints return 200 (swept individually)
- All six legacy pairs run; ground-truth errors equal or better than before
  (`ohrc_crater_field` 0.163 → 0.071 px; `ohrc_low_illumination` 0.288 → 0.138 px);
  `ohrc_disjoint_orbits` still refused; `ohrc_tmc2_extreme` now correctly
  downgraded to *warning* on held-out error
- The headline demo works both ways through the API on window (5888, 88832):
  with the coarse offset → 73.0 % inlier ratio, 1.33 px held-out, **warning**
  (31 % frame overlap); without it → 5/60 inliers, **refused** with 3 reasons
- Python syntax check clean across `backend/`, `scripts/`, `tests/`, `run.py`
- `frontend` builds (`npm run build`, ~290 ms)

---

## 3. Open items, in priority order

### 3.1 `.gitignore` edit was rejected — needs a decision

The current rule `results/*/` **ignores `results/dataset/` and
`results/illumination/`**, but `HANDOFF.md` and `README.md` both cite
`results/illumination/ILLUMINATION.md` as "the committed figures". Either commit
those results or reword the docs. The edit I proposed (rejected) was:

```
data/raw/
results/dataset/tileindex_*.json      # caches, reproducible
results/dataset/_*.png
results/illumination/_*.png
results/showcase/**/_check.png
```

i.e. track the tables, figures, summaries and the tiny alignment cache; ignore
the tile-index caches and scratch images. Committing
`results/illumination/alignment/*.json` is worth it — it makes the first demo run
instant instead of ~50 s.

### 3.2 ~~The final experiment re-run may not have landed~~ — RESOLVED

It landed at 17:04:24 on 2026-09-09, after the `overlap_fraction` check was
added. `results/illumination/ILLUMINATION.md` is current.

Aggregate figures are **unchanged** (the "accepted" column counts
`status != "refused"`, and `warning` is not `refused`). Per-window arm E is now
2 accepted / 3 warning, the three warnings all being the frame-overlap check
firing. Docs need no numeric re-sync.

### 3.3 A server may still be listening on :8000

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | % { Stop-Process -Id $_.OwningProcess -Force }
```

### 3.4 Not done from the brief

- **Phase 6 real terrain.** `TerrainModel` interface, mask plumbing, uncertainty
  feathering and a `MockTerrain` are all implemented; **no DEM is present** and
  `NullTerrain.available` is `False`. Real ray-casting is marked pending
  everywhere it appears.
- **Phase 8 calibration.** The refusal decision is rule-based and transparent;
  `thresholds_calibrated` is `false` in the API and the UI says so. No
  precision/recall measured.
- **UI not opened in a browser.** The Chrome extension was not connected this
  session, so the frontend was verified through its endpoints and by inspecting
  rendered PNGs, not visually. **Open it once before demoing.**
- MD5 checksums in the labels never verified against the 4.6 GB of imagery.

---

## 4. The three findings to not lose

1. **The delivered geolocation of the two 2024 products disagrees by 538 m**
   (≈2 244 px at 0.24 m/px). Correcting it moves the baseline from 2–3 inliers at
   4–8 % to >1 300 inliers at 56–60 %. Without this, "illumination defeats SIFT"
   is the wrong conclusion sitting in plain sight.
2. **Two of our own stages make things worse on this data.** Spatial grid
   selection: held-out 1.41 → 1.71 px. Sub-pixel refinement: 1.41 → 2.36 px, and
   tightening its acceptance only converges back toward not refining. Both are
   now off in the `seleno` preset with the finding recorded.
3. **Self-consistency cannot detect a partially-supported fit.** One window gave
   73 % inliers, 1.22 px reprojection RMSE, 1.33 px held-out and 0.89 overlap
   NCC — with the warped source covering **31 %** of the reference frame and a
   fitted rotation 4.9° off the delivered geometry. This is why
   `overlap_fraction` became a check, and it is the sharpest argument for the
   refusal layer.

Also worth keeping: a local horizon ray-cast is **inapplicable** to three of the
four products (mid-strip solar elevation ≤ 0 yet lit terrain present, because
high ground near the pole is lit over a depressed horizon). The prediction
declares itself inapplicable below 0.05° rather than emitting an
everything-is-shadowed mask. Doing it properly needs wide-area horizon angles
over a real DEM — a different algorithm, and the single most valuable next step.

---

## 5. Reproduce commands

```bash
cd prototype

# dataset layer
python tests/test_ohrc_dataset.py
python scripts/inspect_dataset.py --quick        # add nothing for the full pass

# baseline (Phase 4)
python scripts/run_registration.py --a 20241115T1326 --b 20241115T1525 \
    --window 5888 88832 --size 1024 --preset baseline_verified

# the recommended configuration
python scripts/run_registration.py --a 20241115T1326 --b 20241115T1525 \
    --window 5888 88832 --size 1024 --preset seleno

# the headline finding, live
python scripts/run_registration.py --a 20241115T1326 --b 20241115T1525 \
    --window 5888 88832 --size 1024 --preset seleno --no-offset

# the experiment + full 10-arm ablation
python scripts/illumination_experiment.py --windows 5 --size 1024

# legacy regression
python scripts/run_registration.py --legacy ohrc_crater_field --preset seleno
python scripts/benchmark.py

# the demo
python run.py            # http://127.0.0.1:8000
```

Dataset root defaults to `../datasettesting/dataset`; override with
`SELENO_OHRC_ROOT`.
