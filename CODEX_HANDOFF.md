# Handoff to Codex — 2026-09-25

This picks up from a Claude Code session that continued Codex's own unfinished
"Fix sub-pixel accuracy" work. **Everything below is uncommitted in the working
tree.** Read "Rules" first: two earlier runs took down the user's terminal.

## Rules (non-negotiable)

1. **Memory.** The machine has 15 GB of RAM. `/tmp` is **tmpfs, i.e. RAM**.
   Two uncapped runs made the kernel OOM-kill the user's terminal.
   - Run every registration or suite inside a cap:
     `systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 .venv/bin/python ...`
   - Run one heavy job at a time.
   - Write outputs under `outputs/` (on disk, git-ignored), never under `/tmp`.
2. **Threads.** Always set `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`.
3. **The sealed test fold.** Nothing may fit, select or tune on it:
   - Model terms are chosen by spatial cross-validation over fit cells only.
   - ECC and the balanced refit are adopted on the validation fold only.
   - The check-point screen uses the coarse model, never the exported model.
4. **Don't commit** without asking the user.

## What changed (and why)

| Area | Change | Files |
|---|---|---|
| Test regressions | Codex broke 3 of 35 `tests/test_tool.py` cases. All fixed: 35/35 at last full run | `verify.py`, `register.py` |
| Coarse acceptance | a-contrario **NFA** significance test per candidate (`V.log10_nfa`) + **`V.degenerate`** (fold, scale, anisotropy, area-scale variation over the whole overlap). Unrelated images → `degenerate_transform` | `verify.py`, `register.py` plan loop |
| Coarse dense tie points | `_lattice_tiepoints`: seeds on a lattice inside the overlap, measured by `refine_correspondences` around the locked translation. Falls back to `grid_tiepoints` | `register.py` |
| Fine stage (rewritten) | Lattice seeds over the whole predicted overlap (`_seed_lattice`, ~6000 seeds). Tiles always at native resolution (`native` flag now honest). Probe on tiles holding the most fit-fold seeds. **Outward growth**: weak tiles are retried with an affine refit on fit-fold points | `register.py: _fine_stage` |
| Measurement | `refine_correspondences`: shrinks the patch near borders/mask edges instead of dropping seeds | `methods.py` |
| Dense model fit | `_fit_dense`: loose RANSAC → model-free neighbour consistency (`_consistent`) → cell-balanced LSQ. Then optional **terrain parallax** and a **B-spline local field**, each chosen by CV over fit cells | `register.py`, `local_model.py`, `terrain.py` |
| Terrain parallax | `terrain.py`: LOLA DEM height × fitted coefficient (inverse side). Adopted on OHRC→NAC: CV 1.18 → 0.30 working px | `terrain.py`, `warp_model.py` |
| Model application | `transform.json` carries `local_field` and `parallax` (DEM sampler spec). Everything goes through `warp_model.inverse_points` (export, scoring, preview). `conjugate` handles both | `warp_model.py` |
| Split | Held-out cells laid along the overlap's principal axes when the overlap is elongated (`_overlap_frame`); whole frame if the overlap fills ≥60% of it. Square cells (`_square_cells`) | `evaluation.py`, `register.py` |
| Evaluation | **Check points**: sealed test correspondences that agree with their held-out neighbours (coarse-model displacements). Scored as `check_point_*` beside the unfiltered `held_out_*`. Headline `rmse_px` / `rmse_source_px` / acceptance use check points; `check_point_fraction` ≥ 0.8 is a gate | `evaluation.py`, `register.py` |
| Export | Cropped to the registered footprint (`window=`). k×k supersampling via local Jacobian when the source is finer. Writes all bands (IIRS 256) | `export.py` |
| Memory | `LazyRaster` strided read no longer keeps whole blocks alive (WAC load 3+ GB → ~290 MB). `_available_bytes` respects cgroup limits minus reclaimable page cache | `scene.py`, `register.py` |
| Geometry lattice | See below | `register.py` |
| UI | Check points and local-field rows in the accuracy panel | `frontend/src/ToolView.jsx` |
| Tests | New `tests/test_dense_model.py` (9 tests). `test_validation_fixes` updated for the check-point headline | `tests/` |

Geometry lattice changes:
- Per-axis subsampling that keeps the last row and column. It used to drop the
  last 12% of the TMC-2 strip.
- `_LatticeInverse`: a Newton solve on the bilinear forward lattice, replacing
  the Delaunay inverse. Delaunay slivers mapped a ~1 km band at strip edges to
  column 3999, which showed up as streaks.
- `_Extrapolating`: rim extrapolation within 2 node spacings.

## Results so far (dev runs; held-out check points, source px)

| Pair | Sep-23 baseline RMSE | Best now | Notes |
|---|---:|---:|---|
| OHRC → NAC | 106.5 | **3.0–3.3** (0.78–0.88 NAC px) | terrain parallax + field; patch 61 best |
| TMC-2 → SELENE (evening map) | 12.3 | **2.27** (median 1.37) | default settings best |
| IIRS 2025 → WAC | 12.7 | **1.62** (median 1.16) | patch 61; run **before** the lattice Newton fix, rerun needed |
| IIRS 2024 → WAC | 24.4 | **1.49** (median 0.85, 58% ≤ 1 px) | default settings, after the lattice fix |

Sub-*source*-pixel is not reached. On OHRC it is capped by the reference: one
NAC pixel is 4.17 OHRC pixels.

Measurement-setting sweep (`outputs/dev_20260925/sweep_*.log`, columns: patch,
min_peak, forward_backward):
- **No single setting wins everywhere.** Keep the defaults (41, 0.55, 0.35).
- **Relaxing min_peak/fb** is catastrophic on OHRC (RMSE 23–30) and slightly
  worse on TMC-2 and IIRS 2024.
- **Patch 61** is best for OHRC (4.17× scale gap) and, pre-fix, for IIRS 2025.
  It's worse for TMC-2 and IIRS 2024.

## Pending — do in this order

1. **Run every test suite.** The newest changes are untested: `_LatticeInverse`,
   `_Extrapolating`, `_lattice_tiepoints`, outward growth, terrain.
   ```bash
   for t in test_native_export test_matcher_refinement test_evaluation_acceptance test_validation_fixes test_dense_model; do
     systemd-run --user --scope -q -p MemoryMax=3G .venv/bin/python -m unittest tests.$t; done
   systemd-run --user --scope -q -p MemoryMax=3G .venv/bin/python tests/test_tool.py   # expect 35/35
   systemd-run --user --scope -q -p MemoryMax=3G .venv/bin/python tests/test_ohrc_dataset.py
   ```
   Fix what breaks. Add unit tests for:
   - `_LatticeInverse`: a curved-edge lattice must not return the edge column
     outside the image.
   - `terrain.HeightSampler` / `terrain.term`, including frame conjugation.
   - `_lattice_tiepoints`.
2. **Choose the correlation patch per pair, not globally.** Either:
   - make it a sensor-profile option (`config/sensors/ohrc.yaml`: fine patch 61), or
   - better: pick it by the same fit-cell cross-validation, measuring a probe
     tile at 41 and 61.

   First rerun IIRS 2025 with current code
   (`outputs/dev_20260925/tools/sweep.py iirs2025 <out> '[{}, {"patch": 61}]'`)
   to see whether the lattice fix changed the answer.
3. **Reference for TMC-2.** Keep the **evening** SELENE map (`TCO_MAPe04`) as the
   primary case. The morning map is low-sun (heavy shadows); TMC-2 is high-sun
   (51°), and the morning map registers worse (coverage 9%). In
   `scripts/validate_registration.py` the case `tmc2_selene_morning` is listed
   first. Keep it only as a secondary cross-illumination case, or drop it.
4. **Evaluation edge case.** If any held-out prediction is NaN,
   `error_statistics` returns all-None stats, which hides the numbers. Keep
   failing acceptance when `invalid_n > 0`, but also report stats over the
   valid predictions. It is rarer now with `_Extrapolating`.
5. **Silence the export warning.** `np.nanmedian` on all-NaN tiles in
   `export.py` (the supersampling estimate) warns.
6. **Full validation, capped, outputs on disk.**
   ```bash
   systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0 \
     .venv/bin/python -u scripts/validate_registration.py --suite real \
     --out reports/validation_20260925 --artifacts outputs/validation_20260925
   ```
   Then `--suite synthetic` and `--suite illumination` (14 LROC NAC pairs with
   known homographies; report corner error vs truth, stratified by sun-angle
   difference). The 256-band IIRS export makes each IIRS case ~2.5 min and a
   ~1.4 GB `registered.tif`.
   - `reports/validation_20260925/` currently holds Codex's earlier synthetic
     outputs; replace them with the fresh run.
   - Use `outputs/dev_20260925/tools/summ.py <real.json>` to summarise and
     `tools/diag.py <job_dir>` for per-fold residual structure.
7. **Write the report.** `reports/validation_20260925/REPORT.md`, same style as
   `reports/validation_fixes_20260923/REPORT.md`, plus a regenerated deck table.
   Cover:
   - what changed
   - the per-pair table (check-point RMSE/median/p90/≤1 px in source, reference
     and metres; coverage; extrapolation; model terms adopted)
   - the sweep conclusion
   - honest limits
8. **Update the docs** (`docs/TOOL.md`):
   - §3 accuracy: check points, NFA, the dense model, cross-validated terms.
   - §4 distribution: lattice seeds; `matches.csv` quota vs `matches_all.csv`;
     `tiepoints.npz`.
   - §5 long strips: the field and terrain term replace segments when adopted.
   - A new terrain-parallax section.
   - §6 memory: cgroup-aware budget.
   - §1 artifacts: `matches_all.csv`, `tiepoints.npz`, the `transform.json`
     fields `local_field` and `parallax`.

   Also refresh the numbers in `SESSION_PPT_HANDOFF.md` from the final run.
9. **Ask the user before committing.** Suggested split:
   1. significance + degeneracy checks
   2. lattice fine stage + measurement
   3. dense model + local field
   4. terrain parallax
   5. split + check-point evaluation
   6. export crop/supersample + memory fixes
   7. geometry-lattice fixes
   8. UI
   9. tests
   10. reports/docs

## Known limits (state them, don't hide them)

- OHRC→NAC sub-source-pixel is limited by NAC sampling (1 NAC px = 4.17 OHRC px).
- The terrain term needs a DEM. Only LOLA polar DEMs are in `data/raw/lola`;
  equatorial TMC-2/IIRS scenes have none.
- Large sun-azimuth changes (90–180°) remain hard. The LROC illumination
  suite measures this.
- Real pairs are scored on held-out correspondences, not surveyed control
  points. `scripts/validate_registration.py --ground-truth-directory` accepts
  external control manifests.
