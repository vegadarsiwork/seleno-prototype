"""Diagnostic pass over the four real runs and the two positive controls.

Measurement only: reads each run's saved artifacts, changes nothing in the
pipeline and feeds nothing back into it.

  1. controls: recovered transform vs the known one (from controls.json)
  2. residual field: held-out residual VECTORS, bias, coherence, plots
  3. matcher-independent accuracy: local shift between registered.tif and the
     reference working grid, gradient magnitude, frozen TEST cells only
  4. held-out thresholds in reference px alongside source px

    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u \
      reports/validation_fixes_20260923/diagnostics/analyse.py
"""
import json
import math
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import evaluation as E
from seleno.tool import warp_model as WM
from seleno.tool.coordinates import grid_to_reference, project
from seleno.tool.register import _boxcar_decimate
from seleno.tool.scene import load

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "residuals"
WINDOW, FALLBACK_WINDOW, MIN_WINDOWS = 64, 48, 20
NCC_MARGIN = 8          # NCC search radius, working px
AGREE_PX = 0.5          # phase correlation and NCC agree within this, working px
NEAR_EXCLUDE, NEAR_RADIUS, NEAR_MIN = 8.0, 40.0, 5   # working px; neighbour field


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def stats(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0, "median": None, "p90": None, "rmse": None, "min": None, "max": None}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "rmse": float(np.sqrt(np.mean(v ** 2))), "min": float(v.min()), "max": float(v.max())}


def in_units(s, u):
    """A stats dict in working px -> the same in reference px, source px, metres."""
    out = {}
    for unit, k in (("working_px", 1.0), ("reference_px", u["step"]),
                    ("source_px", u["step"] * u["spr"] if u["spr"] else None),
                    ("m", u["step"] * u["mpr"] if u["mpr"] else None)):
        out[unit] = {key: (None if (val is None or k is None or key == "n") else val * k)
                     for key, val in s.items() if key != "n"}
    out["n"] = s["n"]
    return out


def split_from(ev):
    s = ev["split"]
    split = E.SpatialSplit.__new__(E.SpatialSplit)
    split.shape, split.grid, split.seed = tuple(s["shape"]), s["grid"], s["seed"]
    split.folds = np.asarray(s["cell_folds"], np.uint8)
    return split


def reference_grid(ref_path, frame):
    """The reference working grid exactly as `_target_grid` built it for the run."""
    R = load(str(ref_path))
    h, w = R.array.shape[:2]
    step = int(frame["reference_decimation"])
    r0, c0 = (int(v) for v in frame["reference_origin"])
    win = frame.get("reference_window") or [0, 0, h, w]
    r1, c1 = min(h, int(win[2])), min(w, int(win[3]))
    sub, centre = _boxcar_decimate(R.array, r0, r1, c0, c1, step, R.nodata,
                                   R.meta_scale, R.meta_offset)
    assert list(sub.shape) == list(frame["target_shape"]), (sub.shape, frame["target_shape"])
    assert np.allclose(centre, frame["reference_sample_offset"]), (centre, frame["reference_sample_offset"])
    return sub, np.isfinite(sub) & (sub > -1e30)


def gradmag(img, valid):
    x = np.where(valid, img, np.nan).astype(np.float32)
    lo, hi = np.nanpercentile(x, [2, 98])
    x = np.nan_to_num((x - lo) / max(hi - lo, 1e-6), nan=0.5).astype(np.float32)
    x = cv2.GaussianBlur(x, (0, 0), 1.0)
    g = np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3))
    return g, cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


HANN = {}


def shift_phase(ref_w, reg_w):
    """Upsampled-DFT phase correlation (1/50 px), Hann-windowed. cv2.phaseCorrelate's
    centroid sub-pixel step measured 2x the median error on the calibration below."""
    n = ref_w.shape[0]
    if n not in HANN:
        HANN[n] = cv2.createHanningWindow((n, n), cv2.CV_32F)
    shift, err, _ = phase_cross_correlation(ref_w * HANN[n], reg_w * HANN[n],
                                            upsample_factor=50, normalization="phase")
    return -np.asarray(shift, float)[::-1] * PHASE_SIGN, float(err)


def shift_ncc(ref_w, reg_w):
    m = NCC_MARGIN
    tpl = reg_w[m:-m, m:-m].astype(np.float32)
    res = cv2.matchTemplate(ref_w.astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
    py, px = np.unravel_index(np.argmax(res), res.shape)
    peak = float(res[py, px])
    edge = px in (0, res.shape[1] - 1) or py in (0, res.shape[0] - 1)

    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    sx = px + (0 if edge else para(res[py, px - 1], res[py, px], res[py, px + 1]))
    sy = py + (0 if edge else para(res[py - 1, px], res[py, px], res[py + 1, px]))
    return np.array([sx - m, sy - m]) * NCC_SIGN, peak, edge


def calibrate_signs(trials=200):
    """Fix both estimators' sign so a shift is (registered position - reference
    position) of the same ground, the convention of a forward residual; then
    measure their error on known shifts of noisy gradient-magnitude windows."""
    global PHASE_SIGN, NCC_SIGN
    PHASE_SIGN = NCC_SIGN = 1.0
    rng = np.random.default_rng(3)

    def pair(d, sigma, noise):
        base = cv2.GaussianBlur(rng.normal(size=(200, 200)).astype(np.float32), (0, 0), sigma)
        moved = cv2.warpAffine(base, np.float32([[1, 0, d[0]], [0, 1, d[1]]]), (200, 200),
                               flags=cv2.INTER_CUBIC)     # moved(x) = base(x - d): ground moved by +d
        n = noise * base.std()
        ga = gradmag(base + rng.normal(0, n, base.shape).astype(np.float32), np.ones((200, 200), bool))[0]
        gb = gradmag(moved + rng.normal(0, n, base.shape).astype(np.float32), np.ones((200, 200), bool))[0]
        return ga[68:132, 68:132], gb[68:132, 68:132]
    d = np.array([1.3, -0.7])
    a, b = pair(d, 2.0, 0.0)
    PHASE_SIGN = float(np.sign(shift_phase(a, b)[0][0] / d[0]))
    NCC_SIGN = float(np.sign(shift_ncc(a, b)[0][0] / d[0]))
    ep, en = [], []
    for _ in range(trials):
        d = rng.uniform(-4, 4, 2)
        a, b = pair(d, rng.uniform(1, 3), 0.1)
        ep.append(np.hypot(*(shift_phase(a, b)[0] - d)))
        en.append(np.hypot(*(shift_ncc(a, b)[0] - d)))
    cal = {"trials": trials, "shifts": "uniform +-4 working px, 64 px windows, noise 10% of texture std",
           "phase_error_working_px": stats(ep), "ncc_error_working_px": stats(en)}
    assert cal["phase_error_working_px"]["median"] < 0.15 and cal["ncc_error_working_px"]["median"] < 0.15, cal
    return cal


# --------------------------------------------------------------------------- #
# per run
# --------------------------------------------------------------------------- #

def analyse(run, truth=None):
    art = ROOT / run["artifacts"]
    model = json.loads((art / "transform.json").read_text())
    ev = json.loads((art / "evaluation.json").read_text())
    metrics = json.loads((art / "metrics.json").read_text())
    frame = model["frame"]
    conv = metrics["accuracy"]["units"]
    u = {"step": float(conv["reference_decimation"]), "spr": conv["source_px_per_reference_px"],
         "mpr": conv["metres_per_reference_px"]}
    split = split_from(ev)
    src, ref = np.array(ev["source"], float), np.array(ev["reference"], float)
    out = {"pair": run["pair"], "name": run["name"], "artifacts": run["artifacts"],
           "units": conv, "status": metrics["status"]}

    # ---- 4. held-out thresholds (unfiltered, same set as the RMSE) ----------
    e = WM.residuals(model, src, ref)
    assert float(np.sqrt(np.mean(e ** 2))) == metrics["accuracy"]["rmse_px"]
    e_ref, e_src = e * u["step"], e * u["step"] * u["spr"]
    out["held_out"] = {
        "n": int(e.size), "stats": in_units(stats(e), u),
        # residual below 1e-6 working px: the point lies exactly on the exported model
        "exactly_on_model": int((e < 1e-6).sum()),
        "within_reference_px": {str(k): float(np.mean(e_ref <= k)) for k in (1, 2, 3)},
        "within_source_px": {str(k): float(np.mean(e_src <= k)) for k in (1, 2, 3)}}

    # ---- 2. residual field ---------------------------------------------------
    fwd = (WM.forward_points(model, src) - ref) * u["step"]       # reference px vectors
    mag = np.hypot(fwd[:, 0], fwd[:, 1])
    unit = fwd / np.maximum(mag[:, None], 1e-12)
    bias = fwd.mean(axis=0)
    cells = split.cells(src)
    per_cell = []
    for c in np.unique(cells):
        k = cells == c
        mv = fwd[k].mean(axis=0)
        per_cell.append({"cell": int(c), "n": int(k.sum()), "mean_ref_px": mv.tolist(),
                         "median_abs_ref_px": float(np.median(mag[k])),
                         "coherence": float(np.hypot(*unit[k].mean(axis=0)))})
    # Systematic vs random: predict each residual from held-out neighbours that are
    # NEAR_EXCLUDE..NEAR_RADIUS working px away (never itself or points close
    # enough to share a matching patch). Matcher noise is independent between such
    # neighbours; a model or coordinate error is not.
    tree = cKDTree(ref)
    pred = np.full_like(fwd, np.nan)
    for i, nb in enumerate(tree.query_ball_point(ref, NEAR_RADIUS)):
        nb = [j for j in nb if np.hypot(*(ref[j] - ref[i])) >= NEAR_EXCLUDE]
        if len(nb) >= NEAR_MIN:
            pred[i] = np.median(fwd[nb], axis=0)
    ok = np.isfinite(pred[:, 0])
    decomposition = {
        "definition": "each held-out residual vector predicted by the median of held-out "
                      "neighbours %g-%g working px away (>= %d of them)" % (NEAR_EXCLUDE, NEAR_RADIUS, NEAR_MIN),
        "n": int(ok.sum()),
        "median_abs_residual_ref_px": float(np.median(mag[ok])) if ok.any() else None,
        "systematic_median_abs_ref_px": float(np.median(np.hypot(*pred[ok].T))) if ok.any() else None,
        "random_median_abs_ref_px": float(np.median(np.hypot(*(fwd[ok] - pred[ok]).T))) if ok.any() else None}
    close = e_ref <= 3.0
    out["residual_field"] = {
        "definition": "forward residual model(src) - ref of every held-out correspondence, reference px",
        "bias_mean_ref_px": bias.tolist(), "bias_mean_abs_ref_px": float(np.hypot(*bias)),
        "bias_component_median_ref_px": np.median(fwd, axis=0).tolist(),
        "median_abs_ref_px": float(np.median(mag)),
        "bias_over_median": (float(np.hypot(*bias) / np.median(mag)) if np.median(mag) > 1e-6 else None),
        "direction_coherence": float(np.hypot(*unit.mean(axis=0))),
        "direction_coherence_meaning": "length of the mean unit vector: 0 = directions uniformly random, 1 = all identical",
        "per_test_cell": per_cell,
        "decomposition": decomposition,
        "correction_at_held_out_ref_px": stats(np.hypot(*((WM.forward_points(model, src) - src) * u["step"]).T)),
        "within_3_ref_px_partition": {
            "note": "diagnostic partition only; headline numbers above are unfiltered",
            "fraction": float(close.mean()),
            "bias_mean_ref_px": (fwd[close].mean(axis=0).tolist() if close.any() else None),
            "median_abs_ref_px": (float(np.median(mag[close])) if close.any() else None),
            "direction_coherence": (float(np.hypot(*unit[close].mean(axis=0))) if close.any() else None),
            "outside_median_abs_ref_px": (float(np.median(mag[~close])) if (~close).any() else None)}}

    # ---- 3. matcher-independent local shift ---------------------------------
    B, Bv = reference_grid(run["reference"], frame)
    with rasterio.open(art / "registered.tif") as ds:
        A = ds.read(1).astype(np.float32)
        reg_transform, reg_crs = ds.transform, ds.crs
    Av = np.isfinite(A)
    assert A.shape == B.shape
    ga, gav = gradmag(A, Av)
    gb, gbv = gradmag(B, Bv)
    both = gav & gbv

    def windows(size):
        found = []
        H, W = B.shape
        for y in range(0, H - size + 1, size):
            for x in range(0, W - size + 1, size):
                if not both[y:y + size, x:x + size].all():
                    continue
                pts = np.array([[x, y], [x + size - 1, y], [x, y + size - 1],
                                [x + size - 1, y + size - 1], [x + size / 2, y + size / 2]], float)
                if (split.labels(WM.inverse_points(model, pts)) == split.TEST).all():
                    found.append((y, x))
        return found
    size = WINDOW
    wins = windows(size)
    if len(wins) < MIN_WINDOWS:
        size = FALLBACK_WINDOW
        wins = windows(size)
    rows, flat = [], 0
    for y, x in wins:
        rw, aw = gb[y:y + size, x:x + size], ga[y:y + size, x:x + size]
        if rw.std() < 1e-6 or aw.std() < 1e-6:
            flat += 1                  # no gradient at all on one side: nothing to measure
            continue
        dp, resp = shift_phase(rw, aw)
        dn, peak, edge = shift_ncc(rw, aw)
        rows.append({"y": y, "x": x, "centre": [x + (size - 1) / 2, y + (size - 1) / 2],
                     "phase": dp.tolist(), "phase_error_estimate": resp, "ncc": dn.tolist(),
                     "ncc_peak": peak, "ncc_at_search_edge": bool(edge)})
    dp = np.array([r["phase"] for r in rows]).reshape(-1, 2)
    dn = np.array([r["ncc"] for r in rows]).reshape(-1, 2)
    agree = (np.hypot(*(dp - dn).T) <= AGREE_PX) & ~np.array([r["ncc_at_search_edge"] for r in rows], bool)
    mp, mn = np.hypot(*dp.T), np.hypot(*dn.T)
    # the two independent measurements at the same place: local shift vs the
    # median held-out residual vector inside the same window
    pair_rows = []
    for k, r in enumerate(rows):
        inside = ((ref[:, 0] >= r["x"]) & (ref[:, 0] < r["x"] + size)
                  & (ref[:, 1] >= r["y"]) & (ref[:, 1] < r["y"] + size))
        if inside.sum() >= 3:
            pair_rows.append((dp[k] * u["step"], np.median(fwd[inside], axis=0), bool(agree[k])))
    out["local_shift"] = {
        "definition": "shift of registered.tif relative to the reference working grid, "
                      "gradient magnitude, %d px windows fully inside frozen TEST cells "
                      "(corners and centre mapped back to source working coordinates); "
                      "measured on the working grid, one working px = %g reference px"
                      % (size, u["step"]),
        "window_px": size, "windows": len(rows), "flat_windows_skipped": flat,
        "phase_all": in_units(stats(mp), u),
        "ncc_all": in_units(stats(mn), u),
        "phase_bias_ref_px": (dp.mean(axis=0) * u["step"]).tolist() if len(rows) else None,
        "methods_agree_fraction": float(agree.mean()) if len(rows) else None,
        "phase_where_methods_agree": in_units(stats(mp[agree]), u),
        "vs_held_out": {
            "definition": "windows holding >= 3 held-out points: |local shift - median held-out residual vector|, reference px",
            "windows": len(pair_rows),
            "abs_difference_ref_px": stats([np.hypot(*(a - b)) for a, b, _ in pair_rows]),
            "local_shift_ref_px": stats([np.hypot(*a) for a, _, _ in pair_rows]),
            "held_out_ref_px": stats([np.hypot(*b) for _, b, _ in pair_rows]),
            "where_methods_agree": {
                "windows": sum(1 for *_, g in pair_rows if g),
                "abs_difference_ref_px": stats([np.hypot(*(a - b)) for a, b, g in pair_rows if g])}},
        "per_window": rows}

    # ---- (c) export georeference: registered.tif's grid vs the working grid --
    try:
        with rasterio.open(run["reference"]) as ds:
            ind = np.full(B.shape, np.nan, np.float32)
            reproject(rasterio.band(ds, 1), ind, src_nodata=ds.nodata if ds.nodata is not None else 0,
                      dst_transform=reg_transform, dst_crs=reg_crs, dst_nodata=np.nan,
                      resampling=Resampling.average)
        gi, giv = gradmag(ind, np.isfinite(ind) & (ind != 0))
        ok = giv & gbv
        shifts = []
        H, W = B.shape
        for y in range(0, H - 64 + 1, 64):
            for x in range(0, W - 64 + 1, 64):
                if ok[y:y + 64, x:x + 64].all():
                    shifts.append(shift_phase(gb[y:y + 64, x:x + 64], gi[y:y + 64, x:x + 64])[0])
        shifts = np.array(shifts).reshape(-1, 2)
        out["export_georeference"] = {
            "definition": "reference resampled by GDAL onto registered.tif's declared grid, "
                          "vs the pipeline's own reference working grid; 0 = the exported "
                          "georeference describes the grid the model was fitted on",
            "windows": int(len(shifts)),
            "median_shift_working_px": np.median(shifts, axis=0).tolist() if len(shifts) else None,
            "median_abs_working_px": float(np.median(np.hypot(*shifts.T))) if len(shifts) else None}
    except Exception as exc:                                          # noqa: BLE001
        out["export_georeference"] = {"error": "%s: %s" % (type(exc).__name__, exc)}

    # ---- 1. controls: against the known transform ---------------------------
    if truth is not None:
        G = grid_to_reference(frame)
        Tinv = np.asarray(truth["T_inverse_native_ref_px"], float)
        Htrue = np.linalg.inv(G) @ Tinv @ G           # source working -> reference working
        yy, xx = np.nonzero(Av)
        pick = np.arange(0, yy.size, max(1, yy.size // 200000))
        v = np.column_stack([xx[pick], yy[pick]]).astype(float)
        w = WM.inverse_points(model, v)
        err = np.hypot(*(v - project(Htrue, w)).T)                # working px
        test = split.labels(w) == split.TEST
        Hn = np.asarray(model["matrix_reference_px"], float)

        def params(M):
            return {"scale": float(math.sqrt(abs(np.linalg.det(M[:2, :2])))),
                    "rotation_deg": float(math.degrees(math.atan2(M[1, 0], M[0, 0])))}
        centre = np.array(truth["rotation_scale_centre_ref_px"] + [1.0])
        true_model = {"matrix": Htrue.tolist()}
        e_true = WM.residuals(true_model, src, ref)
        # measured local shift vs the true misregistration at each window centre
        if rows:
            cv = np.array([r["centre"] for r in rows], float)
            d_true = cv - project(Htrue, WM.inverse_points(model, cv))
            meas_err = np.hypot(*(dp - d_true).T)
        else:
            meas_err = np.array([])
        out["control"] = {
            "transform_error_all_overlap": in_units(stats(err), u),
            "transform_error_test_cells": in_units(stats(err[test]), u),
            "recovered_global_ref_px": params(Hn), "true_ref_px": params(Tinv),
            "recovered_translation_at_centre_ref_px": (project(Hn, centre[None, :2]) - project(Tinv, centre[None, :2]))[0].tolist(),
            "held_out_under_true_transform": in_units(stats(e_true), u),
            "held_out_under_true_within_reference_px": {str(k): float(np.mean(e_true * u["step"] <= k)) for k in (1, 2, 3)},
            "local_shift_measurement_error": in_units(stats(meas_err), u),
            "true_misregistration_at_windows": in_units(stats(np.hypot(*d_true.T)) if rows else stats([]), u)}

    plot(run, out, B, Bv, src, ref, fwd, mag, cells, split, u)
    for r in rows:                     # keep the JSON reviewable
        r.pop("centre")
    return out


def plot(run, out, B, Bv, src, ref, fwd, mag, cells, split, u):
    PLOTS.mkdir(exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.45, 1]})
    lo, hi = np.nanpercentile(np.where(Bv, B, np.nan), [2, 98])
    ax.imshow(np.where(Bv, B, np.nan), cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    H, W = split.shape
    for c in np.flatnonzero(split.folds == split.TEST):
        cy, cx = divmod(int(c), split.grid)
        ax.add_patch(plt.Rectangle((cx * W / split.grid, cy * H / split.grid), W / split.grid,
                                   H / split.grid, fill=False, ec="#f2c14e", lw=0.8, ls="--"))
    span = max(np.ptp(ref[:, 0]), np.ptp(ref[:, 1]), 32)
    med = max(float(np.median(mag)), 1e-6)
    k = 0.04 * span / med            # ref px -> drawn working px; median arrow = 4% of span
    q = ax.quiver(ref[:, 0], ref[:, 1], fwd[:, 0] * k, fwd[:, 1] * k,
                  mag * (u["spr"] or 1), cmap="viridis", angles="xy", scale_units="xy", scale=1,
                  width=0.0022, clim=(0, np.percentile(mag * (u["spr"] or 1), 95)))
    for c in np.unique(cells):
        sel = cells == c
        p, mv = ref[sel].mean(axis=0), fwd[sel].mean(axis=0)
        ax.quiver(p[0], p[1], mv[0] * k, mv[1] * k, color="#e4572e", angles="xy",
                  scale_units="xy", scale=1, width=0.006)
    pad = 0.08 * span
    ax.set_xlim(ref[:, 0].min() - pad, ref[:, 0].max() + pad)
    ax.set_ylim(ref[:, 1].max() + pad, ref[:, 1].min() - pad)
    fig.colorbar(q, ax=ax, fraction=0.035, label="|residual| (source px)")
    ax.set_title("Held-out residuals on the reference working grid\n"
                 "arrows drawn at %.0f working px per reference px of residual (median arrow = 4%% of span); red = mean per test cell; "
                 "dashed = test cells" % (k), fontsize=9)
    rf = out["residual_field"]
    lim = float(np.percentile(mag, 95)) * 1.15
    clipped = int((mag > lim).sum())
    bx.scatter(fwd[:, 0], fwd[:, 1], s=4, alpha=0.35, color="#3b6ea5", lw=0)
    for r in (1, 2, 3):
        bx.add_patch(plt.Circle((0, 0), r, fill=False, ec="#888", lw=0.8, ls=":"))
    b = rf["bias_mean_ref_px"]
    bx.plot([0, b[0]], [0, b[1]], color="#e4572e", lw=2.5, label="mean vector (bias)")
    bx.axhline(0, color="#ccc", lw=0.6)
    bx.axvline(0, color="#ccc", lw=0.6)
    bx.set_xlim(-lim, lim)
    bx.set_ylim(lim, -lim)
    bx.set_aspect("equal")
    bx.set_xlabel("dx (reference px)")
    bx.set_ylabel("dy (reference px)")
    bx.legend(loc="upper right", fontsize=8)
    bx.set_title("Residual vectors, reference px (dotted: 1, 2, 3 px)\n"
                 "|bias| %.2f, median |r| %.2f, coherence %.2f; %d beyond p95 not shown"
                 % (rf["bias_mean_abs_ref_px"], rf["median_abs_ref_px"],
                    rf["direction_coherence"], clipped), fontsize=9)
    fig.suptitle("%s   (1 reference px = %.2f source px)" % (run["pair"], u["spr"] or float("nan")),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / (run["name"] + ".png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def fmt(v, nd=2):
    return "–" if v is None else ("%.*f" % (nd, v))


def pct(v):
    return "–" if v is None else "%.1f%%" % (100 * v)


def report(results, sign_check):
    L = ["# Diagnostic pass: held-out error on the corrected runs",
         "",
         "Measurement only. `register.py` is unchanged; every number is read back from each",
         "run's saved artifacts. Generated by `analyse.py` (controls by `control.py`).",
         "One working px = reference decimation × reference px; source px = reference px ×",
         "(source px per reference px).",
         "",
         "Shift estimators on %d known synthetic shifts (%s): phase correlation median error "
         "%.3f / p90 %.3f working px, NCC %.3f / %.3f." % (
             sign_check["trials"], sign_check["shifts"],
             sign_check["phase_error_working_px"]["median"], sign_check["phase_error_working_px"]["p90"],
             sign_check["ncc_error_working_px"]["median"], sign_check["ncc_error_working_px"]["p90"]),
         ""]
    ctrl = [r for r in results if "control" in r]
    L += ["## 1. Positive controls (known transform)", "",
          "| Pair | Status | Held-out median src px | p90 | RMSE | Held-out median ref px | Transform error vs truth, median ref px | p90 ref px | max ref px | median src px | Held-out median under TRUE transform, ref px | Local-shift measurement error, median ref px |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ctrl:
        h, c = r["held_out"]["stats"], r["control"]
        te = c["transform_error_all_overlap"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["pair"], r["status"], fmt(h["source_px"]["median"]), fmt(h["source_px"]["p90"]),
            fmt(h["source_px"]["rmse"]), fmt(h["reference_px"]["median"]),
            fmt(te["reference_px"]["median"], 3), fmt(te["reference_px"]["p90"], 3),
            fmt(te["reference_px"]["max"], 3), fmt(te["source_px"]["median"], 3),
            fmt(c["held_out_under_true_transform"]["reference_px"]["median"]),
            fmt(c["local_shift_measurement_error"]["reference_px"]["median"])))
    L += ["", "Correction the exported model applies at the held-out points (median, reference px),",
          "for comparison with the offsets the controls exercise:", "",
          "| Pair | Correction median ref px | min | max |", "|---|---:|---:|---:|"]
    for r in results:
        c = r["residual_field"]["correction_at_held_out_ref_px"]
        L.append("| %s | %s | %s | %s |" % (r["pair"], fmt(c["median"], 1), fmt(c["min"], 1), fmt(c["max"], 1)))
    L.append("")
    for r in ctrl:
        c = r["control"]
        L.append("- %s: recovered scale %.5f / rotation %.4f° vs true %.5f / %.4f°; "
                 "translation error at the footprint centre %s ref px; test-cell transform error "
                 "median %s, p90 %s ref px." % (
                     r["pair"], c["recovered_global_ref_px"]["scale"],
                     c["recovered_global_ref_px"]["rotation_deg"], c["true_ref_px"]["scale"],
                     c["true_ref_px"]["rotation_deg"],
                     "(%.3f, %.3f)" % tuple(c["recovered_translation_at_centre_ref_px"]),
                     fmt(c["transform_error_test_cells"]["reference_px"]["median"], 3),
                     fmt(c["transform_error_test_cells"]["reference_px"]["p90"], 3)))
    L += ["", "## 2. Residual field (held-out residual vectors)", "",
          "| Pair | n | Bias (mean vector) ref px | abs(bias) ref px | Median abs(residual) ref px | abs(bias) / median | Direction coherence | Within 3 ref px | Median abs(r) within 3 ref px | Coherence within 3 ref px | Plot |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        f = r["residual_field"]
        p = f["within_3_ref_px_partition"]
        L.append("| %s | %d | (%.2f, %.2f) | %s | %s | %s | %s | %s | %s | %s | [png](residuals/%s.png) |" % (
            r["pair"], r["held_out"]["n"], *f["bias_mean_ref_px"], fmt(f["bias_mean_abs_ref_px"]),
            fmt(f["median_abs_ref_px"]), fmt(f["bias_over_median"]), fmt(f["direction_coherence"]),
            pct(p["fraction"]), fmt(p["median_abs_ref_px"]), fmt(p["direction_coherence"]), r["name"]))
    L += ["", "Coherence is the length of the mean unit residual vector: about 0 when directions",
          "are random, 1 when every residual points the same way. A smoothly varying field",
          "(for example a scale error) points opposite ways on either side of its zero line",
          "and scores low even though it is entirely systematic, so the split below is the",
          "better test.", "",
          "**Systematic vs random.** Each residual is predicted from held-out neighbours %g–%g"
          % (NEAR_EXCLUDE, NEAR_RADIUS),
          "working px away. The predicted part is the systematic component and the remainder",
          "is the random component. Matcher noise does not survive the prediction; a model or",
          "coordinate error does.", "",
          "| Pair | Points with ≥%d neighbours | Median abs(residual) ref px | Systematic median ref px | Random median ref px |" % NEAR_MIN,
          "|---|---:|---:|---:|---:|"]
    for r in results:
        d = r["residual_field"]["decomposition"]
        L.append("| %s | %d / %d | %s | %s | %s |" % (
            r["pair"], d["n"], r["held_out"]["n"], fmt(d["median_abs_residual_ref_px"]),
            fmt(d["systematic_median_abs_ref_px"]), fmt(d["random_median_abs_ref_px"])))
    L += ["",
          "", "## 3. Matcher-independent local shift (frozen test cells only)", "",
          "| Pair | Windows | Median ref px | p90 ref px | Median src px | p90 src px | Median m | p90 m | NCC median ref px | Methods agree | Median ref px where they agree | Mean shift ref px | Export georef offset, working px |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in results:
        s = r["local_shift"]
        a, n, g = s["phase_all"], s["ncc_all"], s["phase_where_methods_agree"]
        eg = r["export_georeference"]
        egs = ("error" if "error" in eg else
               "(%.3f, %.3f), n=%d" % (*eg["median_shift_working_px"], eg["windows"])
               if eg.get("median_shift_working_px") else "–")
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["pair"], s["windows"], fmt(a["reference_px"]["median"]), fmt(a["reference_px"]["p90"]),
            fmt(a["source_px"]["median"]), fmt(a["source_px"]["p90"]), fmt(a["m"]["median"], 1),
            fmt(a["m"]["p90"], 1), fmt(n["reference_px"]["median"]), pct(s["methods_agree_fraction"]),
            fmt(g["reference_px"]["median"]),
            "–" if s["phase_bias_ref_px"] is None else "(%.2f, %.2f)" % tuple(s["phase_bias_ref_px"]),
            egs))
    L += ["", "**Local shift vs held-out residual in the same window.** These are two independent",
          "measurements of the same misregistration. Windows holding at least 3 held-out points:", "",
          "| Pair | Windows | abs(shift − residual) median ref px | p90 | Local shift median | Held-out median | Same, where methods agree: windows | abs(diff) median ref px |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        v = r["local_shift"]["vs_held_out"]
        L.append("| %s | %d | %s | %s | %s | %s | %d | %s |" % (
            r["pair"], v["windows"], fmt(v["abs_difference_ref_px"]["median"]),
            fmt(v["abs_difference_ref_px"]["p90"]), fmt(v["local_shift_ref_px"]["median"]),
            fmt(v["held_out_ref_px"]["median"]), v["where_methods_agree"]["windows"],
            fmt(v["where_methods_agree"]["abs_difference_ref_px"]["median"])))
    L += ["", "Phase correlation on gradient magnitude is the primary estimate and NCC the",
          "cross-check. Both run on the working grid. The controls give each estimator's own",
          "error in section 1.",
          "", "## 4. Held-out thresholds, reference px and source px (unfiltered)", "",
          "| Pair | ref px per src px | Median ref px | p90 ref px | ≤1 ref px | ≤2 ref px | ≤3 ref px | ≤1 src px | ≤2 src px | ≤3 src px | Exactly on model (residual < 1e-6) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        h = r["held_out"]
        spr = r["units"]["source_px_per_reference_px"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d (%s) |" % (
            r["pair"], fmt(1 / spr if spr else None, 3), fmt(h["stats"]["reference_px"]["median"]),
            fmt(h["stats"]["reference_px"]["p90"]),
            *[pct(h["within_reference_px"][k]) for k in "123"],
            *[pct(h["within_source_px"][k]) for k in "123"],
            h["exactly_on_model"], pct(h["exactly_on_model"] / h["n"])))
    (HERE / "diagnostics.md").write_text("\n".join(L) + "\n")


def main():
    cv2.setNumThreads(2)
    sign_check = calibrate_signs()
    runs = [dict(r) for r in json.loads((HERE.parent / "acceptance.json").read_text())]
    for r in runs:
        r["reference"] = ROOT / r["reference"]
    controls = json.loads((HERE / "controls.json").read_text())
    results = []
    for r in runs + controls:
        print("ANALYSE", r["pair"], flush=True)
        ref = ROOT / r["reference"] if not isinstance(r["reference"], Path) else r["reference"]
        results.append(analyse({**r, "reference": ref}, r.get("truth")))
        (HERE / "diagnostics.json").write_text(json.dumps(
            {"sign_check": sign_check, "runs": results}, indent=1, allow_nan=False) + "\n")
    report(results, sign_check)


if __name__ == "__main__":
    main()
