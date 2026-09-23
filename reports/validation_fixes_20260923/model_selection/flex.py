"""Model-flexibility experiment. Offline: no matching, no change to register.py.

For each real run, the SAVED fit-fold correspondences (matches.csv) are refitted
with translation-only, rigid (rotation + translation), similarity and affine
models, and every model is scored on the SAME frozen test-cell correspondences
(evaluation.json), unfiltered.

matches.csv stores source points in original source pixels. The model lives in
source working coordinates: the reference-grid position the source geometry
gives that pixel. That position is recovered by inverting the pipeline's own
piecewise-linear lattice map on the SAME triangles, which is exact wherever the
map is (the check below reproduces the pipeline's inlier flags).

    OPENBLAS_NUM_THREADS=2 .venv/bin/python -u \
      reports/validation_fixes_20260923/model_selection/flex.py
"""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import rasterio
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import warp_model as WM
from seleno.tool.coordinates import grid_to_reference, project
from seleno.tool.register import _lattice_interpolators
from seleno.tool.scene import load
from seleno.verify import transfer_error

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analyse", HERE.parent / "diagnostics" / "analyse.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

MODELS = ["translation", "rigid", "similarity", "affine"]
MIN_POINTS = {"translation": 1, "rigid": 2, "similarity": 2, "affine": 3}
ITERS, SEED = 5000, 0


# --------------------------------------------------------------------------- #
# least squares per model class, and one RANSAC for all of them
# --------------------------------------------------------------------------- #

def lsq(model, s, r):
    s, r = np.asarray(s, float), np.asarray(r, float)
    H = np.eye(3)
    if model == "translation":
        H[:2, 2] = (r - s).mean(axis=0)
        return H
    if model == "affine":
        X = np.column_stack([s, np.ones(len(s))])
        sol, *_ = np.linalg.lstsq(X, r, rcond=None)
        H[:2, :] = sol.T
        return H
    # rigid / similarity: Umeyama
    ms, mr = s.mean(axis=0), r.mean(axis=0)
    cs, cr = s - ms, r - mr
    U, D, Vt = np.linalg.svd(cr.T @ cs / len(s))
    Sg = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[1, 1] = -1
    R = U @ Sg @ Vt
    c = 1.0
    if model == "similarity":
        var = (cs ** 2).sum() / len(s)
        c = float(np.trace(np.diag(D) @ Sg) / var) if var > 0 else 1.0
    H[:2, :2] = c * R
    H[:2, 2] = mr - c * R @ ms
    return H


def forward_error(H, s, r):
    return np.hypot(*(project(H, s) - r).T)


def ransac(model, s, r, thr):
    rng = np.random.default_rng(SEED)
    k = MIN_POINTS[model]
    best, best_n = None, -1
    if len(s) < k:
        return None, np.zeros(len(s), bool)
    for _ in range(ITERS):
        idx = rng.choice(len(s), k, replace=False)
        try:
            H = lsq(model, s[idx], r[idx])
        except np.linalg.LinAlgError:
            continue
        n = int((forward_error(H, s, r) <= thr).sum())
        if n > best_n:
            best, best_n = H, n
    inl = forward_error(best, s, r) <= thr
    for _ in range(3):                   # refit on the consensus, as refit()/LM does
        if inl.sum() < k:
            break
        best = lsq(model, s[inl], r[inl])
        inl = forward_error(best, s, r) <= thr
    return best, inl


def params(H):
    a = H[:2, :2]
    return {"scale": math.sqrt(abs(np.linalg.det(a))),
            "rotation_deg": math.degrees(math.atan2(a[1, 0], a[0, 0])),
            "scale_x": float(np.hypot(*a[:, 0])), "scale_y": float(np.hypot(*a[:, 1])),
            "translation_working_px": H[:2, 2].tolist()}


# --------------------------------------------------------------------------- #
# original source pixel -> source working coordinates
# --------------------------------------------------------------------------- #

def source_to_working(S, R, frame, samples, lines):
    """Invert the pipeline's reference-plane -> (sample, line) map exactly: start
    from an approximate inverse, then Newton on the pipeline's own interpolators."""
    projected = R.crs is not None and not R.crs.is_geographic
    f_s, f_l, _ = _lattice_interpolators(S.lonlat, R.crs if projected else None)
    tri = f_s.tri
    SS, LL = f_s.values[:, 0], f_l.values[:, 0]
    approx = LinearNDInterpolator(np.column_stack([SS, LL]), tri.points)
    target = np.column_stack([samples, lines])
    p = approx(target)
    span = np.ptp(tri.points, axis=0) / max(np.ptp(SS), np.ptp(LL), 1.0)
    eps = 1e-3 * span                    # about 1/1000 source px in plane units

    def f(q):
        return np.column_stack([np.ravel(f_s(q)), np.ravel(f_l(q))])
    for _ in range(20):
        v = f(p)
        err = v - target
        if np.nanmax(np.abs(err)) < 1e-7:
            break
        jx = (f(p + [eps[0], 0]) - v) / eps[0]
        jy = (f(p + [0, eps[1]]) - v) / eps[1]
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        d0 = (jy[:, 1] * err[:, 0] - jy[:, 0] * err[:, 1]) / det
        d1 = (-jx[:, 1] * err[:, 0] + jx[:, 0] * err[:, 1]) / det
        p = p - np.column_stack([d0, d1])
    resid = np.hypot(*(f(p) - target).T)
    p[~(resid < 1e-4)] = np.nan          # not inverted to 1e-4 source px: drop
    a, b = p[:, 0], p[:, 1]
    t = R.transform
    if not projected:                    # lattice longitude convention -> the reference's
        centre = t.c + t.a * R.array.shape[1] / 2.0
        a = a + 360.0 * np.round((centre - a) / 360.0)
    inv = ~t
    col = inv.a * a + inv.b * b + inv.c - 0.5
    row = inv.d * a + inv.e * b + inv.f - 0.5
    native = np.column_stack([col, row])
    return project(np.linalg.inv(grid_to_reference(frame)), native)


# --------------------------------------------------------------------------- #

def decomposition(fwd, ref):
    tree = cKDTree(ref)
    pred = np.full_like(fwd, np.nan)
    for i, nb in enumerate(tree.query_ball_point(ref, A.NEAR_RADIUS)):
        nb = [j for j in nb if np.hypot(*(ref[j] - ref[i])) >= A.NEAR_EXCLUDE]
        if len(nb) >= A.NEAR_MIN:
            pred[i] = np.median(fwd[nb], axis=0)
    ok = np.isfinite(pred[:, 0])
    if not ok.any():
        return {"n": 0, "systematic_median_ref_px": None, "random_median_ref_px": None}
    return {"n": int(ok.sum()),
            "systematic_median_ref_px": float(np.median(np.hypot(*pred[ok].T))),
            "random_median_ref_px": float(np.median(np.hypot(*(fwd[ok] - pred[ok]).T)))}


def score(e_working, fwd_ref, test_ref, u):
    e_ref = e_working * u["step"]
    e_src = e_ref * u["spr"]
    return {"median_ref_px": float(np.median(e_ref)), "p90_ref_px": float(np.percentile(e_ref, 90)),
            "rmse_ref_px": float(np.sqrt(np.mean(e_ref ** 2))),
            "median_src_px": float(np.median(e_src)), "p90_src_px": float(np.percentile(e_src, 90)),
            "within_ref_px": {str(k): float(np.mean(e_ref <= k)) for k in (1, 2, 3)},
            "decomposition": decomposition(fwd_ref, test_ref)}


def conditioning(pts, overlap):
    """Fit-point spread against the overlap the model has to cover (working grid)."""
    yy, xx = np.nonzero(overlap)
    area = np.column_stack([xx, yy]).astype(float)
    hull = ConvexHull(pts)
    inside = MplPath(pts[hull.vertices]).contains_points(area)
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))
    ea = np.sort(np.linalg.eigvalsh(np.cov(area.T)))
    return {"n": int(len(pts)), "hull_fraction_of_overlap": float(inside.mean()),
            "axis_ratio_minor_major": float(math.sqrt(ev[0] / ev[1])),
            "fit_spread_sd_working_px": [float(math.sqrt(ev[0])), float(math.sqrt(ev[1]))],
            "overlap_spread_sd_working_px": [float(math.sqrt(ea[0])), float(math.sqrt(ea[1]))],
            "spread_vs_overlap_minor": float(math.sqrt(ev[0] / ea[0])),
            "spread_vs_overlap_major": float(math.sqrt(ev[1] / ea[1]))}


def run(r):
    art = ROOT / r["artifacts"]
    model = json.loads((art / "transform.json").read_text())
    ev = json.loads((art / "evaluation.json").read_text())
    metrics = json.loads((art / "metrics.json").read_text())
    frame, conv = model["frame"], metrics["accuracy"]["units"]
    u = {"step": float(conv["reference_decimation"]), "spr": conv["source_px_per_reference_px"]}
    rows = list(csv.DictReader((art / "matches.csv").open()))
    orig = np.array([[float(x["src_x"]), float(x["src_y"])] for x in rows])
    ref_native = np.array([[float(x["ref_x"]), float(x["ref_y"])] for x in rows])
    inlier = np.array([int(x["inlier"]) for x in rows], bool)
    S, R = load(str(ROOT / r["source"])), load(str(ROOT / r["reference"]))
    fit_src = source_to_working(S, R, frame, orig[:, 0], orig[:, 1])
    fit_ref = project(np.linalg.inv(grid_to_reference(frame)), ref_native)
    # Records whose saved source coordinate is pinned to one value (it recurs
    # exactly, e.g. x = 3500.0 on TMC-2, where the thinned geometry lattice ends)
    # did not record where the point was; they cannot be rebuilt and are dropped.
    pinned = np.zeros(len(rows), bool)
    for k in (0, 1):
        vals, counts = np.unique(orig[:, k], return_counts=True)
        pinned |= np.isin(orig[:, k], vals[counts >= 3])
    good = np.isfinite(fit_src).all(axis=1) & ~pinned
    thr = 3.0 / u["step"]                # the fine stage's verification threshold

    # Check the reconstruction: the exported global matrix should put the
    # pipeline's inliers within its threshold and its outliers beyond it.
    Hx = np.asarray(model["matrix"], float)
    fe = forward_error(Hx, fit_src[good], fit_ref[good])
    ecc = metrics["accuracy"]["ecc"].get("adopted", False)
    L = lsq("affine", fit_src[good & inlier], fit_ref[good & inlier])
    lsq_vs_exported = np.hypot(*(project(L, fit_src[good & inlier]) - project(Hx, fit_src[good & inlier])).T)
    check = {"points": int(len(rows)), "pinned_excluded": int(pinned.sum()),
             "reconstructed": int(good.sum()),
             "lsq_affine_on_flagged_inliers_vs_exported_max_working_px": float(lsq_vs_exported.max()),
             "inlier_flag_reproduced": float(np.mean((fe <= thr) == inlier[good])),
             "inliers_within_threshold": float(np.mean(fe[inlier[good]] <= thr)),
             "inlier_forward_error_max_working_px": float(fe[inlier[good]].max()),
             "threshold_working_px": thr, "ecc_adopted": bool(ecc)}

    s, rf = fit_src[good], fit_ref[good]
    test_s, test_r = np.array(ev["source"], float), np.array(ev["reference"], float)
    out = {"pair": r["pair"], "name": r["name"], "reconstruction_check": check,
           "units": conv, "test_n": int(len(test_s)),
           "conditioning_fit_inliers": None, "models": {}}
    B, Bv = A.reference_grid(ROOT / r["reference"], frame)
    with rasterio.open(art / "registered.tif") as ds:
        overlap = np.isfinite(ds.read(1)) & Bv
    out["conditioning_fit_inliers"] = conditioning(fit_ref[inlier & good], overlap)
    out["conditioning_all_fit_points"] = conditioning(rf, overlap)

    e = WM.residuals(model, test_s, test_r)
    assert float(np.sqrt(np.mean(e ** 2))) == metrics["accuracy"]["rmse_px"]
    fwd = (WM.forward_points(model, test_s) - test_r) * u["step"]
    out["models"]["exported (affine + segments)"] = {
        **score(e, fwd, test_r, u), "fit_inliers": int(inlier.sum()), "params": params(Hx)}
    for name in MODELS:
        H, inl = ransac(name, s, rf, thr)
        e = transfer_error(H, test_s, test_r)
        fwd = (project(H, test_s) - test_r) * u["step"]
        out["models"][name] = {**score(e, fwd, test_r, u), "fit_inliers": int(inl.sum()),
                               "params": params(H)}
    ranked = sorted(MODELS, key=lambda m: (out["models"][m]["median_ref_px"],
                                           out["models"][m]["p90_ref_px"]))
    out["winner_by_median"] = ranked[0]
    out["ranking_by_median"] = ranked
    return out


def fmt(v, nd=2):
    return "–" if v is None else "%.*f" % (nd, v)


def report(results):
    L = ["# Model-flexibility experiment (offline, saved fit points only)", "",
         "Every model is refitted by the same RANSAC (threshold 3 reference px, as in the fine",
         "stage) and least-squares refit on the saved fit-fold points (`matches.csv`), then",
         "scored on the frozen test-cell points (`evaluation.json`). Nothing is filtered.",
         "\"exported\" is the shipped affine plus segments. The systematic/random split uses the",
         "same neighbour method as `diagnostics/`. `register.py` is unchanged.", "",
         "## Reconstruction check", "",
         "The fit points are rebuilt from original source pixels. The rebuilt points should",
         "reproduce the pipeline's own inlier flags under the exported matrix. ECC changes the",
         "matrix after the flags are set, so IIRS is expected to differ slightly. The sharper test",
         "is a least-squares affine on the flagged inliers: OpenCV's refinement converges to it,",
         "so without ECC it should reproduce the exported matrix.", "",
         "| Pair | Fit points | Pinned, excluded | Rebuilt | Inlier flag reproduced | Inliers within threshold | LSQ affine on flagged inliers vs exported, max working px | ECC adopted |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        c = r["reconstruction_check"]
        L.append("| %s | %d | %d | %d | %.1f%% | %.1f%% | %.2e | %s |" % (
            r["pair"], c["points"], c["pinned_excluded"], c["reconstructed"],
            100 * c["inlier_flag_reproduced"], 100 * c["inliers_within_threshold"],
            c["lsq_affine_on_flagged_inliers_vs_exported_max_working_px"],
            "yes" if c["ecc_adopted"] else "no"))
    L += ["", "## Fit-point conditioning (exported model's fit inliers, working grid)", "",
          "| Pair | Inliers | Hull covers overlap | Axis ratio minor/major | Fit spread vs overlap spread, minor axis | major axis |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        c = r["conditioning_fit_inliers"]
        L.append("| %s | %d | %.1f%% | %.3f | %.3f | %.3f |" % (
            r["pair"], c["n"], 100 * c["hull_fraction_of_overlap"], c["axis_ratio_minor_major"],
            c["spread_vs_overlap_minor"], c["spread_vs_overlap_major"]))
    L += ["", "Spread vs overlap is the fit cloud's standard deviation along its own principal axis",
          "divided by the overlap's along ITS principal axis (sorted, not aligned).", ""]
    for r in results:
        L += ["## %s — %d held-out points" % (r["pair"], r["test_n"]), "",
              "| Model | Fit inliers | Median ref px | p90 ref px | RMSE ref px | Median src px | p90 src px | ≤1 ref px | ≤2 ref px | ≤3 ref px | Systematic ref px | Random ref px | Scale | Rotation ° | sx / sy |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, m in r["models"].items():
            d, p = m["decomposition"], m["params"]
            L.append("| %s%s | %d | %s | %s | %s | %s | %s | %.1f%% | %.1f%% | %.1f%% | %s | %s | %.5f | %.4f | %.4f / %.4f |" % (
                name, " **(wins)**" if name == r["winner_by_median"] else "", m["fit_inliers"],
                fmt(m["median_ref_px"]), fmt(m["p90_ref_px"]), fmt(m["rmse_ref_px"]),
                fmt(m["median_src_px"]), fmt(m["p90_src_px"]),
                *[100 * m["within_ref_px"][k] for k in "123"],
                fmt(d["systematic_median_ref_px"]), fmt(d["random_median_ref_px"]),
                p["scale"], p["rotation_deg"], p["scale_x"], p["scale_y"]))
        L.append("")
    L += ["## Winner per pair (lowest held-out median, p90 as tie-break)", "",
          "| Pair | Winner | Ranking |", "|---|---|---|"]
    for r in results:
        L.append("| %s | %s | %s |" % (r["pair"], r["winner_by_median"], " < ".join(r["ranking_by_median"])))
    (HERE / "flex.md").write_text("\n".join(L) + "\n")


def main():
    runs = json.loads((HERE.parent / "acceptance.json").read_text())
    results = []
    for r in runs:
        print("FLEX", r["pair"], flush=True)
        results.append(run(r))
        (HERE / "flex.json").write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    report(results)


if __name__ == "__main__":
    main()
