"""`register(source, reference, out_dir)` - the tool.

Takes two arbitrary image files, puts the source into the reference's frame, and
writes the full artifact set. Nothing about ISRO geometry, `.spm` Sun parameters
or refined corners is assumed; whatever is missing is recorded in
`metrics.json:degraded` and the run continues with reduced capability.

The output contract, in `<out_dir>/<job_id>/`:

    matches.csv      src_x, src_y, ref_x, ref_y, confidence, inlier
    registered.tif   source warped into the reference frame, georeferenced when
                     the reference was, no-data preserved
    transform.json   model type, parameters, per-segment parameters for long strips
    metrics.json     RMSE (m and px), inlier count and ratio, coverage fraction,
                     dispersion, delta Sun azimuth, scale ratio, method used,
                     status and reason
    overlay.png      checkerboard + side-by-side QC
    report.md        human-readable summary

Failure is a first-class outcome. `metrics.json` always exists; on failure it
carries `status: "failed"` and one of `unreadable_input`, `no_overlap`,
`insufficient_matches`, `verification_failed`, `degenerate_transform`.

Accuracy reporting
------------------
RMSE is computed on **held-out** correspondences: the inlier set is split, the
transform is fitted on one part and the error measured on the other. Reporting
the residual of the points a model was fitted to measures self-consistency, not
accuracy, and this project has a written record of that distinction mattering.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import spatial, verify as V
from . import methods as M
from .profiles import Profiles
from .scene import Scene, UnreadableInput, load, normalised

FAILURE_CODES = ("unreadable_input", "no_overlap", "insufficient_matches",
                 "verification_failed", "degenerate_transform")


@dataclass
class Result:
    status: str                                 # pass | warning | failed
    reason: str | None = None
    job_id: str = ""
    out_dir: str = ""
    metrics: dict = field(default_factory=dict)
    transform: dict = field(default_factory=dict)
    n_matches: int = 0

    @property
    def ok(self) -> bool:
        return self.status != "failed"


# --------------------------------------------------------------------------- #
# common frame
# --------------------------------------------------------------------------- #

# Target pixels are transformed in row blocks. rasterio's xy()/transform() take
# and return PYTHON LISTS, so a 6144^2 grid in one call materialises ~2.4 GB of
# float objects - which, with the raster copies that used to happen in scene.py,
# is what made this tool exhaust 15 GB of RAM and get OOM-killed.
_CHUNK_ROWS = 256

# Bytes of working memory per target-grid pixel. The dominant term is NOT the
# image arrays - it is the masked NCC, which correlates in "full" mode and so
# allocates FFT buffers of (2H-1) x (2W-1) in float64/complex128, several at
# once. That is ~4x the target area at 8-16 bytes each. Measured: a 2048^2 grid
# on a TMC-2/SELENE pair peaks at 3.9 GB, i.e. ~930 B per target pixel including
# the memmap pages that get touched.
_BYTES_PER_TARGET_PX = 900


def _available_bytes() -> int:
    """Physical memory we may use, read from the OS rather than assumed."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 4 << 30


def _cap_max_side(max_side: int, budget_fraction: float = 0.5) -> tuple[int, str | None]:
    """Shrink the working grid so the run cannot exhaust memory.

    An earlier version of this tool asked for a 6144^2 working grid on a 15 GB
    machine and was OOM-killed. The size of an unseen reference is not something
    a caller should have to reason about, so the ceiling is computed here from
    what the OS actually has free, and the reduction is reported rather than
    applied silently.
    """
    budget = _available_bytes() * budget_fraction
    allowed_px = budget / _BYTES_PER_TARGET_PX
    allowed_side = int(max(512, allowed_px ** 0.5))
    if max_side <= allowed_side:
        return max_side, None
    return allowed_side, ("working grid capped at %d px a side (asked for %d): "
                          "%.1f GB available, budgeting %.0f%% of it"
                          % (allowed_side, max_side, _available_bytes() / 1e9,
                             100 * budget_fraction))


def _target_grid(ref: Scene, max_side: int, window=None):
    """The reference's pixel grid, optionally cropped, decimated only if needed.

    `window` is a full-resolution (r0, c0, r1, c1) box. Cropping to the region the
    source can actually reach is what makes native-resolution work affordable: a
    strip covering a fifth of a 12 288 px tile needs a 2 000 px window, not a
    6x-decimated whole tile, and the decimation is what puts a floor under the
    reported accuracy.
    """
    h, w = ref.array.shape
    r0, c0, r1, c1 = window if window else (0, 0, h, w)
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    r1, c1 = min(h, int(r1)), min(w, int(c1))
    if r1 - r0 < 32 or c1 - c0 < 32:
        r0, c0, r1, c1 = 0, 0, h, w
    step = max(1, int(math.ceil(max(r1 - r0, c1 - c0) / float(max_side))))
    sub = np.asarray(ref.array[r0:r1:step, c0:c1:step], np.float32)
    sub = sub * ref.meta_scale + ref.meta_offset
    v = np.isfinite(sub) & (sub > -1e30)
    if ref.nodata is not None and ref.meta_scale == 1.0 and ref.meta_offset == 0.0:
        v &= sub != ref.nodata
    return step, (r0, c0), sub, v


def prealign(src: Scene, ref: Scene, max_side: int = 2048, window=None):
    """Put the source on the reference's grid using whatever geometry exists.

    Returns ``(src_on_grid, src_valid, ref_grid, ref_valid, back_x, back_y, info)``
    where `back_x`/`back_y` give, for each target pixel, the ORIGINAL source pixel
    it came from - so match points can always be reported in the source's own
    coordinates, whatever route was taken to get here.
    """
    step, (orow, ocol), R, Rv = _target_grid(ref, max_side, window)
    H, W = R.shape
    info = {"target_shape": [H, W], "reference_decimation": step,
            "reference_origin": [orow, ocol], "route": None, "notes": []}

    # --- route 1: the source carries a lon/lat lattice and the reference a CRS
    if src.lonlat is not None and ref.georeferenced:
        f_s, f_l = _lattice_interpolators(src.lonlat)
        S = np.empty((H, W), np.float64)
        L = np.empty((H, W), np.float64)
        for r0b in range(0, H, _CHUNK_ROWS):
            r1b = min(r0b + _CHUNK_ROWS, H)
            lon, lat = _grid_lonlat(ref, orow, ocol, step, r0b, r1b, W)
            S[r0b:r1b] = f_s(lon, lat)
            L[r0b:r1b] = f_l(lon, lat)
        info["route"] = "source lon/lat lattice -> reference CRS"
        return _sample(src, S, L, R, Rv, info)

    # --- route 2: both georeferenced
    if src.georeferenced and ref.georeferenced:
        from rasterio.transform import rowcol
        from rasterio.warp import transform as warp_transform
        S = np.empty((H, W), np.float64)
        L = np.empty((H, W), np.float64)
        same = str(src.crs) == str(ref.crs)
        for r0b in range(0, H, _CHUNK_ROWS):
            r1b = min(r0b + _CHUNK_ROWS, H)
            xs, ys = _grid_xy(ref, orow, ocol, step, r0b, r1b, W)
            if not same:
                xs, ys = warp_transform(ref.crs, src.crs, xs.ravel().tolist(),
                                        ys.ravel().tolist())
                xs = np.asarray(xs).reshape(r1b - r0b, W)
                ys = np.asarray(ys).reshape(r1b - r0b, W)
            rr, cc = rowcol(src.transform, xs.ravel().tolist(), ys.ravel().tolist(),
                            op=lambda v: v)
            L[r0b:r1b] = np.asarray(rr, np.float64).reshape(r1b - r0b, W)
            S[r0b:r1b] = np.asarray(cc, np.float64).reshape(r1b - r0b, W)
        info["route"] = "both georeferenced: reference CRS -> source CRS"
        return _sample(src, S, L, R, Rv, info)

    # --- route 3: no usable geometry. Pixel space, scale-matched if GSDs are known.
    scale = 1.0
    if src.gsd_m and ref.gsd_m:
        scale = float(src.gsd_m) / float(ref.gsd_m) / step
        info["notes"].append("no shared frame; source resampled by the GSD ratio "
                             "(%.4f) and registered in pixel space" % scale)
    else:
        info["notes"].append("no shared frame and no GSD on at least one side; "
                             "registered in pixel space at native sampling. No "
                             "metre-scale accuracy can be reported.")
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    S = cols / max(scale, 1e-9)
    L = rows / max(scale, 1e-9)
    info["route"] = "pixel space (scale %.4f)" % scale
    return _sample(src, S, L, R, Rv, info)


def _grid_xy(ref, orow, ocol, step, r0b, r1b, W):
    """Projected (x, y) of a block of target pixel centres, as arrays."""
    t = ref.transform
    cols = ocol + (np.arange(W) + 0.5) * step
    rows = orow + (np.arange(r0b, r1b) + 0.5) * step
    C, Rr = np.meshgrid(cols, rows)
    x = t.a * C + t.b * Rr + t.c
    y = t.d * C + t.e * Rr + t.f
    return x, y


def _grid_lonlat(ref, orow, ocol, step, r0b, r1b, W):
    """Same block, converted to lon/lat on the Moon."""
    from rasterio.warp import transform as warp_transform
    x, y = _grid_xy(ref, orow, ocol, step, r0b, r1b, W)
    if ref.crs is not None and not ref.crs.is_geographic:
        lon, lat = warp_transform(ref.crs, "+proj=longlat +R=1737400 +no_defs",
                                  x.ravel().tolist(), y.ravel().tolist())
        return np.asarray(lon).reshape(x.shape), np.asarray(lat).reshape(x.shape)
    return x, y                                  # already degrees


def _lattice_interpolators(grid):
    """(lon, lat) -> (sample, line), built once and reused across blocks."""
    from scipy.interpolate import LinearNDInterpolator
    step = max(1, grid.lon.shape[0] // 200)      # the lattice is far finer than needed
    pts = np.column_stack([grid.lon[::step, ::step].ravel(),
                           grid.lat[::step, ::step].ravel()])
    SS, LL = np.meshgrid(grid.pixels[::step], grid.scans[::step])
    return (LinearNDInterpolator(pts, SS.ravel()),
            LinearNDInterpolator(pts, LL.ravel()))


def _sample(src: Scene, S, L, R, Rv, info):
    """Nearest-neighbour lift of the source onto the target grid.

    Validity is evaluated on the SAMPLED values, so no source-wide boolean mask
    is ever allocated.
    """
    h, w = src.array.shape
    ok = np.isfinite(S) & np.isfinite(L) & (S >= 0) & (S < w) & (L >= 0) & (L < h)
    out = np.zeros(R.shape, np.float32)
    ov = np.zeros(R.shape, bool)
    if ok.any():
        si = np.clip(np.nan_to_num(S).astype(np.int64), 0, w - 1)
        li = np.clip(np.nan_to_num(L).astype(np.int64), 0, h - 1)
        vals = np.asarray(src.array[li[ok], si[ok]], np.float32)
        vals = vals * src.meta_scale + src.meta_offset
        out[ok] = vals
        good = np.isfinite(vals) & (vals > -1e30)
        if src.nodata is not None and src.meta_scale == 1.0 and src.meta_offset == 0.0:
            good &= vals != src.nodata
        ov[ok] = good
    back_x = np.where(ok, np.nan_to_num(S), np.nan).astype(np.float32)
    back_y = np.where(ok, np.nan_to_num(L), np.nan).astype(np.float32)
    info["source_coverage"] = round(float(ov.mean()), 4)
    return out, ov, R, Rv, back_x, back_y, info


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def _job_id(a: str, b: str) -> str:
    h = hashlib.sha1(("%s|%s|%.3f" % (a, b, time.time())).encode()).hexdigest()[:12]
    return h


def _fail(out_dir, job_id, code, message, extra=None):
    os.makedirs(os.path.join(out_dir, job_id), exist_ok=True)
    m = {"status": "failed", "reason": code, "message": message,
         "job_id": job_id, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    if extra:
        m.update(extra)
    assert code in FAILURE_CODES, "undeclared failure code %r" % code
    with open(os.path.join(out_dir, job_id, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    with open(os.path.join(out_dir, job_id, "report.md"), "w") as fh:
        fh.write("# Registration failed\n\n**Reason:** `%s`\n\n%s\n" % (code, message))
    return Result("failed", code, job_id, os.path.join(out_dir, job_id), m)


def register(source: str, reference: str, out_dir: str = "outputs", *,
             model: str = "auto", max_side: int = 2048, grid=(8, 8),
             holdout: float = 0.35, seed: int = 0, profiles: Profiles | None = None,
             segments: int = 0, subpixel: bool = True,
             progress=None, verbose: bool = True) -> Result:
    """Register `source` onto `reference`. Always writes an artifact set.

    `progress`, if given, is called with each log line as it happens, so a
    caller driving this from a server can stream real stage transitions rather
    than inventing a percentage.
    """
    t_start = time.time()
    profiles = profiles or Profiles.load()
    job_id = _job_id(source, reference)
    job_dir = os.path.join(out_dir, job_id)

    def log(*a):
        msg = " ".join(str(x) for x in a)
        if verbose:
            print(msg, flush=True)
        if progress is not None:
            try:
                progress(msg)
            except Exception:                                         # noqa: BLE001
                pass                      # a broken listener must not fail a run
    max_side, cap_note = _cap_max_side(max_side)
    if cap_note:
        log("memory   : %s" % cap_note)

    # ---- 1. read -----------------------------------------------------------
    try:
        S = load(source, profiles)
        R = load(reference, profiles)
    except UnreadableInput as exc:
        return _fail(out_dir, job_id, "unreadable_input", str(exc))
    log("source    : %s" % json.dumps(S.summary()))
    log("reference : %s" % json.dumps(R.summary()))

    sp, rp = profiles.get(S.profile), profiles.get(R.profile)
    degraded = ["source: " + d for d in S.degraded] + ["reference: " + d for d in R.degraded]

    # ---- 2. common frame ---------------------------------------------------
    try:
        A_raw, Am, B_raw, Bm, back_x, back_y, frame = prealign(S, R, max_side=max_side)
        # Second pass: having found where the source actually lands, redo the
        # placement at the finest sampling that fits inside `max_side` over just
        # that region. Without this the reported RMSE is floored by the whole-frame
        # decimation rather than by the method.
        bb = M.overlap_bbox(Am & Bm, pad=8)
        if bb is not None and frame["reference_decimation"] > 1:
            st, (orow, ocol) = frame["reference_decimation"], frame["reference_origin"]
            win = (orow + bb[0] * st, ocol + bb[1] * st,
                   orow + bb[2] * st, ocol + bb[3] * st)
            fine = prealign(S, R, max_side=max_side, window=win)
            if fine[6]["reference_decimation"] < st and (fine[1] & fine[3]).sum() > 4096:
                A_raw, Am, B_raw, Bm, back_x, back_y, frame = fine
                frame["notes"].append(
                    "refined: re-placed at %d x decimation over the overlap instead of "
                    "%d x over the whole reference" % (frame["reference_decimation"], st))
    except Exception as exc:
        return _fail(out_dir, job_id, "no_overlap",
                     "could not place the source in the reference frame: %s: %s"
                     % (type(exc).__name__, exc))
    degraded += frame.get("notes", [])
    if cap_note:
        degraded.append(cap_note)
    both = Am & Bm
    log("frame     : %s, source covers %.1f%% of the reference grid"
        % (frame["route"], 100 * Am.mean()))
    if both.mean() < 0.02 or both.sum() < 4096:
        return _fail(out_dir, job_id, "no_overlap",
                     "the two images share %.2f%% of the reference grid (%d pixels), "
                     "which is below the 2%% / 4096 px floor"
                     % (100 * both.mean(), int(both.sum())),
                     {"frame": frame, "degraded": degraded})

    A = normalised(Scene(path=S.path, array=A_raw, valid=Am, reader=S.reader), sp)
    B = normalised(Scene(path=R.path, array=B_raw, valid=Bm, reader=R.reader), rp)

    # ---- 3. pair character and candidate plan ------------------------------
    gsd_ratio = (S.gsd_m / R.gsd_m) if (S.gsd_m and R.gsd_m) else None
    character = M.pair_character(A, Am, B, Bm, S.profile, R.profile, gsd_ratio)
    from .. import matchers as _mt
    available = {k: v["available"] for k, v in _mt.MATCHERS.items()}
    plan = M.candidate_plan(character, available)
    log("character : %s" % json.dumps(character))
    log("plan      : %s" % " -> ".join(plan))

    # ---- 4. run candidates, verify each ------------------------------------
    # Coverage is scored against cells that BOTH images actually reach. A strip
    # crossing a quarter of a map tile cannot put matches in the other three
    # quarters, and scoring it against the whole frame would refuse a good
    # registration for the wrong reason.
    eligible = spatial.eligible_cells(both, B.shape, grid)
    max_shift = _search_radius_px(S, R, sp, frame)
    attempts, best = [], None
    for name in plan:
        t0 = time.time()
        corr, note = _run_one(name, A, Am, B, Bm, max_shift, grid)
        rec = {"method": name, "seconds": round(time.time() - t0, 2),
               "candidates": 0 if corr is None else len(corr), "note": note}
        if corr is not None and len(corr) >= 4:
            mt = model if model != "auto" else _auto_model(len(corr), character)
            vr = V.verify(corr.src, corr.ref, model_type=mt, threshold=3.0)
            rec.update({"model": mt, "inliers": int(vr.n_inliers),
                        "inlier_ratio": round(vr.inlier_ratio, 4),
                        "verified": bool(vr.ok)})
            if vr.ok and vr.n_inliers >= 6:
                cov = spatial.cell_coverage(corr.ref[vr.inlier_mask], B.shape, grid,
                                            eligible=eligible)
                disp = spatial.dispersion(corr.ref[vr.inlier_mask], B.shape)
                score = vr.n_inliers * (0.5 + cov) * (0.5 + disp)
                rec.update({"coverage": round(cov, 4), "dispersion": round(disp, 4),
                            "score": round(float(score), 2)})
                if best is None or score > best["score"]:
                    best = {"name": name, "corr": corr, "vr": vr, "model": mt,
                            "score": float(score)}
        attempts.append(rec)
        log("  %-16s %s" % (name, json.dumps({k: v for k, v in rec.items()
                                              if k != "note" or v})))
        if best is not None and best["name"] == name and rec.get("inliers", 0) >= 40 \
                and rec.get("coverage", 0) >= 0.5:
            log("  (stopping early: %s is comfortably sufficient)" % name)
            break

    if best is None:
        any_cand = max((a["candidates"] for a in attempts), default=0)
        code = "insufficient_matches" if any_cand < 8 else "verification_failed"
        return _fail(out_dir, job_id, code,
                     "no candidate method produced a geometrically verified "
                     "transform; best attempt had %d correspondences" % any_cand,
                     {"attempts": attempts, "character": character,
                      "frame": frame, "degraded": degraded})

    # ---- 5. plausibility ---------------------------------------------------
    Hm = best["vr"].model
    warn = V.plausibility(Hm, best["model"], expected_scale=1.0)
    d = V.decompose(Hm) if Hm is not None else {}
    if Hm is None or not np.isfinite(Hm).all():
        return _fail(out_dir, job_id, "degenerate_transform",
                     "the estimator returned no usable model",
                     {"attempts": attempts, "degraded": degraded})
    if abs(d.get("est_scale_x", 1.0) - 1.0) > 0.25 or abs(d.get("est_rotation_deg", 0.0)) > 30.0:
        return _fail(out_dir, job_id, "degenerate_transform",
                     "the fitted transform is implausible on a shared frame: "
                     "scale %.3f, rotation %.2f deg"
                     % (d.get("est_scale_x", float("nan")),
                        d.get("est_rotation_deg", float("nan"))),
                     {"attempts": attempts, "decomposition": d, "degraded": degraded})

    # ---- 6. held-out accuracy ---------------------------------------------
    corr, vr = best["corr"], best["vr"]
    idx = np.nonzero(vr.inlier_mask)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idx))
    n_hold = int(round(holdout * len(idx)))
    hold = idx[perm[:n_hold]]
    fit = idx[perm[n_hold:]]
    acc = {}
    Hf = None
    if len(fit) >= V._min_points(best["model"]) and len(hold) >= 3:
        from .. import register as _rg
        Hf = _rg.refit(corr.src[fit], corr.ref[fit], best["model"])
        if Hf is not None:
            res = V.transfer_error(Hf, corr.src[hold].astype(np.float64),
                                   corr.ref[hold].astype(np.float64))
            acc = {"held_out_n": int(len(hold)),
                   "held_out_rmse_px": round(float(np.sqrt((res ** 2).mean())), 4),
                   "held_out_median_px": round(float(np.median(res)), 4),
                   "held_out_p90_px": round(float(np.percentile(res, 90)), 4)}
    else:
        degraded.append("too few inliers to hold points out; RMSE is fit-set "
                        "self-consistency, not accuracy")
        res = vr.residuals[vr.inlier_mask]
        acc = {"held_out_n": 0,
               "fit_rmse_px": round(float(np.sqrt((res ** 2).mean())), 4)}

    # ---- 6b. sub-pixel polish ----------------------------------------------
    # Warm-started from the FIT-SUBSET model, not the all-inlier one, so the
    # held-out points remain points the delivered transform was never fitted to
    # and the improvement below is a real measurement rather than a restatement
    # of the fit.
    ecc = {"attempted": False}
    ecc_base = Hf if Hf is not None else Hm
    if subpixel and ecc_base is not None:
        Hp, ecc = _ecc_polish(A, Am, B, Bm, ecc_base, best["model"])
        if Hp is not None:
            if len(hold) >= 3 and "held_out_rmse_px" in acc:
                resp = V.transfer_error(Hp, corr.src[hold].astype(np.float64),
                                        corr.ref[hold].astype(np.float64))
                r_new = float(np.sqrt((resp ** 2).mean()))
                if r_new < acc["held_out_rmse_px"]:
                    ecc["adopted"] = True
                    ecc["rmse_px_before"] = acc["held_out_rmse_px"]
                    Hm = Hp
                    d = V.decompose(Hm)          # the warp changed; so did these
                    acc.update(held_out_rmse_px=round(r_new, 4),
                               held_out_median_px=round(float(np.median(resp)), 4),
                               held_out_p90_px=round(float(np.percentile(resp, 90)), 4))
                    log("subpixel : ECC polish adopted, held-out RMSE %.4f -> %.4f px"
                        % (ecc["rmse_px_before"], r_new))
                else:
                    ecc["note"] = ("ECC converged but did not improve held-out RMSE "
                                   "(%.4f vs %.4f px); the unpolished model was kept"
                                   % (r_new, acc["held_out_rmse_px"]))
            else:
                # Nothing independent to score it against, so leave the verified
                # model alone rather than adopt an unmeasured change.
                ecc["note"] = ("no held-out points to score the polish against; "
                               "the unpolished model was kept")
    acc["subpixel_method"] = ("parabolic+ecc" if ecc.get("adopted") else "parabolic")
    acc["ecc"] = ecc

    # ---- 7. write artifacts ------------------------------------------------
    os.makedirs(job_dir, exist_ok=True)
    m_per_px = _metres_per_pixel(R, frame)
    warped, wvalid = _warp(A_raw, Am, Hm, B_raw.shape, best["model"])
    seg = _segment_fit(corr, vr, best["model"], B.shape, segments)

    _write_matches(job_dir, corr, vr, back_x, back_y, frame)
    _write_registered(job_dir, warped, wvalid, R, frame)
    tj = _write_transform(job_dir, Hm, best["model"], d, seg, frame, best["name"])
    ov_cov = spatial.cell_coverage(corr.ref[vr.inlier_mask], B.shape, grid,
                                   eligible=eligible)
    ov_disp = spatial.dispersion(corr.ref[vr.inlier_mask], B.shape)
    ov_extrap = spatial.extrapolation_fraction(corr.ref[vr.inlier_mask], B.shape,
                                               grid, eligible=eligible)
    metrics = _write_metrics(job_dir, job_id, S, R, character, frame, attempts, best,
                             vr, acc, ov_cov, ov_disp, m_per_px, degraded, warn, d,
                             time.time() - t_start, grid, eligible, ov_extrap)
    _write_overlay(job_dir, A, Am, B, Bm, warped, wvalid, corr, vr)
    _write_preview_json(job_dir, A, B, corr, vr)
    _write_report(job_dir, job_id, S, R, metrics, tj, attempts)

    status = metrics["status"]
    log("status    : %s  (%s)" % (status, metrics.get("reason") or "all checks passed"))
    log("wrote     : %s" % job_dir)
    return Result(status, metrics.get("reason"), job_id, job_dir, metrics, tj, len(corr))


# --------------------------------------------------------------------------- #
# pieces
# --------------------------------------------------------------------------- #

def _search_radius_px(S: Scene, R: Scene, sp: dict, frame: dict) -> int:
    """How far to search, from the profile's declared geolocation error.

    Phase 2 measured 3.9-4.6 km on unrefined polar OHRC; a radius under that
    cannot find the answer, and the earliest spike in this project failed for
    exactly that reason.
    """
    err_m = sp.get("expected_geolocation_error_m")
    step = frame.get("reference_decimation", 1)
    px = None
    if err_m and R.gsd_m:
        px = err_m / (R.gsd_m * step)
    if px is None:
        px = 0.25 * max(frame["target_shape"])
    return int(max(24, min(px, 0.45 * max(frame["target_shape"]))))


def _auto_model(n: int, character: dict) -> str:
    if character.get("same_sensor") and n >= 30:
        return "homography"
    return "similarity" if n < 30 else "affine"


def _run_one(name, A, Am, B, Bm, max_shift, grid):
    if name == "dense-ncc":
        t = M.dense_translation(A, Am, B, Bm, max_shift_px=max_shift)
        if t is None:
            return None, "no correlation surface"
        bbox = M.overlap_bbox(Am & Bm)
        c = M.grid_tiepoints(A, Am, B, Bm, t["dx"], t["dy"], grid=grid, bbox=bbox)
        if c is None:
            return None, "locked at (%.2f, %.2f) but no tie point survived" % (t["dx"], t["dy"])
        c.detail.update({k: t[k] for k in ("peak", "margin", "subpixel_dx",
                                           "subpixel_dy", "at_search_edge")})
        return c, ""
    return M.sparse(A, Am, B, Bm, name), ""


def _warp(src, svalid, Hm, shape, model):
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = Hm[:2, :] if Hm.shape[0] >= 2 else Hm
    if Hm.shape == (3, 3):
        H = Hm.astype(np.float64)
    out = cv2.warpPerspective(np.nan_to_num(src).astype(np.float32), H,
                              (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vm = cv2.warpPerspective(svalid.astype(np.uint8), H, (shape[1], shape[0]),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0).astype(bool)
    return out, vm


_ECC_MOTION = {"translation": cv2.MOTION_TRANSLATION,
               "euclidean": cv2.MOTION_EUCLIDEAN,
               "similarity": cv2.MOTION_AFFINE,   # OpenCV has no similarity motion
               "affine": cv2.MOTION_AFFINE,
               "homography": cv2.MOTION_HOMOGRAPHY,
               "projective": cv2.MOTION_HOMOGRAPHY}


def _ecc_polish(A, Am, B, Bm, H0, model, iters=60, eps=1e-6,
                max_corner_shift_px=3.0):
    """Sub-pixel refinement of a verified model by ECC, warm-started from H0.

    The discrete stages cannot beat the sampling they search on. Dense NCC
    returns an integer peak - the parabolic fit pulls that back to a fraction of
    a pixel, but only along the correlation surface it was given - and sparse
    keypoints inherit their detector's localisation. ECC (Evangelidis and
    Psarakis 2008) optimises the warp against the image data directly, so it is
    bounded by neither.

    It runs on GRADIENT MAGNITUDE rather than intensity, for the reason that runs
    through this whole project: ECC maximises a correlation coefficient, which is
    invariant to linear photometric change but NOT to inversion, and inversion is
    exactly what a Sun-azimuth reversal does to these scenes - the TMC-2/SELENE
    pair measures -0.41 on raw intensity. Gradient magnitude is polarity-blind,
    so one criterion covers both sides of that flip.

    Returns (H, info). The caller decides whether to adopt it; this function
    refuses only warps that have clearly run away from the verified model.
    """
    motion = _ECC_MOTION.get(model)
    info = {"attempted": True, "converged": False, "adopted": False,
            "correlation": None, "max_corner_shift_px": None, "note": None}
    if motion is None or H0 is None or not np.isfinite(H0).all():
        info.update(attempted=False, note="no ECC motion model for %r" % model)
        return None, info

    ta = M.gradient_magnitude(np.nan_to_num(A).astype(np.float32), Am)
    tb = M.gradient_magnitude(np.nan_to_num(B).astype(np.float32), Bm)
    for t in (ta, tb):
        np.nan_to_num(t, copy=False)

    # Direction matters and is easy to get backwards. Our Hm maps SOURCE pixels
    # to REFERENCE pixels (cv2.warpPerspective(src, Hm) lands in the reference
    # frame). findTransformECC returns the map used with WARP_INVERSE_MAP, i.e.
    # REFERENCE to SOURCE - the inverse. Warm-starting it with Hm instead of
    # Hm^-1 starts the search at roughly double the true displacement, and it
    # diverges rather than refining.
    H0f = np.eye(3, dtype=np.float64)
    H0a = np.asarray(H0, np.float64)
    H0f[:2, :] = H0a[:2, :]
    if H0a.shape == (3, 3):
        H0f = H0a
    try:
        W0 = np.linalg.inv(H0f)
    except np.linalg.LinAlgError:
        info["note"] = "the model is singular; nothing to polish"
        return None, info
    W0 = W0 / W0[2, 2] if abs(W0[2, 2]) > 1e-12 else W0
    W = np.ascontiguousarray(
        W0 if motion == cv2.MOTION_HOMOGRAPHY else W0[:2, :], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    mask = (Am.astype(np.uint8) * 255) if Am is not None else None
    try:
        cc, W = cv2.findTransformECC(tb, ta, W, motion, crit, mask, 5)
    except cv2.error as exc:
        # ECC raises rather than returning when it cannot converge; that is a
        # normal outcome on a low-texture pair, not a bug.
        info["note"] = "ECC did not converge: %s" % str(exc).strip().splitlines()[-1][:120]
        return None, info

    Wp = np.eye(3, dtype=np.float64)
    if motion == cv2.MOTION_HOMOGRAPHY:
        Wp = np.asarray(W, np.float64)
    else:
        Wp[:2, :] = np.asarray(W, np.float64)
    if not np.isfinite(Wp).all():
        info["note"] = "ECC returned a non-finite warp"
        return None, info
    try:
        Hp = np.linalg.inv(Wp)          # back to our source -> reference convention
    except np.linalg.LinAlgError:
        info["note"] = "ECC returned a singular warp"
        return None, info
    if abs(Hp[2, 2]) > 1e-12:
        Hp = Hp / Hp[2, 2]
    if not np.isfinite(Hp).all():
        info["note"] = "ECC warp did not invert cleanly"
        return None, info

    h, w = A.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    def proj(Hx):
        q = Hx @ corners
        return (q[:2] / np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])).T
    shift = float(np.linalg.norm(proj(Hp) - proj(H0f), axis=1).max())
    info.update(converged=True, correlation=round(float(cc), 6),
                max_corner_shift_px=round(shift, 4))
    if shift > max_corner_shift_px:
        info["note"] = ("ECC moved the frame corners by %.2f px, beyond the %.1f px "
                        "this stage is allowed to change a verified model; rejected"
                        % (shift, max_corner_shift_px))
        return None, info
    return Hp, info


def _metres_per_pixel(R: Scene, frame: dict):
    """Metres per working-grid pixel, or None when the reference has no scale.

    Returning a number here when the reference is a bare PNG would turn a pixel
    RMSE into a fake metre RMSE, which is exactly the kind of claim this project
    is supposed to refuse to make.
    """
    if R.gsd_m and R.georeferenced:
        return float(R.gsd_m) * frame.get("reference_decimation", 1)
    if R.gsd_m and R.reader in ("pds4", "pds3"):
        return float(R.gsd_m) * frame.get("reference_decimation", 1)
    return None


def _segment_fit(corr, vr, model, shape, segments):
    """Per-segment transforms for a long strip.

    A single 3x3 homography over a 730 km pushbroom strip is not slightly wrong,
    it is the wrong model class - each line has its own exterior orientation. When
    asked, the frame is split along its long axis and a transform fitted per band,
    so the residual per segment is visible rather than averaged away.
    """
    if segments < 2:
        return None
    from .. import register as _rg
    h, w = shape
    axis = 0 if h >= w else 1
    pts_s, pts_r = corr.src[vr.inlier_mask], corr.ref[vr.inlier_mask]
    coord = pts_r[:, 1] if axis == 0 else pts_r[:, 0]
    edges = np.linspace(0, h if axis == 0 else w, segments + 1)
    out = []
    for k in range(segments):
        sel = (coord >= edges[k]) & (coord < edges[k + 1])
        rec = {"segment": k, "axis": "row" if axis == 0 else "col",
               "from": float(edges[k]), "to": float(edges[k + 1]),
               "n_inliers": int(sel.sum())}
        if sel.sum() >= V._min_points(model):
            Hs = _rg.refit(pts_s[sel], pts_r[sel], model)
            if Hs is not None:
                res = V.transfer_error(Hs, pts_s[sel].astype(np.float64),
                                       pts_r[sel].astype(np.float64))
                rec["matrix"] = np.asarray(Hs, float).tolist()
                rec["rmse_px"] = round(float(np.sqrt((res ** 2).mean())), 4)
        out.append(rec)
    return out


def _write_matches(job_dir, corr, vr, back_x, back_y, frame):
    step = frame.get("reference_decimation", 1)
    orow, ocol = frame.get("reference_origin", [0, 0])
    with open(os.path.join(job_dir, "matches.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["src_x", "src_y", "ref_x", "ref_y", "confidence", "inlier"])
        for k in range(len(corr)):
            sx, sy = corr.src[k]
            # map back to ORIGINAL source pixel coordinates where we can
            j, i = int(round(sy)), int(round(sx))
            if 0 <= j < back_x.shape[0] and 0 <= i < back_x.shape[1] \
                    and np.isfinite(back_x[j, i]):
                osx, osy = float(back_x[j, i]), float(back_y[j, i])
            else:
                osx, osy = float(sx), float(sy)
            w.writerow(["%.3f" % osx, "%.3f" % osy,
                        "%.3f" % (ocol + float(corr.ref[k][0]) * step),
                        "%.3f" % (orow + float(corr.ref[k][1]) * step),
                        "%.4f" % float(corr.confidence[k]),
                        int(bool(vr.inlier_mask[k]))])


def _write_registered(job_dir, warped, wvalid, R: Scene, frame):
    path = os.path.join(job_dir, "registered.tif")
    out = np.where(wvalid, warped, np.nan).astype(np.float32)
    try:
        import rasterio
        from rasterio.transform import Affine
        step = frame.get("reference_decimation", 1)
        tr = None
        if R.georeferenced:
            t, (orow, ocol) = R.transform, frame.get("reference_origin", [0, 0])
            x0, y0 = t * (ocol, orow)
            tr = Affine(t.a * step, t.b, x0, t.d, t.e * step, y0)
        with rasterio.open(path, "w", driver="GTiff", height=out.shape[0],
                           width=out.shape[1], count=1, dtype="float32",
                           crs=R.crs if R.georeferenced else None,
                           transform=tr, nodata=float("nan"),
                           compress="deflate") as ds:
            ds.write(out, 1)
            ds.update_tags(georeferenced=str(bool(R.georeferenced)))
    except Exception:
        cv2.imwrite(path.replace(".tif", ".png"),
                    np.clip(np.nan_to_num(out) * 255, 0, 255).astype(np.uint8))


def _write_transform(job_dir, Hm, model, decomp, seg, frame, method):
    tj = {"model": model, "matrix": np.asarray(Hm, float).tolist(),
          "decomposition": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                            for k, v in (decomp or {}).items()},
          "frame": frame, "method": method,
          "coordinates": "reference pixel coordinates on the working grid; "
                         "multiply by frame.reference_decimation for full-resolution "
                         "reference pixels",
          "segments": seg}
    with open(os.path.join(job_dir, "transform.json"), "w") as fh:
        json.dump(tj, fh, indent=1)
    return tj


def _write_metrics(job_dir, job_id, S, R, character, frame, attempts, best, vr,
                   acc, cov, disp, m_per_px, degraded, warn, decomp, secs, grid,
                   eligible, extrap):
    n = len(best["corr"])
    ratio = vr.inlier_ratio
    rmse_px = acc.get("held_out_rmse_px", acc.get("fit_rmse_px"))
    d_az = None
    if S.sun_azimuth_deg is not None and R.sun_azimuth_deg is not None:
        d = abs(S.sun_azimuth_deg - R.sun_azimuth_deg) % 360.0
        d_az = round(min(d, 360.0 - d), 2)

    reasons = []
    if vr.n_inliers < 12:
        reasons.append("only %d verified inliers" % vr.n_inliers)
    if ratio < 0.15:
        reasons.append("inlier ratio %.1f%% is below 15%%" % (100 * ratio))
    if cov < 0.25:
        reasons.append("matches cover %.0f%% of the reference grid; the transform is "
                       "extrapolated over the rest" % (100 * cov))
    if extrap > 0.5:
        # Coverage alone misses this: tie points crowded into one lit strip can
        # clear the coverage bar against a small eligible set while most of the
        # frame still sits outside their hull, where the fit is extrapolated and
        # its error is unbounded by anything we measured.
        reasons.append("%.0f%% of the reference area lies outside the tie-point hull, "
                       "so the transform is extrapolated there and the quoted RMSE "
                       "does not describe it" % (100 * extrap))
    if character.get("overlap_fraction", 1) < 0.15:
        reasons.append("the images share only %.0f%% of the frame"
                       % (100 * character["overlap_fraction"]))
    reasons += list(warn or [])
    status = "pass" if not reasons else "warning"

    m = {"status": status, "reason": "; ".join(reasons) if reasons else None,
         "job_id": job_id, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "runtime_s": round(secs, 2),
         "method_used": best["name"], "model": best["model"],
         "accuracy": dict(acc, rmse_px=rmse_px,
                          rmse_m=(round(rmse_px * m_per_px, 3)
                                  if (rmse_px is not None and m_per_px) else None),
                          metres_per_pixel=m_per_px,
                          subpixel=bool(rmse_px is not None and rmse_px < 1.0)),
         "matches": {"candidates": n, "inliers": int(vr.n_inliers),
                     "inlier_ratio": round(ratio, 4)},
         "distribution": {"grid": list(grid), "coverage_fraction": round(cov, 4),
                          "dispersion": round(disp, 4),
                          "extrapolation_fraction": round(extrap, 4),
                          "eligible_cells": len(eligible), "total_cells": grid[0] * grid[1],
                          "note": "coverage is the fraction of ELIGIBLE reference grid "
                                  "cells holding at least one inlier, where eligible "
                                  "means both images have data there; dispersion is the "
                                  "mean distance from the centroid over the half-diagonal; "
                                  "extrapolation_fraction is the share of eligible cells "
                                  "OUTSIDE the convex hull of the inliers, where the "
                                  "transform is extrapolated rather than interpolated"},
         "illumination": {"delta_sun_azimuth_deg": d_az,
                          "source_sun_azimuth_deg": S.sun_azimuth_deg,
                          "reference_sun_azimuth_deg": R.sun_azimuth_deg,
                          "source_sun_incidence_deg": S.sun_incidence_deg,
                          "reference_sun_incidence_deg": R.sun_incidence_deg},
         "pair": character, "frame": frame, "decomposition": decomp,
         "attempts": attempts,
         "source": S.summary(), "reference": R.summary(),
         "degraded": degraded}
    with open(os.path.join(job_dir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    return m


def _write_overlay(job_dir, A, Am, B, Bm, warped, wvalid, corr, vr):
    def u8(a, m):
        return M.to_u8(a, m)

    Bu = u8(B, Bm)
    Wu = u8(warped, wvalid)
    h, w = Bu.shape
    tiles = 8
    checker = Bu.copy()
    ty, tx = max(1, h // tiles), max(1, w // tiles)
    for j in range(tiles + 1):
        for i in range(tiles + 1):
            if (i + j) % 2 == 0:
                sl = (slice(j * ty, min((j + 1) * ty, h)), slice(i * tx, min((i + 1) * tx, w)))
                checker[sl] = np.where(wvalid[sl], Wu[sl], Bu[sl])
    side = np.hstack([cv2.cvtColor(u8(A, Am), cv2.COLOR_GRAY2BGR),
                      np.full((h, 12, 3), 255, np.uint8),
                      cv2.cvtColor(Bu, cv2.COLOR_GRAY2BGR)])
    for k in np.nonzero(vr.inlier_mask)[0][:400]:
        p1 = tuple(np.round(corr.src[k]).astype(int))
        p2 = tuple(np.round(corr.ref[k]).astype(int) + np.array([w + 12, 0]))
        cv2.line(side, p1, p2, (90, 220, 120), 1, cv2.LINE_AA)
    ch3 = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    pad = abs(side.shape[1] - ch3.shape[1])
    if side.shape[1] > ch3.shape[1]:
        ch3 = np.hstack([ch3, np.full((h, pad, 3), 255, np.uint8)])
    else:
        side = np.hstack([side, np.full((side.shape[0], pad, 3), 255, np.uint8)])
    cv2.imwrite(os.path.join(job_dir, "overlay.png"),
                np.vstack([side, np.full((14, side.shape[1], 3), 255, np.uint8), ch3]))

    # Separate panels as well as the composite. The composite is what goes in a
    # report; a UI needs the layers apart so it can toggle between them and draw
    # its own match lines at whatever zoom the user is at.
    cv2.imwrite(os.path.join(job_dir, "source.png"), u8(A, Am))
    cv2.imwrite(os.path.join(job_dir, "reference.png"), Bu)
    cv2.imwrite(os.path.join(job_dir, "registered.png"), Wu)


def _write_preview_json(job_dir, A, B, corr, vr, limit=600):
    """Match coordinates in WORKING-GRID pixels, for drawing.

    matches.csv stays exactly as the contract specifies - original source pixels
    and full-resolution reference pixels - because that is what anyone consuming
    the product needs. Those are not the coordinates of the preview panels, and
    recovering one from the other needs the sampling map, which is not part of
    the artifact set. So the display coordinates are written separately rather
    than by bending the deliverable to suit the viewer.
    """
    n = len(corr)
    keep = np.arange(n)
    if n > limit:                       # keep every inlier we can, then fill
        inl = np.nonzero(vr.inlier_mask)[0]
        out = np.nonzero(~vr.inlier_mask)[0]
        keep = np.concatenate([inl[:limit], out[:max(0, limit - len(inl))]])
    pv = {"source": {"width": int(A.shape[1]), "height": int(A.shape[0])},
          "reference": {"width": int(B.shape[1]), "height": int(B.shape[0])},
          "n_total": int(n), "n_shown": int(len(keep)),
          "matches": [{"sx": round(float(corr.src[k][0]), 2),
                       "sy": round(float(corr.src[k][1]), 2),
                       "rx": round(float(corr.ref[k][0]), 2),
                       "ry": round(float(corr.ref[k][1]), 2),
                       "c": round(float(corr.confidence[k]), 4),
                       "inlier": bool(vr.inlier_mask[k])} for k in keep]}
    with open(os.path.join(job_dir, "preview.json"), "w") as fh:
        json.dump(pv, fh)


def _write_report(job_dir, job_id, S, R, m, tj, attempts):
    a, d = m["accuracy"], m["distribution"]
    L = ["# Registration report\n",
         "Job `%s`, %s.\n" % (job_id, m["generated_utc"]),
         "| | source | reference |", "|---|---|---|",
         "| file | `%s` | `%s` |" % (os.path.basename(S.path), os.path.basename(R.path)),
         "| reader | %s | %s |" % (S.reader, R.reader),
         "| instrument | %s | %s |" % (S.instrument, R.instrument),
         "| GSD (m) | %s | %s |" % (S.gsd_m, R.gsd_m),
         "| georeferenced | %s | %s |" % (S.georeferenced, R.georeferenced),
         "\n## Result\n",
         "**Status: %s**%s\n" % (m["status"].upper(),
                                 ("  \n" + m["reason"]) if m.get("reason") else ""),
         "| metric | value |", "|---|--:|",
         "| method used | `%s` |" % m["method_used"],
         "| model | %s |" % m["model"],
         "| candidates | %d |" % m["matches"]["candidates"],
         "| inliers | %d |" % m["matches"]["inliers"],
         "| inlier ratio | %.1f%% |" % (100 * m["matches"]["inlier_ratio"]),
         "| RMSE (px) | %s |" % a.get("rmse_px"),
         "| RMSE (m) | %s |" % a.get("rmse_m"),
         "| held-out points | %s |" % a.get("held_out_n"),
         "| sub-pixel | %s |" % a.get("subpixel"),
         "| coverage fraction | %.2f |" % d["coverage_fraction"],
         "| dispersion | %.2f |" % d["dispersion"],
         "| extrapolated area | %.0f%% |" % (100 * d.get("extrapolation_fraction", 0.0)),
         "| Δ Sun azimuth | %s |" % m["illumination"]["delta_sun_azimuth_deg"],
         "| scale ratio | %s |" % m["pair"]["scale_ratio"],
         "| cross-correlation | %s |" % m["pair"]["cross_correlation"],
         "| runtime (s) | %s |" % m["runtime_s"],
         "\n## Methods tried\n",
         "| method | candidates | inliers | ratio | coverage | seconds |",
         "|---|--:|--:|--:|--:|--:|"]
    for t in attempts:
        L.append("| `%s` | %d | %s | %s | %s | %s |"
                 % (t["method"], t.get("candidates", 0), t.get("inliers", "-"),
                    t.get("inlier_ratio", "-"), t.get("coverage", "-"), t.get("seconds")))
    L += ["\nThe winner is chosen by geometric verification, not by a hardcoded",
          "preference: Phase 2 measured that sparse learned matching wins on",
          "same-sensor illumination change while dense NCC wins on anti-correlated",
          "cross-sensor pairs, so the tool tries both and reports which won.\n"]
    if m.get("degraded"):
        L += ["## What was missing\n"]
        L += ["- %s" % x for x in m["degraded"]]
        L.append("")
    L += ["\n## Artifacts\n",
          "`matches.csv`, `registered.tif`, `transform.json`, `metrics.json`,",
          "`overlay.png`, `report.md`\n"]
    with open(os.path.join(job_dir, "report.md"), "w") as fh:
        fh.write("\n".join(L))
