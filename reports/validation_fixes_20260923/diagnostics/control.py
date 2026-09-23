"""Positive control for the whole registration chain. Measurement only.

A synthetic reference is rendered from the REAL source product through the
source's own geometry lattice and a KNOWN affine, on the real reference's grid
(same CRS, transform, shape, GSD, sensor profile), then blurred to the
reference GSD and given noise. It is registered by `register()` with the same
options as the real pair, so every stage - placement, working grid, split,
matching, fine stage, export, scoring - runs exactly as it does for real data.
Only the image content differs: same sensor, same illumination, known answer.

Truth. A synthetic reference pixel p (native, zero-based centre coordinates)
shows the ground the source geometry puts at T p. The prealigned source puts at
working pixel w the ground at G w, with G = grid_to_reference(frame). So the
source-working -> reference-working map the run should recover is
G^-1 T^-1 G, i.e. T^-1 in native reference pixels.

    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u \
      reports/validation_fixes_20260923/diagnostics/control.py
"""
import contextlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import rasterio
import torch
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import register
from seleno.tool.register import _lattice_interpolators
from seleno.tool.scene import load

HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/raw"
OUT = ROOT / "outputs/validation_fixes_20260923/diagnostics"
OPTIONS = {"max_side": 2048, "grid": (12, 12), "segments": 6}   # as the real pairs
# Names must not carry a source-sensor token: profiles match on the filename and
# "..._tmc2.tif" would be read as TMC-2, not SELENE.
CASES = [
    {"name": "control_tmc2_selene", "label": "Control: TMC-2 → synthetic SELENE (1.35x)",
     "source": DATA / "ch2/tmc2/extracted/data/calibrated/20260813/ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml",
     "grid_of": DATA / "selene/TCO_MAPe04_S15E141S18E144SC.lbl",
     "file": "synthetic_selene_control_a.tif", "dtype": "uint16",
     "rotation_deg": 0.20, "scale": 1.003, "shift_px": (9.37, -4.62)},
    {"name": "control_ohrc_nac", "label": "Control: OHRC → synthetic NAC (4.17x)",
     "source": DATA / "ch2/ohrc/data/calibrated/20241115/ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml",
     "grid_of": DATA / "nac/NAC_CM355_P892S2250_pole_6km.tif",
     "file": "synthetic_nac_control_b.tif", "dtype": "uint8",
     "rotation_deg": -0.15, "scale": 0.997, "shift_px": (-6.71, 3.28)},
    # Same, but with a correction as large as the real OHRC run's: its median
    # correction at the held-out points is (-2536.3, +3799.9) reference px, i.e.
    # 4.6 km, which sends the fine stage through a km-scale prewarp.
    {"name": "control_ohrc_nac_km", "label": "Control: OHRC → synthetic NAC (4.17x, 4.6 km offset)",
     "source": DATA / "ch2/ohrc/data/calibrated/20241115/ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml",
     "grid_of": DATA / "nac/NAC_CM355_P892S2250_pole_6km.tif",
     "file": "synthetic_nac_control_c.tif", "dtype": "uint8",
     "rotation_deg": -0.15, "scale": 0.997, "shift_px": (2536.37, -3799.62)},
]
SNR = 40.0          # scene standard deviation / noise standard deviation
COARSE = 8          # geometry evaluated every 8 reference px, then bilinear
TILE = 512


def known_affine(centre, rotation_deg, scale, shift):
    """T in native reference centre coordinates: rotate/scale about `centre`, then shift."""
    a = math.radians(rotation_deg)
    A = np.array([[scale * math.cos(a), -scale * math.sin(a), 0],
                  [scale * math.sin(a), scale * math.cos(a), 0], [0, 0, 1]])
    c = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]], float)
    back = np.array([[1, 0, centre[0] + shift[0]], [0, 1, centre[1] + shift[1]], [0, 0, 1]], float)
    return back @ A @ c


def geometry(S, R):
    """Native reference centre (col, row) -> source centre (sample, line), as the
    pipeline's route 1 computes it."""
    projected = R.crs is not None and not R.crs.is_geographic
    f_s, f_l, prep = _lattice_interpolators(S.lonlat, R.crs if projected else None)
    t = R.transform

    def exact(cols, rows):
        x = t.a * (cols + 0.5) + t.b * (rows + 0.5) + t.c
        y = t.d * (cols + 0.5) + t.e * (rows + 0.5) + t.f
        a, b = prep(x, y)
        return f_s(a, b), f_l(a, b)
    return exact


def coarse_geometry(exact, S, row0, row1, col0, col1):
    """Source (sample, line) every COARSE reference px over [row0, row1) x [col0, col1)."""
    gr = np.arange(row0, row1 + COARSE, COARSE, dtype=np.float64)
    gc = np.arange(col0, col1 + COARSE, COARSE, dtype=np.float64)
    CS, CL = np.empty((gr.size, gc.size)), np.empty((gr.size, gc.size))
    for i in range(0, gr.size, 64):
        C, Rr = np.meshgrid(gc, gr[i:i + 64])
        CS[i:i + 64], CL[i:i + 64] = exact(C, Rr)
    ok = np.isfinite(CS) & np.isfinite(CL)
    ok &= (CS >= 0) & (CS <= S.array.shape[1] - 1) & (CL >= 0) & (CL <= S.array.shape[0] - 1)
    CS[~ok], CL[~ok] = np.nan, np.nan
    return gr, gc, CS, CL, ok


def render(case, S, R):
    h, w = R.array.shape[:2]
    exact = geometry(S, R)
    # The source footprint on the reference grid, per its own geometry, sets the
    # centre that the known rotation and scale act about.
    gr, gc, _, _, ok = coarse_geometry(exact, S, 0, h, 0, w)
    rows, cols = np.nonzero(ok)
    r0, r1 = int(gr[rows.min()]), int(min(h, gr[rows.max()] + 1))
    c0, c1 = int(gc[cols.min()]), int(min(w, gc[cols.max()] + 1))
    centre = ((c0 + c1) / 2.0, (r0 + r1) / 2.0)
    T = known_affine(centre, case["rotation_deg"], case["scale"], case["shift_px"])
    # Geometry is needed wherever T sends the grid, which a km-scale shift puts
    # outside the reference tile.
    corners = T @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], float)
    gy0 = int(np.floor(corners[1].min())) - 2 * COARSE
    gx0 = int(np.floor(corners[0].min())) - 2 * COARSE
    gy1 = int(np.ceil(corners[1].max())) + 2 * COARSE
    gx1 = int(np.ceil(corners[0].max())) + 2 * COARSE
    _, _, CS, CL, _ = coarse_geometry(exact, S, gy0, gy1, gx0, gx1)

    def lookup(qr, qc):
        at = [(qr - gy0) / COARSE, (qc - gx0) / COARSE]
        return (map_coordinates(CS, at, order=1, cval=np.nan),
                map_coordinates(CL, at, order=1, cval=np.nan))

    # How far the bilinear geometry departs from the exact interpolator.
    rng = np.random.default_rng(1)
    pc = rng.uniform(gx0, gx1, 20000)
    pr = rng.uniform(gy0, gy1, 20000)
    es, el = exact(pc, pr)
    bs, bl = lookup(pr, pc)
    d = np.hypot(es - bs, el - bl)
    d = d[np.isfinite(d)]

    f = case["source_px_per_ref_px"]
    sigma = 0.5 * f                       # the coarser sensor's footprint, in source px
    pad = int(math.ceil(4 * sigma)) + 3
    out = np.full((h, w), np.nan, np.float32)
    raw_nodata = S.nodata
    for tr in range(0, h, TILE):
        for tc in range(0, w, TILE):
            rr, cc = np.mgrid[tr:min(tr + TILE, h), tc:min(tc + TILE, w)].astype(np.float64)
            qc = T[0, 0] * cc + T[0, 1] * rr + T[0, 2]
            qr = T[1, 0] * cc + T[1, 1] * rr + T[1, 2]
            ss, ll = lookup(qr, qc)
            good = np.isfinite(ss) & np.isfinite(ll)
            if good.sum() < 16:
                continue
            l0 = max(0, int(np.floor(ll[good].min())) - pad)
            l1 = min(S.array.shape[0], int(np.ceil(ll[good].max())) + pad + 1)
            s0 = max(0, int(np.floor(ss[good].min())) - pad)
            s1 = min(S.array.shape[1], int(np.ceil(ss[good].max())) + pad + 1)
            raw = np.asarray(S.array[l0:l1, s0:s1], np.float32)
            valid = np.isfinite(raw) & (raw > -1e30)
            if raw_nodata is not None:
                valid &= raw != raw_nodata
            val = np.where(valid, raw * S.meta_scale + S.meta_offset, 0).astype(np.float32)
            k = (0, 0)
            num = cv2.GaussianBlur(val * valid, k, sigma, borderType=cv2.BORDER_REFLECT)
            den = cv2.GaussianBlur(valid.astype(np.float32), k, sigma, borderType=cv2.BORDER_REFLECT)
            img = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
            coords = [ll[good] - l0, ss[good] - s0]
            v = map_coordinates(img, coords, order=1, mode="nearest")
            m = map_coordinates(den, coords, order=1, mode="nearest")
            tile = np.full(rr.shape, np.nan, np.float32)
            tile[good] = np.where(m > 0.99, v, np.nan)
            out[tr:tr + rr.shape[0], tc:tc + rr.shape[1]] = tile

    finite = np.isfinite(out)
    std = float(np.std(out[finite]))
    noise_sigma = std / SNR
    out[finite] += np.random.default_rng(0).normal(0.0, noise_sigma, int(finite.sum())).astype(np.float32)
    hi = 255 if case["dtype"] == "uint8" else 65535
    q = np.zeros((h, w), case["dtype"])
    q[finite] = np.clip(np.rint(out[finite]), 1, hi).astype(case["dtype"])   # 0 is nodata
    path = OUT / "synthetic" / case["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype=case["dtype"], crs=R.crs, transform=R.transform, nodata=0,
                       tiled=True, blockxsize=512, blockysize=512, compress="deflate") as ds:
        ds.write(q, 1)
    return path, T, {"geometry_footprint_rows": [r0, r1], "geometry_footprint_cols": [c0, c1],
                     "rotation_scale_centre_ref_px": list(centre),
                     "blur_sigma_source_px": sigma, "noise_sigma_dn": noise_sigma,
                     "snr": SNR, "scene_std_dn": std, "valid_pixels": int(finite.sum()),
                     "geometry_bilinear_vs_exact_source_px": {
                         "median": float(np.median(d)), "max": float(d.max()), "n": int(d.size)}}


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    results = []
    for case in CASES:
        print("BUILD", case["label"], flush=True)
        S, R = load(str(case["source"])), load(str(case["grid_of"]))
        case["source_px_per_ref_px"] = R.gsd_m / S.gsd_m
        path, T, info = render(case, S, R)
        del S, R
        truth = {"T_native_ref_px": T.tolist(), "T_inverse_native_ref_px": np.linalg.inv(T).tolist(),
                 "meaning": "synthetic reference pixel p shows the ground the source geometry "
                            "places at T p; the recovered model should equal T^-1 in native "
                            "reference pixels",
                 "rotation_deg": case["rotation_deg"], "scale": case["scale"],
                 "shift_ref_px": list(case["shift_px"]),
                 "source_px_per_ref_px": case["source_px_per_ref_px"], **info}
        print("  wrote", path.relative_to(ROOT), json.dumps(info), flush=True)
        print("START", case["label"], OPTIONS, flush=True)
        with (HERE / (case["name"] + ".log")).open("w") as log, contextlib.redirect_stdout(log):
            result = register(str(case["source"]), str(path), str(OUT / case["name"]), **OPTIONS)
        row = {"pair": case["label"], "name": case["name"],
               "source": str(case["source"].relative_to(ROOT)),
               "reference": str(path.relative_to(ROOT)),
               "grid_of": str(case["grid_of"].relative_to(ROOT)), "options": OPTIONS,
               "truth": truth, "status": result.status, "reason": result.reason,
               "artifacts": str(Path(result.out_dir).relative_to(ROOT))}
        results.append(row)
        (HERE / "controls.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print("DONE", case["label"], result.status,
              (result.metrics or {}).get("accuracy_statement"), flush=True)


if __name__ == "__main__":
    main()
