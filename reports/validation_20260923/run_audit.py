"""Reproducible functional audit; does not change application code.

Run: OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u
     reports/validation_20260923/run_audit.py {real,lroc,synthetic}
"""
import contextlib
import csv
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import register
from seleno.metrics import corner_error

HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "validation_20260923"
CRS = "+proj=stere +lat_0=-90 +R=1737400 +units=m"
EXISTING = {}


def project(H, p):
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H).T
    return q[:, :2] / q[:, 2:]


def run(name, src, ref, truth=None, shape=None, **kw):
    if name in EXISTING:
        print("PRESERVED", name, flush=True)
        return EXISTING[name]
    started = time.time()
    row = {"name": name, "source": str(src), "reference": str(ref), "options": kw}
    print("START", name, flush=True)
    try:
        with open(HERE / (name + ".log"), "w") as log, contextlib.redirect_stdout(log):
            res = register(str(src), str(ref), str(OUT / name), **kw)
        row.update(status=res.status, reason=res.reason, out_dir=res.out_dir)
        m = res.metrics
        for key in ("accuracy", "matches", "distribution", "fine_stage", "method_used", "frame", "degraded"):
            row[key] = m.get(key)
        if res.status != "failed":
            with rasterio.open(Path(res.out_dir) / "registered.tif") as ds:
                row["export"] = {"shape": list(ds.shape), "bands": ds.count,
                                 "transform": list(ds.transform), "crs": str(ds.crs)}
            pts = np.genfromtxt(Path(res.out_dir) / "matches.csv", delimiter=",", names=True)
            pts = np.atleast_1d(pts)
            pts = pts[pts["inlier"] == 1]
            if truth is not None:
                H = np.asarray(res.transform["matrix_reference_px"])
                # Exported matrix is a correction AFTER prealignment. Compose
                # the prealignment before comparing an original-source truth.
                base = np.eye(3)
                if res.transform["frame"].get("placement") is not None:
                    base = np.linalg.inv(np.asarray(res.transform["frame"]["placement"]))
                elif str(src).endswith(".tif") and str(ref).endswith(".tif"):
                    with rasterio.open(src) as ss, rasterio.open(ref) as rr:
                        if ss.crs == rr.crs:
                            af = Affine.translation(-.5, -.5) * ~rr.transform * ss.transform * Affine.translation(.5, .5)
                            base = np.asarray(af).reshape(3, 3)
                row["direct_export_corner_error_px"] = corner_error(H, truth, shape)["corner_error_px"]
                row.update(corner_error(H @ base, truth, shape))
                ps = np.c_[pts["src_x"], pts["src_y"]]
                pr = np.c_[pts["ref_x"], pts["ref_y"]]
                err = np.linalg.norm(project(truth, ps) - pr, axis=1)
                row["csv_truth_rmse_px"] = float(np.sqrt(np.mean(err ** 2)))
                row["csv_truth_median_px"] = float(np.median(err))
                row["csv_matrix_direct_rmse_px"] = float(np.sqrt(np.mean(
                    np.linalg.norm(project(H, ps) - pr, axis=1) ** 2)))
    except Exception:
        row.update(status="crashed", traceback=traceback.format_exc())
    row["seconds"] = round(time.time() - started, 2)
    print("DONE", name, row["status"], "src_rmse", (row.get("accuracy") or {}).get("rmse_source_px"),
          "truth_corner", row.get("corner_error_px"), "seconds", row["seconds"], flush=True)
    return row


def real():
    d = ROOT / "data/raw"
    o = d / "ch2/ohrc/data/calibrated/20241115/ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml"
    t = d / "ch2/tmc2/extracted/data/calibrated/20260813/ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml"
    i = d / "ch2/iirs/extracted/ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.xml"
    i2 = d / "ch2/iirs/extracted/ch2_iir_ndi_20240120T1432235872_d_rfl_d18_srd.xml"
    cases = [("real_ohrc_nac", o, d / "nac/NAC_CM355_P892S2250_pole_6km.tif"),
             ("real_tmc2_selene", t, d / "selene/TCO_MAPe04_S15E141S18E144SC.lbl"),
             ("real_iirs_wac", i, d / "wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"),
             ("real_iirs_ohrc_disjoint", i2, o)]
    for name, src, ref in cases:
        yield run(name, src, ref, max_side=2048, grid=(12, 12), segments=6)


def lroc():
    manifest = json.loads((ROOT / "data/lroc/manifest.json").read_text())
    for p in manifest["pairs"]:
        row = run(p["id"], ROOT / "data/lroc/pairs" / p["source_file"],
                  ROOT / "data/lroc/pairs" / p["reference_file"],
                  truth=p["gt_homography"], shape=p["source_shape"], max_side=1024)
        row.update(negative=p["negative"], delta_sun_lon_deg=p["delta_sun_lon_deg"])
        yield row


def write_tif(path, arr, transform):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=CRS, transform=transform) as ds:
        ds.write(arr.astype(np.float32), 1)
    return path


def synthetic():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_tool import terrain
    fixtures = OUT / "inputs"
    fixtures.mkdir(parents=True, exist_ok=True)
    n = 512
    a = 0.1 + 0.8 * terrain(147, n)
    ref = fixtures / "reference.png"
    cv2.imwrite(str(ref), (255 * a).astype(np.uint8))
    for name, angle, scale, mode in [
        ("translation", 0, 1, "normal"), ("brightness", 0, 1, "gamma"),
        ("contrast_inversion", 0, 1, "inverted"), ("rotation_20", 20, 1, "normal"),
        ("rotation_60", 60, 1, "normal"), ("scale_1p5", 0, 1.5, "normal"),
        ("perspective", 0, 1, "perspective")]:
        G = np.eye(3)
        G[:2] = cv2.getRotationMatrix2D((n / 2, n / 2), angle, scale)
        G[:2, 2] += (7.35, -4.65)
        if mode == "perspective":
            G[2, :2] = (0.00015, -0.0001)
        b = cv2.warpPerspective(a, G, (n, n))
        if mode == "gamma":
            b = 0.1 + 0.75 * b ** 1.8
        if mode == "inverted":
            b = np.where(b > 0, 1 - b, 0)
        src = fixtures / (name + ".png")
        cv2.imwrite(str(src), np.clip(255 * b, 0, 255).astype(np.uint8))
        yield run("synthetic_" + name, src, ref, truth=np.linalg.inv(G), shape=a.shape, max_side=512)

    # Same ground at different pixel sizes, with exact centre-coordinate truth.
    master = 0.1 + 0.8 * terrain(149, 1024)
    tr = Affine(1, 0, 0, 0, -1, 0)
    src = write_tif(fixtures / "fine_source.tif", master, tr)
    coarse = cv2.resize(master, (256, 256), interpolation=cv2.INTER_AREA)
    ref4 = write_tif(fixtures / "coarse_reference.tif", coarse, tr * Affine.scale(4))
    H = np.array([[.25, 0, -.375], [0, .25, -.375], [0, 0, 1]])
    yield run("synthetic_scale4_geo", src, ref4, truth=H, shape=master.shape, max_side=512)
    yield run("synthetic_decimated_identity", src, src, truth=np.eye(3),
              shape=master.shape, max_side=256, segments=4)

    # Isolate the GeoTIFF writer on a rotated/sheared reference grid.
    from seleno.tool.scene import load
    from seleno.tool.register import _write_registered
    ref_rot = write_tif(fixtures / "rotated_geogrid.tif", a, Affine(2, .3, 100, .2, -2, 200))
    scene = load(str(ref_rot))
    dest = OUT / "geogrid_writer"
    dest.mkdir(parents=True, exist_ok=True)
    frame = {"reference_decimation": 4, "reference_origin": [8, 12]}
    _write_registered(str(dest), a[::4, ::4], np.ones((128, 128), bool), scene, frame)
    with rasterio.open(dest / "registered.tif") as ds:
        expected = scene.transform * Affine.translation(12, 8) * Affine.scale(4)
        got = ds.transform
        actual_xy, expected_xy = got * (127, 127), expected * (127, 127)
        yield {"name": "rotated_geogrid_writer", "actual": list(got), "expected": list(expected),
               "corner_displacement_m": float(np.linalg.norm(np.array(actual_xy) - expected_xy))}


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    mode = sys.argv[1]
    if "--resume" in sys.argv[2:]:
        previous = HERE / (mode + ".json")
        if previous.exists():
            EXISTING = {r["name"]: r for r in json.loads(previous.read_text())}
    rows = []
    for row in {"real": real, "lroc": lroc, "synthetic": synthetic}[mode]():
        rows.append(row)
        (HERE / (mode + ".json")).write_text(json.dumps(rows, indent=2, default=str))
