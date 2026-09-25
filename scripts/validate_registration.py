"""Fixed-threshold registration acceptance, including independent synthetic truth.

systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 \
    env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python \
    scripts/validate_registration.py --suite synthetic

Each run writes fresh evidence under --out. A completed registration, a low
matcher residual and independent verification are distinct outcomes. The exit
code is nonzero when any selected positive case misses the acceptance gates.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import register
from seleno.tool import evaluation as E
from seleno.tool.coordinates import project, grid_to_reference
from seleno.tool import warp_model
from seleno.tool.scene import load
from seleno.tool.export import reference_to_source


def terrain(seed, size):
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 6.)
    a += .6 * cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 1.5)
    for _ in range(max(18, size * size // 5000)):
        y, x = rng.integers(20, size - 20, 2)
        radius = int(rng.integers(5, 16))
        cv2.circle(a, (int(x), int(y)), radius, float(rng.uniform(.2, .9)), -1)
        cv2.circle(a, (int(x), int(y)), radius, float(rng.uniform(0, .3)), 2)
    return .1 + .8 * (a - a.min()) / np.ptp(a)


def write_tif(path, array, transform):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
            count=1, dtype="float32", crs="+proj=stere +lat_0=-90 +R=1737400 +units=m",
            transform=transform) as dst:
        dst.write(array.astype(np.float32), 1)
    return path


def truth_manifest(directory, case):
    """Freeze controls on a regular native grid before invoking registration."""
    directory.mkdir(parents=True, exist_ok=True)
    height, width = case["source_shape"]
    yy, xx = np.meshgrid(np.linspace(8, height - 9, 17), np.linspace(8, width - 9, 17), indexing="ij")
    source = np.column_stack([xx.ravel(), yy.ravel()])
    reference = project(case["truth"], source)
    rh, rw = case["reference_shape"]
    valid = ((reference[:, 0] >= 0) & (reference[:, 0] <= rw - 1) &
             (reference[:, 1] >= 0) & (reference[:, 1] <= rh - 1))
    provenance = case.get("provenance", {"kind": "synthetic_transform",
        "reference": "scripts/validate_registration.py deterministic image construction, seed 147",
        "independent_of_registration": True, "uncertainty_source_px": 0.})
    document = {"schema_version": 1, "coordinates": "native_pixel_centres",
        "provenance": provenance,
        "points": [{"id": "grid_%03d" % i, "source": p.tolist(), "reference": q.tolist()}
                   for i, (p, q) in enumerate(zip(source[valid], reference[valid]))]}
    for name in ("source", "reference"):
        path = case[name].resolve()
        document[name] = {"path": str(path), "sha256": E.file_sha256(path),
                          "shape": list(case[name + "_shape"])}
    manifest = directory / (case["name"] + "_truth.json")
    manifest.write_text(json.dumps(document, indent=2) + "\n")
    return manifest


def synthetic_cases(directory):
    directory.mkdir(parents=True, exist_ok=True)
    size = 512
    a = terrain(147, size)
    reference = directory / "reference.png"
    cv2.imwrite(str(reference), (255 * a).astype(np.uint8))
    for name, angle, scale, illumination in [
        ("translation", 0., 1., "normal"), ("rotation_60", 60., 1., "normal"),
        ("scale_1p5_no_metadata", 0., 1.5, "normal"),
        ("scale_0p5_no_metadata", 0., .5, "normal"),
        ("gamma", 0., 1., "gamma"), ("contrast_inversion", 0., 1., "inverted"),
        ("rotation_120_scale_1p5", 120., 1.5, "normal")]:
        h = np.eye(3)
        h[:2] = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
        h[:2, 2] += (7.35, -4.65)
        b = cv2.warpPerspective(a, h, (size, size))
        if illumination == "gamma":
            b = np.where(b > 0, .1 + .75 * b ** 1.8, 0)
        if illumination == "inverted":
            b = np.where(b > 0, 1 - b, 0)
        source = directory / (name + ".png")
        cv2.imwrite(str(source), np.clip(255 * b, 0, 255).astype(np.uint8))
        yield {"name": name, "source": source, "reference": reference, "truth": np.linalg.inv(h),
               "source_shape": (size, size), "reference_shape": (size, size),
               "options": {"max_side": 512}}
    master = terrain(149, 1024)
    tr = Affine(1, 0, 0, 0, -1, 0)
    source = write_tif(directory / "native_source.tif", master, tr)
    coarse = cv2.resize(master, (256, 256), interpolation=cv2.INTER_AREA)
    ref4 = write_tif(directory / "coarse_reference.tif", coarse, tr * Affine.scale(4))
    yield {"name": "scale4_georeferenced", "source": source, "reference": ref4,
           "truth": np.array([[.25, 0, -.375], [0, .25, -.375], [0, 0, 1.]]),
           "source_shape": (1024, 1024), "reference_shape": (256, 256), "options": {"max_side": 512}}
    yield {"name": "decimated_identity", "source": source, "reference": source,
           "truth": np.eye(3), "source_shape": (1024, 1024), "reference_shape": (1024, 1024),
           "options": {"max_side": 256, "segments": 4}}


def real_cases():
    data = ROOT / "data/raw"
    wac = data / "wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    pairs = [
        ("ohrc_nac", "ch2/ohrc/data/calibrated/20241115/ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml",
         data / "nac/NAC_CM355_P892S2250_pole_6km.tif", {"max_side": 2048, "grid": (12, 12), "segments": 6}),
        # The evening map is the primary case. The low-sun morning map has
        # heavier shadows than this high-sun (51 degree) TMC-2 observation.
        ("tmc2_selene", "ch2/tmc2/extracted/data/calibrated/20260813/ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml",
         data / "selene/TCO_MAPe04_S15E141S18E144SC.lbl", {"max_side": 2048, "grid": (12, 12), "segments": 6}),
        ("iirs_20250729_wac", "ch2/iirs/extracted/ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.xml", wac, {}),
        ("iirs_20240120_wac", "ch2/iirs/extracted/ch2_iir_ndi_20240120T1432235872_d_rfl_d18_srd.xml", wac, {}),
        ("tmc2_selene_morning", "ch2/tmc2/extracted/data/calibrated/20260813/ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml",
         data / "selene/TCO_MAPm04_S15E141S18E144SC.lbl", {"max_side": 2048, "grid": (12, 12), "segments": 6})]
    for name, source, reference, options in pairs:
        yield {"name": name, "source": data / source, "reference": reference, "options": options,
               "role": "secondary cross-illumination" if name.endswith("morning") else "primary"}


def illumination_cases():
    document = json.loads((ROOT / "data/lroc/manifest.json").read_text())
    for pair in document["pairs"]:
        yield {"name": pair["id"], "source": ROOT / "data/lroc/pairs" / pair["source_file"],
               "reference": ROOT / "data/lroc/pairs" / pair["reference_file"],
               "source_shape": pair["source_shape"], "reference_shape": pair["reference_shape"],
               "truth": pair["gt_homography"], "negative": pair["negative"],
               "sun_angle_difference_deg": pair["delta_sun_lon_deg"], "options": {"max_side": 1024},
               "provenance": {"kind": "synthetic_transform",
                   "reference": "data/lroc/manifest.json controlled-mosaic grid plus known image warp; nominal map controls, not surveyed detector points",
                   "independent_of_registration": True, "uncertainty_source_px": .8}}


def corner_error(case, directory):
    """Forward error of the complete exported model at four native source corners.

    These controlled cases use an affine metadata placement (or pixel space),
    followed by the serialized residual model, including any local terms.
    Corners may be outside the measured overlap; this exposes extrapolation.
    """
    model = json.loads((Path(directory) / "transform.json").read_text())
    source, reference = load(str(case["source"])), load(str(case["reference"]))
    if source.lonlat is not None or (source.georeferenced and reference.georeferenced
                                      and source.crs != reference.crs):
        raise ValueError("corner controls require affine metadata placement")
    basis = np.array([[0., 0.], [1., 0.], [0., 1.]])
    mapped = reference_to_source(source, reference, model["frame"], basis)
    placement = np.eye(3)
    placement[:2, 0], placement[:2, 1] = mapped[1] - mapped[0], mapped[2] - mapped[0]
    placement[:2, 2] = mapped[0]
    height, width = case["source_shape"]
    corners = np.array([[0., 0.], [width - 1., 0.], [width - 1., height - 1.], [0., height - 1.]])
    c = grid_to_reference(model["frame"])
    initial = project(np.linalg.inv(c) @ np.linalg.inv(placement), corners)
    predicted = project(c, warp_model.forward_points(model, initial))
    expected = project(case["truth"], corners)
    stats = E.error_statistics(predicted - expected)
    return {"definition": "forward transfer at four source corners in native reference pixels; includes extrapolation",
            "statistics": stats, "source_corners": corners.tolist(),
            "expected_reference": expected.tolist(),
            "predicted_reference": np.where(np.isfinite(predicted), predicted, None).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("synthetic", "real", "illumination"), default="synthetic")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/validation_20260925")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/validation_20260925")
    parser.add_argument("--case", action="append", help="run only the named case; repeatable")
    parser.add_argument("--ground-truth-directory", type=Path, help="externally acquired <case>.json manifests for real pairs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    cases = {"synthetic": lambda: synthetic_cases(args.artifacts / "inputs"),
             "real": real_cases, "illumination": illumination_cases}[args.suite]()
    results = []
    for case in cases:
        if args.case and case["name"] not in args.case:
            continue
        name = case["name"]
        print("START", name, flush=True)
        started = time.monotonic()
        row = {"name": name, "suite": args.suite, "source": str(case["source"]),
               "reference": str(case["reference"]), "negative": case.get("negative", False),
               "role": case.get("role", "controlled experiment"),
               "sun_angle_difference_deg": case.get("sun_angle_difference_deg")}
        try:
            # Negative controls carry no transform (manifest gt_homography null).
            truth = (truth_manifest(args.artifacts / "inputs", case)
                     if case.get("truth") is not None else None)
            if args.ground_truth_directory:
                candidate = args.ground_truth_directory / (name + ".json")
                if candidate.exists():
                    truth = candidate
            options = dict(case["options"])
            if truth:
                options["ground_truth"] = str(truth)
            with (args.out / (name + ".log")).open("w") as log, contextlib.redirect_stdout(log):
                result = register(str(case["source"]), str(case["reference"]),
                                  str(args.artifacts / name), **options)
            metrics = result.metrics
            row.update(status=result.status, reason=result.reason, metrics=metrics, out_dir=result.out_dir)
            acceptance = metrics.get("acceptance", {})
            if case.get("negative"):
                row["passed"] = result.status == "failed" or acceptance.get("passed") is False
            else:
                row["passed"] = acceptance.get("passed") is True
                if truth:
                    external = metrics.get("accuracy", {}).get("independent_ground_truth", {})
                    row["passed"] = row["passed"] and external.get("passed") is True
            if result.ok:
                evidence = Path(result.out_dir)
                if "truth" in case and not case.get("negative"):
                    row["corner_error"] = corner_error(case, evidence)
                with rasterio.open(evidence / "registered.tif") as ds:
                    row["export"] = {"shape": list(ds.shape), "bands": ds.count,
                                     "transform": list(ds.transform), "crs": str(ds.crs)}
                review = args.out / "evidence" / name
                review.mkdir(parents=True, exist_ok=True)
                for filename in ("metrics.json", "transform.json", "evaluation.json", "ground_truth_evaluation.json"):
                    if (evidence / filename).exists():
                        (review / filename).write_bytes((evidence / filename).read_bytes())
        except Exception:
            row.update(status="crashed", passed=False, traceback=traceback.format_exc())
        row["seconds"] = round(time.monotonic() - started, 2)
        row["process_peak_rss_mb_so_far"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024., 1)
        results.append(row)
        (args.out / (args.suite + ".json")).write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print("DONE", name, row["status"], "accepted", row["passed"], row["seconds"], "s", flush=True)
    if not results:
        parser.error("no cases selected")
    runtime = {"threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
               "tmpdir": os.environ.get("TMPDIR"),
               "process_peak_rss_mb": results[-1]["process_peak_rss_mb_so_far"]}
    group = Path("/sys/fs/cgroup") / Path(Path("/proc/self/cgroup").read_text().strip().split("::")[1]).relative_to("/")
    for key in ("memory.max", "memory.swap.max", "memory.peak", "memory.events"):
        if (group / key).exists():
            runtime[key] = (group / key).read_text().strip()
    (args.out / (args.suite + "_runtime.json")).write_text(json.dumps(runtime, indent=2) + "\n")
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)


if __name__ == "__main__":
    main()
