"""Run all four real pairs and build the only current deck table.

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u \
  reports/validation_fixes_20260923/run_acceptance.py
"""
import contextlib
import csv
import json
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import register
from seleno.tool import warp_model as WM

HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/raw"
OUT = ROOT / "outputs/validation_fixes_20260923"
WAC = DATA / "wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
CASES = [
    ("OHRC → NAC", "ohrc_nac", DATA / "ch2/ohrc/data/calibrated/20241115/ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml",
     DATA / "nac/NAC_CM355_P892S2250_pole_6km.tif", {"max_side": 2048, "grid": (12, 12), "segments": 6}),
    ("TMC-2 → SELENE", "tmc2_selene", DATA / "ch2/tmc2/extracted/data/calibrated/20260813/ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml",
     DATA / "selene/TCO_MAPe04_S15E141S18E144SC.lbl", {"max_side": 2048, "grid": (12, 12), "segments": 6}),
    ("IIRS 2025-07-29 → WAC", "iirs_20250729_wac", DATA / "ch2/iirs/extracted/ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.xml", WAC, {}),
    ("IIRS 2024-01-20 → WAC", "iirs_20240120_wac", DATA / "ch2/iirs/extracted/ch2_iir_ndi_20240120T1432235872_d_rfl_d18_srd.xml", WAC, {}),
]
FIELDS = ["pair", "rmse_source_px", "rmse_reference_px", "rmse_working_px", "rmse_m",
          "median_source_px", "p90_source_px", "within_1_source_px", "within_2_source_px",
          "within_3_source_px", "held_out_n",
          "coverage", "extrapolated_fraction", "status", "subpixel", "accuracy_statement"]
ROBUST = FIELDS[5:11]


def robust_stats(e, units):
    """Median, p90 and within-k fractions over every held-out residual.

    Same sealed test set and residuals as the RMSE, no filtering; converted to
    source pixels exactly as rmse_source_px is.
    """
    sampling = units.get("source_px_per_reference_px")
    if len(e) < 3 or not sampling:
        return {k: None for k in ROBUST[:-1]} | {"held_out_n": len(e)}
    s = e * units["reference_decimation"] * sampling
    return {"median_source_px": float(np.median(s)),
            "p90_source_px": float(np.percentile(s, 90)),
            **{"within_%d_source_px" % k: float(np.mean(s <= k)) for k in (1, 2, 3)},
            "held_out_n": len(e)}


def persist(results):
    (HERE / "acceptance.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    table = []
    for result in results:
        m = result.get("metrics", {})
        a, d = m.get("accuracy", {}), m.get("distribution", {})
        r = result.get("robust_accuracy", {})
        table.append({"pair": result["pair"], **{k: a.get(k) for k in FIELDS[1:5]},
                      **{k: r.get(k) for k in ROBUST},
                      "coverage": d.get("coverage_fraction"),
                      "extrapolated_fraction": d.get("extrapolation_fraction"),
                      "status": result["status"], "subpixel": a.get("subpixel"),
                      "accuracy_statement": m.get("accuracy_statement", "Accuracy unavailable")})
    with (HERE / "deck_summary.csv").open("w") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    lines = ["| Pair | RMSE source px | RMSE reference px | RMSE working px | RMSE m | Median source px | P90 source px | ≤1 source px | ≤2 source px | ≤3 source px | Held-out n | Coverage | Extrapolated | Status | Subpixel (source) | Accuracy statement |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for row in table:
        values = []
        for key in FIELDS:
            value = row[key]
            if value is None:
                value = "unknown"
            elif key in ("coverage", "extrapolated_fraction") or key.startswith("within_"):
                value = "%.2f%%" % (100 * value)
            elif isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, float):
                value = "%.4f" % value
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    (HERE / "deck_summary.md").write_text("\n".join(lines) + "\n")


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    results = []
    for label, name, source, reference, options in CASES:
        print("START", label, options or "bare sensor defaults", flush=True)
        row = {"pair": label, "name": name, "source": str(source.relative_to(ROOT)),
               "reference": str(reference.relative_to(ROOT)), "options": options}
        try:
            with (HERE / (name + ".log")).open("w") as log, contextlib.redirect_stdout(log):
                result = register(str(source), str(reference), str(OUT / name), **options)
            row.update(status=result.status, reason=result.reason, metrics=result.metrics,
                       artifacts=str(Path(result.out_dir).relative_to(ROOT)))
            if result.ok:
                path = Path(result.out_dir)
                model = json.loads((path / "transform.json").read_text())
                ev = json.loads((path / "evaluation.json").read_text())
                # Independent artifact read-back, after the completed run. No
                # evaluated result is fed back into fitting or model selection.
                e = WM.residuals(model, np.array(ev["source"]), np.array(ev["reference"]))
                recomputed = float(np.sqrt(np.mean(e ** 2))) if len(e) >= 3 else None
                assert recomputed == result.metrics["accuracy"]["rmse_px"]
                assert WM.digest(model) == ev["transform_sha256"]
                acc = result.metrics["accuracy"]
                assert len(e) == acc["held_out_n"]
                if len(e) >= 3:
                    assert float(np.median(e)) == acc["held_out_median_px"]
                    assert float(np.percentile(e, 90)) == acc["held_out_p90_px"]
                row["robust_accuracy"] = robust_stats(e, acc["units"])
                row["artifact_checks"] = {"rmse_recomputed_exactly": True,
                                          "transform_sha256": WM.digest(model)}
                with rasterio.open(path / "registered.tif") as ds:
                    row["export"] = {"shape": ds.shape, "bands": ds.count,
                                     "transform": tuple(ds.transform), "crs": str(ds.crs)}
                # Small, reviewable evidence in Git; rasters remain in outputs/.
                evidence_dir = HERE / "evidence" / name
                evidence_dir.mkdir(parents=True, exist_ok=True)
                for artifact in ("metrics.json", "transform.json", "evaluation.json"):
                    (evidence_dir / artifact).write_bytes((path / artifact).read_bytes())
        except Exception:
            row.update(status="crashed", traceback=traceback.format_exc())
        results.append(row)
        persist(results)
        print("DONE", label, row["status"], row.get("metrics", {}).get("accuracy_statement"), flush=True)
    if any(r["status"] in ("failed", "crashed") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
