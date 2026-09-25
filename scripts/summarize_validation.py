"""Regenerate review/deck tables from completed validation, never from dev bests.

Run after validate_registration.py, in the same capped, two-thread environment.
No fitting, selection or registration is performed here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import warp_model as WM
from seleno.tool.coordinates import grid_to_reference


def number(value, digits=3):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join("—" if value is None else str(value) for value in row)
                        + " |" for row in rows]) + "\n"


def distance_stats(values):
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    return {"n": len(values), "invalid_n": int(len(values) - len(valid)),
            "rmse": float(np.sqrt(np.mean(valid ** 2))) if len(valid) else None,
            "median": float(np.median(valid)) if len(valid) else None,
            "p90": float(np.percentile(valid, 90)) if len(valid) else None,
            "within_1": float(np.mean(valid <= 1.)) if len(valid) else None}


def real_rows(rows, out):
    result = []
    for row in rows:
        name, m = row["name"], row.get("metrics", {})
        a, f, d = m.get("accuracy", {}), m.get("fine_stage", {}), m.get("distribution", {})
        path = out / "evidence" / name
        ev = (json.loads((path / "evaluation.json").read_text())
              if (path / "evaluation.json").exists() else {"source": [], "reference": []})
        model = (json.loads((path / "transform.json").read_text())
                 if (path / "transform.json").exists() else {"matrix": np.eye(3).tolist(), "frame": {}})
        cp = np.asarray(ev.get("check_point", [True] * len(ev["source"])), bool)
        missing = np.full((len(cp), 2), np.nan)
        source_error = (np.asarray(ev.get("predicted_native_source", missing), float).reshape(-1, 2)
                        - np.asarray(ev.get("native_source", missing), float).reshape(-1, 2))
        reference_error = ((WM.forward_points(model, np.asarray(ev["source"], float).reshape(-1, 2))
                            - np.asarray(ev["reference"], float).reshape(-1, 2))
                           @ grid_to_reference(model["frame"])[:2, :2].T)
        src = np.linalg.norm(source_error, axis=1)
        ref = np.linalg.norm(reference_error, axis=1)
        units = a.get("units", {})
        metres = units.get("metres_per_reference_px")
        terms = [m.get("model", "global")]
        if model.get("parallax"):
            terms.append("terrain")
        if model.get("local_field"):
            terms.append("field")
        if model.get("segments"):
            terms.append("segments")
        result.append({"name": name, "role": row.get("role"), "status": row["status"],
                       "accepted": row.get("passed", False), "seconds": row["seconds"],
                       "source": distance_stats(src[cp]), "reference": distance_stats(ref[cp]),
                       "metres": distance_stats(ref[cp] * metres) if metres else None,
                       "all_source": distance_stats(src), "check_fraction": a.get("check_point_fraction"),
                       "check_n": int(cp.sum()), "held_n": len(cp),
                       "coverage": d.get("coverage_fraction"), "extrapolation": d.get("extrapolation_fraction"),
                       "models": " + ".join(terms) if row.get("export") else "none (refused)", "patch": f.get("patch_px"),
                       "patch_selection": f.get("patch_selection"), "patch_cv": f.get("patch_cv"),
                       "points": f.get("points"), "inliers": m.get("matches", {}).get("inliers"),
                       "inlier_ratio": m.get("matches", {}).get("inlier_ratio"),
                       "fit": f.get("model_fit"), "export": row.get("export"),
                       "reasons": m.get("acceptance", {}).get("reasons", [m.get("message") or row.get("reason")]),
                       "out_dir": row.get("out_dir")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/validation_20260925")
    args = parser.parse_args()
    out = args.out
    suites = {name: json.loads((out / (name + ".json")).read_text())
              for name in ("real", "synthetic", "illumination")}
    real = real_rows(suites["real"], out)
    flat = []
    for r in real:
        item = {k: r[k] for k in ("name", "role", "status", "accepted", "check_n", "held_n", "check_fraction",
                                 "coverage", "extrapolation", "models", "patch", "points", "inliers", "seconds")}
        for unit in ("source", "reference", "metres", "all_source"):
            item.update({unit + "_" + key: value for key, value in (r[unit] or {}).items()})
        flat.append(item)
    with (out / "deck_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    deck = "# Registration results — 2026-09-25\n\n"
    deck += "Fresh final runs. Held-out check points are matcher measurements, not surveyed controls. "
    deck += "Source errors use backward transfer; reference/metre errors use forward transfer. "
    deck += "Percentages in this table use ≤1 source pixel. All real cases remain unverified independently.\n\n"
    deck += table(["Pair", "Role", "Source RMSE / median / p90 (px)", "≤1 src px", "Reference RMSE (px)",
                   "RMSE (m)", "Coverage / extrapolation", "Check / held", "Patch", "Model", "Accepted"],
                  [[r["name"], r["role"], " / ".join(number(r["source"][k]) for k in ("rmse", "median", "p90")),
                    percent(r["source"]["within_1"]), number(r["reference"]["rmse"]),
                    number((r["metres"] or {}).get("rmse")),
                    percent(r["coverage"]) + " / " + percent(r["extrapolation"]),
                    f'{r["check_n"]}/{r["held_n"]}', r["patch"], r["models"], str(r["accepted"])] for r in real])
    deck += "\n[Full report](REPORT.md) · [CSV](deck_summary.csv) · [Raw real results](real.json)\n"
    (out / "deck_summary.md").write_text(deck)
    detailed = "## Real-pair check points\n\n"
    detailed += table(["Pair", "Unit", "RMSE", "Median", "p90", "≤1 unit", "Invalid / n"],
                       [[r["name"], unit, number(s["rmse"]), number(s["median"]), number(s["p90"]),
                         percent(s["within_1"]), f'{s["invalid_n"]}/{s["n"]}']
                        for r in real for unit in ("source", "reference", "metres")
                        if (s := r[unit]) is not None])
    detailed += "\n## Spatial support and unscreened evidence\n\n"
    detailed += table(["Pair", "Coverage", "Extrapolation", "Check-point retention", "All-held source RMSE",
                        "All-held invalid", "Fit inliers", "Native points", "Patch", "Terms", "Accepted"],
                       [[r["name"], percent(r["coverage"]), percent(r["extrapolation"]),
                         percent(r["check_fraction"]), number(r["all_source"]["rmse"]), r["all_source"]["invalid_n"],
                         r["inliers"], r["points"], r["patch"], r["models"], r["accepted"]] for r in real])
    for suite in ("synthetic", "illumination"):
        detailed += f"\n## {suite.title()} controls\n\n"
        detailed += table(["Case", "Sun Δ°", "Status", "Truth RMSE (src px)",
                            "Corner RMSE / max (ref px)", "Acceptance"],
                           [[r["name"], r.get("sun_angle_difference_deg", "—"),
                             r["status"] + (" (negative)" if r.get("negative") else ""),
                             number(r.get("metrics", {}).get("accuracy", {}).get("independent_ground_truth", {}).get("statistics", {}).get("rmse_px")),
                             " / ".join(number(r.get("corner_error", {}).get("statistics", {}).get(k)) for k in ("rmse_px", "max_px")),
                             r.get("passed", False)] for r in suites[suite]])
    bins = []
    for lo, hi in ((0, 60), (60, 90), (90, 135), (135, 180)):
        rows = [r for r in suites["illumination"] if lo < r["sun_angle_difference_deg"] <= hi
                or lo == 0 and r["sun_angle_difference_deg"] == 0]
        positive = [r for r in rows if not r.get("negative")]
        errors = [r["corner_error"]["statistics"]["rmse_px"] for r in positive
                  if r.get("corner_error", {}).get("statistics", {}).get("rmse_px") is not None]
        bins.append({"sun_range": f"{lo}–{hi}", "positive_n": len(positive), "negative_n": len(rows) - len(positive),
                     "exported_n": sum(r["status"] not in ("failed", "crashed") for r in positive),
                     "accepted_n": sum(r.get("passed", False) for r in positive),
                     "corner_median": float(np.median(errors)) if errors else None,
                     "corner_worst": max(errors) if errors else None,
                     "negative_rejected_n": sum(r.get("passed", False) for r in rows if r.get("negative"))})
    detailed += "\n## Illumination by sun-angle difference\n\n"
    detailed += "Corner aggregates include exported positive cases only; refusal counts are shown beside them. "
    detailed += "Bins include their upper endpoint; adjacent lower endpoints are excluded.\n\n"
    detailed += table(["Sun Δ°", "Positive cases", "Exported", "Accepted", "Median / worst corner RMSE (ref px)", "Negative rejected or flagged / n"],
                       [[r["sun_range"], r["positive_n"], r["exported_n"], r["accepted_n"],
                         number(r["corner_median"]) + " / " + number(r["corner_worst"]),
                         f'{r["negative_rejected_n"]}/{r["negative_n"]}'] for r in bins])
    (out / "report_tables.md").write_text(detailed)
    summary = {"real": real, "illumination_bins": bins,
               "suites": {name: {"n": len(rows), "exported": sum(r["status"] not in ("failed", "crashed") for r in rows),
                                  "accepted": sum(r.get("passed", False) for r in rows),
                                  "crashed": sum(r["status"] == "crashed" for r in rows)} for name, rows in suites.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(deck)
    print(json.dumps(summary["suites"]))


if __name__ == "__main__":
    main()
