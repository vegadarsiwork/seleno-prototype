#!/usr/bin/env python
"""Summarize registrations of one source against different references.

Reads finished job directories only (metrics.json, quality.json); it never
fits or reruns anything. For each run it reports the held-out error in native
SOURCE pixels, so references of different resolution are compared in the one
unit they share, and splits that error into a part predictable from nearby
held-out points (spatially systematic: a better model could remove it) and the
remainder (random at the point level: set by how precisely features can be
located, i.e. by the images, not the model).

    python reports/registration_review_20260926/compare_references.py \
        NAME=outputs/<job> NAME=outputs/<job> ... --out comparison.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def neighbour_split(residuals, k=8):
    rows = [r for r in residuals if r["dx"] is not None and r["dy"] is not None]
    if len(rows) < k + 2:
        return None
    xy = np.array([[r["x"], r["y"]] for r in rows])
    e = np.array([[r["dx"], r["dy"]] for r in rows])
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    neighbours = e[idx[:, 1:]].mean(axis=1)                  # leave-one-out neighbour mean
    out = {}
    for c, name in ((0, "sample"), (1, "line")):
        rest = e[:, c] - neighbours[:, c]
        out[name] = {"explained_by_neighbours": float(max(0., 1 - rest.var() / e[:, c].var())),
                     "random_nmad_px": float(1.4826 * np.median(np.abs(rest - np.median(rest))))}
    return out


def summarize(name, job):
    job = Path(job)
    m = json.loads((job / "metrics.json").read_text())
    q = json.loads((job / "quality.json").read_text())
    a, src, ref = m["accuracy"], q["source"], m["reference"]
    conv = {c["id"]: c["ours"] for c in q["conventions"]}
    ci = (q.get("intervals") or {}).get("source", {}).get("all", {})
    below = lambda s: next(t["fraction_all"] for t in s["thresholds"] if t["below_px"] == 1.)
    return {"name": name, "job": str(job), "reference": ref["path"], "reference_gsd_m": ref["gsd_m"],
            "status": m["status"], "acceptance": m["acceptance"]["status"],
            "model": m["model"], "method": m["method_used"], "split": m["distribution"]["grid"],
            "working_grid": m.get("working_grid"), "transform_sha256": a["transform_sha256"],
            "held_out": src["all"]["n"], "invalid": src["all"]["invalid_n"], "screened": src["screened"]["n"],
            "all_rmse_px": src["all"]["rmse_px"], "all_rmse_ci": ci.get("rmse_px"),
            "all_p95_px": src["all"]["p95_px"], "all_below_1": below(src["all"]),
            "screened_rmse_px": src["screened"]["rmse_px"], "screened_median_px": src["screened"]["median_px"],
            "screened_p95_px": src["screened"]["p95_px"], "screened_below_1": below(src["screened"]),
            "nmad_xy_px": src["all"]["nmad_xy_px"], "surface_rmse_m": a.get("rmse_ground_m"),
            "kaguya_form": conv.get("kaguya"), "ce90_m": (conv.get("circular_error") or {}).get("ce90_m"),
            "asp_mean_median_px": [conv["asp"]["mean_px"], conv["asp"]["median_px"]] if "asp" in conv else None,
            "support_inside_hull": (m["distribution"].get("valid_pixel_support") or {}).get("supported_fraction"),
            "noise": neighbour_split(q["residuals"]), "runtime_s": m["runtime_s"],
            "degraded": m["degraded"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=job_directory")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = [summarize(*r.split("=", 1)) for r in args.runs]
    fmt = lambda v, d=3: "–" if v is None else ("%.*f" % (d, v))
    print("| | " + " | ".join(r["name"] for r in rows) + " |")
    print("|---|" + "---:|" * len(rows))
    lines = [("reference GSD (m)", lambda r: fmt(r["reference_gsd_m"], 1)),
             ("model; split", lambda r: "%s; %s" % (r["model"], "×".join(map(str, r["split"])))),
             ("held-out / invalid / screened", lambda r: "%d / %d / %d" % (r["held_out"], r["invalid"], r["screened"])),
             ("all RMSE [95% CI] (source px)", lambda r: "%s [%s]" % (fmt(r["all_rmse_px"], 2), "–".join(fmt(v, 2) for v in r["all_rmse_ci"] or []))),
             ("all p95 (source px)", lambda r: fmt(r["all_p95_px"], 2)),
             ("all below 1 source px", lambda r: "%.1f%%" % (100 * r["all_below_1"])),
             ("screened RMSE / median (source px)", lambda r: "%s / %s" % (fmt(r["screened_rmse_px"], 2), fmt(r["screened_median_px"], 2))),
             ("screened below 1 source px", lambda r: "%.1f%%" % (100 * r["screened_below_1"])),
             ("random part, NMAD sample / line (source px)", lambda r: "–" if not r["noise"] else "%.2f / %.2f" % (r["noise"]["sample"]["random_nmad_px"], r["noise"]["line"]["random_nmad_px"])),
             ("variance explained by neighbours, sample / line", lambda r: "–" if not r["noise"] else "%.0f%% / %.0f%%" % (100 * r["noise"]["sample"]["explained_by_neighbours"], 100 * r["noise"]["line"]["explained_by_neighbours"])),
             ("surface RMSE (m, screened)", lambda r: fmt(r["surface_rmse_m"], 1)),
             ("ASP form mean / median (source px)", lambda r: "–" if not r["asp_mean_median_px"] else "%.2f / %.2f" % tuple(r["asp_mean_median_px"])),
             ("acceptance", lambda r: r["acceptance"]),
             ("runtime (s)", lambda r: fmt(r["runtime_s"], 0))]
    for label, f in lines:
        print("| %s | %s |" % (label, " | ".join(f(r) for r in rows)))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
