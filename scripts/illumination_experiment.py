"""Phases 5 and 7 - the OHRC illumination-stress experiment and its ablation.

The pair: 20241115T1326321339 against 20241115T1525004388. Consecutive orbits
(23328, 23329) two hours apart, 95.3 % mutual footprint, and - measured from the
ancillary Sun series at mid-strip - solar azimuth 177.9 deg vs 245.2 deg with
elevation -0.17 deg vs +0.79 deg. Cast shadows move bodily between them.

What this script does, in order:

1. Establishes the footprints and the common ground.
2. **Measures the coarse offset between the two products' delivered
   geolocation.** This is not optional bookkeeping: the disagreement is several
   hundred metres, and without correcting it every window pair is mislocated by
   thousands of pixels and every matcher looks broken for the wrong reason.
3. Samples candidate windows from the mutually usable ground.
4. Runs each ablation arm on each window.
5. Writes the tables, the per-window diagnostics and the figures.

There is **no ground truth** for any of these pairs. Reprojection RMSE and
held-out RMSE measure self-consistency; disagreement with the delivered geometry
is an independent check on a prior that is itself only good to
metres-to-decametres. The script never prints an accuracy figure it does not have.

    python scripts/illumination_experiment.py
    python scripts/illumination_experiment.py --windows 6 --size 1024
    python scripts/illumination_experiment.py --arms baseline_verified,seleno
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from seleno import alignment, ohrc, pairs, registration, theme       # noqa: E402
from seleno.experiments import ABLATION, build_options               # noqa: E402
from seleno.ohrc import tiles as T                                   # noqa: E402

OUT = os.path.join(ROOT, "results", "illumination")
PAIR_A, PAIR_B = "20241115T1326", "20241115T1525"


def fmt(v, d=3):
    if v is None:
        return "--"
    try:
        return ("%." + str(d) + "f") % float(v)
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #

def pick_windows(a, b, offset, size: int, n: int, cache_dir: str) -> list[dict]:
    """Windows that are usable in A and land inside B once the offset is applied."""
    idx = T.StripIndex.cached(a, cache_dir, tile=512, stride=1024, step=8)
    cands = sorted(idx.usable(), key=lambda t: -t.std)
    picked, used_lines = [], []
    for st in cands:
        if len(picked) >= n * 3:
            break
        s0 = int(np.clip(st.sample0 - (size - 512) // 2, 0, a.samples - size))
        l0 = int(np.clip(st.line0 - (size - 512) // 2, 0, a.lines - size))
        # keep the windows spread along the strip
        if any(abs(l0 - p) < 3 * size for p in used_lines):
            continue
        try:
            pr = pairs.ohrc_window_pair(a, b, s0, l0, size, offset_stereo_m=offset)
        except ValueError:
            continue
        ref_std = float(pr.reference.std())
        if ref_std < 4.0:
            continue                      # reference window is shadowed: nothing to match
        used_lines.append(l0)
        picked.append({"sample0": s0, "line0": l0,
                       "source_mean": st.mean, "source_std": st.std,
                       "reference_mean": round(float(pr.reference.mean()), 3),
                       "reference_std": round(ref_std, 3),
                       "min_std": round(min(st.std, ref_std), 3)})
    # Rank by the weaker of the two sides: a correspondence needs texture in both
    # images, and the reference side is the binding constraint on this pair.
    picked.sort(key=lambda w: -w["min_std"])
    return picked[:n]


def run_arms(a, b, win: dict, size: int, offset, align_rec, arms) -> list[dict]:
    rows = []
    for code, preset, label in arms:
        pr = pairs.ohrc_window_pair(a, b, win["sample0"], win["line0"], size,
                                    offset_stereo_m=offset,
                                    coarse_alignment=align_rec)
        opts = build_options(preset, pr)
        t0 = time.time()
        res = registration.run(pr, opts)
        m = res.metrics
        rows.append({
            "code": code, "preset": preset, "label": label,
            "status": res.status, "confidence": res.confidence,
            "candidates_before_mask": m.get("candidates_before_mask"),
            "dropped_by_mask": m.get("dropped_by_mask"),
            "candidates": m["candidate_matches"],
            "inliers": m["inliers"], "inlier_ratio": m["inlier_ratio"],
            "retained": m["selected_matches"],
            "reprojection_rmse_px": (m.get("reprojection") or {}).get("rmse_px"),
            "heldout_rmse_px": (m.get("heldout") or {}).get("rmse_px"),
            "heldout_count": m.get("heldout_count"),
            "coverage": m["spatial_coverage"],
            "coverage_whole_window": m.get("spatial_coverage_whole_window"),
            "matchable_cells": m.get("matchable_cells"),
            "usable_fraction": m.get("usable_fraction"),
            "prior_disagreement_m": (m.get("geometry_prior_disagreement") or {})
            .get("corner_disagreement_m"),
            "ncc_before": m.get("ncc_before"), "ncc_after": m.get("ncc_after"),
            "runtime_s": round(time.time() - t0, 3),
            "reasons": res.reasons,
            "warnings": res.warnings,
            "mask_is_synthetic": m.get("mask_is_synthetic_terrain"),
        })
    return rows


# --------------------------------------------------------------------------- #

def chart_arms(per_arm: dict, path: str) -> None:
    """Inlier ratio and held-out RMSE per ablation arm, averaged over windows.

    Two panels rather than two y-axes on one plot: the quantities have different
    units and a dual axis invites exactly the misreading this project is trying
    to avoid.
    """
    theme.apply_matplotlib(plt)
    codes = list(per_arm.keys())
    ratio = [100 * np.mean([r["inlier_ratio"] for r in per_arm[c]]) for c in codes]
    hold = [np.mean([r["heldout_rmse_px"] for r in per_arm[c]
                     if r["heldout_rmse_px"] is not None] or [np.nan]) for c in codes]
    acc = [sum(1 for r in per_arm[c] if r["status"] != "refused") for c in codes]
    n_win = max(len(v) for v in per_arm.values())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    x = np.arange(len(codes))

    ax = axes[0]
    ax.bar(x, ratio, width=0.62, color=theme.SERIES[0],
           edgecolor=theme.SURFACE, linewidth=2)
    for i, v in enumerate(ratio):
        ax.text(i, v + 1.2, "%.0f" % v, ha="center", fontsize=8,
                color=theme.TEXT_SECONDARY)
    ax.set_xticks(x, codes)
    ax.set_ylabel("mean inlier ratio (%)")
    ax.set_title("Inlier ratio by arm", color=theme.TEXT_PRIMARY, fontsize=10,
                 loc="left", pad=16)
    ax.text(0.0, 1.02, "mean over %d windows" % n_win, transform=ax.transAxes,
            fontsize=8, color=theme.TEXT_MUTED)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ok = ~np.isnan(hold)
    ax.bar(x[ok], np.array(hold)[ok], width=0.62, color=theme.SERIES[2],
           edgecolor=theme.SURFACE, linewidth=2)
    for i in np.where(ok)[0]:
        ax.text(i, hold[i], " %.2f" % hold[i], ha="center", va="bottom",
                fontsize=8, color=theme.TEXT_SECONDARY)
    ax.set_xticks(x, codes)
    ax.set_ylabel("mean held-out RMSE (px)")
    ax.set_title("Held-out error by arm", color=theme.TEXT_PRIMARY, fontsize=10,
                 loc="left", pad=16)
    ax.text(0.0, 1.02, "self-consistency off the fitted points - not accuracy",
            transform=ax.transAxes, fontsize=8, color=theme.TEXT_MUTED)
    ax.grid(axis="x", visible=False)

    fig.text(0.005, 0.005,
             "accepted windows per arm: " + "  ".join(
                 "%s=%d/%d" % (c, a_, n_win) for c, a_ in zip(codes, acc)),
             fontsize=7.5, color=theme.TEXT_FAINT)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def window_figure(a, b, win, size, offset, align_rec, path) -> None:
    """Source, reference and the registered overlay for one window."""
    pr = pairs.ohrc_window_pair(a, b, win["sample0"], win["line0"], size,
                                offset_stereo_m=offset, coarse_alignment=align_rec)
    opts = build_options("seleno", pr)
    res = registration.run(pr, opts)
    cells = []
    for key, tag in (("mask", "usability mask"), ("verified", "inliers / outliers"),
                     ("overlay", "registered overlay")):
        img = res.images.get(key)
        if img is None:
            continue
        w = 430
        h = int(img.shape[0] * w / img.shape[1])
        cell = cv2.resize(img, (w, h))
        cv2.putText(cell, tag, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (60, 200, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    if not cells:
        return
    hmax = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(25, 25, 26))
             for c in cells]
    sheet = np.hstack(cells)
    banner = np.full((30, sheet.shape[1], 3), (25, 25, 26), np.uint8)
    sa, sb = pr.sun_source, pr.sun_reference
    cv2.putText(banner, "s%d l%d  |  sun el %.2f->%.2f  az %.1f->%.1f  |  %s"
                % (win["sample0"], win["line0"], sa["elevation_deg"], sb["elevation_deg"],
                   sa["azimuth_deg"], sb["azimuth_deg"], res.status.upper()),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (183, 194, 195), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([banner, sheet]))


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=PAIR_A)
    ap.add_argument("--b", default=PAIR_B)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--arms", default=None,
                    help="comma-separated preset names (default: the full ablation)")
    ap.add_argument("--rebuild-alignment", action="store_true")
    ap.add_argument("--no-offset", action="store_true",
                    help="deliberately skip the coarse-offset correction")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    products = ohrc.discover()
    if not products:
        print("no OHRC products found; set SELENO_OHRC_ROOT")
        return 1
    a = ohrc.by_timestamp(products, args.a)
    b = ohrc.by_timestamp(products, args.b)

    arms = ABLATION
    if args.arms:
        want = [x.strip() for x in args.arms.split(",") if x.strip()]
        by_preset = {p: (c, p, l) for c, p, l in ABLATION}
        arms = [by_preset.get(w, (w[:3].upper(), w, w)) for w in want]

    print("== %s  ->  %s" % (a.timestamp, b.timestamp))
    ov = ohrc.footprint_overlap(a.geometry, b.geometry, cell_m=100.0)
    sa = a.sun_at_line(a.lines // 2)
    sb = b.sun_at_line(b.lines // 2)
    print("   footprint overlap %.1f%% of the smaller strip (%.1f km2)"
          % (100 * ov["fraction_of_smaller"], ov["area_overlap_km2"]))
    print("   sun A: el %+.3f deg  az %.1f deg   (1 m of relief -> %.0f m shadow)"
          % (sa["elevation_deg"], sa["azimuth_deg"],
             ohrc.shadow_length_per_metre(sa["elevation_deg"])))
    print("   sun B: el %+.3f deg  az %.1f deg   (1 m of relief -> %.0f m shadow)"
          % (sb["elevation_deg"], sb["azimuth_deg"],
             ohrc.shadow_length_per_metre(sb["elevation_deg"])))
    print("   delta: az %+.1f deg  el %+.3f deg"
          % (sb["azimuth_deg"] - sa["azimuth_deg"],
             sb["elevation_deg"] - sa["elevation_deg"]))

    print("\n== coarse alignment of the delivered geolocation")
    t0 = time.time()
    align_rec = alignment.get_offset(a, b, OUT, rebuild=args.rebuild_alignment)
    offset = (0.0, 0.0) if args.no_offset else alignment.offset_or_zero(align_rec)
    print("   %s" % ("cached" if align_rec.get("from_cache") else
                     "computed in %.0f s" % (time.time() - t0)))
    if align_rec.get("ok"):
        print("   offset %s m  (|d| = %.0f m = %.0f source pixels)"
              % (align_rec["offset_stereo_m"], align_rec["offset_m"],
                 align_rec["offset_px_source_gsd"]))
        print("   peak NCC %.3f vs runner-up %.3f (margin %.3f) on %s -> confident=%s"
              % (align_rec["peak_ncc"], align_rec["second_peak_ncc"],
                 align_rec["peak_margin"], align_rec["representation"],
                 align_rec["confident"]))
    else:
        print("   FAILED: %s" % align_rec.get("reason"))
    if args.no_offset:
        print("   --no-offset given: running WITHOUT the correction")
    print("   applied offset: %s" % (list(offset),))

    print("\n== sampling windows")
    wins = pick_windows(a, b, offset, args.size,
                        args.windows, os.path.join(ROOT, "results", "dataset"))
    if not wins:
        print("   no usable window found")
        return 1
    for w in wins:
        print("   s%-6d l%-7d  src mean %6.1f sd %5.1f | ref mean %6.1f sd %5.1f"
              % (w["sample0"], w["line0"], w["source_mean"], w["source_std"],
                 w["reference_mean"], w["reference_std"]))

    print("\n== ablation (%d arms x %d windows)" % (len(arms), len(wins)))
    per_window, per_arm = [], {c: [] for c, _, _ in arms}
    for w in wins:
        rows = run_arms(a, b, w, args.size, offset, align_rec, arms)
        per_window.append({"window": w, "arms": rows})
        for r in rows:
            per_arm[r["code"]].append(r)
        print("   s%-6d l%-7d  %s" % (
            w["sample0"], w["line0"],
            "  ".join("%s:%s/%.0f%%" % (r["code"], r["inliers"],
                                        100 * r["inlier_ratio"]) for r in rows)))

    # -------------------------------------------------------------- tables
    lines = [
        "# OHRC illumination experiment",
        "",
        "Generated by `scripts/illumination_experiment.py` on %s."
        % time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "Pair: `%s` -> `%s` (orbits %s, %s), %d windows of %d px, %.2f m/px."
        % (a.product_id, b.product_id, a.label.imaging_orbit, b.label.imaging_orbit,
           len(wins), args.size, a.gsd_m or 0.24),
        "",
        "**There is no ground truth for any pair in this experiment.** Both windows",
        "are real acquisitions and no correct transform between them is known.",
        "Reprojection RMSE and held-out RMSE measure self-consistency. Disagreement",
        "with the delivered geometry is an independent check against a prior that is",
        "itself only accurate to metres-to-decametres.",
        "",
        "## Illumination",
        "",
        "| | source | reference | delta |",
        "|---|--:|--:|--:|",
        "| solar elevation (deg) | %+.3f | %+.3f | %+.3f |"
        % (sa["elevation_deg"], sb["elevation_deg"],
           sb["elevation_deg"] - sa["elevation_deg"]),
        "| solar azimuth (deg) | %.1f | %.1f | %+.1f |"
        % (sa["azimuth_deg"], sb["azimuth_deg"],
           sb["azimuth_deg"] - sa["azimuth_deg"]),
        "| shadow cast by 1 m of relief (m) | %.0f | %.0f | |"
        % (ohrc.shadow_length_per_metre(sa["elevation_deg"]),
           ohrc.shadow_length_per_metre(sb["elevation_deg"])),
        "",
        "Footprint overlap: %.1f %% of the smaller strip (%.1f km2)."
        % (100 * ov["fraction_of_smaller"], ov["area_overlap_km2"]),
        "",
        "## Coarse alignment of the delivered geolocation",
        "",
    ]
    if align_rec.get("ok"):
        lines += [
            "| quantity | value |",
            "|---|--:|",
            "| measured offset | %s m |" % (align_rec["offset_stereo_m"],),
            "| magnitude | %.0f m (%.0f source pixels) |"
            % (align_rec["offset_m"], align_rec["offset_px_source_gsd"]),
            "| correlation peak | %.4f |" % align_rec["peak_ncc"],
            "| runner-up peak | %.4f |" % align_rec["second_peak_ncc"],
            "| margin | %.4f |" % align_rec["peak_margin"],
            "| representation | %s |" % align_rec["representation"],
            "| mutually lit pixels | %d |" % align_rec["lit_pixels"],
            "| confident | %s |" % align_rec["confident"],
            "",
            "The two products' delivered geolocation disagrees by **%.0f m**, which is"
            % align_rec["offset_m"],
            "**%.0f pixels** at this ground sampling. Without correcting it a 1024 px"
            % align_rec["offset_px_source_gsd"],
            "window and its geometry-predicted counterpart share almost no ground, and",
            "every matcher looks broken for a reason that has nothing to do with",
            "illumination. Correcting it is what makes the rest of this table meaningful.",
            "",
        ]
    else:
        lines += ["Coarse alignment failed: %s" % align_rec.get("reason"), ""]

    lines += ["## Ablation, averaged over %d windows" % len(wins), "",
              "| arm | configuration | cand. | inliers | inlier ratio | retained "
              "| reproj RMSE | held-out RMSE | coverage | accepted |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for code, preset, label in arms:
        rs = per_arm[code]
        if not rs:
            continue

        def mean(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        acc = sum(1 for r in rs if r["status"] != "refused")
        lines.append("| %s | %s | %s | %s | %s %% | %s | %s | %s | %s %% | %d/%d |" % (
            code, label, fmt(mean("candidates"), 0), fmt(mean("inliers"), 0),
            fmt(100 * (mean("inlier_ratio") or 0), 1), fmt(mean("retained"), 0),
            fmt(mean("reprojection_rmse_px")), fmt(mean("heldout_rmse_px")),
            fmt(100 * (mean("coverage") or 0), 0), acc, len(rs)))
    lines.append("")

    lines += ["## Per window", ""]
    for entry in per_window:
        w = entry["window"]
        lines += ["### window sample %d, line %d" % (w["sample0"], w["line0"]), "",
                  "source mean DN %.1f (sd %.1f), reference mean DN %.1f (sd %.1f)"
                  % (w["source_mean"], w["source_std"],
                     w["reference_mean"], w["reference_std"]), "",
                  "| arm | cand. | inliers | ratio | reproj RMSE | held-out | coverage "
                  "| vs geometry | status |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|---|"]
        for r in entry["arms"]:
            lines.append("| %s | %d | %d | %.1f %% | %s | %s | %.0f %% | %s m | %s |" % (
                r["code"], r["candidates"], r["inliers"], 100 * r["inlier_ratio"],
                fmt(r["reprojection_rmse_px"]), fmt(r["heldout_rmse_px"]),
                100 * r["coverage"], fmt(r["prior_disagreement_m"], 0), r["status"]))
        lines.append("")

    with open(os.path.join(OUT, "ILLUMINATION.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "illumination_raw.json"), "w", encoding="utf-8") as fh:
        json.dump({"a": a.describe(), "b": b.describe(),
                   "footprint_overlap": ov,
                   "sun_source_mid": sa, "sun_reference_mid": sb,
                   "coarse_alignment": align_rec,
                   "applied_offset_stereo_m": list(offset),
                   "windows": per_window}, fh, indent=2, default=str)

    print("\n== figures")
    chart_arms(per_arm, os.path.join(OUT, "ablation_arms.png"))
    for i, entry in enumerate(per_window[:3]):
        window_figure(a, b, entry["window"], args.size, offset, align_rec,
                      os.path.join(OUT, "window_%d.png" % i))
    print("wrote %s" % OUT)
    a.close(); b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
