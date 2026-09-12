"""Phase 2 - look at the actual OHRC data before doing anything to it.

Produces, under ``results/dataset/``:

  strip_<ts>_thumb.png          whole-strip preview (strided, never a bulk load)
  strip_<ts>_tiles.png          representative lit / transition / shadow windows
  pair_2024_regions.png         the two 2024 strips over the same ground
  dn_histograms.png             DN distribution, log counts, all products
  cross_track_profile.png       mean DN per sample - the swath gradient
  along_track_profile.png       mean DN per line - the traverse into shadow
  illumination_classes.png      tile-class composition per strip
  dataset_summary.json          every number behind those figures
  tileindex_<ts>_*.json         cached per-tile statistics

Usage:
    python scripts/inspect_dataset.py                  # everything
    python scripts/inspect_dataset.py --quick          # coarser scan
    python scripts/inspect_dataset.py --only 20241115T1326
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

from seleno import theme                                            # noqa: E402
from seleno import ohrc                                             # noqa: E402
from seleno.ohrc import tiles as T                                  # noqa: E402

OUT = os.path.join(ROOT, "results", "dataset")


def short_name(ts: str) -> str:
    """Compact, unambiguous series label.

    Both 2024 products share the date, so a date-only label collides. MMDD.HHMM
    separates every product in this archive.
    """
    return "%s.%s" % (ts[4:8], ts[9:13])


def legend_name(ts: str) -> str:
    return "%s-%s-%s %s:%s" % (ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13])


def _label(ax, x, y, text):
    """Direct label at the line end. Text wears an ink token, not the series hue -
    position next to the line carries the identity."""
    ax.annotate(text, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                color=theme.TEXT_SECONDARY, fontsize=7.5, va="center",
                fontweight="600")


def _titles(fig, ax, title: str, subtitle: str) -> None:
    """Title above, subtitle beneath it, neither overlapping the plot."""
    ax.set_title(title, color=theme.TEXT_PRIMARY, fontsize=11, loc="left", pad=20)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
            color=theme.TEXT_MUTED, va="bottom")


# --------------------------------------------------------------------------- #
# per-strip imagery
# --------------------------------------------------------------------------- #

def strip_thumbnail(p, path: str, max_side: int = 1500) -> dict:
    """Whole-strip preview. Rotated to landscape so it is legible on a page."""
    th = p.thumbnail(max_side)
    disp = T.normalise_for_display(th)
    disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    pad = 34
    canvas = np.full((h + pad, w, 3), (25, 25, 26), np.uint8)
    canvas[pad:] = img
    txt = ("%s   %d x %d px   %.2f m/px   sun el %.3f deg   az %.1f deg"
           % (p.timestamp, p.lines, p.samples, p.gsd_m or 0,
              p.label.sun_elevation_deg or 0, p.label.sun_azimuth_deg or 0))
    cv2.putText(canvas, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (183, 194, 195), 1, cv2.LINE_AA)
    cv2.putText(canvas, "along-track ->", (w - 130, h + pad - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (122, 107, 98), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)
    return {"file": os.path.basename(path), "thumb_shape": list(th.shape),
            "note": "per-tile percentile stretch, visualisation only"}


def tile_montage(p, index: T.StripIndex, path: str, tile_px: int = 512,
                 per_class: int = 3) -> dict:
    """Representative windows per illumination class, at full resolution."""
    picks = T.pick_representative(index, per_class=per_class)
    rows, meta = [], {}
    for klass in ("lit", "transition", "shadow"):
        cells, entries = [], []
        for st in picks.get(klass, []):
            raw = p.read_tile(st.sample0, st.line0, tile_px, tile_px)
            disp = T.normalise_for_display(raw)
            cell = cv2.cvtColor(cv2.resize(disp, (256, 256)), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(cell, (0, 0), (255, 255), (49, 55, 66), 1)
            cv2.putText(cell, "%s s%d l%d" % (klass, st.sample0, st.line0),
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 200, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(cell, "mean %.1f sd %.1f dark %.0f%% lv %d"
                        % (st.mean, st.std, 100 * st.frac_dark, st.distinct),
                        (6, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (183, 194, 195), 1,
                        cv2.LINE_AA)
            cells.append(cell)
            entries.append(st.to_dict())
        while len(cells) < per_class:
            blank = np.full((256, 256, 3), (25, 25, 26), np.uint8)
            cv2.putText(blank, "no %s tile" % klass, (60, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (98, 107, 122), 1, cv2.LINE_AA)
            cells.append(blank)
        rows.append(np.hstack(cells))
        meta[klass] = entries
    cv2.imwrite(path, np.vstack(rows))
    return {"file": os.path.basename(path), "tiles": meta,
            "note": "each window stretched independently; raw DN stats printed on the tile"}


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #

def dn_histograms(hists: dict, path: str) -> None:
    """DN distribution per product.

    Log counts: the distribution spans four orders of magnitude between the
    shadow floor and the lit tail, and a linear axis shows one spike at DN 0-4
    and nothing else. One axis only.
    """
    theme.apply_matplotlib(plt)
    colours = theme.series_for(hists.keys())
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    dn = np.arange(256)
    for ts, h in hists.items():
        frac = h / max(h.sum(), 1)
        ax.plot(dn, np.maximum(frac, 1e-9), color=colours[ts], label=legend_name(ts))
    ax.set_yscale("log")
    ax.set_xlim(0, 255)
    ax.set_ylim(1e-8, 1.0)
    ax.set_xlabel("digital number (8-bit)")
    ax.set_ylabel("fraction of sampled pixels (log)")
    _titles(fig, ax, "OHRC DN distribution",
            "two thirds of every strip sits at or below DN 10; the 2021 strip also "
            "saturates at 255")
    ax.axvline(T.DARK_DN, color=theme.TEXT_FAINT, lw=1, ls=":")
    ax.annotate("DN %d\nshadow floor" % T.DARK_DN, xy=(T.DARK_DN, 3e-7),
                xytext=(18, 3e-7), color=theme.TEXT_MUTED, fontsize=8)
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def profile_chart(profiles: dict, path: str, xlabel: str, title: str,
                  subtitle: str) -> None:
    theme.apply_matplotlib(plt)
    colours = theme.series_for(profiles.keys())
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for ts, prof in profiles.items():
        x = np.linspace(0, 1, len(prof))
        ax.plot(x, prof, color=colours[ts], label=legend_name(ts))
        _label(ax, x[-1], prof[-1], short_name(ts))
    ax.set_xlim(0, 1.10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean DN")
    _titles(fig, ax, title, subtitle)
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def illumination_chart(summaries: dict, path: str) -> None:
    """Tile-class composition. Stacked bars with a 2 px surface gap between fills."""
    theme.apply_matplotlib(plt)
    order = ["lit", "transition", "shadow"]
    cols = {"lit": theme.SERIES[3], "transition": theme.SERIES[1],
            "shadow": theme.SERIES[0]}
    labels = list(summaries.keys())
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    left = np.zeros(len(labels))
    y = np.arange(len(labels))
    for k in order:
        vals = np.array([summaries[t]["fractions"][k] for t in labels])
        ax.barh(y, vals, left=left, height=0.52, color=cols[k], label=k,
                edgecolor=theme.SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            if v > 0.07:
                ax.text(left[i] + v / 2, y[i], "%.0f%%" % (100 * v),
                        ha="center", va="center", fontsize=8,
                        color=theme.TEXT_PRIMARY)
        left += vals
    ax.set_yticks(y, [legend_name(t) for t in labels])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of 512 px tiles")
    _titles(fig, ax, "Tile illumination classes",
            "class from the fraction of pixels at or below DN %d" % T.DARK_DN)
    ax.grid(axis="y", visible=False)
    # Above the plot: inside the axes it collides with the bottom bar's label.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# the 2024 pair over shared ground
# --------------------------------------------------------------------------- #

def pair_regions(a, b, path: str, n: int = 3, tile_px: int = 512) -> dict:
    """Same ground, two illuminations.

    Windows are chosen in A, converted to projected metres via A's delivered
    geometry, then mapped back into B's pixel grid. Because that geometry is
    system-level and unrefined, the two crops are only approximately co-located -
    this figure is a visual sanity check, not evidence of registration.
    """
    ga, gb = a.geometry, b.geometry
    idx = T.StripIndex.cached(a, OUT, tile=tile_px, stride=tile_px * 6, step=8)
    cands = sorted(idx.usable(), key=lambda t: -t.std)
    rows, meta = [], []
    for st in cands:
        if len(rows) >= n:
            break
        cx, cy = st.centre
        x, y = ga.stereo(cx, cy)
        sb, lb, ok = gb.stereo_to_pixel(x, y)
        if not ok:
            continue
        sb0, lb0 = int(sb - tile_px / 2), int(lb - tile_px / 2)
        if not (0 <= sb0 <= b.samples - tile_px and 0 <= lb0 <= b.lines - tile_px):
            continue
        ta = a.read_tile(st.sample0, st.line0, tile_px, tile_px)
        tb = b.read_tile(sb0, lb0, tile_px, tile_px)
        sa_sun = a.sun_at_line(st.line0 + tile_px // 2)
        sb_sun = b.sun_at_line(lb0 + tile_px // 2)

        def cell(t, tag, sun, stat):
            d = cv2.cvtColor(cv2.resize(T.normalise_for_display(t), (300, 300)),
                             cv2.COLOR_GRAY2BGR)
            cv2.rectangle(d, (0, 0), (299, 299), (49, 55, 66), 1)
            cv2.putText(d, tag, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (60, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(d, "sun el %.2f az %.1f" % (sun["elevation_deg"], sun["azimuth_deg"]),
                        (6, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (183, 194, 195), 1,
                        cv2.LINE_AA)
            cv2.putText(d, "mean %.1f sd %.1f" % (float(t.mean()), float(t.std())),
                        (6, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (183, 194, 195), 1,
                        cv2.LINE_AA)
            return d

        rows.append(np.hstack([
            cell(ta, "A %s s%d l%d" % (a.timestamp[:13], st.sample0, st.line0), sa_sun, st),
            cell(tb, "B %s s%d l%d" % (b.timestamp[:13], sb0, lb0), sb_sun, None)]))
        meta.append({
            "a": {"sample0": st.sample0, "line0": st.line0, "mean": st.mean,
                  "std": st.std, "sun": sa_sun},
            "b": {"sample0": sb0, "line0": lb0, "mean": round(float(tb.mean()), 3),
                  "std": round(float(tb.std()), 3), "sun": sb_sun},
            "stereo_m": [round(float(x), 2), round(float(y), 2)],
            "d_azimuth_deg": round(sb_sun["azimuth_deg"] - sa_sun["azimuth_deg"], 3),
            "d_elevation_deg": round(sb_sun["elevation_deg"] - sa_sun["elevation_deg"], 4),
        })
    if not rows:
        return {"file": None, "note": "no shared usable window found"}
    cv2.imwrite(path, np.vstack(rows))
    return {"file": os.path.basename(path), "windows": meta,
            "note": ("windows co-located through the delivered system-level geometry; "
                     "alignment is approximate by construction")}


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset root (default: auto-detect)")
    ap.add_argument("--only", default=None, help="restrict to one product timestamp")
    ap.add_argument("--quick", action="store_true", help="coarser scan, faster")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached tile indices")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    products = ohrc.discover(args.root)
    if not products:
        print("no OHRC products found. Set SELENO_OHRC_ROOT or pass --root.")
        return 1
    if args.only:
        products = [ohrc.by_timestamp(products, args.only)]

    tile_px = 512
    stride = tile_px * (4 if args.quick else 2)
    step = 8 if args.quick else 4
    rows_prof = 300 if args.quick else 800

    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "dataset_root": products[0].root,
              "tile_scan": {"tile_px": tile_px, "stride_px": stride, "read_step": step},
              "thresholds": {"dark_dn": T.DARK_DN, "shadow_fraction": T.SHADOW_FRACTION,
                             "lit_fraction": T.LIT_FRACTION,
                             "usable_std": T.USABLE_STD,
                             "usable_distinct": T.USABLE_DISTINCT},
              "products": {}, "figures": {}}

    hists, cross, along, summaries = {}, {}, {}, {}

    for p in products:
        t0 = time.time()
        print("\n== %s  %d x %d  %.2f m/px" % (p.timestamp, p.lines, p.samples, p.gsd_m or 0))
        entry = p.describe()

        idx = T.StripIndex.cached(p, OUT, tile=tile_px, stride=stride, step=step,
                                  rebuild=args.rebuild)
        entry["tile_index"] = idx.summary()
        summaries[p.timestamp] = idx.summary()
        print("   tiles %d  lit/trans/shadow %s  usable %.1f%%"
              % (len(idx.tiles), idx.summary()["counts"],
                 100 * idx.summary()["usable_fraction"]))

        h = T.dn_histogram(p, rows=rows_prof)
        hists[p.timestamp] = h
        entry["dn_stats"] = T.histogram_stats(h)
        print("   DN mean %.2f  median %d  <=10DN %.1f%%  distinct %d"
              % (entry["dn_stats"]["mean"], entry["dn_stats"]["median"],
                 100 * entry["dn_stats"]["frac_le_10"], entry["dn_stats"]["distinct_levels"]))

        cp = T.cross_track_profile(p, rows=rows_prof)
        ap_ = T.along_track_profile(p, rows=rows_prof)
        cross[p.timestamp] = cp
        along[p.timestamp] = ap_
        entry["cross_track"] = {"sample0_mean_dn": round(float(cp[0]), 3),
                                "sample_last_mean_dn": round(float(cp[-1]), 3),
                                "min": round(float(cp.min()), 3),
                                "max": round(float(cp.max()), 3)}
        entry["along_track"] = {"min": round(float(ap_.min()), 3),
                                "max": round(float(ap_.max()), 3)}
        print("   cross-track %.1f -> %.1f DN   along-track %.1f -> %.1f DN"
              % (cp[0], cp[-1], ap_.min(), ap_.max()))

        entry["figures"] = {}
        entry["figures"]["thumbnail"] = strip_thumbnail(
            p, os.path.join(OUT, "strip_%s_thumb.png" % p.timestamp))
        entry["figures"]["tiles"] = tile_montage(
            p, idx, os.path.join(OUT, "strip_%s_tiles.png" % p.timestamp), tile_px)

        if p.has_sun_series:
            entry["sun_along_strip"] = {
                "start": p.sun_at_line(0),
                "middle": p.sun_at_line(p.lines // 2),
                "end": p.sun_at_line(p.lines - 1),
                "shadow_length_per_metre_at_middle": (
                    ohrc.shadow_length_per_metre(
                        p.sun_at_line(p.lines // 2)["elevation_deg"])),
            }
        report["products"][p.timestamp] = entry
        p.close()
        print("   %.1fs" % (time.time() - t0))

    # ------------------------------------------------------------- charts
    print("\n== charts")
    dn_histograms(hists, os.path.join(OUT, "dn_histograms.png"))
    profile_chart(cross, os.path.join(OUT, "cross_track_profile.png"),
                  "sample (cross-track), 0 to 11999 normalised",
                  "Cross-track brightness gradient",
                  "a real terrain and illumination gradient across the swath, not detector shading "
                  "- tiles from opposite edges are not comparable")
    profile_chart(along, os.path.join(OUT, "along_track_profile.png"),
                  "line (along-track), start to end normalised",
                  "Along-track brightness - the traverse into shadow",
                  "any global normalisation destroys one end of the strip")
    illumination_chart(summaries, os.path.join(OUT, "illumination_classes.png"))
    for k in ("dn_histograms", "cross_track_profile", "along_track_profile",
              "illumination_classes"):
        report["figures"][k] = k + ".png"

    # ------------------------------------------- the 2024 illumination pair
    try:
        a = ohrc.by_timestamp(ohrc.discover(args.root), "20241115T1326")
        b = ohrc.by_timestamp(ohrc.discover(args.root), "20241115T1525")
    except KeyError:
        report["pair_2024"] = {"note": "the 2024-11-15 pair is not present"}
    else:
        print("== 2024 pair regions")
        ov = ohrc.footprint_overlap(a.geometry, b.geometry, cell_m=100.0)
        sa = a.sun_at_line(a.lines // 2)
        sb = b.sun_at_line(b.lines // 2)
        report["pair_2024"] = {
            "a": a.timestamp, "b": b.timestamp,
            "footprint_overlap": ov,
            "sun_a_mid": sa, "sun_b_mid": sb,
            "d_azimuth_deg": round(sb["azimuth_deg"] - sa["azimuth_deg"], 3),
            "d_elevation_deg": round(sb["elevation_deg"] - sa["elevation_deg"], 4),
            "shadow_length_per_metre": {
                "a": ohrc.shadow_length_per_metre(sa["elevation_deg"]),
                "b": ohrc.shadow_length_per_metre(sb["elevation_deg"])},
            "regions": pair_regions(a, b, os.path.join(OUT, "pair_2024_regions.png")),
        }
        print("   overlap %.1f%% of smaller;  d_az %.1f deg  d_el %.3f deg"
              % (100 * ov["fraction_of_smaller"],
                 report["pair_2024"]["d_azimuth_deg"],
                 report["pair_2024"]["d_elevation_deg"]))
        a.close(); b.close()

    report["overlap_matrix"] = ohrc.overlap_matrix(ohrc.discover(args.root), cell_m=100.0)

    with open(os.path.join(OUT, "dataset_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
