"""Run the staged registration pipeline on an OHRC window pair or a legacy pair.

Baseline (Phase 4) - minimal preprocessing, SIFT, ratio test, MAGSAC++:

    python scripts/run_registration.py --a 20241115T1326 --b 20241115T1525 \
        --window 6000 50000 --size 1024 --preset baseline_verified

Geometry-aware arm:

    python scripts/run_registration.py --a 20241115T1326 --b 20241115T1525 \
        --window 6000 50000 --size 1024 --preset seleno

Legacy demo pair, numbers unchanged from the earlier prototype:

    python scripts/run_registration.py --legacy ohrc_crater_field --preset seleno
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from seleno import alignment, ohrc, pairs, registration, store   # noqa: E402
from seleno.experiments import PRESETS, build_options            # noqa: E402

ALIGN_CACHE = os.path.join(ROOT, "results", "illumination")


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out or [""]


def show(res) -> None:
    m = res.metrics
    o = res.options
    print("\n" + "=" * 74)
    print("pair    : %s" % res.pair_id)
    print("config  : matcher=%s model=%s ransac=%.1f mask=%s terrain=%s subpixel=%s"
          % (o["matcher"], o["model_type"], o["ransac_threshold"],
             o["mask_mode"], o["terrain_model"], o["subpixel_enabled"]))
    print("-" * 74)
    for s in res.stages:
        print("  [%-8s] %-38s %6.3f s" % (s["status"], s["name"], s["seconds"]))
        if s["note"]:
            for line in _wrap(s["note"], 64):
                print("             %s" % line)
    print("-" * 74)
    print("  candidates (pre-mask)  %8s" % m.get("candidates_before_mask"))
    print("  dropped by mask        %8s" % m.get("dropped_by_mask"))
    print("  candidates             %8d" % m["candidate_matches"])
    print("  inliers                %8d" % m["inliers"])
    print("  inlier ratio           %8.1f %%" % (100 * m["inlier_ratio"]))
    print("  retained               %8d" % m["selected_matches"])
    r = (m.get("reprojection") or {}).get("rmse_px")
    print("  reprojection RMSE      %8s px  (self-consistency, NOT accuracy)"
          % ("%.3f" % r if r is not None else "n/a"))
    ho = (m.get("heldout") or {}).get("rmse_px")
    print("  held-out RMSE          %8s px  (%d points excluded from the fit)"
          % ("%.3f" % ho if ho is not None else "n/a", m.get("heldout_count", 0)))
    gt = m.get("ground_truth")
    if gt and gt.get("corner_error_px") is not None:
        print("  GROUND-TRUTH error     %8.3f px = %.3f m"
              % (gt["corner_error_px"], gt.get("corner_error_m", float("nan"))))
    else:
        print("  GROUND-TRUTH error     %8s     none exists for this pair" % "--")
    dis = m.get("geometry_prior_disagreement")
    if dis:
        print("  vs delivered geometry  %8.1f m   independent check, not truth"
              % dis["corner_disagreement_m"])
    print("  spatial coverage       %8.1f %%  (top-K by confidence: %.1f %%)"
          % (100 * m["spatial_coverage"], 100 * m["spatial_coverage_before"]))
    print("  usable fraction        %8s" % m.get("usable_fraction"))
    print("  overlap fraction       %8s   (of the reference frame)"
          % m.get("overlap_fraction"))
    print("  overlap NCC            %8s -> %s" % (m.get("ncc_before"), m.get("ncc_after")))
    print("  runtime                %8.3f s" % m["runtime_total_s"])
    print("-" * 74)
    print("  STATUS: %s   confidence %.3f" % (res.status.upper(), res.confidence))
    for reason in res.reasons:
        for line in _wrap("- " + reason, 68):
            print("    %s" % line)
    for w in res.warnings:
        for line in _wrap("! " + w, 68):
            print("    %s" % line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="source product timestamp")
    ap.add_argument("--b", help="reference product timestamp")
    ap.add_argument("--window", nargs=2, type=int, metavar=("SAMPLE0", "LINE0"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--legacy", help="a legacy manifest pair id instead")
    ap.add_argument("--preset", default="seleno", choices=sorted(PRESETS))
    ap.add_argument("--matcher", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mask", default=None, help="none|observed|predicted|both")
    ap.add_argument("--terrain", default=None, help="none|mock")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-subpixel", action="store_true")
    ap.add_argument("--no-offset", action="store_true",
                    help="skip the measured coarse-offset correction between the two "
                         "products' geolocation (it is applied by default)")
    ap.add_argument("--save", default=None, help="write images + JSON under results/<dir>")
    args = ap.parse_args()

    if args.legacy:
        st = store.Store(os.path.join(ROOT, "data"))
        pair = pairs.legacy_pair(st, args.legacy)
    else:
        if not (args.a and args.b and args.window):
            ap.error("need --a, --b and --window, or --legacy")
        products = ohrc.discover()
        if not products:
            print("no OHRC products found; set SELENO_OHRC_ROOT")
            return 1
        a = ohrc.by_timestamp(products, args.a)
        b = ohrc.by_timestamp(products, args.b)
        # The coarse offset between the two products' delivered geolocation is
        # applied by DEFAULT. Without it the reference window lands hundreds of
        # metres off the source window's ground and the fit degenerates - which
        # is exactly what `--no-offset` exists to demonstrate.
        align = alignment.get_offset(a, b, ALIGN_CACHE)
        off = (0.0, 0.0) if args.no_offset else alignment.offset_or_zero(align)
        if align.get("ok"):
            print("coarse alignment: offset %.0f m (%.0f source px), peak %.3f, "
                  "margin %.3f, confident=%s%s"
                  % (align["offset_m"], align["offset_px_source_gsd"],
                     align["peak_ncc"], align["peak_margin"], align["confident"],
                     "  [SKIPPED via --no-offset]" if args.no_offset else ""))
        else:
            print("coarse alignment unavailable: %s" % align.get("reason"))
        pair = pairs.ohrc_window_pair(a, b, args.window[0], args.window[1], args.size,
                                      offset_stereo_m=off, coarse_alignment=align)

    over = {}
    for key, val in (("matcher", args.matcher), ("model_type", args.model),
                     ("mask_mode", args.mask), ("terrain_model", args.terrain),
                     ("ransac_threshold", args.threshold)):
        if val is not None:
            over[key] = val
    if args.no_subpixel:
        over["subpixel_enabled"] = False

    opts = build_options(args.preset, pair, over)
    res = registration.run(pair, opts)
    show(res)

    if args.save:
        out = os.path.join(ROOT, "results", args.save, res.pair_id)
        os.makedirs(out, exist_ok=True)
        for k, img in res.images.items():
            cv2.imwrite(os.path.join(out, k + ".png"), img)
        with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(res.json(), fh, indent=2, default=str)
        print("\n  saved -> %s" % out)
    return 0 if res.status != "refused" else 2


if __name__ == "__main__":
    sys.exit(main())
