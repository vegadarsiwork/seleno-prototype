"""Command-line harness: run one pair (or all of them) and print real numbers.

    python scripts/run_pipeline.py --pair ohrc_crater_field
    python scripts/run_pipeline.py --all --matcher sift
    python scripts/run_pipeline.py --pair ohrc_crater_field --save results/
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

from seleno import pipeline, store  # noqa: E402


def show(res: dict) -> None:
    m = res["metrics"]
    print("\n" + "=" * 68)
    print("pair    : %s" % res["pair_id"])
    print("matcher : %s   model: %s   ransac: %.1f px"
          % (res["options"]["matcher"], res["options"]["model_type"],
             res["options"]["ransac_threshold"]))
    print("-" * 68)
    for s in res["stages"]:
        print("  [%-7s] %-42s %6.3f s" % (s["status"], s["name"], s["seconds"]))
        if s["note"]:
            print("            %s" % s["note"])
    print("-" * 68)
    print("  candidate matches   %8d" % m["candidate_matches"])
    print("  inliers             %8d" % m["inliers"])
    print("  outliers rejected   %8d" % m["outliers_rejected"])
    print("  inlier ratio        %8.1f %%" % (100 * m["inlier_ratio"]))
    print("  selected matches    %8d" % m["selected_matches"])
    r = m["reprojection"].get("rmse_px")
    print("  reprojection RMSE   %8s px   (self-consistency, not absolute accuracy)"
          % ("%.3f" % r if r is not None else "n/a"))
    gt = m.get("ground_truth")
    if gt and gt.get("corner_error_px") is not None:
        print("  GT corner error     %8.3f px   (against the known transform)" % gt["corner_error_px"])
        print("  GT RMSE             %8.3f px" % gt["gt_rmse_px"])
    else:
        print("  ground truth        %8s" % "none for this pair")
    print("  coverage            %8.1f %%  (top-K by confidence alone: %.1f %%)"
          % (100 * m["spatial_coverage"], 100 * m["spatial_coverage_before"]))
    print("  overlap NCC         %8s -> %s" % (m.get("ncc_before"), m.get("ncc_after")))
    print("  runtime             %8.3f s" % m["runtime_total_s"])
    v = res["verdict"]
    print("-" * 68)
    print("  VERDICT: %s" % ("REGISTERED" if v["registered"] else "REJECTED"))
    for r_ in v["reasons"]:
        print("    - %s" % r_)
    if v.get("expected_outcome"):
        print("    expected: %s" % v["expected_outcome"])
    for w in res.get("warnings", []):
        print("    ! %s" % w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--matcher", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--no-mutual", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    st = store.Store(os.path.join(ROOT, "data"))
    ids = [p["id"] for p in st.pairs] if args.all else [args.pair or st.pairs[0]["id"]]

    for pid in ids:
        pair = st.pair(pid)
        opts = pipeline.Options()
        opts.model_type = args.model or pair.get("recommended_model", "homography")
        if args.matcher:
            opts.matcher = args.matcher
        if args.threshold is not None:
            opts.ransac_threshold = args.threshold
        if args.ratio is not None:
            opts.ratio = args.ratio
        if args.no_mutual:
            opts.mutual_check = False
        opts.preprocess_enabled = not args.no_preprocess
        opts.spatial_enabled = not args.no_spatial
        opts.verify_enabled = not args.no_verify

        src, ref = st.images(pid)
        res = pipeline.run(pair, src, ref, opts)
        show(res)
        if args.save:
            out = os.path.join(ROOT, args.save, pid)
            os.makedirs(out, exist_ok=True)
            for k, img in res["_images"].items():
                cv2.imwrite(os.path.join(out, k + ".png"), img)
            with open(os.path.join(out, "result.json"), "w") as fh:
                json.dump({k: v for k, v in res.items() if k != "_images"}, fh, indent=2)
            print("  saved -> %s" % out)


if __name__ == "__main__":
    main()
