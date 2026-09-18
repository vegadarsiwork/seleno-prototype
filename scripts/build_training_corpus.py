"""Fetch the LROC illumination corpus used to fine-tune LightGlue.

    python scripts/build_training_corpus.py            # fetch everything (~430 MB)
    python scripts/build_training_corpus.py --tiles 6  # a small subset first

Why these products are a training set with no labelling cost
------------------------------------------------------------
Every `NAC_POLE_SOUTH_CM_<bin>_<tile>` product is a subset of **one** ISIS jigsaw
solution resampled onto **one** 1 m polar stereographic grid. Two bins of the
same tile are therefore the same ground at the same pixel coordinates, to about
the control network's 0.8 px (2 sigma). The ground truth relating them is the
**identity transform, exact by construction** - no annotation, no synthetic
warp, no matcher involved in defining it.

What differs between the two is real: a different sub-solar longitude, different
cast shadows, and a different set of underlying NAC frames.

40 controlled tiles carry 862 single-illumination products between them, which is
tens of thousands of real illumination-change pairs.

Resolution, stated plainly
--------------------------
This corpus is built from the **browse pyramids** (~32 m/px), not the
full-resolution products. Two reasons, and one caveat:

* the full-resolution GeoTIFFs are striped, so cutting chips out of them costs
  ~93 MB of scanline reads per tile-bin - about 13 GB for this corpus, versus
  430 MB for the pyramids;
* the Phase 2 benchmark that established the 0/12 vs 7/12 result was run at this
  same sampling, so training and evaluation stay on one footing;
* **caveat**: a matcher fine-tuned at 32 m/px is not thereby validated at 1 m/px.
  Re-running at full resolution on a subset is the obvious follow-up and is not
  claimed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from seleno import nac                                            # noqa: E402

CATALOG = os.path.join(ROOT, "data", "index", "nac_south_catalog.json")
CORPUS = os.path.join(ROOT, "data", "processed", "nac_illum")
GAP_S = 1.0                         # the archive 429s on bursts; see docs/DATA_INVENTORY.md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=0, help="limit to the N best-covered tiles")
    ap.add_argument("--bands", nargs="*", default=["P892", "P878", "P863", "P848"])
    ap.add_argument("--gap", type=float, default=GAP_S)
    ap.add_argument("--rebuild-index", action="store_true",
                    help="write index.json from files already on disk, fetching nothing")
    args = ap.parse_args()

    if not os.path.exists(CATALOG):
        sys.exit("missing %s - run scripts/build_footprint_index.py first" % CATALOG)
    cat = json.load(open(CATALOG))
    single = {b: v for b, v in cat.items() if b.isdigit()}

    per_tile: dict[str, list[str]] = {}
    for b, entries in single.items():
        for t, _ in entries:
            if t[:4] in args.bands:
                per_tile.setdefault(t, []).append(b)
    order = sorted(per_tile, key=lambda t: -len(per_tile[t]))
    if args.tiles:
        order = order[:args.tiles]

    os.makedirs(CORPUS, exist_ok=True)
    todo = [(t, b) for t in order for b in sorted(per_tile[t])]
    print("corpus: %d tiles, %d tile-bin products" % (len(order), len(todo)))

    index, fetched, failed = [], 0, []
    t_start = time.time()
    for i, (tile, b) in enumerate(todo):
        dst = os.path.join(CORPUS, "browse_CM_%s_%s.png" % (b, tile))
        if args.rebuild_index:
            if os.path.exists(dst):
                index.append({"tile": tile, "bin": b, "path": os.path.relpath(dst, ROOT)})
            continue
        if not os.path.exists(dst):
            try:
                time.sleep(args.gap)
                nac.load_browse(b, tile, CORPUS)
                fetched += 1
            except Exception as exc:
                failed.append((tile, b, "%s" % type(exc).__name__))
                continue
        index.append({"tile": tile, "bin": b, "path": os.path.relpath(dst, ROOT)})
        if (i + 1) % 50 == 0:
            print("  %d/%d  (%d fetched, %d failed, %.0fs)"
                  % (i + 1, len(todo), fetched, len(failed), time.time() - t_start), flush=True)

    meta = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_products": len(index), "n_tiles": len({r["tile"] for r in index}),
            "bands": args.bands, "source": "LROC NAC controlled south polar mosaics, "
                                            "browse pyramids (~32 m/px)",
            "ground_truth": "identity transform between bins of the same tile",
            "failed": failed, "products": index}
    out = os.path.join(CORPUS, "index.json")
    json.dump(meta, open(out, "w"), indent=1)
    size = sum(os.path.getsize(os.path.join(ROOT, r["path"])) for r in index)
    print("\n%d products over %d tiles, %.0f MB" % (len(index), meta["n_tiles"], size / 1e6))
    if failed:
        print("%d failed: %s" % (len(failed), failed[:5]))
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
