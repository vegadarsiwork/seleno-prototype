"""CLI:  python -m seleno register --source X --reference Y --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="seleno",
                                 description="Lunar image correspondence and registration.")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("register", help="register a source image onto a reference")
    r.add_argument("--source", required=True)
    r.add_argument("--reference", required=True)
    r.add_argument("--out", default="outputs")
    r.add_argument("--model", default="auto",
                   choices=["auto", "similarity", "affine", "homography"])
    r.add_argument("--max-side", type=int, default=2048,
                   help="working grid cap; the reference is decimated to fit")
    r.add_argument("--grid", type=int, default=8,
                   help="N x N grid for tie points and the coverage metric")
    r.add_argument("--segments", type=int, default=0,
                   help="fit this many per-segment transforms along the long axis "
                        "(for pushbroom strips); 0 disables")
    r.add_argument("--no-fine", action="store_true",
                   help="skip the native-resolution fine stage (faster; the "
                        "reported accuracy is then floored by the working grid)")
    r.add_argument("--fine-tiles", type=int, default=0,
                   help="cap on native-resolution windows; 0 means as many as "
                        "needed to tile the overlap (max 24)")
    r.add_argument("--no-subpixel", action="store_true",
                   help="skip the ECC sub-pixel polish (faster; the discrete "
                        "stages' own accuracy is then what is reported)")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--json", action="store_true", help="print metrics.json to stdout")

    p = sub.add_parser("profiles", help="list the sensor profiles that are loaded")

    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if a.command == "profiles":
        from seleno.tool import Profiles
        pr = Profiles.load()
        for n in pr.names():
            d = pr.get(n)
            print("  %-12s %-10s gsd %-8s bits %-5s shadow %-6s texture %s"
                  % (n, d.get("instrument"), d.get("gsd_m"), d.get("bit_depth"),
                     d.get("shadow_threshold"), d.get("texture_threshold")))
        return 0

    from seleno.tool import register
    res = register(a.source, a.reference, a.out, model=a.model, max_side=a.max_side,
                   grid=(a.grid, a.grid), segments=a.segments,
                   subpixel=not a.no_subpixel, fine=not a.no_fine,
                   fine_tiles=a.fine_tiles, verbose=not a.quiet)
    if a.json:
        print(json.dumps(res.metrics, indent=1))
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
