"""Candidate source->reference pairs, discovered from data/index/footprints.gpkg.

Every intersection is computed in the south polar stereographic plane.  Doing it
in lon/lat here would be wrong, not merely imprecise: one OHRC strip spans
222 deg -> 110 deg -> 22 deg of longitude while covering 25 km of ground.

Two illumination-difference columns are emitted for every candidate, and the
difference between them is the main result of Phase 1:

* `d_incidence_deg` - the quantity the problem statement asks to stratify by.
  South of about 89 deg it is degenerate: the whole archive sits in a 1 deg band
  around 90 deg, so every pair lands in the 0-10 deg bin and the stratification
  carries no information.
* `d_sub_solar_lon_deg` - the Sun *azimuth* change.  This is the variable that
  actually moves at the pole, it spans the full 0-180 deg, and it is what
  decides whether cast shadows agree between two images.

    python scripts/discover_pairs.py
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def circ_delta(a, b):
    """Smallest absolute difference between two angles, degrees."""
    if a is None or b is None:
        return None
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data/index/footprints.gpkg"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/index/pairs.csv"))
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="keep candidates covering at least this fraction of the source")
    args = ap.parse_args()

    import geopandas as gpd
    import pandas as pd

    g = gpd.read_file(args.index, layer="footprints_stereo")
    src = g[g.role == "source"]
    ref = g[g.role == "reference"]
    print("sources %d, references %d" % (len(src), len(ref)))

    # spatial index so 4 x ~15 000 is not a full cross product
    sidx = ref.sindex
    rows = []
    for _, s in src.iterrows():
        sgeom = s.geometry
        sarea = sgeom.area
        hits = ref.iloc[sorted(sidx.query(sgeom, predicate="intersects"))]
        for _, r in hits.iterrows():
            inter = sgeom.intersection(r.geometry)
            if inter.is_empty:
                continue
            a = inter.area
            rows.append({
                "source_id": s.product_id,
                "source_instrument": s.instrument,
                "source_gsd_m": s.gsd_m,
                "reference_id": r.product_id,
                "reference_mission": r.mission,
                "reference_instrument": r.instrument,
                "reference_level": r.level,
                "reference_gsd_m": r.gsd_m,
                "pair_type": "%s-%s" % (s.instrument, r.instrument),
                "scale_ratio": (r.gsd_m / s.gsd_m) if s.gsd_m else None,
                "overlap_km2": a / 1e6,
                "coverage_of_source": a / sarea if sarea else 0.0,
                "coverage_of_reference": a / r.geometry.area if r.geometry.area else 0.0,
                "d_incidence_deg": circ_delta(s.sun_incidence_deg, r.sun_incidence_deg),
                "d_sub_solar_lon_deg": circ_delta(s.sub_solar_lon_deg, r.sub_solar_lon_deg),
                "source_sub_solar_lon_deg": s.sub_solar_lon_deg,
                "reference_sub_solar_lon_deg": r.sub_solar_lon_deg,
                "source_sun_trusted": bool(s.sun_geometry_trusted),
                "reference_sun_trusted": bool(r.sun_geometry_trusted),
                "reference_bytes": r.bytes,
                "reference_url": r.source_url,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no overlaps found")
        return
    df = df[df.coverage_of_source >= args.min_coverage]
    df = df.sort_values(["source_id", "coverage_of_source"], ascending=[True, False])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("wrote %s  (%d candidate pairs)\n" % (args.out, len(df)))

    print("── candidates per pair type")
    print(df.groupby("pair_type").agg(
        pairs=("reference_id", "size"),
        median_coverage=("coverage_of_source", "median"),
        max_coverage=("coverage_of_source", "max")).to_string())

    print("\n── coverage of the source footprint")
    for lo, hi in [(0.0, .1), (.1, .5), (.5, .9), (.9, 1.01)]:
        n = ((df.coverage_of_source >= lo) & (df.coverage_of_source < hi)).sum()
        print("   %4.0f%% - %4.0f%% : %5d" % (lo * 100, hi * 100, n))

    print("\n── Sun-angle difference, the PS stratification (incidence)")
    d = df.d_incidence_deg.dropna()
    if len(d):
        for lo, hi, lbl in [(0, 10, "0-10"), (10, 30, "10-30"), (30, 1e9, ">30")]:
            print("   %-6s deg : %5d   (%.1f%%)" % (lbl, ((d >= lo) & (d < hi)).sum(),
                                                    100 * ((d >= lo) & (d < hi)).mean()))
        print("   observed range: %.3f .. %.3f deg" % (d.min(), d.max()))

    print("\n── Sun-azimuth difference, what actually varies at the pole")
    d = df.d_sub_solar_lon_deg.dropna()
    if len(d):
        for lo, hi, lbl in [(0, 10, "0-10"), (10, 45, "10-45"), (45, 90, "45-90"),
                            (90, 1e9, ">90")]:
            print("   %-6s deg : %5d   (%.1f%%)" % (lbl, ((d >= lo) & (d < hi)).sum(),
                                                    100 * ((d >= lo) & (d < hi)).mean()))
        print("   observed range: %.1f .. %.1f deg" % (d.min(), d.max()))

    print("\n── best illumination-matched reference per source (controlled NAC only)")
    cm = df[(df.reference_instrument == "NAC") & df.d_sub_solar_lon_deg.notna()
            & (df.coverage_of_source > 0.2)]
    for sid, grp in cm.groupby("source_id"):
        best = grp.sort_values("d_sub_solar_lon_deg").iloc[0]
        print("   %-44s -> %-34s d_az %5.1f deg, covers %4.0f%% of source"
              % (sid[:44], best.reference_id, best.d_sub_solar_lon_deg,
                 100 * best.coverage_of_source))


if __name__ == "__main__":
    main()
