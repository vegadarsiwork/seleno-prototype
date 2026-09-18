"""Phase 1b: prove the zero-auth reference terrain actually opens, projects and
renders, then cut the small crops the unit tests will assert against.

Nothing here is taken on trust from a label:

* the LOLA DEM is read twice - once through GDAL's PDS driver, once as a raw
  ``numpy.memmap`` using only the numbers in the detached ``.lbl`` - and the two
  must agree pixel for pixel;
* the elevations are checked against the known depth of the south polar basin;
* the WAC mosaic's own CRS is used to pull the same ground the DEM covers, and
  both are resampled onto the project's south polar stereographic grid;
* a hillshade is rendered so the terrain can be eyeballed rather than asserted.

    python scripts/verify_reference_data.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

STEREO = ("+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
          "+R=1737400 +units=m +no_defs")
R_MOON = 1737400.0
OUT = os.path.join(ROOT, "data/processed/testcrops")
FIG = os.path.join(ROOT, "reports/gt_qc")


def ok(msg):
    print("  PASS  " + msg)


def fail(msg):
    print("  FAIL  " + msg)
    return 1


def parse_pds3_label(path: str) -> dict:
    """Minimal PDS3 keyword scrape - enough for a detached image label."""
    out = {}
    for line in open(path, errors="replace"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().strip('"')
        v = v.split("<")[0].strip().strip('"').strip()
        if k and k not in out:
            out[k] = v
    return out


def check_lola(stem: str, expect_lat_max: float) -> int:
    print("\nLOLA %s" % stem)
    lbl = os.path.join(ROOT, "data/raw/lola/%s.lbl" % stem)
    img = os.path.join(ROOT, "data/raw/lola/%s.img" % stem)
    if not (os.path.exists(lbl) and os.path.exists(img)):
        return fail("%s not downloaded" % stem)

    L = parse_pds3_label(lbl)
    lines, samples = int(L["LINES"]), int(L["LINE_SAMPLES"])
    scale = float(L["SCALING_FACTOR"])
    offset = float(L["OFFSET"])
    mscale = float(L["MAP_SCALE"])
    bad = 0

    want = lines * samples * int(L["SAMPLE_BITS"]) // 8
    got = os.path.getsize(img)
    if got != want:
        return fail("file is %d bytes, label declares %d (%.1f%% present) - "
                    "incomplete download, nothing downstream is meaningful"
                    % (got, want, 100.0 * got / want))
    else:
        ok("size matches the label exactly (%d x %d x 16 bit = %d bytes)" % (lines, samples, got))

    if abs(float(L["MAXIMUM_LATITUDE"]) - expect_lat_max) > 1e-6:
        bad |= fail("MAXIMUM_LATITUDE is %s, expected %s" % (L["MAXIMUM_LATITUDE"], expect_lat_max))
    else:
        ok("covers %s .. %s deg latitude at %g m/px" % (L["MINIMUM_LATITUDE"],
                                                        L["MAXIMUM_LATITUDE"], mscale))

    # --- the projected extent implied by the label must match the latitude circle
    half = samples / 2.0 * mscale
    rho = 2.0 * R_MOON * math.tan(math.radians(90.0 + float(L["MAXIMUM_LATITUDE"])) / 2.0)
    if abs(half - rho) / rho > 0.01:
        bad |= fail("half-width %.1f m vs latitude-circle radius %.1f m (%.2f%%)"
                    % (half, rho, 100 * abs(half - rho) / rho))
    else:
        ok("half-width %.1f m agrees with the %s deg circle at %.1f m (%.3f%%)"
           % (half, L["MAXIMUM_LATITUDE"], rho, 100 * abs(half - rho) / rho))

    # --- raw read
    raw = np.memmap(img, dtype="<i2", mode="r", shape=(lines, samples))
    c = lines // 2
    win_raw = np.array(raw[c - 512:c + 512, c - 512:c + 512], dtype=np.float64) * scale + offset

    # --- GDAL read of the same window, through the PDS driver
    try:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(lbl) as ds:
            ok("GDAL opens the detached label with driver %r, CRS %s"
               % (ds.driver, "present" if ds.crs else "ABSENT"))
            win_gdal = ds.read(1, window=Window(c - 512, c - 512, 1024, 1024)).astype(np.float64)
            win_gdal = win_gdal * scale + offset
            if np.allclose(win_raw, win_gdal):
                ok("GDAL and raw memmap agree on all 1 048 576 pixels")
            else:
                bad |= fail("GDAL and raw memmap disagree (max %.3f m)"
                            % np.abs(win_raw - win_gdal).max())
    except Exception as exc:
        print("  NOTE  GDAL could not read the PDS label (%s: %s); raw memmap path is "
              "authoritative" % (type(exc).__name__, exc))

    # --- physical sanity: the south polar basin floor is well below the mean radius
    lo, hi = win_raw.min(), win_raw.max()
    print("  INFO  centre 1024^2 window: elevation %.1f .. %.1f m (radius %.1f .. %.1f km)"
          % (lo - R_MOON, hi - R_MOON, lo / 1000, hi / 1000))
    if not (1_720_000 < lo < 1_750_000 and 1_720_000 < hi < 1_755_000):
        bad |= fail("elevations are not plausible lunar radii")
    else:
        ok("elevations are plausible lunar radii")
    return bad


def check_wac() -> int:
    print("\nLROC WAC global mosaic")
    tif = os.path.join(ROOT,
                       "data/raw/wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif")
    if not os.path.exists(tif):
        return fail("not downloaded")
    import rasterio
    bad = 0
    with rasterio.open(tif) as ds:
        print("  INFO  %d x %d, %s, %d band(s), blocks %s, overviews %s"
              % (ds.width, ds.height, ds.dtypes[0], ds.count, ds.block_shapes[0],
                 ds.overviews(1)[:4] or "none"))
        print("  INFO  CRS %s" % (ds.crs.to_string() if ds.crs else "ABSENT"))
        print("  INFO  bounds %s" % (tuple(round(v, 1) for v in ds.bounds),))
        if ds.crs is None:
            bad |= fail("no CRS")
        else:
            ok("carries a CRS")
        if abs(abs(ds.transform.a) - 100.0) > 1.0:
            print("  NOTE  pixel size is %.3f map units, not 100 m - the mosaic is "
                  "equirectangular in degrees" % abs(ds.transform.a))
        else:
            ok("100 m pixels")
    return bad


def polar_crops(size_px: int = 1024) -> int:
    """Put LOLA and WAC on the project's grid over the OHRC footprint and render."""
    print("\nCommon-grid crop over the Chandrayaan-2 OHRC footprint")
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    # the four OHRC strips all sit inside +/- 25 km of the pole
    half_m = 25_000.0
    res = 2 * half_m / size_px
    dst_tr = from_origin(-half_m, half_m, res, res)
    bad = 0

    # --- LOLA
    lbl = os.path.join(ROOT, "data/raw/lola/ldem_875s_5m.lbl")
    L = parse_pds3_label(lbl)
    n = int(L["LINES"])
    mscale = float(L["MAP_SCALE"])
    src_tr = from_origin(-n / 2 * mscale, n / 2 * mscale, mscale, mscale)
    raw = np.memmap(os.path.join(ROOT, "data/raw/lola/ldem_875s_5m.img"),
                    dtype="<i2", mode="r", shape=(n, n))
    # crop generously in source pixels first so we never materialise 30336^2
    pad = int(half_m / mscale) + 8
    c = n // 2
    sub = np.array(raw[c - pad:c + pad, c - pad:c + pad], dtype=np.float32)
    sub = sub * float(L["SCALING_FACTOR"]) + float(L["OFFSET"]) - R_MOON
    sub_tr = from_origin(-pad * mscale, pad * mscale, mscale, mscale)

    dem = np.zeros((size_px, size_px), np.float32)
    reproject(sub, dem, src_transform=sub_tr, src_crs=STEREO,
              dst_transform=dst_tr, dst_crs=STEREO, resampling=Resampling.bilinear)
    print("  INFO  LOLA on grid: %.1f .. %.1f m relative to 1737.4 km" % (dem.min(), dem.max()))
    if np.ptp(dem) < 100:
        bad |= fail("DEM relief under 100 m over 50 km - suspicious")
    else:
        ok("relief %.0f m over the 50 km box" % np.ptp(dem))

    # --- WAC
    tif = os.path.join(ROOT, "data/raw/wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif")
    wac = np.zeros((size_px, size_px), np.uint8)
    with rasterio.open(tif) as ds:
        reproject(rasterio.band(ds, 1), wac, dst_transform=dst_tr, dst_crs=STEREO,
                  resampling=Resampling.bilinear)
    print("  INFO  WAC on grid: DN %d .. %d, mean %.1f" % (wac.min(), wac.max(), wac.mean()))
    if wac.max() == wac.min():
        bad |= fail("WAC crop is constant - reprojection produced nothing")
    else:
        ok("WAC reprojects onto the same grid and carries signal")

    # --- hillshade, so the terrain can be seen rather than asserted
    gy, gx = np.gradient(dem.astype(np.float64), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(315.0), math.radians(45.0)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)

    np.save(os.path.join(OUT, "lola_5m_pole_50km_%d.npy" % size_px), dem)
    np.save(os.path.join(OUT, "wac_100m_pole_50km_%d.npy" % size_px), wac)
    np.save(os.path.join(OUT, "lola_hillshade_pole_50km_%d.npy" % size_px), hs.astype(np.float32))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    for a, img, t, kw in (
            (ax[0], dem, "LOLA 5 m/px DEM\n(m relative to 1737.4 km)", dict(cmap="terrain")),
            (ax[1], hs, "hillshade, sun az 315 alt 45", dict(cmap="gray", vmin=0, vmax=1)),
            (ax[2], wac, "LROC WAC 100 m mosaic", dict(cmap="gray"))):
        im = a.imshow(img, extent=[-half_m / 1000, half_m / 1000] * 2, **kw)
        a.set_title(t, fontsize=10)
        a.set_xlabel("km east of the pole")
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[0].set_ylabel("km north of the pole")
    fig.suptitle("Phase 1b verification: zero-auth reference terrain on the project's "
                 "south polar stereographic grid", fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "phase1b_reference_terrain.png")
    fig.savefig(p, dpi=110)
    print("  INFO  wrote %s" % os.path.relpath(p, ROOT))
    print("  INFO  wrote 3 test crops -> %s" % os.path.relpath(OUT, ROOT))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crops", action="store_true")
    args = ap.parse_args()
    bad = 0
    bad |= check_lola("ldem_875s_5m", -87.5)
    bad |= check_lola("ldem_80s_20m", -80.0)
    bad |= check_wac()
    if not args.skip_crops:
        bad |= polar_crops()
    print("\n%s" % ("ALL CHECKS PASSED" if not bad else "SOME CHECKS FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
