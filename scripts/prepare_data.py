"""Build the Seleno demo pairs from real Chandrayaan-2 OHRC products.

Source products (ISRO, PDS4, redistributed by the Internet Archive - see
data/README.md):

  ch2_ohr_ncp_20200229T0739312111_d_img_d18   orbit 2297, calibrated
  ch2_ohr_nrp_20200229T0739312111_d_img_d18   orbit 2297, raw (same acquisition)
  ch2_ohr_ncp_20200229T0938004033_d_img_d32   orbit 2298, calibrated

Nothing here synthesises lunar terrain. Every pixel comes from a delivered ISRO
product. What *is* synthetic - and is labelled as such in the manifest - is the
geometric transform applied to build a controlled pair with a known ground truth,
and the resampling used to simulate a TMC-2-class ground sampling distance.

Run:  python scripts/prepare_data.py --raw <dir-with-downloaded-products>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from seleno.geodesy import GeoGrid, ground_distance_m  # noqa: E402

OUT = os.path.join(ROOT, "data", "pairs")

# --- delivered product facts, read from the PDS4 labels ---------------------- #
OHRC_GSD_A = 0.22977000623605362      # m/px, orbit 2297  (isda:pixel_resolution)
OHRC_GSD_B = 0.2302230600054759       # m/px, orbit 2298
BROWSE_DECIMATION = 10                # 93693x12000 full res -> 9369x1200 browse
TMC2_GSD = 5.0                        # m/px, TMC-2 nominal
FULL_SAMPLES = 12000


def T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)


def similarity_about(cx, cy, deg, scale):
    c = np.cos(np.radians(deg)) * scale
    s = np.sin(np.radians(deg)) * scale
    M = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
    return T(cx, cy) @ M @ T(-cx, -cy)


def perspective_about(cx, cy, gx, gy):
    P = np.array([[1, 0, 0], [0, 1, 0], [gx, gy, 1]], np.float64)
    return T(cx, cy) @ P @ T(-cx, -cy)


def radiometric(img, gamma=1.0, gain=1.0, bias=0.0, noise_sigma=0.0, seed=7):
    """Apply an illumination-like intensity change. Geometry untouched."""
    f = img.astype(np.float32) / 255.0
    f = np.power(np.clip(f, 0, 1), gamma) * gain + bias
    if noise_sigma > 0:
        f = f + np.random.default_rng(seed).normal(0, noise_sigma, f.shape).astype(np.float32)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def synth_pair(tile, *, size, src_xy, deg, scale, persp=(0.0, 0.0), radio=None):
    """Cut a source window and a geometrically transformed reference window.

    Returns (src, ref, H_gt) where H_gt maps *source pixel* -> *reference pixel*
    exactly, by construction.
    """
    th, tw = tile.shape[:2]
    sx, sy = src_xy
    src = tile[sy:sy + size, sx:sx + size]
    if src.shape[:2] != (size, size):
        raise ValueError("source window falls outside the tile")

    cx, cy = tw / 2.0, th / 2.0
    H_world = perspective_about(cx, cy, *persp) @ similarity_about(cx, cy, deg, scale)

    # where the source-window centre lands, so the reference window contains it
    p = H_world @ np.array([sx + size / 2.0, sy + size / 2.0, 1.0])
    p = p[:2] / p[2]
    rx = int(round(p[0] - size / 2.0))
    ry = int(round(p[1] - size / 2.0))
    rx = int(np.clip(rx, 0, max(tw - size, 0)))
    ry = int(np.clip(ry, 0, max(th - size, 0)))

    ref_full = cv2.warpPerspective(tile, H_world, (tw, th), flags=cv2.INTER_LINEAR)
    ref = ref_full[ry:ry + size, rx:rx + size]
    if radio:
        ref = radiometric(ref, **radio)

    H_gt = T(-rx, -ry) @ H_world @ T(sx, sy)
    H_gt = H_gt / H_gt[2, 2]
    return src, ref, H_gt


def simulate_coarse(tile, src_gsd, target_gsd, deg=0.0):
    """Anti-aliased resample of a real tile to a coarser GSD, plus optional rotation.

    This SIMULATES a coarser sensor's sampling. It is not TMC-2 data and does not
    reproduce TMC-2's optics, MTF, noise or viewing geometry.
    """
    f = src_gsd / target_gsd
    blurred = cv2.GaussianBlur(tile, (0, 0), 0.5 / f)
    h, w = tile.shape[:2]
    nw, nh = max(int(round(w * f)), 16), max(int(round(h * f)), 16)
    small = cv2.resize(blurred, (nw, nh), interpolation=cv2.INTER_AREA)
    H = np.diag([f, f, 1.0]).astype(np.float64)
    if deg:
        R = similarity_about(nw / 2.0, nh / 2.0, deg, 1.0)
        small = cv2.warpPerspective(small, R, (nw, nh), flags=cv2.INTER_LINEAR)
        H = R @ H
        # Trim to the rectangle that stays inside the rotated support, so the
        # reference contains only real resampled data and no synthetic black
        # corners for the matcher to latch onto.
        a = abs(np.radians(deg))
        keep = 1.0 / (abs(np.cos(a)) + abs(np.sin(a)))
        cw, ch = int(nw * keep) // 2 * 2, int(nh * keep) // 2 * 2
        ox, oy = (nw - cw) // 2, (nh - ch) // 2
        small = small[oy:oy + ch, ox:ox + cw]
        H = T(-ox, -oy) @ H
    return small, H / H[2, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory holding the downloaded ISRO products")
    args = ap.parse_args()
    raw = args.raw
    os.makedirs(OUT, exist_ok=True)

    def R(name):
        p = os.path.join(raw, name)
        if not os.path.exists(p):
            raise SystemExit("missing input: " + p)
        return p

    band78 = np.load(R("ohrcA_rows78000_82096.npy"))     # crater field, well lit
    band48 = np.load(R("ohrcA_rows48000_52096.npy"))     # lit terrain beside deep shadow
    band02 = np.load(R("ohrcA_rows2000_6096.npy"))       # polar, effectively unlit
    brw_cal = cv2.imread(R("ch2_ohr_ncp_20200229T0739312111_b_brw_d18.png"), 0)
    brw_raw = cv2.imread(R("ch2_ohr_nrp_20200229T0739312111_b_brw_d18.png"), 0)
    brw_b = cv2.imread(R("ch2_ohr_ncp_20200229T0938004033_b_brw_d32.png"), 0)

    geoA = GeoGrid.from_csv(R("ch2_ohr_ncp_20200229T0739312111_g_grd_d18.csv"))
    geoB = GeoGrid.from_csv(R("ch2_ohr_ncp_20200229T0938004033_g_grd_d32.csv"))

    pairs = []

    def emit(pid, src, ref, H_gt, meta):
        cv2.imwrite(os.path.join(OUT, pid + "_source.png"), src)
        cv2.imwrite(os.path.join(OUT, pid + "_reference.png"), ref)
        meta = dict(meta)
        meta["id"] = pid
        meta["source_file"] = pid + "_source.png"
        meta["reference_file"] = pid + "_reference.png"
        meta["source_shape"] = list(src.shape[:2])
        meta["reference_shape"] = list(ref.shape[:2])
        meta["ground_truth_available"] = H_gt is not None
        meta["gt_homography"] = None if H_gt is None else [[float(v) for v in r] for r in H_gt]
        pairs.append(meta)
        print("  %-24s src%s ref%s gt=%s" % (pid, src.shape[:2], ref.shape[:2], H_gt is not None))

    # ---------------------------------------------------------------- pair 1
    # Well-lit crater field at native OHRC sampling. Moderate view change.
    SIZE = 1280
    tile = band78[:, 1200:1200 + 3600]
    src, ref, H = synth_pair(tile, size=SIZE, src_xy=(900, 900), deg=11.0, scale=1.12,
                             persp=(2.0e-5, -1.2e-5),
                             radio=dict(gamma=1.28, gain=0.94, bias=0.03, noise_sigma=0.006))
    fp = geoA.footprint(1200 + 900, 78000 + 900, SIZE, SIZE)
    emit("ohrc_crater_field", src, ref, H, {
        "name": "OHRC crater field - view change",
        "short": "Well-lit mare/highland crater field at native OHRC sampling.",
        "difficulty": "baseline",
        "sensor_source": "Chandrayaan-2 OHRC (calibrated, orbit 2297)",
        "sensor_reference": "Same OHRC product, synthetic view change + illumination change",
        "gsd_source_m": OHRC_GSD_A, "gsd_reference_m": OHRC_GSD_A,
        "footprint_m": round(SIZE * OHRC_GSD_A, 1),
        "recommended_model": "homography",
        "synthetic_component": ("reference is the same real imagery re-projected by a known "
                                "11 deg rotation, 1.12x scale and small perspective, then "
                                "gamma/gain adjusted"),
        "real_component": "all pixels are delivered ISRO OHRC data",
        "geo": fp,
    })

    # ---------------------------------------------------------------- pair 2
    # Fully real: raw vs calibrated product of the SAME acquisition, offset crops.
    # Ground truth is an exact translation - no resampling of either side.
    S2 = 1024
    ax, ay = 90, 6300
    bx, by = 90 + 37, 6300 - 24
    src2 = brw_cal[ay:ay + S2, ax:ax + S2]
    ref2 = brw_raw[by:by + S2, bx:bx + S2]
    H2 = T(ax - bx, ay - by)
    fp2 = geoA.footprint(ax * BROWSE_DECIMATION, ay * BROWSE_DECIMATION,
                         S2 * BROWSE_DECIMATION, S2 * BROWSE_DECIMATION)
    emit("ohrc_raw_vs_calibrated", src2, ref2, H2, {
        "name": "OHRC calibrated vs raw - same acquisition",
        "short": "Two ISRO products of the identical scene; only radiometry differs.",
        "difficulty": "real pair, exact ground truth",
        "sensor_source": "OHRC calibrated browse (ncp), orbit 2297",
        "sensor_reference": "OHRC raw browse (nrp), same acquisition 2020-02-29T07:39:31Z",
        "gsd_source_m": OHRC_GSD_A * BROWSE_DECIMATION,
        "gsd_reference_m": OHRC_GSD_A * BROWSE_DECIMATION,
        "footprint_m": round(S2 * OHRC_GSD_A * BROWSE_DECIMATION, 1),
        "recommended_model": "similarity",
        "synthetic_component": ("none - the two windows are cut from two delivered products "
                                "at a known pixel offset, so ground truth is an exact translation"),
        "real_component": "both sides are delivered ISRO OHRC data; radiometric difference is real",
        "geo": fp2,
    })

    # ---------------------------------------------------------------- pair 3
    # Cross-resolution, moderate ratio: OHRC browse (2.30 m/px) -> TMC-2 GSD (5 m/px)
    S3 = 1100
    tile3 = brw_cal[7100:7100 + S3, 50:50 + S3]
    ref3, H3 = simulate_coarse(tile3, OHRC_GSD_A * BROWSE_DECIMATION, TMC2_GSD, deg=6.0)
    ref3 = radiometric(ref3, gamma=1.15, gain=0.97, bias=0.02)
    fp3 = geoA.footprint(50 * BROWSE_DECIMATION, 7100 * BROWSE_DECIMATION,
                         S3 * BROWSE_DECIMATION, S3 * BROWSE_DECIMATION)
    emit("ohrc_tmc2_moderate", tile3, ref3, H3, {
        "name": "Cross-resolution 2.3 -> 5.0 m/px (TMC-2 class)",
        "short": "OHRC browse against a simulated TMC-2 ground sampling distance (2.2x).",
        "difficulty": "cross-resolution",
        "sensor_source": "OHRC calibrated browse, orbit 2297 (~2.30 m/px)",
        "sensor_reference": "SIMULATED TMC-2-class product at 5.0 m/px - not real TMC-2 data",
        "gsd_source_m": OHRC_GSD_A * BROWSE_DECIMATION, "gsd_reference_m": TMC2_GSD,
        "footprint_m": round(S3 * OHRC_GSD_A * BROWSE_DECIMATION, 1),
        "recommended_model": "similarity",
        "synthetic_component": ("reference produced by anti-aliased resampling of the real tile "
                                "to 5.0 m/px plus a 6 deg rotation; TMC-2 optics, MTF, noise and "
                                "viewing geometry are NOT simulated"),
        "real_component": "source is delivered ISRO OHRC data; reference derives from it",
        "geo": fp3,
    })

    # ---------------------------------------------------------------- pair 4
    # Cross-resolution, the real OHRC-to-TMC-2 ratio: 0.23 -> 5.0 m/px (21.8x)
    S4 = 3072
    tile4 = band78[512:512 + S4, 1500:1500 + S4]
    ref4, H4 = simulate_coarse(tile4, OHRC_GSD_A, TMC2_GSD, deg=4.0)
    ref4 = radiometric(ref4, gamma=1.10, gain=0.98)
    fp4 = geoA.footprint(1500, 78000 + 512, S4, S4)
    emit("ohrc_tmc2_extreme", tile4, ref4, H4, {
        "name": "Cross-resolution 0.23 -> 5.0 m/px (21.8x)",
        "short": "The actual OHRC-to-TMC-2 scale gap. Expected to be hard.",
        "difficulty": "hard - extreme scale ratio",
        "sensor_source": "Chandrayaan-2 OHRC full resolution, orbit 2297",
        "sensor_reference": "SIMULATED TMC-2-class product at 5.0 m/px - not real TMC-2 data",
        "gsd_source_m": OHRC_GSD_A, "gsd_reference_m": TMC2_GSD,
        "footprint_m": round(S4 * OHRC_GSD_A, 1),
        "recommended_model": "similarity",
        "synthetic_component": "reference is the real tile resampled to 5.0 m/px plus 4 deg rotation",
        "real_component": "source is delivered ISRO OHRC data at native 0.23 m/px",
        "geo": fp4,
    })

    # ---------------------------------------------------------------- pair 5
    # Real low-illumination polar terrain. Near the poles OHRC images terrain that
    # receives almost no direct sun; DN range collapses and descriptors starve.
    S5 = 1280
    # Window chosen by scanning band48 for a genuine shadow boundary: ~35 percent
    # of pixels below DN 20 with the lit remainder still carrying texture.
    tile5 = band48[:, 900:900 + 3000]
    src5, ref5, H5 = synth_pair(tile5, size=S5, src_xy=(380, 1280), deg=9.0, scale=1.08,
                                radio=dict(gamma=1.6, gain=0.8, bias=0.0, noise_sigma=0.012))
    fp5 = geoA.footprint(900 + 380, 48000 + 1280, S5, S5)
    emit("ohrc_low_illumination", src5, ref5, H5, {
        "name": "Low-illumination polar terrain",
        "short": "Real shadowed south-polar terrain with a strong illumination change.",
        "difficulty": "hard - low contrast and shadow",
        "sensor_source": "Chandrayaan-2 OHRC full resolution, orbit 2297 (shadowed band)",
        "sensor_reference": "Same imagery, known view change plus strong gamma/gain change",
        "gsd_source_m": OHRC_GSD_A, "gsd_reference_m": OHRC_GSD_A,
        "footprint_m": round(S5 * OHRC_GSD_A, 1),
        "recommended_model": "homography",
        "synthetic_component": "known 9 deg rotation, 1.08x scale, gamma 1.6 with added noise",
        "real_component": "all pixels are delivered ISRO OHRC data from a genuinely shadowed region",
        "geo": fp5,
    })

    # ---------------------------------------------------------------- pair 6
    # True negative. Two real strips from consecutive orbits whose ground tracks
    # do not overlap. Correct behaviour is to refuse to register them.
    S6 = 1024
    src6 = brw_cal[7600:7600 + S6, 90:90 + S6]
    ref6 = brw_b[5200:5200 + S6, 90:90 + S6]
    fpa = geoA.footprint(90 * BROWSE_DECIMATION, 7600 * BROWSE_DECIMATION,
                         S6 * BROWSE_DECIMATION, S6 * BROWSE_DECIMATION)
    fpb = geoB.footprint(90 * BROWSE_DECIMATION, 5200 * BROWSE_DECIMATION,
                         S6 * BROWSE_DECIMATION, S6 * BROWSE_DECIMATION)
    sep = ground_distance_m(*fpa["center_lon_lat"], *fpb["center_lon_lat"])
    emit("ohrc_disjoint_orbits", src6, ref6, None, {
        "name": "Disjoint orbits - no true overlap",
        "short": "Two real OHRC strips %.1f km apart. There is no correct registration." % (sep / 1000.0),
        "difficulty": "true negative",
        "sensor_source": "OHRC calibrated browse, orbit 2297 (2020-02-29T07:39Z)",
        "sensor_reference": "OHRC calibrated browse, orbit 2298 (2020-02-29T09:38Z)",
        "gsd_source_m": OHRC_GSD_A * BROWSE_DECIMATION,
        "gsd_reference_m": OHRC_GSD_B * BROWSE_DECIMATION,
        "footprint_m": round(S6 * OHRC_GSD_A * BROWSE_DECIMATION, 1),
        "recommended_model": "similarity",
        "synthetic_component": "none",
        "real_component": "both sides are delivered ISRO OHRC data from different orbits",
        "expected_outcome": ("no valid registration; a high inlier count here would indicate the "
                            "pipeline is fitting noise"),
        "separation_km": round(sep / 1000.0, 2),
        "geo": {"source": fpa, "reference": fpb},
    })

    manifest = {
        "generated_by": "scripts/prepare_data.py",
        "products": {
            "ohrc_orbit_2297_calibrated": {
                "lid": "urn:isro:isda:ch2_cho.ohr:data_calibrated:"
                       "ch2_ohr_ncp_20200229t0739312111_d_img_d18",
                "lines": 93693, "samples": FULL_SAMPLES,
                "pixel_resolution_m": OHRC_GSD_A,
                "spacecraft_altitude_km": 90.405660145954926,
                "acquired": "2020-02-29T07:39:31.2111Z",
            },
            "ohrc_orbit_2297_raw": {
                "lid": "urn:isro:isda:ch2_cho.ohr:data_raw:"
                       "ch2_ohr_nrp_20200229t0739312111_d_img_d18",
                "pixel_resolution_m": OHRC_GSD_A,
            },
            "ohrc_orbit_2298_calibrated": {
                "lid": "urn:isro:isda:ch2_cho.ohr:data_calibrated:"
                       "ch2_ohr_ncp_20200229t0938004033_d_img_d32",
                "lines": 93693, "samples": FULL_SAMPLES,
                "pixel_resolution_m": OHRC_GSD_B,
                "spacecraft_altitude_km": 90.583919379077614,
                "acquired": "2020-02-29T09:38:00.4033Z",
            },
        },
        "pairs": pairs,
    }
    with open(os.path.join(ROOT, "data", "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("\nwrote %d pairs -> %s" % (len(pairs), OUT))


if __name__ == "__main__":
    main()
