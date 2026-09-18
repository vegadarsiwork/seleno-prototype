"""Build real-illumination-change pairs from the LROC controlled polar mosaics.

Source: LROC `NAC_POLE_SOUTH_CM_XXX` - controlled 1 m/px south polar mosaics,
one per 10-degree bin of sub-solar longitude (Archinal et al. 2023, LPSC 54
#2333). Public, no credentials; served from NASA PDS by HTTP range reads.

Why these make a useful second dataset:

* Every bin is a subset of **one** ISIS jigsaw control network (18 323 NAC
  images, LOLA-tied, bundle residual 0.8 px at 2 sigma) and is projected onto
  the **same** 1 m polar stereographic grid. The same (col, row) in two bins is
  the same ground to about a pixel. So a pair cut from two bins has a
  **real** illumination change - different Sun azimuth, different cast shadows,
  different NAC images - and a ground truth that is known without any matcher.
* The legacy OHRC pairs only have *synthetic* illumination changes (gamma/gain).
  The OHRC repeat-pass windows have real ones but no ground truth. This fills
  the gap between the two.

What *is* synthetic, and recorded as such in the manifest: a known
similarity + mild perspective warp applied to the reference window, so the
pipeline cannot win by returning the identity. Ground truth is that warp,
exact by construction, on top of the ~0.8 px control-network residual.

Negative controls: pairs whose two windows are different ground. The correct
answer for those is a refusal.

    python scripts/fetch_lroc_pairs.py              # build data/lroc/
    python scripts/fetch_lroc_pairs.py --list       # show chosen windows only

Full-resolution reads are cached under results/lroc_cache/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "lroc")
CACHE = os.path.join(ROOT, "results", "lroc_cache")

STORE = ("https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/"
         "pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/")
TILE = "P892S2250"          # lon 180-270 E, lat -90..-88.5
NS = NL = 45488             # samples, lines (PDS4 label)
RECORD = NS * 4             # IEEE754LSBSingle
HDR = RECORD                # array offset 181952 bytes (PDS4 label)
NODATA_BELOW = -1e30        # missing_constant 0xFF7FFFFB = -3.4028e38
GSD = 1.0

SIZE = 1024                 # window side, px = m
BROWSE_STEP = 32            # 45488 / 1421 browse pixels

# (source bin, reference bin). Sub-solar longitude is the Sun azimuth proxy;
# the difference spans 50 to 180 degrees.
PAIRS = [
    ("005", "055"),
    ("095", "145"),
    ("145", "235"),
    ("235", "325"),
    ("055", "145"),
    ("145", "325"),
    ("055", "235"),
]
PER_PAIR = 2
NEGATIVES = 2


def img_url(b: str) -> str:
    return (STORE + "DATA/BDR/NAC_POLE/NAC_POLE_SOUTH_CM_%s/"
            "NAC_POLE_SOUTH_CM_%s_%s.IMG" % (b, b, TILE))


def browse_url(b: str) -> str:
    return (STORE + "EXTRAS/BROWSE/NAC_POLE/NAC_POLE_SOUTH_CM_%s/"
            "NAC_POLE_SOUTH_CM_%s_%s.BROWSE.PNG" % (b, b, TILE))


def curl(url: str, dst: str, byte_range: str | None = None) -> None:
    cmd = ["curl", "-sfL", "--retry", "3", "--max-time", "1800", "-o", dst]
    if byte_range:
        cmd += ["-r", byte_range]
    subprocess.run(cmd + [url], check=True)


def curl_range(url: str, dst: str, lo: int, hi: int, chunk: int = 32 << 20) -> None:
    """Fetch [lo, hi] in chunks, retrying each one.

    One 186 MB request is at the mercy of a single dropped connection (curl
    exit 56 was seen mid-band); small chunks make a reset cost seconds.
    """
    with open(dst, "wb") as out:
        for s in range(lo, hi + 1, chunk):
            e = min(s + chunk - 1, hi)
            part = dst + ".part"
            for attempt in range(6):
                try:
                    curl(url, part, "%d-%d" % (s, e))
                    if os.path.getsize(part) == e - s + 1:
                        break
                except subprocess.CalledProcessError:
                    pass
                time.sleep(2 * (attempt + 1))
            else:
                raise IOError("range %d-%d failed after retries: %s" % (s, e, url))
            with open(part, "rb") as fh:
                out.write(fh.read())
            os.remove(part)


def browse(b: str) -> np.ndarray:
    d = os.path.join(CACHE, "browse")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "CM_%s_%s.png" % (b, TILE))
    if not os.path.exists(p):
        curl(browse_url(b), p)
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError("could not read browse %s" % p)
    return im


# --------------------------------------------------------------------------- #
# window selection on the browse images

def cell_scores(im: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Per SIZE-sized cell: lit fraction and texture (std of lit pixels)."""
    k = SIZE // BROWSE_STEP
    lit = np.zeros((cells, cells))
    tex = np.zeros((cells, cells))
    for r in range(cells):
        for c in range(cells):
            blk = im[r * k:(r + 1) * k, c * k:(c + 1) * k].astype(np.float32)
            if blk.size == 0:
                continue
            m = blk > 25
            lit[r, c] = m.mean()
            tex[r, c] = blk[m].std() if m.sum() > 16 else 0.0
    return lit, tex


def choose_windows(bins: list[str]) -> list[dict]:
    """Pick windows on a SIZE grid so pairs sharing a row share one download."""
    cells = NS // SIZE
    scores = {b: cell_scores(browse(b), cells) for b in bins}
    used: set[tuple[int, int]] = set()
    out = []
    for a, b in PAIRS:
        la, ta = scores[a]
        lb, tb = scores[b]
        got = 0
        # Both lit and textured. Opposite-Sun bins rarely share much lit
        # ground, so a looser second pass keeps the hardest pairs in the set.
        for min_lit in (0.55, 0.35):
            ok = (la > min_lit) & (lb > min_lit) & (ta > 6) & (tb > 6)
            score = np.where(ok, np.minimum(la, lb) * np.minimum(ta, tb), -1)
            # keep away from tile edges so the warp has margin
            score[:1, :] = score[-1:, :] = score[:, :1] = score[:, -1:] = -1
            order = np.dstack(np.unravel_index(np.argsort(-score, axis=None),
                                               score.shape))[0]
            for r, c in order:
                if score[r, c] <= 0 or got >= PER_PAIR:
                    break
                if (r, c) in used:
                    continue
                used.add((int(r), int(c)))
                out.append({"a": a, "b": b, "row0": int(r) * SIZE, "col0": int(c) * SIZE,
                            "lit_a": round(float(la[r, c]), 3),
                            "lit_b": round(float(lb[r, c]), 3), "negative": False})
                got += 1
        if got < PER_PAIR:
            print("  note: only %d usable window(s) for %s/%s" % (got, a, b))

    # negatives: source and reference on different, both-textured ground,
    # reusing rows already being fetched so they cost nothing extra
    pos = [w for w in out]
    for i in range(min(NEGATIVES, len(pos) // 2)):
        w1, w2 = pos[i], pos[-1 - i]
        if (w1["row0"], w1["col0"]) == (w2["row0"], w2["col0"]):
            continue
        out.append({"a": w1["a"], "b": w2["b"], "row0": w1["row0"], "col0": w1["col0"],
                    "row0_b": w2["row0"], "col0_b": w2["col0"],
                    "lit_a": w1["lit_a"], "lit_b": w2["lit_b"], "negative": True})
    return out


# --------------------------------------------------------------------------- #
# full-resolution reads

def fetch_band(b: str, row0: int, cols: list[int]) -> dict[int, np.ndarray]:
    """One contiguous range request for SIZE rows; slice out each window."""
    os.makedirs(CACHE, exist_ok=True)
    need = [c for c in cols
            if not os.path.exists(os.path.join(CACHE, "CM_%s_r%d_c%d.npy" % (b, row0, c)))]
    if need:
        lo = HDR + row0 * RECORD
        hi = HDR + (row0 + SIZE) * RECORD - 1
        tmp = os.path.join(CACHE, "_band_%s_%d.bin" % (b, row0))
        t0 = time.time()
        curl_range(img_url(b), tmp, lo, hi)
        if os.path.getsize(tmp) != hi - lo + 1:
            os.remove(tmp)
            raise IOError("short read for CM_%s row %d" % (b, row0))
        buf = np.fromfile(tmp, dtype="<f4").reshape(SIZE, NS)
        for c in need:
            np.save(os.path.join(CACHE, "CM_%s_r%d_c%d.npy" % (b, row0, c)),
                    np.ascontiguousarray(buf[:, c:c + SIZE]))
        del buf
        os.remove(tmp)
        print("  CM_%s row %-6d %3.0f MB in %4.0fs" % (b, row0, (hi - lo + 1) / 1e6,
                                                        time.time() - t0))
    return {c: np.load(os.path.join(CACHE, "CM_%s_r%d_c%d.npy" % (b, row0, c)))
            for c in cols}


def to_u8(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-window percentile stretch. Returns image and nodata fraction."""
    valid = a > NODATA_BELOW
    bad = 1.0 - valid.mean()
    if valid.sum() < 100:
        return np.zeros(a.shape, np.uint8), bad
    lo, hi = np.percentile(a[valid], [0.5, 99.5])
    f = np.clip((np.where(valid, a, lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    return (f * 255 + 0.5).astype(np.uint8), bad


# --------------------------------------------------------------------------- #
# known warp

def warp_params(i: int) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    rng = np.random.default_rng(1000 + i)
    deg = float(rng.uniform(-20, 20))
    scale = float(rng.uniform(0.9, 1.1))
    shift = (float(rng.uniform(-60, 60)), float(rng.uniform(-60, 60)))
    persp = (float(rng.uniform(-3e-5, 3e-5)), float(rng.uniform(-3e-5, 3e-5)))
    return deg, scale, shift, persp


def warp_matrix(deg, scale, shift, persp) -> np.ndarray:
    c = SIZE / 2.0
    th = np.deg2rad(deg)
    T1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], float)
    S = np.array([[scale * np.cos(th), -scale * np.sin(th), 0],
                  [scale * np.sin(th), scale * np.cos(th), 0],
                  [persp[0], persp[1], 1]], float)
    T2 = np.array([[1, 0, c + shift[0]], [0, 1, c + shift[1]], [0, 0, 1]], float)
    H = T2 @ S @ T1
    return H / H[2, 2]


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="choose windows, fetch nothing big")
    ap.add_argument("--jobs", type=int, default=2, help="parallel range requests")
    args = ap.parse_args()

    bins = sorted({x for p in PAIRS for x in p})
    print("browse images for %d bins, tile %s" % (len(bins), TILE))
    wins = choose_windows(bins)
    for w in wins:
        print("  %s %s -> %s  row %-6d col %-6d lit %.2f/%.2f"
              % ("NEG" if w["negative"] else "pos", w["a"], w["b"], w["row0"], w["col0"],
                 w["lit_a"], w["lit_b"]))
    if args.list:
        return 0

    # group reads: (bin, row0) -> columns
    jobs: dict[tuple[str, int], set[int]] = {}
    for w in wins:
        jobs.setdefault((w["a"], w["row0"]), set()).add(w["col0"])
        jobs.setdefault((w["b"], w.get("row0_b", w["row0"])), set()).add(w.get("col0_b", w["col0"]))
    total = len(jobs) * SIZE * RECORD / 1e9
    print("\n%d range requests, up to %.1f GB (cached ones skipped)" % (len(jobs), total))
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda kv: fetch_band(kv[0][0], kv[0][1], sorted(kv[1])), jobs.items()))

    pairs_dir = os.path.join(OUT, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    manifest = {
        "generated_by": "scripts/fetch_lroc_pairs.py",
        "source": {
            "product": "LROC NAC_POLE_SOUTH_CM_XXX controlled illumination mosaics, tile %s" % TILE,
            "url": STORE + "DATA/BDR/NAC_POLE/",
            "gsd_m": GSD,
            "control": "ISIS jigsaw, 18 323 NAC images, LOLA-tied, 0.8 px at 2 sigma",
            "citation": "Archinal et al. (2023), LPSC 54, abstract #2333",
        },
        "pairs": [],
    }
    for i, w in enumerate(wins):
        a = fetch_band(w["a"], w["row0"], [w["col0"]])[w["col0"]]
        rb, cb = w.get("row0_b", w["row0"]), w.get("col0_b", w["col0"])
        b = fetch_band(w["b"], rb, [cb])[cb]
        src, bad_a = to_u8(a)
        ref_raw, bad_b = to_u8(b)
        if max(bad_a, bad_b) > 0.02:
            print("  skip %s/%s r%d c%d: %.0f%% nodata"
                  % (w["a"], w["b"], w["row0"], w["col0"], 100 * max(bad_a, bad_b)))
            continue
        deg, scale, shift, persp = warp_params(i)
        H = warp_matrix(deg, scale, shift, persp)
        ref = cv2.warpPerspective(ref_raw, H, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        daz = abs(int(w["a"]) - int(w["b"])) % 360
        daz = min(daz, 360 - daz)
        neg = w["negative"]
        pid = "lroc_%s_%s_r%d_c%d%s" % (w["a"], w["b"], w["row0"], w["col0"],
                                        "_neg" if neg else "")
        cv2.imwrite(os.path.join(pairs_dir, pid + "_source.png"), src)
        cv2.imwrite(os.path.join(pairs_dir, pid + "_reference.png"), ref)
        manifest["pairs"].append({
            "id": pid,
            "name": ("LROC NAC %s vs %s - different ground (negative)" % (w["a"], w["b"])
                     if neg else "LROC NAC sun %s vs %s (d-az %d deg)" % (w["a"], w["b"], daz)),
            "short": ("Two windows of different terrain; the correct answer is a refusal."
                      if neg else
                      "Same controlled ground under two sub-solar longitudes %d deg apart."
                      % daz),
            "difficulty": "negative" if neg else ("hard" if daz >= 120 else "moderate"),
            "sensor_source": "LRO NAC controlled mosaic, sub-solar lon bin %s" % w["a"],
            "sensor_reference": "LRO NAC controlled mosaic, sub-solar lon bin %s, warped" % w["b"],
            "gsd_source_m": GSD, "gsd_reference_m": GSD,
            "footprint_m": SIZE * GSD,
            "recommended_model": "homography",
            "delta_sun_lon_deg": daz,
            "negative": neg,
            "window": {"tile": TILE, "row0": w["row0"], "col0": w["col0"],
                       "row0_b": rb, "col0_b": cb},
            "lit_fraction_browse": [w["lit_a"], w["lit_b"]],
            "warp": {"deg": deg, "scale": scale, "shift_px": shift, "persp": persp},
            "real_component": "all pixels are delivered LROC NAC mosaic data; the "
                              "illumination change between source and reference is real",
            "synthetic_component": ("none - different ground, no transform exists" if neg else
                                    "known similarity + perspective warp applied to the "
                                    "reference; truth also carries the ~0.8 px control residual"),
            "source_file": pid + "_source.png",
            "reference_file": pid + "_reference.png",
            "source_shape": list(src.shape), "reference_shape": list(ref.shape),
            "ground_truth_available": not neg,
            "gt_homography": None if neg else [[float(v) for v in r] for r in H],
        })
        print("  wrote %s" % pid)

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("\n%d pairs -> %s" % (len(manifest["pairs"]), os.path.relpath(OUT, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
