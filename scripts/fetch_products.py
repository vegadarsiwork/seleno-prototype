"""Download the Chandrayaan-2 OHRC products this prototype uses.

Source: the Internet Archive mirror of ISRO's OHRC release, which is public and
needs no credentials. See data/README.md for provenance and licensing.

    python scripts/fetch_products.py --out data/raw

Small members (browse PNG, geometry CSV, PDS4 label) are pulled individually
from inside the remote ZIP. The full-resolution image is a 1.12 GB deflated
member, so random access into it is impossible; instead the compressed stream is
inflated only as far as the requested row bands and then abandoned.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import urllib.parse
import urllib.request
import zlib

import numpy as np

try:
    from remotezip import RemoteZip
except ImportError:
    sys.exit("remotezip is required:  pip install remotezip")

BASE = "https://archive.org/download/chandrayaan-2-high-resolution-images-of-the-moon/"
FOLDER = "Optical High Resolution Camera (OHRC)/"

ZIPS = {
    "A_cal": "ch2_ohr_ncp_20200229T0739312111_d_img_d18.zip",
    "A_raw": "ch2_ohr_nrp_20200229T0739312111_d_img_d18.zip",
    "B_cal": "ch2_ohr_ncp_20200229T0938004033_d_img_d32.zip",
}

SAMPLES = 12000
# Full-resolution row bands cut from the orbit-2297 calibrated strip.
BANDS = [(78000, 82096), (48000, 52096), (2000, 6096)]


def url_for(zip_name: str) -> str:
    return BASE + urllib.parse.quote(FOLDER + zip_name)


def fetch_small(zip_name: str, out_dir: str, suffixes=(".png", ".csv")) -> None:
    """Pull browse/geometry/label members - each only a few MB compressed."""
    url = url_for(zip_name)
    with RemoteZip(url) as z:
        for n in z.namelist():
            base = os.path.basename(n)
            keep = n.endswith(suffixes) or (n.endswith(".xml") and "/data/" in "/" + n)
            if not keep or not base:
                continue
            dst = os.path.join(out_dir, base)
            if os.path.exists(dst):
                print("  have  %s" % base)
                continue
            t = time.time()
            with open(dst, "wb") as fh:
                fh.write(z.read(n))
            print("  got   %s  (%.1f MB, %.1fs)"
                  % (base, os.path.getsize(dst) / 1e6, time.time() - t))


def fetch_bands(zip_name: str, out_dir: str, bands=BANDS) -> None:
    """Stream-inflate the .img member, keeping only the requested row bands."""
    targets = {b: os.path.join(out_dir, "ohrcA_rows%d_%d.npy" % b) for b in bands}
    todo = [b for b, p in targets.items() if not os.path.exists(p)]
    if not todo:
        print("  have  all full-resolution bands")
        return

    url = url_for(zip_name)
    with RemoteZip(url) as z:
        info = [i for i in z.infolist() if i.filename.endswith(".img")][0]

    # The local file header length is variable, so read it to find the data start.
    req = urllib.request.Request(
        url, headers={"Range": "bytes=%d-%d" % (info.header_offset, info.header_offset + 29)})
    lh = urllib.request.urlopen(req).read(30)
    name_len, extra_len = struct.unpack("<HH", lh[26:30])
    data_off = info.header_offset + 30 + name_len + extra_len

    need = max(b[1] for b in todo) * SAMPLES
    print("  streaming %.0f MB of decompressed image data" % (need / 1e6))

    parts = {b: [] for b in todo}
    dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
    produced, t0, last = 0, time.time(), 0
    req = urllib.request.Request(
        url, headers={"Range": "bytes=%d-%d" % (data_off, data_off + info.compress_size - 1)})
    with urllib.request.urlopen(req) as r:
        while produced < need:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out = dec.decompress(chunk) if dec else chunk
            if not out:
                continue
            start, produced = produced, produced + len(out)
            for b in todo:
                s, e = b[0] * SAMPLES, b[1] * SAMPLES
                lo, hi = max(start, s), min(produced, e)
                if hi > lo:
                    parts[b].append((lo, out[lo - start:hi - start]))
            if produced - last > 100e6:
                last = produced
                print("    %4.0f / %4.0f MB   %3.0fs"
                      % (produced / 1e6, need / 1e6, time.time() - t0))

    for b in todo:
        s, e = b
        raw = b"".join(x[1] for x in sorted(parts[b]))
        expect = (e - s) * SAMPLES
        if len(raw) != expect:
            print("  FAILED %s: got %d of %d bytes" % (targets[b], len(raw), expect))
            continue
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(e - s, SAMPLES)
        np.save(targets[b], arr)
        print("  got   %s  %s  mean %.1f"
              % (os.path.basename(targets[b]), arr.shape, arr.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--skip-fullres", action="store_true",
                    help="browse products only (fast); the full-resolution pairs will not build")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    for key, zn in ZIPS.items():
        print("%s  %s" % (key, zn))
        fetch_small(zn, out)

    if not args.skip_fullres:
        print("full-resolution bands from orbit 2297 (calibrated)")
        fetch_bands(ZIPS["A_cal"], out)

    print("\ndone -> %s" % out)
    print("next:  python scripts/prepare_data.py --raw %s" % args.out)


if __name__ == "__main__":
    main()
