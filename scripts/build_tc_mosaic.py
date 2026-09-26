#!/usr/bin/env python
"""Mosaic SELENE/Kaguya TC map tiles into one GDAL VRT reference.

A pushbroom strip can cross dozens of 3 x 3 degree TC tiles, and the
registration tool takes one reference raster. A VRT references the tiles in
place, so nothing is copied, and the tool reads it lazily in windows like any
other large GeoTIFF. Every tile must share one CRS and pixel size; gaps between
tiles are nodata (0, as in the tiles themselves).

    python scripts/build_tc_mosaic.py data/raw/selene/tc_morning_iirs20240119 \
        --out data/raw/selene/tc_morning_iirs20240119/TCO_MAPm04_mosaic.vrt

Keep "TCO_MAP" in the output name: the tool picks the selene_tc sensor profile
from it.

`--average N` writes a GeoTIFF of N x N block means instead (nodata ignored; a
block keeps a value only where at least half of it is valid). The fine stage
measures at the reference's own resolution, so a 55 m IIRS strip against 7.4 m
TC would be upsampled 7.4x and correlated over patches only ~8 source pixels
wide. Averaging TC to about half the source pixel gives both sides comparable
sampling ("common-resolution matching") while keeping TC's sharper, better
controlled geometry:

    python scripts/build_tc_mosaic.py data/raw/selene/tc_morning_iirs20240119 \
        --average 4 --out data/raw/selene/tc_morning_iirs20240119/TCO_MAPm04_mosaic_x4.tif
"""
import argparse
import html
import os
import sys
from pathlib import Path

import numpy as np
import rasterio


def build(labels, out):
    tiles = []
    for path in labels:
        with rasterio.open(path) as ds:
            if ds.count != 1 or ds.crs is None:
                raise SystemExit("%s: expected one georeferenced band" % path)
            need = ds.width * ds.height * np.dtype(ds.dtypes[0]).itemsize
            if Path(path).with_suffix(".img").stat().st_size < need:
                print("skipping %s: image file incomplete" % Path(path).name)
                continue
            tiles.append((path, ds.crs, ds.transform, ds.width, ds.height, ds.dtypes[0]))
    if not tiles:
        raise SystemExit("no complete tiles")
    crs, t0 = tiles[0][1], tiles[0][2]
    for path, c, t, *_ in tiles[1:]:
        if c != crs or abs(t.a - t0.a) > 1e-9 * abs(t0.a) or abs(t.e - t0.e) > 1e-9 * abs(t0.e):
            raise SystemExit("%s: CRS or pixel size differs from %s" % (path, tiles[0][0]))
    px, py = t0.a, t0.e
    x0 = min(t.c for _, _, t, *_ in tiles)
    y0 = max(t.f for _, _, t, *_ in tiles)
    cols, rows, sources = 0, 0, []
    for path, _, t, w, h, dtype in tiles:
        xoff, yoff = (t.c - x0) / px, (t.f - y0) / py
        if abs(xoff - round(xoff)) > 1e-3 or abs(yoff - round(yoff)) > 1e-3:
            raise SystemExit("%s: not aligned to the mosaic grid" % path)
        xoff, yoff = int(round(xoff)), int(round(yoff))
        cols, rows = max(cols, xoff + w), max(rows, yoff + h)
        sources.append((path, xoff, yoff, w, h, dtype))
    gdal_type = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16",
                 "float32": "Float32"}[tiles[0][5]]
    lines = ['<VRTDataset rasterXSize="%d" rasterYSize="%d">' % (cols, rows),
             "  <SRS>%s</SRS>" % html.escape(crs.to_wkt()),
             "  <GeoTransform>%.17g, %.17g, 0, %.17g, 0, %.17g</GeoTransform>" % (x0, px, y0, py),
             '  <VRTRasterBand dataType="%s" band="1">' % gdal_type,
             "    <NoDataValue>0</NoDataValue>"]
    for path, xoff, yoff, w, h, _ in sources:
        rel = os.path.relpath(os.path.abspath(path), os.path.dirname(os.path.abspath(out)))
        lines += ["    <ComplexSource>",
                  '      <SourceFilename relativeToVRT="1">%s</SourceFilename>' % html.escape(str(rel)),
                  "      <SourceBand>1</SourceBand>",
                  '      <SrcRect xOff="0" yOff="0" xSize="%d" ySize="%d"/>' % (w, h),
                  '      <DstRect xOff="%d" yOff="%d" xSize="%d" ySize="%d"/>' % (xoff, yoff, w, h),
                  "      <NODATA>0</NODATA>",
                  "    </ComplexSource>"]
    lines += ["  </VRTRasterBand>", "</VRTDataset>", ""]
    Path(out).write_text("\n".join(lines))
    return cols, rows, len(sources)


def build_averaged(labels, out, factor):
    """Block-mean mosaic on the same grid lines, every `factor` native pixels."""
    from rasterio.transform import Affine
    from rasterio.windows import Window

    tiles = []
    for path in labels:
        with rasterio.open(path) as ds:
            need = ds.width * ds.height * np.dtype(ds.dtypes[0]).itemsize
            if Path(path).with_suffix(".img").stat().st_size < need:
                print("skipping %s: image file incomplete" % Path(path).name)
                continue
            if ds.width % factor or ds.height % factor:
                raise SystemExit("%s: %d x %d is not divisible by %d" % (path, ds.width, ds.height, factor))
            tiles.append((path, ds.crs, ds.transform, ds.width, ds.height))
    if not tiles:
        raise SystemExit("no complete tiles")
    crs, t0 = tiles[0][1], tiles[0][2]
    x0 = min(t.c for _, _, t, _, _ in tiles)
    y0 = max(t.f for _, _, t, _, _ in tiles)
    place = [(p, int(round((t.c - x0) / t0.a)), int(round((t.f - y0) / t0.e)), w, h)
             for p, c, t, w, h in tiles]
    if any(c != crs for _, c, *_ in tiles):
        raise SystemExit("tiles do not share one CRS")
    cols = max(x + w for _, x, _, w, _ in place) // factor
    rows = max(y + h for _, _, y, _, h in place) // factor
    profile = dict(driver="GTiff", width=cols, height=rows, count=1, dtype="uint16", nodata=0,
                   crs=crs, transform=Affine(t0.a * factor, 0, x0, 0, t0.e * factor, y0),
                   tiled=True, blockxsize=256, blockysize=256, compress="deflate",
                   predictor=2, BIGTIFF="IF_SAFER")
    band = 1024 - 1024 % factor                                    # rows per read
    with rasterio.open(out, "w", **profile) as dst:
        for path, xoff, yoff, w, h in place:
            with rasterio.open(path) as src:
                for r in range(0, h, band):
                    n = min(band, h - r)
                    a = src.read(1, window=Window(0, r, w, n)).astype(np.float64)
                    valid = a > 0
                    shape = (n // factor, factor, w // factor, factor)
                    total = np.where(valid, a, 0).reshape(shape).sum(axis=(1, 3))
                    count = valid.reshape(shape).sum(axis=(1, 3))
                    mean = np.where(count >= factor * factor / 2, total / np.maximum(count, 1), 0)
                    dst.write(np.clip(np.round(mean), 0, 65535).astype(np.uint16), 1,
                              window=Window(xoff // factor, (yoff + r) // factor, w // factor, n // factor))
    return cols, rows, len(place)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="directory holding TCO_MAP*.lbl + .img tiles")
    ap.add_argument("--out", type=Path, required=True,
                    help="output .vrt, or .tif with --average (keep TCO_MAP in the name)")
    ap.add_argument("--average", type=int, default=0, metavar="N",
                    help="write an N x N block-mean GeoTIFF instead of a VRT")
    args = ap.parse_args()
    labels = sorted(args.directory.glob("TCO_MAP*SC.lbl"))
    labels = [p for p in labels if p.with_suffix(".img").exists()]
    if not labels:
        sys.exit("no complete TCO_MAP*SC.lbl/.img tile pairs in %s" % args.directory)
    if args.average > 1:
        cols, rows, n = build_averaged(labels, args.out, args.average)
    else:
        cols, rows, n = build(labels, args.out)
    print("%s: %d tiles, %d x %d px" % (args.out, n, cols, rows))


if __name__ == "__main__":
    main()
