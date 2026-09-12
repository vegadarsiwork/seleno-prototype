"""Correctness and safety tests for the OHRC dataset layer.

Runs under pytest if it is installed, and standalone otherwise:

    python tests/test_ohrc_dataset.py

Every test is read-only against the archive. The suite skips cleanly when the
dataset is not present, so it is safe to run in a checkout without the 4.6 GB of
imagery.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from seleno import ohrc                                            # noqa: E402
from seleno.ohrc.geometry import (lonlat_to_south_stereo,          # noqa: E402
                                  south_stereo_to_lonlat)

PRODUCTS = ohrc.discover()
HAVE_DATA = len(PRODUCTS) > 0
PAIR_A, PAIR_B = "20241115T1326", "20241115T1525"


class SkipTest(Exception):
    pass


def _require_data():
    if not HAVE_DATA:
        raise SkipTest("OHRC dataset not found (set SELENO_OHRC_ROOT)")


# --------------------------------------------------------------------------- #
# projection maths - no dataset needed
# --------------------------------------------------------------------------- #

def test_stereographic_round_trip():
    """lon/lat -> projected -> lon/lat is identity to sub-millimetre."""
    lons = np.array([0.0, 22.45, 110.27, 222.26, 359.9])
    lats = np.array([-89.2, -89.5, -89.9468, -89.99, -88.0])
    x, y = lonlat_to_south_stereo(lons, lats)
    lon2, lat2 = south_stereo_to_lonlat(x, y)
    assert np.allclose(lats, lat2, atol=1e-9), (lats, lat2)
    assert np.allclose(lons % 360, lon2 % 360, atol=1e-7), (lons, lon2)


def test_south_pole_maps_to_origin():
    x, y = lonlat_to_south_stereo(137.0, -90.0)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6


def test_longitude_is_degenerate_near_pole():
    """The reason the projected plane exists at all.

    Two points 100 deg apart in longitude at 89.95 S are only a few hundred
    metres apart on the ground. Any overlap logic in lon/lat would be wrong.
    """
    x1, y1 = lonlat_to_south_stereo(22.45, -89.9468)
    x2, y2 = lonlat_to_south_stereo(122.45, -89.9468)
    sep = float(np.hypot(x2 - x1, y2 - y1))
    assert sep < 2500.0, "expected sub-2.5 km separation, got %.1f m" % sep


def test_shadow_length_scaling():
    """Grazing illumination: 1 m of relief casts a very long shadow."""
    assert ohrc.shadow_length_per_metre(0.786) > 60.0
    assert abs(ohrc.shadow_length_per_metre(45.0) - 1.0) < 1e-9
    assert ohrc.shadow_length_per_metre(0.0) == float("inf")
    assert ohrc.shadow_length_per_metre(-0.2) == float("inf")


# --------------------------------------------------------------------------- #
# labels and raster geometry
# --------------------------------------------------------------------------- #

def test_discovery_finds_products():
    _require_data()
    assert len(PRODUCTS) >= 1
    ids = {p.timestamp for p in PRODUCTS}
    assert len(ids) == len(PRODUCTS), "duplicate timestamps in discovery"


def test_declared_size_equals_lines_times_samples():
    """The headless-raster contract: bytes on disk == Line x Sample x 1."""
    _require_data()
    for p in PRODUCTS:
        actual = os.path.getsize(p.img_path)
        assert p.label.data_type == "UnsignedByte", p.product_id
        assert p.label.offset == 0, p.product_id
        assert p.lines > 0 and p.samples > 0, p.product_id
        assert actual == p.lines * p.samples, (
            "%s: %d bytes on disk != %d x %d = %d"
            % (p.product_id, actual, p.lines, p.samples, p.lines * p.samples))
        assert p.label.declared_file_size == actual, p.product_id
        ok, why = p.verify()
        assert ok, why


def test_dimensions_are_per_product_not_hardcoded():
    """The products genuinely differ in line count, so nothing may assume one."""
    _require_data()
    if len(PRODUCTS) < 3:
        raise SkipTest("needs at least 3 products")
    assert len({p.lines for p in PRODUCTS}) > 1, "expected differing line counts"
    assert {p.samples for p in PRODUCTS} == {12000}


def test_label_flags_unrefined_geolocation():
    """Refined corners equal system corners in this archive; we must say so."""
    _require_data()
    for p in PRODUCTS:
        assert p.label.corners_refined is False, (
            "%s unexpectedly has genuinely refined corners - update the docs"
            % p.product_id)
        assert any("no photogrammetric refinement" in n for n in p.label.notes)


def test_label_flags_line_exposure_unit_defect():
    _require_data()
    hit = [p for p in PRODUCTS
           if any("unit attribute is wrong" in n for n in p.label.notes)]
    assert hit, "expected the line_exposure_duration unit defect to be flagged"


# --------------------------------------------------------------------------- #
# tile access
# --------------------------------------------------------------------------- #

def test_tile_read_matches_raw_file_bytes():
    """Tile contents verified against independent seek/read on the file."""
    _require_data()
    p = PRODUCTS[0]
    x0, y0, w, h = 4321, 12345, 64, 8
    tile = p.read_tile(x0, y0, w, h)
    assert tile.shape == (h, w)
    with open(p.img_path, "rb") as fh:
        for r in range(h):
            fh.seek((y0 + r) * p.samples + x0)
            row = np.frombuffer(fh.read(w), dtype=np.uint8)
            assert np.array_equal(row, tile[r]), "row %d differs" % r


def test_tile_is_a_copy_not_a_view_on_the_map():
    _require_data()
    p = PRODUCTS[0]
    t = p.read_tile(0, 0, 32, 32)
    assert isinstance(t, np.ndarray) and not isinstance(t, np.memmap)
    assert t.base is None or not isinstance(t.base, np.memmap)
    assert t.flags.writeable, "caller-owned copy should be writeable"


def test_tile_clipped_at_raster_edges():
    _require_data()
    p = PRODUCTS[0]
    t = p.read_tile(p.samples - 10, p.lines - 5, 512, 512)
    assert t.shape == (5, 10)
    assert p.read_tile(-50, -50, 16, 16).shape == (16, 16)


def test_tile_rejects_nonpositive_size():
    _require_data()
    p = PRODUCTS[0]
    for w, h in ((0, 8), (8, 0), (-4, 8)):
        try:
            p.read_tile(0, 0, w, h)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for size (%d, %d)" % (w, h))


def test_striding_decimates_correctly():
    _require_data()
    p = PRODUCTS[0]
    full = p.read_tile(1000, 2000, 64, 64, step=1)
    dec = p.read_tile(1000, 2000, 64, 64, step=4)
    assert dec.shape == (16, 16)
    assert np.array_equal(dec, full[::4, ::4])


def test_no_whole_image_is_loaded():
    """Resident memory must stay far below the raster size.

    The memmap lets the OS cache pages, so RSS does grow with access; the
    contract is that it never approaches the ~1.2 GB file size for the kind of
    windowed access the pipeline performs.
    """
    _require_data()
    try:
        import psutil
    except ImportError:
        raise SkipTest("psutil not installed")
    proc = psutil.Process()
    p = [q for q in PRODUCTS if q.lines > 90000][0]
    p.close()
    before = proc.memory_info().rss
    rng = np.random.default_rng(0)
    for _ in range(40):
        y = int(rng.integers(0, p.lines - 512))
        x = int(rng.integers(0, p.samples - 512))
        p.read_tile(x, y, 512, 512).mean()
    p.thumbnail(800)
    grown = proc.memory_info().rss - before
    file_bytes = os.path.getsize(p.img_path)
    assert grown < 0.25 * file_bytes, (
        "RSS grew %.0f MB against a %.0f MB raster - looks like a bulk load"
        % (grown / 1e6, file_bytes / 1e6))


# --------------------------------------------------------------------------- #
# immutability
# --------------------------------------------------------------------------- #

def test_archive_is_opened_read_only():
    _require_data()
    p = PRODUCTS[0]
    assert p._map.mode == "r"
    assert p._map.flags.writeable is False
    try:
        p._map[0, 0] = 123
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("memmap accepted a write - archive is not protected")


def test_dataset_files_unmodified_by_a_full_access_pass():
    """mtime, size and a content hash of a sampled region are all stable."""
    _require_data()
    p = PRODUCTS[0]

    def fingerprint():
        st = os.stat(p.img_path)
        with open(p.img_path, "rb") as fh:
            fh.seek(st.st_size // 2)
            mid = fh.read(65536)
        return st.st_mtime_ns, st.st_size, hashlib.md5(mid).hexdigest()

    before = fingerprint()
    p.read_tile(100, 100, 256, 256)
    p.thumbnail(400)
    _ = p.describe()
    if p.has_geometry:
        _ = p.geometry.lonlat(500, 500)
    if p.has_sun_series:
        _ = p.sun_at_line(10)
    assert fingerprint() == before, "dataset file changed during a read-only pass"


# --------------------------------------------------------------------------- #
# geometry indexing
# --------------------------------------------------------------------------- #

def test_geometry_lattice_shape_and_extent():
    _require_data()
    for p in PRODUCTS:
        if not p.has_geometry:
            continue
        g = p.geometry
        assert g.pixels[0] == 0 and g.pixels[-1] == p.samples - 1
        assert g.scans[0] == 0
        # The lattice's last scan node must be the last image line, which is
        # what proves the grid indexes the raster 1:1 with no crop or offset.
        assert g.scans[-1] == p.lines - 1, (
            "%s: last scan node %d != last line %d"
            % (p.product_id, int(g.scans[-1]), p.lines - 1))
        assert g.lon.shape == (len(g.scans), len(g.pixels))


def test_geometry_corners_reproduce_the_label():
    """Grid node (0,0) must equal the label's upper-left corner, and so on."""
    _require_data()
    for p in PRODUCTS:
        if not p.has_geometry:
            continue
        g, c = p.geometry, p.label.corners_system
        checks = [
            ((0, 0), "upper_left_longitude", "upper_left_latitude"),
            ((p.samples - 1, 0), "upper_right_longitude", "upper_right_latitude"),
            ((0, p.lines - 1), "lower_left_longitude", "lower_left_latitude"),
            ((p.samples - 1, p.lines - 1), "lower_right_longitude", "lower_right_latitude"),
        ]
        for (s, l), klon, klat in checks:
            lon, lat = g.lonlat(s, l)
            assert abs(lon - c[klon]) < 1e-5, (p.product_id, klon, lon, c[klon])
            assert abs(lat - c[klat]) < 1e-5, (p.product_id, klat, lat, c[klat])


def test_geometry_node_interpolation_is_exact_at_nodes():
    _require_data()
    p = PRODUCTS[0]
    g = p.geometry
    for jj in (0, len(g.scans) // 2, len(g.scans) - 1):
        for ii in (0, len(g.pixels) // 2, len(g.pixels) - 1):
            lon, lat = g.lonlat(g.pixels[ii], g.scans[jj])
            assert abs(lon - g.lon[jj, ii]) < 1e-9
            assert abs(lat - g.lat[jj, ii]) < 1e-9


def test_geometry_inverse_round_trip():
    """projected -> (sample, line) -> projected closes to well under a lattice cell."""
    _require_data()
    p = PRODUCTS[0]
    g = p.geometry
    for s, l in ((3000, 20000), (6000, p.lines // 2), (11000, p.lines - 3000)):
        x, y = g.stereo(s, l)
        s2, l2, ok = g.stereo_to_pixel(x, y)
        assert ok
        x2, y2 = g.stereo(s2, l2)
        err = float(np.hypot(x2 - x, y2 - y))
        assert err < 5.0, "closure error %.2f m at (%d, %d)" % (err, s, l)
        assert abs(s2 - s) < 60 and abs(l2 - l) < 220, (s, s2, l, l2)


def test_window_footprint_has_four_corners_and_a_centre():
    _require_data()
    p = PRODUCTS[0]
    fp = p.window_footprint(5000, 40000, 1024, 1024)
    assert len(fp["corners_lon_lat"]) == 4
    assert len(fp["center_stereo_m"]) == 2
    xs = [c[0] for c in fp["corners_stereo_m"]]
    ys = [c[1] for c in fp["corners_stereo_m"]]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    expected = 1024 * (p.gsd_m or 0.24)
    assert 0.3 * expected < span < 3.0 * expected, (
        "window ground span %.1f m implausible for 1024 px at %.2f m/px"
        % (span, p.gsd_m or 0.24))


def test_2024_pair_overlaps_heavily():
    """The primary illumination-stress pair must actually share ground."""
    _require_data()
    try:
        a = ohrc.by_timestamp(PRODUCTS, PAIR_A)
        b = ohrc.by_timestamp(PRODUCTS, PAIR_B)
    except KeyError:
        raise SkipTest("the 2024-11-15 pair is not present")
    ov = ohrc.footprint_overlap(a.geometry, b.geometry, cell_m=100.0)
    assert ov["fraction_of_smaller"] > 0.85, ov


# --------------------------------------------------------------------------- #
# sun geometry
# --------------------------------------------------------------------------- #

def test_sun_series_cadence_is_derived_not_assumed():
    """readme.txt claims 512 ms; the records are 40 ms apart."""
    _require_data()
    for p in PRODUCTS:
        if not p.has_sun_series:
            continue
        c = p.sun.cadence_s
        assert 0.01 < c < 0.2, "%s: implausible cadence %.4f s" % (p.product_id, c)
        assert abs(c - 0.512) > 0.1, "cadence looks like the readme's wrong 512 ms"


def test_sun_phase_column_equals_ninety_minus_elevation():
    _require_data()
    for p in PRODUCTS:
        if not p.has_sun_series:
            continue
        s = p.sun
        assert np.allclose(s.phase_deg, 90.0 - s.elevation_deg, atol=0.05), p.product_id


def test_label_sun_angles_match_the_mid_strip_series_sample():
    """Confirms the label scalar is a scene-centre sample of the .spm series."""
    _require_data()
    for p in PRODUCTS:
        if not (p.has_sun_series and p.label.sun_elevation_deg is not None):
            continue
        mid = p.sun_at_line(p.lines // 2)
        assert abs(mid["elevation_deg"] - p.label.sun_elevation_deg) < 0.01, (
            "%s: mid-strip elevation %.4f vs label %.4f"
            % (p.product_id, mid["elevation_deg"], p.label.sun_elevation_deg))


def test_sun_azimuth_sweeps_along_the_strip():
    """Justifies interpolating per line instead of using the label scalar."""
    _require_data()
    sweeps = [p.sun.summary()["azimuth_sweep_deg"] for p in PRODUCTS if p.has_sun_series]
    assert sweeps and max(sweeps) > 20.0, (
        "expected tens of degrees of azimuth sweep near the pole, got %r" % sweeps)


def test_sun_at_line_is_monotonic_in_time():
    _require_data()
    p = [q for q in PRODUCTS if q.has_sun_series][0]
    ts = [p.sun_at_line(l)["t_since_start_s"] for l in (0, p.lines // 2, p.lines - 1)]
    assert ts[0] < ts[1] < ts[2]
    assert abs(ts[-1] - (p.label.dwell_seconds or 0)) < 1e-6


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    npass = nfail = nskip = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            print("SKIP  %-52s %s" % (name, exc))
            nskip += 1
        except Exception as exc:                                  # noqa: BLE001
            print("FAIL  %-52s %s: %s" % (name, type(exc).__name__, exc))
            failures.append(name)
            nfail += 1
        else:
            print("pass  %s" % name)
            npass += 1
    print("\n%d passed, %d failed, %d skipped" % (npass, nfail, nskip))
    if failures:
        print("failed: %s" % ", ".join(failures))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(_main())
