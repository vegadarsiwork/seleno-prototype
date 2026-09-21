"""Adversarial tests for the registration tool. Phase 8 of the brief.

Every case here is a way the tool could produce a confident wrong answer or
crash on input it will genuinely meet. They run on small synthetic rasters so
the whole file finishes in seconds and uses a few megabytes - the tool's own
memory ceiling is exercised separately by `test_memory_ceiling_shrinks_grid`.

    python tests/test_tool.py

The contract being tested is the one in the brief: on failure `metrics.json`
exists and carries `status: "failed"` plus one of

    insufficient_matches | verification_failed | no_overlap
    degenerate_transform | unreadable_input
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cv2                                                        # noqa: E402
from seleno.tool import FAILURE_CODES, register                    # noqa: E402
from seleno.tool.register import _cap_max_side                     # noqa: E402
from seleno.tool.scene import UnreadableInput, load                 # noqa: E402

TMP = None
PASS, FAIL, SKIP = [], [], []


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def terrain(seed=0, n=256):
    """A crater-ish field with enough structure for any matcher to bite on."""
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 6.0)
    a += 0.6 * cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 1.5)
    for _ in range(18):
        cy, cx = rng.integers(20, n - 20, 2)
        r = int(rng.integers(5, 16))
        cv2.circle(a, (int(cx), int(cy)), r, float(rng.uniform(0.2, 0.9)), -1)
        cv2.circle(a, (int(cx), int(cy)), r, float(rng.uniform(0.0, 0.3)), 2)
    a -= a.min()
    return (a / max(a.max(), 1e-6))


def terrain_hw(seed, h, w):
    """`terrain`, at any shape - pushbroom strips are rarely square."""
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 6.0)
    a += 0.6 * cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    for _ in range(max(18, h * w // 4000)):
        cy = int(rng.integers(10, max(11, h - 10)))
        cx = int(rng.integers(10, max(11, w - 10)))
        r = int(rng.integers(5, 16))
        cv2.circle(a, (cx, cy), r, float(rng.uniform(0.2, 0.9)), -1)
        cv2.circle(a, (cx, cy), r, float(rng.uniform(0.0, 0.3)), 2)
    a -= a.min()
    return (a / max(a.max(), 1e-6))


def write_png(path, arr01, bits=8):
    if bits == 8:
        cv2.imwrite(path, np.clip(arr01 * 255, 0, 255).astype(np.uint8))
    else:
        cv2.imwrite(path, np.clip(arr01 * 65535, 0, 65535).astype(np.uint16))
    return path


def geo_pair(master, res_src, res_ref, tag, shift=(0, 0)):
    """Two GeoTIFFs of the SAME ground at different resolutions.

    Writing two rasters with the same pixel count but different pixel sizes
    gives them different extents, which is a no-overlap test, not a scale test.
    Both of these cover the master's full extent; only the sampling differs.
    """
    import rasterio
    from rasterio.transform import Affine
    n = master.shape[0]
    out = []
    for i, res_m in enumerate((res_src, res_ref)):
        k = max(2, int(round(res_m)))
        arr = master[::k, ::k]
        if i == 0 and shift != (0, 0):
            arr = np.roll(arr, shift, axis=(0, 1))
        path = os.path.join(TMP, "%s_%d.tif" % (tag, i))
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                           width=arr.shape[1], count=1, dtype="uint8",
                           crs="+proj=stere +lat_0=-90 +R=1737400 +units=m",
                           transform=Affine(res_m, 0, 0, 0, -res_m, 0)) as ds:
            ds.write(np.clip(arr * 255, 0, 255).astype(np.uint8), 1)
        out.append(path)
    return out


def run(src, ref, **kw):
    out = os.path.join(TMP, "out")
    kw.setdefault("verbose", False)
    kw.setdefault("max_side", 512)
    return register(src, ref, out, **kw)


def metrics_of(res):
    p = os.path.join(res.out_dir, "metrics.json")
    assert os.path.exists(p), "metrics.json must exist even on failure"
    return json.load(open(p))


def check(name, fn):
    try:
        note = fn()
        PASS.append(name)
        print("pass  %-52s %s" % (name, note or ""))
    except AssertionError as exc:
        FAIL.append((name, str(exc)))
        print("FAIL  %-52s %s" % (name, exc))
    except Exception:
        FAIL.append((name, traceback.format_exc(limit=2)))
        print("ERROR %-52s" % name)
        traceback.print_exc(limit=3)


# --------------------------------------------------------------------------- #
# the cases
# --------------------------------------------------------------------------- #

def t_identical_pair_is_near_identity():
    a = terrain(1)
    s = write_png(os.path.join(TMP, "id_a.png"), a)
    r = write_png(os.path.join(TMP, "id_b.png"), a)
    res = run(s, r)
    assert res.status != "failed", "identical images must register, got %s" % res.reason
    m = metrics_of(res)
    rmse = m["accuracy"]["rmse_px"]
    assert rmse is not None and rmse < 1.0, "identity RMSE should be sub-pixel, got %s" % rmse
    d = m["decomposition"]
    assert abs(d["est_scale_x"] - 1.0) < 0.02, "scale %s" % d["est_scale_x"]
    assert abs(d["est_rotation_deg"]) < 1.0, "rotation %s" % d["est_rotation_deg"]
    return "RMSE %.4f px, scale %.4f, rot %.3f deg" % (rmse, d["est_scale_x"],
                                                       d["est_rotation_deg"])


def t_zero_overlap_reports_no_overlap():
    s = write_png(os.path.join(TMP, "zo_a.png"), terrain(2))
    r = write_png(os.path.join(TMP, "zo_b.png"), terrain(999))
    res = run(s, r)
    m = metrics_of(res)
    assert res.status == "failed", "unrelated images must not register, got %s" % res.status
    assert m["reason"] in FAILURE_CODES, "undeclared code %r" % m["reason"]
    # two unrelated rasters of the same size DO overlap geometrically; the honest
    # codes are the matching ones, not no_overlap
    assert m["reason"] in ("insufficient_matches", "verification_failed",
                           "degenerate_transform"), m["reason"]
    return m["reason"]


def t_disjoint_footprints_report_no_overlap():
    """Geometrically disjoint georeferenced inputs -> no_overlap specifically."""
    import rasterio
    from rasterio.transform import Affine
    paths = []
    for i, x0 in enumerate((0.0, 5000.0)):          # 1 km apart, 200 m wide
        p = os.path.join(TMP, "dj_%d.tif" % i)
        with rasterio.open(p, "w", driver="GTiff", height=256, width=256, count=1,
                           dtype="uint8", crs="+proj=stere +lat_0=-90 +R=1737400 +units=m",
                           transform=Affine(1.0, 0, x0, 0, -1.0, 0.0)) as ds:
            ds.write(np.clip(terrain(3 + i) * 255, 0, 255).astype(np.uint8), 1)
        paths.append(p)
    res = run(*paths)
    m = metrics_of(res)
    assert res.status == "failed", res.status
    assert m["reason"] == "no_overlap", "expected no_overlap, got %r" % m["reason"]
    return m["reason"]


def t_truncated_file_is_unreadable_input():
    a = terrain(4)
    good = write_png(os.path.join(TMP, "tr_ref.png"), a)
    src = os.path.join(TMP, "tr_src.img")
    lbl = os.path.join(TMP, "tr_src.lbl")
    # a PDS3 label that declares far more data than the file holds
    open(lbl, "w").write("PDS_VERSION_ID = PDS3\r\nOBJECT = IMAGE\r\n  LINES = 4000\r\n"
                         "  LINE_SAMPLES = 4000\r\n  SAMPLE_BITS = 16\r\n"
                         "  SAMPLE_TYPE = LSB_UNSIGNED_INTEGER\r\nEND_OBJECT = IMAGE\r\nEND\r\n")
    open(src, "wb").write(os.urandom(4096))
    res = run(src, good)
    m = metrics_of(res)
    assert res.status == "failed", res.status
    assert m["reason"] == "unreadable_input", "expected unreadable_input, got %r" % m["reason"]
    return m["message"][:60]


def t_empty_file_is_unreadable_input():
    good = write_png(os.path.join(TMP, "ef_ref.png"), terrain(5))
    empty = os.path.join(TMP, "empty.png")
    open(empty, "wb").close()
    res = run(empty, good)
    m = metrics_of(res)
    assert m["reason"] == "unreadable_input", m["reason"]
    return "empty file handled"


def t_missing_file_is_unreadable_input():
    good = write_png(os.path.join(TMP, "mf_ref.png"), terrain(6))
    res = run(os.path.join(TMP, "does_not_exist.png"), good)
    assert metrics_of(res)["reason"] == "unreadable_input"
    return "missing file handled"


def t_8bit_and_16bit_agree():
    """The same scene at two bit depths must give the same answer."""
    a = terrain(7)
    ref = write_png(os.path.join(TMP, "bd_ref.png"), a)
    s8 = write_png(os.path.join(TMP, "bd_8.png"), a, bits=8)
    s16 = write_png(os.path.join(TMP, "bd_16.png"), a, bits=16)
    r8, r16 = run(s8, ref), run(s16, ref)
    assert r8.status != "failed" and r16.status != "failed", "%s / %s" % (r8.status, r16.status)
    m8, m16 = metrics_of(r8), metrics_of(r16)
    e8, e16 = m8["accuracy"]["rmse_px"], m16["accuracy"]["rmse_px"]
    assert m16["source"]["bit_depth"] == 16, "16-bit input read as %s" % m16["source"]["bit_depth"]
    assert abs(e8 - e16) < 0.5, "8-bit %.4f vs 16-bit %.4f px disagree" % (e8, e16)
    return "8-bit %.4f px, 16-bit %.4f px" % (e8, e16)


def t_rotated_source_registers_or_fails_cleanly():
    a = terrain(8)
    ref = write_png(os.path.join(TMP, "rot_ref.png"), a)
    src = write_png(os.path.join(TMP, "rot_src.png"), np.rot90(a).copy())
    res = run(src, ref)
    m = metrics_of(res)
    if res.status == "failed":
        assert m["reason"] in FAILURE_CODES, m["reason"]
        return "failed cleanly: %s" % m["reason"]
    d = m["decomposition"]
    rot = abs(d["est_rotation_deg"])
    assert rot > 45 or m["status"] == "warning", \
        "a 90 deg rotation was registered as %0.2f deg with status %s" % (rot, m["status"])
    return "registered, rotation %.1f deg" % rot


def t_flipped_source_registers_or_fails_cleanly():
    a = terrain(9)
    ref = write_png(os.path.join(TMP, "flip_ref.png"), a)
    src = write_png(os.path.join(TMP, "flip_src.png"), a[::-1].copy())
    res = run(src, ref)
    m = metrics_of(res)
    assert res.status == "failed" or m["status"] in ("pass", "warning"), m
    if res.status == "failed":
        assert m["reason"] in FAILURE_CODES, m["reason"]
        return "failed cleanly: %s" % m["reason"]
    # a mirror is not in any of the model families, so it must not look confident
    assert m["matches"]["inlier_ratio"] < 0.9, \
        "a flipped image produced %.0f%% inliers, which is implausible" \
        % (100 * m["matches"]["inlier_ratio"])
    return "inlier ratio %.2f" % m["matches"]["inlier_ratio"]


def t_mostly_shadow_crop_fails_or_is_honest():
    """Polar reality: two thirds at the noise floor, a lit sliver at one edge."""
    a = terrain(10) * 0.02                      # noise floor
    a[:, 200:] = terrain(11)[:, 200:]           # a lit strip
    ref = write_png(os.path.join(TMP, "sh_ref.png"), a)
    b = a.copy()
    b[:, 200:] = np.roll(b[:, 200:], 3, axis=0)
    src = write_png(os.path.join(TMP, "sh_src.png"), b)
    res = run(src, ref)
    m = metrics_of(res)
    if res.status == "failed":
        assert m["reason"] in FAILURE_CODES, m["reason"]
        return "failed cleanly: %s" % m["reason"]
    cov = m["distribution"]["coverage_fraction"]
    assert m["status"] == "warning" or cov > 0.5, \
        "matches confined to %.0f%% of the frame but status is %r" % (100 * cov, m["status"])
    return "status %s, coverage %.2f" % (m["status"], cov)


def t_no_metadata_input_still_runs():
    a = terrain(12)
    s = write_png(os.path.join(TMP, "nm_a.png"), a)
    r = write_png(os.path.join(TMP, "nm_b.png"), np.roll(a, (4, -3), axis=(0, 1)))
    res = run(s, r)
    assert res.status != "failed", res.reason
    m = metrics_of(res)
    assert m["accuracy"]["rmse_m"] is None, \
        "a bare PNG has no scale; rmse_m must be null, got %s" % m["accuracy"]["rmse_m"]
    assert m["accuracy"]["metres_per_pixel"] is None
    assert any("pixel space" in d for d in m["degraded"]), \
        "the run must record that it had no geometry: %s" % m["degraded"]
    return "ran in pixel space, refused to report metres"


def t_full_artifact_set_is_written():
    a = terrain(13)
    s = write_png(os.path.join(TMP, "art_a.png"), a)
    r = write_png(os.path.join(TMP, "art_b.png"), np.roll(a, 5, axis=1))
    res = run(s, r)
    assert res.status != "failed", res.reason
    want = ["matches.csv", "metrics.json", "transform.json", "overlay.png", "report.md"]
    missing = [f for f in want if not os.path.exists(os.path.join(res.out_dir, f))]
    assert not missing, "missing artifacts: %s" % missing
    reg = [f for f in os.listdir(res.out_dir) if f.startswith("registered.")]
    assert reg, "no registered raster written"
    rows = open(os.path.join(res.out_dir, "matches.csv")).read().strip().splitlines()
    assert rows[0] == "src_x,src_y,ref_x,ref_y,confidence,inlier", rows[0]
    assert len(rows) > 5, "only %d match rows" % (len(rows) - 1)
    return "%d matches, %d artifacts" % (len(rows) - 1, len(want) + len(reg))


def t_rmse_is_reported_in_four_units():
    """One error, four units, and they must agree with each other."""
    master = terrain(21, n=1024)
    src, ref = geo_pair(master, 2.0, 4.0, "u")
    res = run(src, ref, max_side=256)
    assert res.status != "failed", res.reason
    m = metrics_of(res)
    a_ = m["accuracy"]
    for k in ("rmse_reference_px", "rmse_working_px", "rmse_source_px", "rmse_m"):
        assert k in a_ and a_[k] is not None, \
            "%s is missing or null on a fully georeferenced pair" % k
    u = a_["units"]
    ref_px = a_["rmse_reference_px"]
    assert abs(a_["rmse_m"] - ref_px * u["metres_per_reference_px"]) < 0.05, \
        "metres disagree with reference px"
    assert abs(a_["rmse_source_px"] - ref_px * u["source_px_per_reference_px"]) < 0.05, \
        "source px disagree with reference px"
    assert abs(a_["rmse_working_px"] - ref_px / u["reference_decimation"]) < 0.05, \
        "working px disagree with reference px"
    return "ref %.3f px = %.2f m = %.3f src px = %.3f working px" % (
        ref_px, a_["rmse_m"], a_["rmse_source_px"], a_["rmse_working_px"])


def t_subpixel_flag_is_about_source_pixels():
    a = terrain(22)
    s = write_png(os.path.join(TMP, "sp_a.png"), a)
    r = write_png(os.path.join(TMP, "sp_b.png"), np.roll(a, 3, axis=1))
    res = run(s, r)
    m = metrics_of(res)["accuracy"]
    assert m["subpixel_basis"] == "source pixels", m["subpixel_basis"]
    if m["rmse_source_px"] is None:
        assert m["subpixel"] is None, \
            "no scale to convert with, so the flag must be null, not False"
    else:
        assert m["subpixel"] == (m["rmse_source_px"] < 1.0)
    assert "subpixel_working_grid" in m, "the working-grid flag must still be visible"
    return "subpixel=%s basis=%s floor=%s" % (
        m["subpixel"], m["subpixel_basis"], m.get("subpixel_floor_source_px"))


def t_unreachable_subpixel_is_a_note_not_a_failure():
    """A coarse reference limits accuracy; that is not a defect in the run."""
    master = terrain(23, n=1024)
    src, ref = geo_pair(master, 2.0, 8.0, "fl")     # reference 4x coarser
    res = run(src, ref, max_side=512)
    m = metrics_of(res)
    if res.status == "failed":
        return "failed cleanly: %s" % m["reason"]
    acc = m["accuracy"]
    assert acc["subpixel_floor_source_px"] is not None
    assert acc["subpixel_floor_source_px"] > 1.0, acc["subpixel_floor_source_px"]
    assert acc["subpixel_attainable"] is False, \
        "a coarser reference cannot give sub-source-pixel; that must be stated"
    joined = " ".join(m.get("notes") or [])
    assert "not reachable" in joined, "the limit must be stated in notes: %r" % joined
    assert "reference pixel is" not in (m.get("reason") or ""), \
        "a limit of the reference must not be counted against the run's status"
    return "floor %.2f src px, attainable=%s, status %s" % (
        acc["subpixel_floor_source_px"], acc["subpixel_attainable"], m["status"])


def t_fine_stage_beats_the_working_grid():
    """A decimated solve must be refined at native resolution, not trusted."""
    import rasterio
    from rasterio.transform import Affine
    a = terrain(24, n=1024)
    b = np.roll(a, (5, -4), axis=(0, 1))
    paths = []
    for i, arr in enumerate((b, a)):
        p = os.path.join(TMP, "fs_%d.tif" % i)
        with rasterio.open(p, "w", driver="GTiff", height=1024, width=1024, count=1,
                           dtype="uint8",
                           crs="+proj=stere +lat_0=-90 +R=1737400 +units=m",
                           transform=Affine(1.0, 0, 0, 0, -1.0, 0)) as ds:
            ds.write(np.clip(arr * 255, 0, 255).astype(np.uint8), 1)
        paths.append(p)
    # max_side 256 forces a 4x decimated working grid
    coarse = run(paths[0], paths[1], max_side=256, fine=False)
    fine = run(paths[0], paths[1], max_side=256, fine=True)
    if coarse.status == "failed" or fine.status == "failed":
        return "pair did not register; nothing to compare"
    mc, mf = metrics_of(coarse), metrics_of(fine)
    assert mc["fine_stage"]["attempted"] is False, "fine=False must not run the stage"
    if not mf["fine_stage"].get("adopted"):
        return "fine stage ran but was not adopted: %s" % mf["fine_stage"].get("note")
    rc = mc["accuracy"]["rmse_reference_px"]
    rf = mf["accuracy"]["rmse_reference_px"]
    assert rf < rc, "fine stage made it worse: %.3f -> %.3f reference px" % (rc, rf)
    return "%.3f -> %.3f reference px (%d native points)" % (
        rc, rf, mf["fine_stage"]["points"])


def _native_error(res, truth_dx, truth_dy, h, w):
    """Worst error, in reference px, of the delivered full-resolution transform
    against a known translation, over the source's corners and centre."""
    Hn = np.asarray(res.transform["matrix_reference_px"], float)
    pts = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
                    [(w - 1) / 2, (h - 1) / 2]], float)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ Hn.T
    p = p[:, :2] / p[:, 2:]
    return float(np.abs(p - (pts + [truth_dx, truth_dy])).max())


def t_pixel_space_strip_fine_windows_line_up():
    """A long strip with no geometry, registered onto itself.

    The fine stage tiles native windows down the strip. In pixel space each
    one used to be filled from the TOP of the source whatever its position, so
    every window past the first compared unrelated ground; on a real IIRS strip
    that took a 249/253 coarse solve to 10/226 and a warning.
    """
    a = terrain_hw(41, 3000, 200)
    p = write_png(os.path.join(TMP, "strip.png"), a)
    res = run(p, p, max_side=256)                  # 12x decimated working grid
    m = metrics_of(res)
    fs = m["fine_stage"]
    assert fs.get("attempted"), "the fine stage must run on a decimated strip"
    assert fs.get("adopted"), "fine stage not adopted on a same-file strip: %s" % fs.get("note")
    assert res.status == "pass", "same-file strip must pass, got %s: %s" % (
        res.status, res.reason)
    ratio = m["matches"]["inlier_ratio"]
    assert ratio > 0.5, "native inlier ratio %.3f on identical images" % ratio
    err = _native_error(res, 0, 0, *a.shape)
    assert err < 1.0, "same-file strip is off identity by %.2f px" % err
    return "%d/%d native inliers from %d windows, off identity by %.2f px" % (
        m["matches"]["inliers"], m["matches"]["candidates"], fs["windows_used"], err)


def t_fine_stage_cannot_replace_a_better_coarse_set():
    """A native set that verifies but is worse must not displace the coarse one.

    Twelve consistent points clustered in one corner among 188 random ones
    clear the old bar (>= 8 inliers) while failing the inlier-ratio, coverage
    and hull checks the coarse set passes.
    """
    from seleno.tool.methods import Correspondences
    mod = sys.modules["seleno.tool.register"]
    real = mod._fine_stage

    def poisoned(S, R, corr, vr, frame, *a, **k):
        rng = np.random.default_rng(5)
        H, W = R.array.shape
        good = rng.uniform(0, 0.15, (12, 2)) * [W, H]
        src = np.vstack([good, rng.uniform(0, 1, (188, 2)) * [W, H]])
        ref = np.vstack([good, rng.uniform(0, 1, (188, 2)) * [W, H]])
        c = Correspondences(src=src.astype(np.float32), ref=ref.astype(np.float32),
                            confidence=np.ones(len(src), np.float32),
                            method="poisoned@native", kind="dense",
                            detail={"back_x": src[:, 0], "back_y": src[:, 1]})
        return c, {"attempted": True, "points": len(src), "windows_used": 1,
                   "native": True, "note": None}

    a = terrain(42, n=1024)
    p = write_png(os.path.join(TMP, "poison.png"), a)
    mod._fine_stage = poisoned
    try:
        res = run(p, p, max_side=256)
    finally:
        mod._fine_stage = real
    m = metrics_of(res)
    fs = m["fine_stage"]
    assert not fs.get("adopted"), "a worse native set replaced the coarse solution"
    assert "coarse solution was kept" in (fs.get("note") or ""), fs.get("note")
    assert res.status == "pass", "coarse set should still pass, got %s: %s" % (
        res.status, res.reason)
    return "refused: %s" % fs["note"][:90]


def t_random_pixel_space_pairs_are_right_or_refused():
    """Random shapes, sizes, shifts and working grids, no geometry at all.

    Whatever the input, a result is either within a pixel of the known shift or
    a declared failure. A confident wrong answer is the one outcome not allowed.
    """
    rng = np.random.default_rng(2026)
    lines, wrong, failed = [], [], []
    for i in range(6):
        kind = ("tall", "wide", "square")[i % 3]
        if kind == "tall":
            h, w = int(rng.integers(1500, 3500)), int(rng.integers(150, 320))
        elif kind == "wide":
            h, w = int(rng.integers(150, 320)), int(rng.integers(1500, 3500))
        else:
            h, w = int(rng.integers(600, 1400)), int(rng.integers(600, 1400))
        dy, dx = int(rng.integers(0, max(2, h // 60))), int(rng.integers(0, max(2, w // 60)))
        master = terrain_hw(100 + i, h + dy, w + dx)
        ref = write_png(os.path.join(TMP, "rnd%d_ref.png" % i), master[:h, :w])
        src = write_png(os.path.join(TMP, "rnd%d_src.png" % i), master[dy:dy + h, dx:dx + w])
        max_side = int(rng.choice([256, 512]))
        res = run(src, ref, max_side=max_side)
        tag = "%s %dx%d shift(%d,%d) side %d" % (kind, h, w, dx, dy, max_side)
        if res.status == "failed":
            code = metrics_of(res).get("reason")
            assert code in FAILURE_CODES, "%s failed without a declared code: %s" % (tag, code)
            failed.append(tag)
            continue
        err = _native_error(res, dx, dy, h, w)
        lines.append("%s -> %.2f px" % (tag, err))
        if err >= 1.0:
            wrong.append("%s: off by %.2f px (%s)" % (tag, err, res.status))
    assert not wrong, "confident wrong answers: " + "; ".join(wrong)
    # The thinnest strips at a 256 px grid fall under the 4096 px overlap floor
    # and are refused; that is allowed, but most of these must still register or
    # the test says nothing about accuracy.
    assert len(lines) >= 4, "only %d of 6 registered; refused: %s" % (
        len(lines), "; ".join(failed))
    return "%d right, %d refused; worst %s" % (
        len(lines), len(failed), max(lines, key=lambda s: float(s.split()[-2])) if lines else "-")


def iirs_like_cube(tag, nbands=64, n=192, lo=0.8, hi=4.8, order="bsq",
                  with_centres=True):
    """A synthetic PDS4 spectrometer cube shaped like IIRS.

    Reflected-light bands carry the terrain. Bands past 2.5 um carry an
    INVERTED, smoother field standing in for thermal emission, so a pseudo-pan
    that fails to exclude them is measurably worse than one that does.
    """
    ground = terrain(31, n=n)
    thermal = cv2.GaussianBlur(1.0 - ground, (0, 0), 9.0)
    centres = [lo + i * (hi - lo) / (nbands - 1) for i in range(nbands)]
    cube = np.zeros((nbands, n, n), np.float32)
    rng = np.random.default_rng(7)
    for i, c in enumerate(centres):
        base = ground if c < 2.5 else thermal
        gain = 0.3 + 2.0 * np.exp(-((c - 1.2) ** 2) / 0.5)
        cube[i] = base * gain + 0.02 * rng.standard_normal((n, n)).astype(np.float32)
    raw = np.clip(cube * 8000, 0, 65535).astype("<u2")

    img = os.path.join(TMP, "%s.img" % tag)
    raw.tofile(img)
    axes = {"bsq": ["Band", "Line", "Sample"]}[order]
    els = {"Band": nbands, "Line": n, "Sample": n}
    ax = "".join("<Axis_Array><axis_name>%s</axis_name><elements>%d</elements>"
                 "</Axis_Array>" % (a, els[a]) for a in axes)
    bands_xml = "".join('<sp:center_value unit="micrometer">%.6f</sp:center_value>'
                        % c for c in centres) if with_centres else ""
    xml = ('<?xml version="1.0"?><Product_Observational>'
           '<logical_identifier>urn:isro:isda:ch2_iir:test</logical_identifier>'
           '<File_Area_Observational><Array_3D_Spectrum>'
           '<offset unit="byte">0</offset>'
           '<axis_index_order>Last Index Fastest</axis_index_order>'
           '<data_type>UnsignedLSB2</data_type>' + ax +
           '</Array_3D_Spectrum></File_Area_Observational>'
           '<Spectral_Characteristics>' + bands_xml + '</Spectral_Characteristics>'
           '</Product_Observational>')
    lbl = os.path.join(TMP, "%s.xml" % tag)
    open(lbl, "w").write(xml)
    return lbl, ground


def t_spectrometer_cube_collapses_to_reflected_light():
    from seleno.tool import spectral
    lbl, ground = iirs_like_cube("iirs_a")
    sc = load(lbl)
    assert sc.array.ndim == 2, "a cube must be collapsed before use, got %r" % (
        sc.array.shape,)
    assert sc.instrument.lower() in ("iirs", "unknown"), sc.instrument
    note = " ".join(sc.degraded)
    assert "pseudo-pan" in note, "the collapse must be recorded: %s" % note
    # it must resemble the reflected-light ground, not the thermal stand-in
    a = np.asarray(sc.array, np.float32)
    g = ground[:a.shape[0], :a.shape[1]]
    r = float(np.corrcoef(a.ravel(), g.ravel())[0, 1])
    assert r > 0.7, "pseudo-pan does not track the reflected-light terrain (r=%.3f)" % r
    return "%s, r=%.3f vs terrain" % (note.split(" (")[0], r)


def t_thermal_bands_are_excluded():
    """Past ~2.5 um the signal is emission, not reflectance; it must be dropped."""
    from seleno.tool import spectral
    lbl, ground = iirs_like_cube("iirs_b")
    from seleno.ohrc.label import parse_label
    L = parse_label(lbl)
    assert L.bands == 64, "band axis not parsed: %r" % L.bands
    assert len(L.band_centres_um) == 64, "band centres not parsed"
    idx = spectral.select_bands(L.band_centres_um)
    assert idx, "no band selected in the reflected-light window"
    used = [L.band_centres_um[i] for i in idx]
    assert max(used) <= 1.6 + 1e-6, "a band past 1.6 um was selected: %.3f" % max(used)
    assert min(used) >= 0.8 - 1e-6, "a band below 0.8 um was selected: %.3f" % min(used)
    assert not any(c > 2.5 for c in used), "thermal band selected"
    return "%d of %d bands, %.2f-%.2f um" % (len(idx), L.bands, min(used), max(used))


def t_cube_without_band_centres_says_so():
    """No wavelengths in the label means the window cannot be applied."""
    lbl, _ = iirs_like_cube("iirs_c", with_centres=False)
    sc = load(lbl)
    assert sc.array.ndim == 2
    note = " ".join(sc.degraded)
    assert "no band centre" in note or "carried no band centre" in note, \
        "a guessed band selection must be declared: %s" % note
    return "declared: %s" % note.split(";")[0][:70]


def envi_cube(tag, nbands=32, lines=300, samples=64, lo_nm=712.3, step_nm=16.84,
              ext=".qub", with_loc=True):
    """A PDS4 spectrometer product shaped like a real IIRS delivery.

    The synthetic cube the first version of these tests used was wrong in three
    ways that only a real product exposed: the data file is a `.qub` and is
    named in the label rather than implied by the extension; the band centres
    live in `<center_wavelength unit="nm">` inside `Band_Bin`, in NANOMETRES;
    and the geometry arrives as an ENVI `_loc_` backplane rather than the
    tabular CSV the framing cameras ship.
    """
    ground = terrain(41, n=max(lines, samples))[:lines, :samples]
    thermal = cv2.GaussianBlur(1.0 - ground, (0, 0), 7.0)
    centres = [lo_nm + i * step_nm for i in range(nbands)]
    cube = np.zeros((nbands, lines, samples), np.float32)
    rng = np.random.default_rng(11)
    for i, c_nm in enumerate(centres):
        c = c_nm / 1000.0
        base = ground if c < 2.5 else thermal
        cube[i] = base * (0.05 + 0.3 * np.exp(-((c - 1.2) ** 2) / 0.6)) \
            + 0.004 * rng.standard_normal((lines, samples)).astype(np.float32)
    stem = os.path.join(TMP, "ch2_iir_ndi_20240120T1432235872_d_rfl_%s" % tag)
    cube.astype("<f4").tofile(stem + ext)

    if with_loc:
        loc = np.zeros((4, lines, samples), np.float32)
        lon = 40.0 + np.linspace(0, 0.4, samples)[None, :] * np.ones((lines, 1))
        lat = np.linspace(-70.0, -66.0, lines)[:, None] * np.ones((1, samples))
        loc[0], loc[1] = lon, lat
        loc[2] = 1737400.0
        loc_path = stem.replace("_d_rfl_", "_d_loc_") + "_ard.img"
        open(loc_path, "wb").write(loc.astype("<f4").tobytes())

    bins = "".join(
        '<Band_Bin><band_number>%d</band_number>'
        '<band_width unit="nm">19.8</band_width>'
        '<center_wavelength unit="nm">%.1f</center_wavelength></Band_Bin>'
        % (i + 1, c) for i, c in enumerate(centres))
    xml = ('<?xml version="1.0"?><Product_Observational>'
           '<logical_identifier>urn:isro:isda:ch2_cho.iir:data_derived:%s</logical_identifier>'
           '<File_Area_Observational><File><file_name>%s</file_name></File>'
           '<Array_3D_Spectrum><offset unit="byte">0</offset>'
           '<axis_index_order>Last Index Fastest</axis_index_order>'
           '<Element_Array><data_type>IEEE754LSBSingle</data_type></Element_Array>'
           '<Axis_Array><axis_name>BAND</axis_name><elements>%d</elements>'
           '<sequence_number>1</sequence_number>%s</Axis_Array>'
           '<Axis_Array><axis_name>LINE</axis_name><elements>%d</elements></Axis_Array>'
           '<Axis_Array><axis_name>SAMPLE</axis_name><elements>%d</elements></Axis_Array>'
           '</Array_3D_Spectrum></File_Area_Observational>'
           '<isda:pixel_resolution unit="m/pixel">56.73</isda:pixel_resolution>'
           '</Product_Observational>'
           % (tag, os.path.basename(stem + ext), nbands, bins, lines, samples))
    open(stem + ".xml", "w").write(xml)
    return stem + ".xml", ground


def t_label_names_the_data_file_not_the_extension():
    """IIRS ships a .qub; guessing `.img` beside the label simply misses it."""
    lbl, _ = envi_cube("qb", ext=".qub")
    assert not os.path.exists(lbl.replace(".xml", ".img")), "fixture should have no .img"
    sc = load(lbl)
    assert sc.array.ndim == 2, sc.array.shape
    assert sc.path.endswith(".qub"), "the label's own file_name must be used: %s" % sc.path
    return os.path.basename(sc.path)


def t_band_centres_in_nanometres_are_converted():
    """Real labels give centres in nm inside Band_Bin; the window is in um."""
    from seleno.ohrc.label import parse_label
    lbl, _ = envi_cube("nm", nbands=32, lo_nm=712.3, step_nm=16.84)
    L = parse_label(lbl)
    assert L.bands == 32, L.bands
    c = L.band_centres_um
    assert len(c) == 32, "band centres not parsed: %d" % len(c)
    assert 0.70 < c[0] < 0.72, "nm not converted to um: first centre %r" % c[0]
    assert 1.2 < c[-1] < 1.3, "last centre %r" % c[-1]
    sc = load(lbl)
    note = " ".join(sc.degraded)
    assert "band centres from the label" in note, \
        "real centres must not fall back to a guess: %s" % note
    return "%.4f..%.4f um, %s" % (c[0], c[-1], note.split("(")[1].rstrip(")"))


def t_geometry_comes_from_the_loc_backplane():
    """IIRS has no geometry CSV; lon/lat arrive as an ENVI _loc_ cube."""
    lbl, _ = envi_cube("loc", with_loc=True)
    sc = load(lbl)
    assert sc.lonlat is not None, "no lattice built from the backplane"
    lon, lat = sc.lonlat.lon, sc.lonlat.lat
    assert 39.9 < np.nanmin(lon) < 40.5, np.nanmin(lon)
    assert -70.5 < np.nanmin(lat) < -65.5, np.nanmin(lat)
    assert "backplane" in " ".join(sc.degraded), sc.degraded
    return "lattice %s, lon %.2f..%.2f lat %.2f..%.2f" % (
        lon.shape, np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat))


def t_large_raster_is_read_lazily():
    """A global mosaic must not be pulled into RAM to be opened."""
    from seleno.tool.scene import LazyRaster, _EAGER_BYTES
    import rasterio
    from rasterio.transform import Affine
    n = 2048
    p = os.path.join(TMP, "big.tif")
    a = np.clip(terrain(42, n=n) * 255, 0, 255).astype(np.uint8)
    with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="uint8", crs="+proj=eqc +R=1737400 +units=m",
                       transform=Affine(100.0, 0, 0, 0, -100.0, 0)) as ds:
        ds.write(a, 1)
    lz = LazyRaster(p)
    assert lz.shape == (n, n) and lz.ndim == 2 and lz.size == n * n
    # strided slicing must agree with the eager read
    ref = a.astype(np.float32)[::7, ::5]
    got = lz[0:n:7, 0:n:5]
    assert got.shape == ref.shape, "%s vs %s" % (got.shape, ref.shape)
    assert np.allclose(got, ref), "strided read disagrees with the eager read"
    # paired fancy indexing must agree too
    rng = np.random.default_rng(3)
    rr = rng.integers(0, n, 500)
    cc = rng.integers(0, n, 500)
    assert np.allclose(lz[rr, cc], a[rr, cc].astype(np.float32)), \
        "fancy indexing disagrees with the eager read"
    return "%dx%d lazy reads match eager, threshold %d MB" % (n, n, _EAGER_BYTES >> 20)


def t_failure_codes_are_declared():
    assert set(FAILURE_CODES) == {"unreadable_input", "no_overlap", "insufficient_matches",
                                  "verification_failed", "degenerate_transform"}, FAILURE_CODES
    return ", ".join(FAILURE_CODES)


def t_memory_ceiling_shrinks_grid():
    """The tool must not ask for more working memory than the machine has."""
    small, note = _cap_max_side(64000)
    assert note is not None, "a 64000 px grid should have been capped"
    est_gb = small * small * 900 / 1e9
    assert est_gb < 8, "capped grid still estimates %.1f GB" % est_gb
    same, none_note = _cap_max_side(256)
    assert same == 256 and none_note is None, "a small grid must not be capped"
    return "64000 -> %d (est %.1f GB)" % (small, est_gb)


def t_uniform_distribution_is_reported():
    a = terrain(14)
    s = write_png(os.path.join(TMP, "ud_a.png"), a)
    r = write_png(os.path.join(TMP, "ud_b.png"), np.roll(a, 3, axis=0))
    res = run(s, r, grid=(6, 6))
    m = metrics_of(res)
    d = m["distribution"]
    for k in ("grid", "coverage_fraction", "dispersion", "eligible_cells"):
        assert k in d, "missing %r in distribution" % k
    assert d["grid"] == [6, 6]
    assert 0.0 <= d["coverage_fraction"] <= 1.0
    return "coverage %.2f, dispersion %.2f over %d eligible cells" % (
        d["coverage_fraction"], d["dispersion"], d["eligible_cells"])


def _view_fixture(name, h=700, w=1000):
    import rasterio
    from rasterio.transform import Affine
    p = os.path.join(TMP, name)
    a = np.clip(terrain_hw(21, h, w) * 255, 0, 255).astype(np.uint8)
    with rasterio.open(p, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", crs="+proj=eqc +R=1737400 +units=m",
                       transform=Affine(100.0, 0, 0, 0, -100.0, 0)) as ds:
        ds.write(a, 1)
    return p, a


def t_view_tiles_are_the_size_the_viewer_expects():
    """Every tile at every level must be exactly ceil(level size) - 256*index,
    or the viewer stretches edge tiles and the image shears at its borders."""
    from seleno.tool import view as V
    p, a = _view_fixture("view_sizes.tif")
    v = V.open_view(p, os.path.join(TMP, "viewcache"))
    assert isinstance(v.array, np.memmap), "uncompressed TIFF should be memory-mapped"
    h, w = a.shape
    n = 0
    s = 1
    while s <= v.max_scale:
        lh, lw = -(-h // s), -(-w // s)
        for ty in range(-(-lh // V.TILE)):
            for tx in range(-(-lw // V.TILE)):
                t = V.tile(v, s, tx, ty)
                want = (min(V.TILE, lh - ty * V.TILE), min(V.TILE, lw - tx * V.TILE))
                assert t.shape == want, "s=%d tile %d,%d is %s, want %s" % (s, tx, ty, t.shape, want)
                n += 1
        s *= 2
    # at 1:1 a tile is the file's own pixels through the one fixed stretch
    got = V.tile(v, 1, 1, 1).astype(np.float32)
    raw = a[256:512, 256:512].astype(np.float32)
    want = np.rint(np.clip((raw - v.lo) * (255.0 / (v.hi - v.lo)), 0, 255))
    assert np.array_equal(got, want), "1:1 tile is not the native pixels"
    geo = v.georef()
    assert geo and geo["kind"] == "eqc" and geo["R"] == 1737400.0, geo
    return "%d tiles over %d levels, georef %s" % (n, int(math.log2(v.max_scale)) + 1, geo["kind"])


def t_view_overview_agrees_with_native_reads():
    """Tiles switch from native reads to the cached overview at one scale. The
    two must show the same thing there, or zooming makes the image jump."""
    from seleno.tool import view as V
    saved = V.OVERVIEW_MAX_SIDE
    V.OVERVIEW_MAX_SIDE = 128
    try:
        p, a = _view_fixture("view_ov.tif")
        v = V.open_view(p, os.path.join(TMP, "viewcache"))
        f = v.factor
        assert f == 8 and v.overview is not None, "expected an 8x overview, got %d" % f
        worst = 0.0
        for s in (f, 2 * f):
            ov = V.render(v, 0, 0, v.width, v.height, s)
            keep, v.overview = v.overview, None
            try:
                nat = V.render(v, 0, 0, v.width, v.height, s)
            finally:
                v.overview = keep
            assert ov.shape == nat.shape, "%s vs %s" % (ov.shape, nat.shape)
            # The last row and column are partial blocks. The overview averages
            # its partial blocks with equal weight, native reads by pixel count,
            # so only the interior is expected to agree exactly.
            d = np.abs(ov.astype(int) - nat.astype(int))[:-1, :-1]
            worst = max(worst, float(d.max()))
        assert worst <= 1, "overview and native reads differ by %d grey levels" % worst
        # and a second open reuses the cached overview rather than rebuilding
        cached = [x for x in os.listdir(os.path.join(TMP, "viewcache")) if x.endswith(".npy")]
        assert cached, "overview was not cached"
        region, box, s = V.region(v, -50, -50, 400, 300)
        assert box == (0, 0, 400, 300) and s == 1 and region.shape == (300, 400), (box, s, region.shape)
    finally:
        V.OVERVIEW_MAX_SIDE = saved
    return "factor %d, max difference %d grey level(s)" % (f, worst)


def t_upload_names_stay_inside_their_batch():
    """Upload names come from the browser and must never escape data/uploads."""
    from fastapi import HTTPException
    import tool_routes as R
    base = os.path.join(R.UPLOADS, "abcdef12")
    for name in ("../../etc/passwd", "a/../../b.png", "/etc/passwd", "..\\..\\x.tif",
                 ".bashrc", "ok.png", "dir/sub/scene.xml"):
        dest = R._upload_target("abcdef12", name)
        assert dest.startswith(base + os.sep), "%r escaped to %s" % (name, dest)
    for bad_batch in ("../x", "ABC", "a", "abc/def"):
        try:
            R._upload_target(bad_batch, "a.png")
        except HTTPException:
            continue
        raise AssertionError("batch id %r was accepted" % bad_batch)
    for bad_name in ("", "..", "/", "a/" * 12 + "b"):
        try:
            R._upload_target("abcdef12", bad_name)
        except HTTPException:
            continue
        raise AssertionError("name %r was accepted" % bad_name)
    return "traversal, absolute and hidden names all contained"


def main():
    global TMP
    TMP = tempfile.mkdtemp(prefix="seleno_tool_")
    print("Phase 8 adversarial tests  (%s)\n" % TMP)
    try:
        for name, fn in sorted((k[2:], v) for k, v in globals().items()
                               if k.startswith("t_") and callable(v)):
            check(name, fn)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d passed, %d failed, %d skipped" % (len(PASS), len(FAIL), len(SKIP)))
    for n, why in FAIL:
        print("  FAILED %s: %s" % (n, str(why).splitlines()[-1][:120]))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
