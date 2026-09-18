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


def write_png(path, arr01, bits=8):
    if bits == 8:
        cv2.imwrite(path, np.clip(arr01 * 255, 0, 255).astype(np.uint8))
    else:
        cv2.imwrite(path, np.clip(arr01 * 65535, 0, 65535).astype(np.uint16))
    return path


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
