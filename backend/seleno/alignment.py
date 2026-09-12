"""Cached coarse alignment between two OHRC products.

`ohrc.reproject.coarse_align` costs ~50 s per pair because it reprojects both
strips onto a common grid. The answer is a property of the pair, not of the
experiment, so it is computed once and cached on disk.

This is the stage that makes everything downstream possible. Measured on the two
2024-11-15 products, the delivered system-level geolocation disagrees between
strips by roughly 540 m - about 2 200 pixels at 0.24 m/px. Cutting a 1024 px
window from one strip and its geometry-predicted counterpart from the other
therefore yields two windows that barely share ground, and a matcher fed those
produces a handful of spurious correspondences. Reading that as "the
illumination change defeats the matcher" would be wrong, and it is the kind of
wrong conclusion this module exists to prevent.
"""
from __future__ import annotations

import json
import os

from .ohrc import reproject as R

CACHE_DIRNAME = "alignment"


def cache_path(cache_dir: str, a, b) -> str:
    return os.path.join(cache_dir, CACHE_DIRNAME,
                        "coarse_%s__%s.json" % (a.timestamp, b.timestamp))


def get_offset(a, b, cache_dir: str, *, rebuild: bool = False,
               require_confident: bool = False, **kwargs) -> dict:
    """Coarse ground offset between two products, cached.

    Returns the full `coarse_align` record. Callers should look at
    ``confident`` before using ``offset_stereo_m``: an unconfident peak means
    the two strips could not be aligned by translation at all, which is itself a
    result worth reporting rather than papering over.
    """
    path = cache_path(cache_dir, a, b)
    if os.path.exists(path) and not rebuild:
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            if rec.get("a") == a.timestamp and rec.get("b") == b.timestamp:
                rec["from_cache"] = True
                return rec
        except Exception:
            pass

    rec = R.coarse_align(a, b, **kwargs)
    rec["from_cache"] = False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    if require_confident and not rec.get("confident"):
        raise RuntimeError(
            "coarse alignment between %s and %s is not confident (peak %s, "
            "margin %s): %s" % (a.timestamp, b.timestamp, rec.get("peak_ncc"),
                                rec.get("peak_margin"), rec.get("reason", "weak peak")))
    return rec


def offset_or_zero(rec: dict | None) -> tuple[float, float]:
    """The offset to apply, or (0, 0) when there is no trustworthy measurement."""
    if not rec or not rec.get("ok") or not rec.get("confident"):
        return (0.0, 0.0)
    ox, oy = rec.get("offset_stereo_m", (0.0, 0.0))
    return (float(ox), float(oy))
