"""Stage 2 - feature extraction and candidate correspondence.

Matchers sit behind a single interface so a learned matcher (LoFTR / SuperGlue /
LightGlue) can be dropped in without touching the rest of the pipeline.
Availability is probed at import time and reported honestly to the UI - an
unavailable matcher is never silently swapped for another one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class MatchResult:
    kp_src: np.ndarray                    # (N,2) float32 pixel coords in preprocessed source
    kp_ref: np.ndarray                    # (N,2) float32 pixel coords in preprocessed reference
    score: np.ndarray                     # (N,) higher = more confident
    n_features_src: int = 0
    n_features_ref: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# classical detector / descriptor matchers
# --------------------------------------------------------------------------- #

def _detector(name: str, n_features: int):
    if name == "sift":
        return cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02, edgeThreshold=12)
    if name == "akaze":
        return cv2.AKAZE_create(threshold=0.0008)
    if name == "orb":
        return cv2.ORB_create(nfeatures=n_features, fastThreshold=8, scaleFactor=1.2, nlevels=10)
    raise ValueError("unknown detector " + name)


def _match_descriptors(desc_a, desc_b, binary: bool, ratio: float, mutual_check: bool = True):
    """Lowe ratio test, optionally enforced in both directions (mutual best).

    `ratio` and `mutual_check` are the two knobs that decide how clean the
    candidate set is *before* geometric verification sees it. Turning them down
    is what a naive pipeline does, and it is the honest way to show what the
    robust estimator is actually protecting against.
    """
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return [], {}
    norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)

    def one_way(d1, d2):
        keep = {}
        for p in bf.knnMatch(d1, d2, k=2):
            if len(p) < 2:
                continue
            m, n = p
            if m.distance < ratio * n.distance:
                keep[m.queryIdx] = (m.trainIdx, m.distance, n.distance)
        return keep

    fwd = one_way(desc_a, desc_b)
    stats = {"ratio_test_survivors_fwd": len(fwd), "mutual_check": mutual_check}

    out = []
    if mutual_check:
        bwd = one_way(desc_b, desc_a)
        stats["ratio_test_survivors_bwd"] = len(bwd)
        for qi, (ti, d1, d2) in fwd.items():
            back = bwd.get(ti)
            if back is not None and back[0] == qi:
                # confidence = how decisively the best beat the runner-up
                out.append((qi, ti, float(1.0 - d1 / max(d2, 1e-6))))
        stats["mutual_consistent"] = len(out)
    else:
        for qi, (ti, d1, d2) in fwd.items():
            out.append((qi, ti, float(1.0 - d1 / max(d2, 1e-6))))
    return out, stats


def classical_match(img_src, img_ref, *, detector="sift", ratio=0.80, n_features=8000,
                    mutual_check=True):
    det = _detector(detector, n_features)
    kp1, d1 = det.detectAndCompute(img_src, None)
    kp2, d2 = det.detectAndCompute(img_ref, None)
    binary = detector in ("orb", "akaze")
    if binary and d1 is not None:
        d1 = d1.astype(np.uint8)
        d2 = d2.astype(np.uint8) if d2 is not None else None
    mutual, stats = _match_descriptors(d1, d2, binary, ratio, mutual_check)

    if mutual:
        p1 = np.float32([kp1[i].pt for i, _, _ in mutual])
        p2 = np.float32([kp2[j].pt for _, j, _ in mutual])
        sc = np.float32([s for _, _, s in mutual])
    else:
        p1 = np.zeros((0, 2), np.float32)
        p2 = np.zeros((0, 2), np.float32)
        sc = np.zeros((0,), np.float32)

    detail = {"detector": detector, "ratio": ratio}
    detail.update(stats)
    return MatchResult(p1, p2, sc, len(kp1), len(kp2), detail)


def pyramid_match(img_src, img_ref, *, detector="sift", ratio=0.80, n_features=8000,
                  mutual_check=True, levels=(1.0, 0.5, 0.25)):
    """Multi-scale correspondence.

    Detectors are only approximately scale invariant. When a pair spans a large
    scale ratio (OHRC at 0.23 m/px against a TMC-2-like 5 m/px product) matching
    the source at several decimation levels and pooling the results recovers
    matches a single-scale pass misses. Coordinates are mapped back to level 0.
    """
    all_p1, all_p2, all_sc = [], [], []
    per_level, nf1, nf2 = {}, 0, 0
    for lv in levels:
        if lv == 1.0:
            small = img_src
        else:
            small = cv2.resize(cv2.GaussianBlur(img_src, (0, 0), 0.5 / lv), (0, 0),
                               fx=lv, fy=lv, interpolation=cv2.INTER_AREA)
        if min(small.shape[:2]) < 48:
            continue
        r = classical_match(small, img_ref, detector=detector, ratio=ratio,
                            n_features=n_features, mutual_check=mutual_check)
        per_level["level_%g" % lv] = len(r.kp_src)
        nf1 = max(nf1, r.n_features_src)
        nf2 = max(nf2, r.n_features_ref)
        if len(r.kp_src):
            all_p1.append(r.kp_src / lv)      # back to level-0 source coords
            all_p2.append(r.kp_ref)
            all_sc.append(r.score)

    detail = {"detector": detector, "levels": list(levels), "multiscale": True}
    detail.update(per_level)
    if not all_p1:
        return MatchResult(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                           np.zeros((0,), np.float32), nf1, nf2, detail)
    return MatchResult(np.vstack(all_p1), np.vstack(all_p2), np.concatenate(all_sc),
                       nf1, nf2, detail)


# --------------------------------------------------------------------------- #
# learned matcher slot - probed, never faked
# --------------------------------------------------------------------------- #

def _probe_learned():
    try:
        import torch  # noqa: F401
        import kornia  # noqa: F401
    except Exception as exc:      # environment dependent
        return False, "torch/kornia not importable (%s)" % type(exc).__name__
    return True, "kornia LoFTR available"


LEARNED_AVAILABLE, LEARNED_REASON = _probe_learned()
_LOFTR = None


def loftr_match(img_src, img_ref, *, max_side=640, **_):
    """Detector-free semi-dense matcher (LoFTR) via kornia.

    Raises if the dependency is absent, so the UI surfaces that rather than
    quietly falling back to SIFT and mislabelling the result.
    """
    global _LOFTR
    if not LEARNED_AVAILABLE:
        raise RuntimeError("LoFTR unavailable: " + LEARNED_REASON)
    import torch
    import kornia.feature as KF

    def prep(g):
        h, w = g.shape[:2]
        s = min(1.0, max_side / float(max(h, w)))
        r = cv2.resize(g, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
        return torch.from_numpy(r).float()[None, None] / 255.0, s

    t1, s1 = prep(img_src)
    t2, s2 = prep(img_ref)
    if _LOFTR is None:
        _LOFTR = KF.LoFTR(pretrained="outdoor").eval()
    with torch.no_grad():
        out = _LOFTR({"image0": t1, "image1": t2})
    p1 = out["keypoints0"].cpu().numpy() / s1
    p2 = out["keypoints1"].cpu().numpy() / s2
    sc = out["confidence"].cpu().numpy()
    return MatchResult(p1.astype(np.float32), p2.astype(np.float32), sc.astype(np.float32),
                       len(p1), len(p2), {"detector": "loftr", "max_side": max_side})


MATCHERS: dict[str, dict[str, Any]] = {
    "sift": {
        "fn": lambda a, b, **k: classical_match(a, b, detector="sift", **k),
        "label": "SIFT + ratio test + mutual check",
        "kind": "classical", "available": True,
    },
    "sift_pyramid": {
        "fn": lambda a, b, **k: pyramid_match(a, b, detector="sift", **k),
        "label": "SIFT multi-scale pyramid",
        "kind": "classical", "available": True,
    },
    "akaze": {
        "fn": lambda a, b, **k: classical_match(a, b, detector="akaze", **k),
        "label": "AKAZE (nonlinear scale space)",
        "kind": "classical", "available": True,
    },
    "orb": {
        "fn": lambda a, b, **k: classical_match(a, b, detector="orb", **k),
        "label": "ORB (fast binary baseline)",
        "kind": "classical", "available": True,
    },
    "loftr": {
        "fn": loftr_match,
        "label": "LoFTR (kornia, outdoor weights)",
        "kind": "learned", "available": LEARNED_AVAILABLE, "reason": LEARNED_REASON,
    },
}


def run_matcher(name: str, img_src, img_ref, **kwargs) -> MatchResult:
    spec = MATCHERS.get(name)
    if spec is None:
        raise ValueError("unknown matcher '%s'" % name)
    if not spec["available"]:
        raise RuntimeError("matcher '%s' unavailable: %s" % (name, spec.get("reason", "not installed")))
    fn: Callable = spec["fn"]
    return fn(img_src, img_ref, **kwargs)
