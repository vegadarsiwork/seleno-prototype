"""Where a registration problem comes from.

One interface, two sources:

* **OHRC windows** - a window in one real strip and the corresponding window in
  another, located through the delivered geometry. No ground truth exists for
  these. The geometry prior is recorded separately so the pipeline can be scored
  against it without ever calling it truth.
* **Legacy demo pairs** - the six manifest pairs the earlier prototype shipped,
  five of which carry a synthetic ground-truth transform. Kept so the existing
  regression numbers stay reproducible.

The distinction the whole project rests on is encoded in two separate fields:

``gt_transform``       a transform known to be correct by construction. Only ever
                       set for synthetically-warped pairs.
``geometry_prior``     the transform implied by ISRO's delivered geolocation.
                       System-level, unrefined, metre-to-decametre absolute. A
                       useful independent check; **not** truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .ohrc import geometry as G


@dataclass
class PairSpec:
    """One registration problem, plus everything known about it."""

    pair_id: str
    name: str
    short: str
    source: np.ndarray                      # uint8 window
    reference: np.ndarray
    gsd_source_m: float
    gsd_reference_m: float
    kind: str                               # "ohrc" | "legacy"
    recommended_model: str = "similarity"

    gt_transform: np.ndarray | None = None          # truth, or None
    gt_origin: str = "none"                         # how truth was obtained
    geometry_prior: np.ndarray | None = None        # delivered-geometry estimate
    geometry_prior_origin: str = "none"

    sun_source: dict | None = None
    sun_reference: dict | None = None
    footprint_source: dict | None = None
    footprint_reference: dict | None = None
    stereo_xy: tuple[np.ndarray, np.ndarray] | None = None   # per-pixel projected metres
    difficulty: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_transform is not None

    def meta(self) -> dict:
        """JSON-safe description for the API and the UI."""
        d = {
            "id": self.pair_id, "name": self.name, "short": self.short,
            "kind": self.kind, "difficulty": self.difficulty,
            "source_shape": list(self.source.shape[:2]),
            "reference_shape": list(self.reference.shape[:2]),
            "gsd_source_m": self.gsd_source_m,
            "gsd_reference_m": self.gsd_reference_m,
            "recommended_model": self.recommended_model,
            "ground_truth_available": self.has_ground_truth,
            "ground_truth_origin": self.gt_origin,
            "geometry_prior_available": self.geometry_prior is not None,
            "geometry_prior_origin": self.geometry_prior_origin,
            "sun_source": self.sun_source,
            "sun_reference": self.sun_reference,
            "footprint_source": self.footprint_source,
            "footprint_reference": self.footprint_reference,
            "provenance": self.provenance,
            "warnings": list(self.warnings),
        }
        if self.sun_source and self.sun_reference:
            d["d_azimuth_deg"] = round(
                self.sun_reference["azimuth_deg"] - self.sun_source["azimuth_deg"], 3)
            d["d_elevation_deg"] = round(
                self.sun_reference["elevation_deg"] - self.sun_source["elevation_deg"], 4)
        return d


# --------------------------------------------------------------------------- #
# OHRC windows
# --------------------------------------------------------------------------- #

def _fit_transform(src_pts: np.ndarray, dst_pts: np.ndarray,
                   model: str = "similarity") -> np.ndarray | None:
    src = np.asarray(src_pts, np.float64)
    dst = np.asarray(dst_pts, np.float64)
    if model == "homography" and len(src) >= 4:
        M, _ = cv2.findHomography(src, dst, 0)
    elif model == "affine" and len(src) >= 3:
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    elif len(src) >= 2:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    else:
        return None
    if M is None:
        return None
    out = np.eye(3, dtype=np.float64)
    M = np.asarray(M)
    if M.shape == (2, 3):
        out[:2, :] = M
        return out
    return M.astype(np.float64)


def stereo_grid(product, sample0: int, line0: int, width: int, height: int,
                stride: int = 8):
    """Per-pixel projected (x, y) metres for a window, upsampled from a coarse grid.

    The delivered lattice is ~22-28 m on the ground, so evaluating the
    interpolator every `stride` pixels and resizing costs nothing in accuracy and
    keeps a 1024 px window's coordinate computation trivial.
    """
    ss = np.arange(sample0, sample0 + width, stride, dtype=np.float64)
    ll = np.arange(line0, line0 + height, stride, dtype=np.float64)
    S, L = np.meshgrid(ss, ll)
    X, Y = product.geometry.stereo(S.ravel(), L.ravel())
    X = np.asarray(X, np.float64).reshape(S.shape)
    Y = np.asarray(Y, np.float64).reshape(S.shape)
    X = cv2.resize(X, (width, height), interpolation=cv2.INTER_LINEAR)
    Y = cv2.resize(Y, (width, height), interpolation=cv2.INTER_LINEAR)
    return X, Y


def ohrc_window_pair(a, b, sample0: int, line0: int, size: int = 1024,
                     *, pair_id: str | None = None,
                     model: str = "affine",
                     offset_stereo_m: tuple[float, float] | None = None,
                     coarse_alignment: dict | None = None) -> PairSpec:
    """A window in strip `a` and the same ground in strip `b`.

    The reference window is centred on wherever `a`'s window centre lands in
    `b` according to the delivered geometry, and the corner-to-corner mapping
    between the two windows becomes the geometry prior.

    That prior is what the pipeline is asked to improve on. It is not assumed
    correct: the two 2024 strips overlap 95 % by footprint, but their
    geolocation is system-level with no photogrammetric refinement, so the true
    offset between these two crops is unknown at the outset and is one of the
    things the experiment measures.

    The default transform model is **affine**, not similarity. The two strips
    were acquired at different roll angles (14.6 deg and 15.2 deg for the 2024
    pair), so the residual relationship between two pushbroom views of the same
    ground carries a small shear and anisotropic scale that a 4-dof similarity
    cannot absorb. Measured on one window: similarity gave 52 inliers at 22.5 %,
    affine gave 133 at 57.6 % on identical correspondences.
    """
    ga, gb = a.geometry, b.geometry
    # Correction for the measured disagreement between the two products'
    # delivered geolocation. Without it the reference window lands hundreds of
    # metres off the source window's ground - see ohrc.reproject.coarse_align.
    ox, oy = (offset_stereo_m or (0.0, 0.0))
    cx, cy = sample0 + size / 2.0, line0 + size / 2.0
    x, y = ga.stereo(cx, cy)
    sb, lb, ok = gb.stereo_to_pixel(x + ox, y + oy)
    if not ok:
        raise ValueError("window centre (%d, %d) of %s does not fall inside %s"
                         % (sample0, line0, a.timestamp, b.timestamp))
    sb0 = int(round(sb - size / 2.0))
    lb0 = int(round(lb - size / 2.0))
    sb0 = int(np.clip(sb0, 0, max(0, b.samples - size)))
    lb0 = int(np.clip(lb0, 0, max(0, b.lines - size)))

    src = a.read_tile(sample0, line0, size, size)
    ref = b.read_tile(sb0, lb0, size, size)

    # geometry prior: source-window corners -> projected -> reference-window pixels
    corners = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float64)
    xs, ys = ga.stereo(corners[:, 0] + sample0, corners[:, 1] + line0)
    sB, lB, okB = gb.stereo_to_pixel(np.atleast_1d(xs) + ox, np.atleast_1d(ys) + oy)
    prior = None
    prior_origin = "none"
    if np.all(okB):
        dst = np.column_stack([np.atleast_1d(sB) - sb0, np.atleast_1d(lB) - lb0])
        prior = _fit_transform(corners, dst, model)
        prior_origin = ("delivered system-level geometry grids of both products "
                        "(unrefined; metre-to-decametre absolute accuracy)")
        if ox or oy:
            prior_origin += (" plus a measured coarse offset of (%.1f, %.1f) m "
                             "between the two products' geolocation" % (ox, oy))

    sun_a = a.sun_at_line(line0 + size // 2) if a.has_sun_series else None
    sun_b = b.sun_at_line(lb0 + size // 2) if b.has_sun_series else None

    pid = pair_id or ("ohrc_%s__%s_s%d_l%d" % (a.timestamp[:13], b.timestamp[:13],
                                               sample0, line0))
    warnings = [
        "no ground truth exists for this pair: both windows are real acquisitions "
        "and no correct transform between them is known",
    ]
    if sun_a and sun_b:
        d_az = sun_b["azimuth_deg"] - sun_a["azimuth_deg"]
        d_el = sun_b["elevation_deg"] - sun_a["elevation_deg"]
        warnings.append(
            "illumination differs by %.1f deg azimuth and %.2f deg elevation at a "
            "grazing incidence - cast shadows move and shading reverses"
            % (d_az, d_el))

    return PairSpec(
        pair_id=pid,
        name="OHRC %s -> %s" % (a.timestamp[:13], b.timestamp[:13]),
        short=("window (%d, %d) of %s against the same ground in %s"
               % (sample0, line0, a.timestamp[:13], b.timestamp[:13])),
        source=src, reference=ref,
        gsd_source_m=a.gsd_m or 0.24, gsd_reference_m=b.gsd_m or 0.24,
        kind="ohrc", recommended_model=model,
        gt_transform=None, gt_origin="none - real repeat-pass imagery",
        geometry_prior=prior, geometry_prior_origin=prior_origin,
        sun_source=sun_a, sun_reference=sun_b,
        footprint_source=a.window_footprint(sample0, line0, size, size),
        footprint_reference=b.window_footprint(sb0, lb0, size, size),
        stereo_xy=stereo_grid(a, sample0, line0, size, size),
        difficulty="real repeat-pass, differing illumination",
        provenance={
            "source_product": a.product_id, "reference_product": b.product_id,
            "source_window": {"sample0": sample0, "line0": line0, "size": size},
            "reference_window": {"sample0": sb0, "line0": lb0, "size": size},
            "coarse_offset_stereo_m": [round(float(ox), 2), round(float(oy), 2)],
            "coarse_alignment": coarse_alignment,
            "source_orbit": a.label.imaging_orbit,
            "reference_orbit": b.label.imaging_orbit,
            "source_time": a.label.start_time.isoformat() if a.label.start_time else None,
            "reference_time": b.label.start_time.isoformat() if b.label.start_time else None,
        },
        warnings=warnings)


def prior_offset_m(pair: PairSpec) -> float | None:
    """Ground distance between the two window centres under the geometry prior.

    Non-zero because the reference window is snapped to integer pixels and
    clipped at the strip edge; useful as a sanity figure in the UI.
    """
    if pair.geometry_prior is None:
        return None
    h, w = pair.source.shape[:2]
    c = np.array([w / 2.0, h / 2.0, 1.0])
    q = pair.geometry_prior @ c
    q = q[:2] / q[2]
    d_px = float(np.hypot(q[0] - w / 2.0, q[1] - h / 2.0))
    return d_px * pair.gsd_reference_m


# --------------------------------------------------------------------------- #
# legacy manifest pairs
# --------------------------------------------------------------------------- #

def legacy_pair(store, pair_id: str) -> PairSpec:
    """Adapt one of the earlier prototype's manifest pairs onto PairSpec."""
    p = store.pair(pair_id)
    src, ref = store.images(pair_id)
    gt = p.get("gt_homography")
    return PairSpec(
        pair_id=p["id"], name=p.get("name", p["id"]), short=p.get("short", ""),
        source=src, reference=ref,
        gsd_source_m=float(p.get("gsd_source_m") or 0.24),
        gsd_reference_m=float(p.get("gsd_reference_m") or 0.24),
        kind="legacy",
        recommended_model=p.get("recommended_model", "homography"),
        gt_transform=np.asarray(gt, np.float64) if gt is not None else None,
        gt_origin=(p.get("synthetic_component") or "none")
        if gt is not None else "none",
        difficulty=p.get("difficulty", ""),
        provenance={"manifest": True,
                    "real_component": p.get("real_component"),
                    "synthetic_component": p.get("synthetic_component")},
        warnings=([] if gt is not None else
                  ["no ground truth for this pair"]))
