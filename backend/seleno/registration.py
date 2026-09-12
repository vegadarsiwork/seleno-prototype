"""The Seleno registration pipeline.

    INPUT
      -> metadata / geometry
      -> reference and overlap estimation
      -> illumination-aware preprocessing (usability mask)
      -> correspondence
      -> robust geometric verification
      -> correspondence selection
      -> sub-pixel refinement
      -> trust / refusal decision
      -> RESULT

Two rules govern what comes out:

* **A transform is not a result.** `RegistrationResult.status` is
  ``accepted`` / ``warning`` / ``refused``, with the reasons listed. Returning a
  confident transform for a pair that cannot be registered is the failure mode
  this project is built to avoid.
* **Self-consistency is never reported as accuracy.** Reprojection RMSE,
  held-out error, ground-truth error and disagreement with the delivered
  geometry are four separate fields and are never merged.

The stage functions themselves are the earlier prototype's modules, reused
unchanged: `preprocess`, `matchers`, `verify`, `spatial`, `register`, `metrics`,
`viz`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import (illumination, matchers, metrics, preprocess, register, spatial,
               subpixel, verify, viz)
from .pairs import PairSpec

# Rule-based decision thresholds. NOT CALIBRATED. They are hand-picked from the
# behaviour observed on this archive and are stated here, in one place, so the
# UI and the docs can say so. Calibrating them against labelled overlapping and
# non-overlapping pairs, and reporting precision/recall for the decision itself,
# is future work.
THRESHOLDS = {
    "min_inliers": 12,
    "min_inlier_ratio": 0.15,
    "min_spatial_coverage": 0.15,
    "max_rmse_multiple_of_threshold": 4.0,
    "max_heldout_rmse_px": 6.0,
    "warn_heldout_rmse_px": 3.0,
    "warn_usable_fraction": 0.10,
    "max_prior_disagreement_m": 250.0,
    # A transform fitted from correspondences covering only part of the frame is
    # extrapolated over the rest. Measured case: 73 % inlier ratio, 1.2 px
    # reprojection RMSE and 0.89 overlap NCC, yet the warped source covered 31 %
    # of the reference frame and the fitted rotation disagreed with the delivered
    # geometry by 4.9 deg. Self-consistency cannot detect that; overlap can.
    "warn_overlap_fraction": 0.50,
}


@dataclass
class Options:
    # correspondence
    matcher: str = "sift"
    ratio: float = 0.80
    mutual_check: bool = True
    max_features: int = 8000
    # preprocessing
    preprocess_enabled: bool = True
    use_clahe: bool = True
    use_flatten: bool = True
    harmonise_gsd: bool = True
    # illumination handling
    mask_mode: str = "none"                 # none | observed | predicted | both
    terrain_model: str = "none"             # none | mock | (a real DEM once wired)
    remove_shading: bool = False
    # verification and selection
    verify_enabled: bool = True
    model_type: str = "similarity"
    ransac_threshold: float = 3.0
    # "grid" applies the spatial quota, "topk" keeps the same number of matches
    # ranked by confidence alone (the fair A/B for the spatial stage), "all"
    # keeps every verified inlier (what a plain baseline would do).
    selection_mode: str = "grid"
    grid: int = 6
    per_cell: int = 3
    # refinement and evaluation
    subpixel_enabled: bool = True
    holdout_fraction: float = 0.30
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Options":
        d = dict(d or {})
        # Accept the older boolean spelling so saved configs keep working.
        if "spatial_enabled" in d and "selection_mode" not in d:
            d["selection_mode"] = "grid" if d.pop("spatial_enabled") else "all"
        d.pop("spatial_enabled", None)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Stage:
    name: str
    status: str = "ok"                      # ok | skipped | failed | warning
    note: str = ""
    seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    """What the pipeline returns. A transform is only part of it."""

    pair_id: str
    status: str                             # accepted | warning | refused
    transform: list | None
    model_type: str
    confidence: float
    reasons: list[str]
    metrics: dict
    stages: list[dict]
    options: dict
    pair: dict
    warnings: list[str] = field(default_factory=list)
    images: dict = field(default_factory=dict)   # not JSON-safe; stripped by the API

    def json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "images"}
        d["available_images"] = sorted(self.images.keys())
        return d


class _Timer:
    def __init__(self):
        self.t: dict[str, float] = {}

    def __call__(self, key):
        self._k = key
        return self

    def __enter__(self):
        self._s = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.t[self._k] = self.t.get(self._k, 0.0) + (time.perf_counter() - self._s)
        return False


# --------------------------------------------------------------------------- #

def _transform_disagreement_m(H: np.ndarray | None, prior: np.ndarray | None,
                              shape, gsd_m: float) -> dict | None:
    """How far the estimate moves the image corners away from the prior.

    An **independent** check, not an accuracy figure: the prior itself is
    system-level geolocation with metre-to-decametre error. A large
    disagreement means one of the two is wrong and the result deserves scrutiny.
    """
    if H is None or prior is None:
        return None
    h, w = shape[:2]
    c = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    ch = np.hstack([c, np.ones((4, 1))]).T

    def proj(M):
        q = M @ ch
        return (q[:2] / np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])).T

    d = np.linalg.norm(proj(H) - proj(prior), axis=1)
    return {"corner_disagreement_px": round(float(d.mean()), 3),
            "corner_disagreement_m": round(float(d.mean() * gsd_m), 2),
            "max_corner_disagreement_px": round(float(d.max()), 3)}


def _split_holdout(n: int, fraction: float, seed: int):
    """Deterministic fit/hold-out split of the selected correspondences."""
    if n < 8 or fraction <= 0:
        return np.ones(n, bool), np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = max(2, int(round(n * fraction)))
    k = min(k, n - 4)                       # always leave enough to fit
    hold = np.zeros(n, bool)
    hold[idx[:k]] = True
    return ~hold, hold


def _build_mask(pair: PairSpec, opts: Options, src_img, ref_img) -> illumination.UsabilityMask:
    shape = src_img.shape[:2]
    if opts.mask_mode == "none":
        return illumination.all_usable(shape)

    observed = illumination.observed_signal_mask(src_img)
    if opts.mask_mode == "observed":
        return observed

    terrain = illumination.get_terrain(opts.terrain_model)
    if not terrain.available:
        m = illumination.all_usable(shape)
        m.source = "predicted-shadow (unavailable)"
        m.warnings.append(
            "predicted-shadow masking requested but terrain model %r is unavailable: %s"
            % (terrain.name, terrain.unavailable_reason))
        return m.combine(observed) if opts.mask_mode == "both" else m
    if pair.stereo_xy is None or not (pair.sun_source and pair.sun_reference):
        m = illumination.all_usable(shape)
        m.source = "predicted-shadow (insufficient geometry)"
        m.warnings.append(
            "predicted-shadow masking needs per-pixel projected coordinates and Sun "
            "geometry for both images; this pair has neither")
        return m.combine(observed) if opts.mask_mode == "both" else m

    X, Y = pair.stereo_xy
    predicted = illumination.predicted_shadow_mask(
        terrain, X, Y, pair.sun_source, pair.sun_reference,
        gsd_m=pair.gsd_source_m)
    if opts.mask_mode == "both":
        return predicted.combine(observed)
    return predicted


# --------------------------------------------------------------------------- #

def run(pair: PairSpec, opts: Options) -> RegistrationResult:
    timer = _Timer()
    stages: list[Stage] = []
    images: dict[str, np.ndarray] = {}
    warnings: list[str] = list(pair.warnings)

    img_src, img_ref = pair.source, pair.reference
    gsd_s, gsd_r = pair.gsd_source_m, pair.gsd_reference_m

    # -------------------------------------------------- 1. metadata / geometry
    with timer("metadata"):
        prior = pair.geometry_prior
        meta_detail = {
            "kind": pair.kind,
            "gsd_source_m": gsd_s, "gsd_reference_m": gsd_r,
            "ground_truth_available": pair.has_ground_truth,
            "ground_truth_origin": pair.gt_origin,
            "geometry_prior_available": prior is not None,
            "geometry_prior_origin": pair.geometry_prior_origin,
            "sun_source": pair.sun_source, "sun_reference": pair.sun_reference,
            "provenance": pair.provenance,
        }
        if pair.sun_source and pair.sun_reference:
            meta_detail["d_azimuth_deg"] = round(
                pair.sun_reference["azimuth_deg"] - pair.sun_source["azimuth_deg"], 3)
            meta_detail["d_elevation_deg"] = round(
                pair.sun_reference["elevation_deg"] - pair.sun_source["elevation_deg"], 4)
    stages.append(Stage(
        "Metadata and geometry", "ok",
        ("%s pair, %.3f / %.3f m/px, ground truth: %s, geometry prior: %s"
         % (pair.kind, gsd_s, gsd_r,
            "yes" if pair.has_ground_truth else "none",
            "yes" if prior is not None else "none")),
        timer.t["metadata"], meta_detail))

    # ------------------------------------- 2. reference / overlap estimation
    with timer("overlap"):
        ov = {"source_shape": list(img_src.shape[:2]),
              "reference_shape": list(img_ref.shape[:2]),
              "gsd_ratio": round(gsd_r / gsd_s, 4) if gsd_s else None}
        if pair.footprint_source and pair.footprint_reference:
            ov["footprint_source"] = pair.footprint_source
            ov["footprint_reference"] = pair.footprint_reference
        from .pairs import prior_offset_m
        off = prior_offset_m(pair)
        if off is not None:
            ov["prior_centre_offset_m"] = round(off, 2)
        coarse = (pair.provenance or {}).get("coarse_offset_stereo_m")
        if coarse:
            ov["coarse_offset_stereo_m"] = coarse
            ov["coarse_offset_m"] = round(float((coarse[0] ** 2 + coarse[1] ** 2) ** 0.5), 2)
        ov["coarse_alignment"] = (pair.provenance or {}).get("coarse_alignment")
    stages.append(Stage("Reference and overlap estimation", "ok",
                        ("source %dx%d, reference %dx%d, GSD ratio %s"
                         % (img_src.shape[1], img_src.shape[0],
                            img_ref.shape[1], img_ref.shape[0], ov["gsd_ratio"])),
                        timer.t["overlap"], ov))

    # ------------------------------- 3. illumination-aware preprocessing
    with timer("preprocess"):
        mask = _build_mask(pair, opts, img_src, img_ref)
        warnings.extend(mask.warnings)

        src_for_pp = img_src
        shading_note = ""
        if opts.remove_shading:
            terrain = illumination.get_terrain(opts.terrain_model)
            if terrain.available and pair.stereo_xy is not None and pair.sun_source:
                X, Y = pair.stereo_xy
                sh = terrain.slope_shading(X, Y, pair.sun_source["azimuth_deg"],
                                           pair.sun_source["elevation_deg"])
                src_for_pp = illumination.remove_predicted_shading(img_src, sh)
                shading_note = " | broad terrain shading divided out (%s)" % terrain.name
                if terrain.is_synthetic:
                    warnings.append(
                        "shading removal used a SYNTHETIC terrain model; the result is "
                        "not evidence about the real surface")
            else:
                warnings.append(
                    "shading removal requested but no usable terrain model / geometry")

        target = None
        if opts.harmonise_gsd and gsd_s and gsd_r and gsd_r > gsd_s * 1.05:
            target = gsd_r
        pre_s = preprocess.preprocess(
            src_for_pp, enabled=opts.preprocess_enabled, use_clahe=opts.use_clahe,
            use_flatten=opts.use_flatten, src_gsd=gsd_s, target_gsd=target)
        pre_r = preprocess.preprocess(
            img_ref, enabled=opts.preprocess_enabled, use_clahe=opts.use_clahe,
            use_flatten=opts.use_flatten)
    a, b = pre_s.image, pre_r.image
    inv_scale = 1.0 / pre_s.scale if pre_s.scale else 1.0

    mask_status = "ok" if opts.mask_mode != "none" else "skipped"
    if mask.usable_fraction < THRESHOLDS["warn_usable_fraction"] and opts.mask_mode != "none":
        mask_status = "warning"
    stages.append(Stage(
        "Illumination-aware preprocessing", mask_status,
        ("%s%s | usable %.1f%% of the source window"
         % (" | ".join(pre_s.steps), shading_note, 100 * mask.usable_fraction)),
        timer.t["preprocess"],
        {"source": pre_s.stats, "reference": pre_r.stats,
         "steps_source": pre_s.steps, "steps_reference": pre_r.steps,
         "coordinate_scale": pre_s.scale, "mask": mask.summary()}))

    images["mask"] = viz_mask(img_src, mask)

    # ------------------------------------------------- 4. correspondence
    try:
        with timer("match"):
            kw = ({} if opts.matcher == "loftr"
                  else {"ratio": opts.ratio, "n_features": opts.max_features,
                        "mutual_check": opts.mutual_check})
            mres = matchers.run_matcher(opts.matcher, a, b, **kw)
    except Exception as exc:                                        # noqa: BLE001
        stages.append(Stage("Correspondence", "failed", str(exc),
                            timer.t.get("match", 0.0)))
        return _refused(pair, opts, stages, timer, [str(exc)], images, warnings)

    p_src = mres.kp_src * inv_scale
    p_ref = mres.kp_ref.astype(np.float64)
    n_raw = len(p_src)

    # Drop correspondences whose source point lies in a predicted-unusable
    # region. Filtering after detection rather than blanking the image first is
    # deliberate: zeroing a region manufactures a hard edge that the detector
    # then fires on.
    if opts.mask_mode != "none" and n_raw:
        yy = np.clip(p_src[:, 1].astype(int), 0, mask.usable.shape[0] - 1)
        xx = np.clip(p_src[:, 0].astype(int), 0, mask.usable.shape[1] - 1)
        keep = mask.usable[yy, xx]
        n_dropped = int((~keep).sum())
        p_src, p_ref = p_src[keep], p_ref[keep]
        scores = mres.score[keep] if len(mres.score) == n_raw else np.ones(len(p_src), np.float32)
        kp_disp = mres.kp_src[keep]
    else:
        n_dropped = 0
        scores = (mres.score if len(mres.score) == n_raw
                  else np.ones(n_raw, np.float32))
        kp_disp = mres.kp_src
    n_cand = len(p_src)

    stages.append(Stage(
        "Correspondence", "ok",
        ("%s: %d + %d keypoints -> %d candidates%s"
         % (opts.matcher, mres.n_features_src, mres.n_features_ref, n_cand,
            (" (%d dropped as unmatchable)" % n_dropped) if n_dropped else "")),
        timer.t["match"],
        {**mres.detail, "candidates_before_mask": n_raw,
         "dropped_by_mask": n_dropped, "candidates": n_cand}))

    images["candidates"] = viz.draw_matches(
        a, b, kp_disp, p_ref, np.ones(n_cand, bool), max_lines=300,
        color_in=(200, 200, 210), draw_outliers=False)

    # -------------------------------------- 5. robust geometric verification
    with timer("verify"):
        vres = verify.verify(p_src, p_ref, model_type=opts.model_type,
                             threshold=opts.ransac_threshold,
                             enabled=opts.verify_enabled)
    expected_scale = 1.0 if target is not None else ((gsd_s / gsd_r) if (gsd_s and gsd_r) else 1.0)
    if vres.ok:
        warnings.extend(verify.plausibility(vres.model, opts.model_type, expected_scale))
    stages.append(Stage(
        "Robust geometric verification",
        "ok" if vres.ok else ("skipped" if not opts.verify_enabled else "failed"),
        ("%d candidates -> %d inliers (%.1f%%), %d rejected"
         % (vres.n_candidates, vres.n_inliers, 100 * vres.inlier_ratio, vres.n_outliers))
        if vres.ok else vres.reason,
        timer.t["verify"], vres.detail))
    images["verified"] = viz.draw_matches(a, b, kp_disp, p_ref, vres.inlier_mask,
                                          max_lines=350)

    # --------------------------------------- 6. correspondence selection
    inl = vres.inlier_mask
    p_src_in, p_ref_in, sc_in = p_src[inl], p_ref[inl], scores[inl]
    with timer("select"):
        # The usability mask also fixes the coverage METRIC, not just the
        # matching: two thirds of an OHRC window is shadow, so scoring coverage
        # against every grid cell would refuse a good registration whose matches
        # sit in the only matchable region. Cells with nothing matchable in them
        # are excluded from the denominator.
        cov_mask = mask.usable if opts.mask_mode != "none" else None
        sres = spatial.select(p_src_in, sc_in, img_src.shape[:2],
                              mode=opts.selection_mode,
                              grid=(opts.grid, opts.grid), per_cell=opts.per_cell,
                              usable_mask=cov_mask)
    p_src_sel, p_ref_sel = p_src_in[sres.keep], p_ref_in[sres.keep]
    stages.append(Stage(
        "Correspondence selection",
        "ok" if opts.selection_mode == "grid" else "skipped",
        ("%d inliers -> %d retained (%s); coverage %.0f%% of %d matchable cells "
         "(top-K by confidence: %.0f%%; whole window: %.0f%%)"
         % (len(p_src_in), len(p_src_sel), sres.detail.get("status"),
            100 * sres.coverage_after, sres.n_eligible_cells,
            100 * sres.coverage_before, 100 * sres.coverage_whole_window)),
        timer.t["select"], sres.detail))

    grid = (opts.grid, opts.grid)
    p_top = p_src_in[sres.keep_topk] if sres.keep_topk is not None else p_src_in
    images["select_before"] = viz.draw_points_grid(
        a, p_top * pre_s.scale, grid, color=viz.INLIER,
        label="top-%d by confidence | coverage %.0f%%"
              % (len(p_top), 100 * sres.coverage_before))
    images["select_after"] = viz.draw_points_grid(
        a, p_src_sel * pre_s.scale, grid,
        color=viz.SELECTED if opts.selection_mode == "grid" else viz.INLIER,
        label="retained %d | coverage %.0f%%"
              % (len(p_src_sel), 100 * sres.coverage_after))

    # ------------------------------------------- 7. sub-pixel refinement
    H_coarse = register.refit(p_src_sel, p_ref_sel, opts.model_type)
    if H_coarse is None:
        H_coarse = vres.model
    rmse_coarse = metrics.reprojection_metrics(H_coarse, p_src_sel, p_ref_sel).get("rmse_px")
    refine_detail: dict = {"enabled": bool(opts.subpixel_enabled)}
    with timer("subpixel"):
        if opts.subpixel_enabled and H_coarse is not None and len(p_src_sel) >= 4:
            rr = subpixel.refine(img_src, img_ref, p_src_sel, p_ref_sel, H_coarse)
            if rr.n_accepted >= 4:
                p_src_sel, p_ref_sel = rr.p_src, rr.p_ref
            refine_detail.update(rr.detail)
            refine_detail["applied"] = rr.n_accepted >= 4
        else:
            refine_detail["applied"] = False
            refine_detail["reason"] = ("disabled" if not opts.subpixel_enabled
                                       else "no coarse transform or too few matches")
    stages.append(Stage(
        "Sub-pixel refinement",
        "ok" if refine_detail.get("applied") else "skipped",
        ("%d of %d correspondences refined, median shift %s px"
         % (refine_detail.get("n_refined", 0), len(p_src_sel),
            refine_detail.get("median_shift_px")))
        if refine_detail.get("applied") else str(refine_detail.get("reason", "not applied")),
        timer.t["subpixel"], refine_detail))

    # ------------------------------------- final transform + hold-out split
    with timer("estimate"):
        fit_m, hold_m = _split_holdout(len(p_src_sel), opts.holdout_fraction, opts.seed)
        H_final = register.refit(p_src_sel[fit_m], p_ref_sel[fit_m], opts.model_type)
        if H_final is None:
            H_final = register.refit(p_src_sel, p_ref_sel, opts.model_type)
        if H_final is None:
            H_final = vres.model
        heldout = None
        if hold_m.any() and H_final is not None:
            heldout = metrics.reprojection_metrics(H_final,
                                                   p_src_sel[hold_m], p_ref_sel[hold_m])
        reg = register.register(img_src, img_ref, H_final, opts.model_type) \
            if H_final is not None else None
    if reg is not None:
        images["overlay"] = reg["anaglyph"]
        images["checker"] = reg["checker"]
        images["difference"] = reg["diff"]
        images["footprint"] = viz.draw_footprint(b, H_final, img_src.shape[:2])

    # -------------------------------------------------------- 8. metrics
    with timer("metrics"):
        repro = metrics.reprojection_metrics(H_final, p_src_sel[fit_m], p_ref_sel[fit_m])
        gt = None
        if pair.gt_transform is not None and H_final is not None:
            gt = {}
            gt.update(metrics.corner_error(H_final, pair.gt_transform, img_src.shape[:2]))
            gt.update(metrics.ground_truth_residuals(H_final, pair.gt_transform, p_src_sel))
            if gt.get("corner_error_px") is not None:
                gt["corner_error_m"] = round(gt["corner_error_px"] * gsd_r, 4)
        disagree = _transform_disagreement_m(H_final, prior, img_src.shape[:2], gsd_r)
        ncc_before = register.identity_baseline_ncc(
            preprocess.to_gray_u8(img_src), preprocess.to_gray_u8(img_ref))

        m = metrics.summarise(
            candidates=n_cand, verified=int(inl.sum()), selected=len(p_src_sel),
            timings=timer.t,
            coverage={
                "spatial_coverage": round(sres.coverage_after, 4),
                "spatial_coverage_before": round(sres.coverage_before, 4),
                "spatial_coverage_all_inliers": round(sres.coverage_all_inliers, 4),
                "spatial_coverage_whole_window": round(sres.coverage_whole_window, 4),
                "matchable_cells": sres.n_eligible_cells,
                "grid_cells": sres.n_cells,
                "spatial_dispersion": round(sres.dispersion_after, 4),
                "spatial_dispersion_before": round(sres.dispersion_before, 4),
            },
            extra={
                "candidates_before_mask": n_raw,
                "dropped_by_mask": n_dropped,
                "usable_fraction": round(mask.usable_fraction, 4),
                "mask_source": mask.source,
                "mask_is_prediction": mask.is_prediction,
                "mask_is_synthetic_terrain": mask.is_synthetic_terrain,
                "reprojection": repro,
                "reprojection_rmse_px_before_subpixel": rmse_coarse,
                "heldout": heldout,
                "heldout_count": int(hold_m.sum()),
                "ground_truth": gt,
                "ground_truth_available": pair.has_ground_truth,
                "geometry_prior_disagreement": disagree,
                "ncc_before": None if np.isnan(ncc_before) else round(float(ncc_before), 4),
                "ncc_after": None if (reg is None or np.isnan(reg["ncc_after"]))
                             else round(float(reg["ncc_after"]), 4),
                "overlap_fraction": None if reg is None else round(reg["overlap_fraction"], 4),
                "keypoints_source": mres.n_features_src,
                "keypoints_reference": mres.n_features_ref,
                "fit_count": int(fit_m.sum()),
            })
    stages.append(Stage("Metrics", "ok", _metric_note(m), timer.t["metrics"], {}))

    # ------------------------------------------- 9. trust / refusal decision
    status, confidence, reasons = decide(m, vres, opts, mask, H_final)
    stages.append(Stage(
        "Trust and refusal decision",
        {"accepted": "ok", "warning": "warning", "refused": "failed"}[status],
        "%s: %s" % (status.upper(), "; ".join(reasons)), 0.0,
        {"thresholds": THRESHOLDS, "calibrated": False}))

    return RegistrationResult(
        pair_id=pair.pair_id, status=status,
        transform=None if H_final is None else [[float(v) for v in r] for r in H_final],
        model_type=opts.model_type, confidence=confidence, reasons=reasons,
        metrics=m, stages=[s.__dict__ for s in stages], options=opts.__dict__,
        pair=pair.meta(), warnings=warnings, images=images)


# --------------------------------------------------------------------------- #

def decide(m: dict, vres, opts: Options, mask, H) -> tuple[str, float, list[str]]:
    """Rule-based accept / warn / refuse, with every reason stated.

    Deliberately transparent and deliberately uncalibrated. The intended
    research step is to fit this decision on labelled overlapping and
    non-overlapping pairs and report its precision and recall; until that is
    done the thresholds are engineering judgement and are labelled as such
    everywhere they appear.
    """
    hard: list[str] = []
    soft: list[str] = []
    T = THRESHOLDS

    if H is None or not vres.ok:
        hard.append(vres.reason or "no transform could be estimated")
    if m["inliers"] < T["min_inliers"]:
        hard.append("only %d geometrically consistent matches (need >= %d)"
                    % (m["inliers"], T["min_inliers"]))
    if m["inlier_ratio"] < T["min_inlier_ratio"]:
        hard.append("inlier ratio %.1f%% below the %.0f%% floor"
                    % (100 * m["inlier_ratio"], 100 * T["min_inlier_ratio"]))
    if m["spatial_coverage"] < T["min_spatial_coverage"]:
        hard.append("matches cover only %.0f%% of the grid"
                    % (100 * m["spatial_coverage"]))
    r = (m.get("reprojection") or {}).get("rmse_px")
    if r is not None and r > T["max_rmse_multiple_of_threshold"] * opts.ransac_threshold:
        hard.append("reprojection RMSE %.2f px exceeds %gx the RANSAC threshold"
                    % (r, T["max_rmse_multiple_of_threshold"]))

    ho = (m.get("heldout") or {}).get("rmse_px")
    if ho is not None:
        if ho > T["max_heldout_rmse_px"]:
            hard.append("held-out RMSE %.2f px exceeds %.1f px - the transform does "
                        "not generalise off the fitted points"
                        % (ho, T["max_heldout_rmse_px"]))
        elif ho > T["warn_heldout_rmse_px"]:
            soft.append("held-out RMSE %.2f px is above the %.1f px comfort level"
                        % (ho, T["warn_heldout_rmse_px"]))
    else:
        soft.append("no hold-out set was large enough to test generalisation")

    dis = m.get("geometry_prior_disagreement")
    if dis and dis.get("corner_disagreement_m") is not None:
        d = dis["corner_disagreement_m"]
        if d > T["max_prior_disagreement_m"]:
            soft.append("estimate disagrees with the delivered geometry by %.0f m; "
                        "one of the two is wrong (the delivered geometry is "
                        "system-level and unrefined, so this is not proof of error)"
                        % d)

    ovl = m.get("overlap_fraction")
    if ovl is not None and ovl < T["warn_overlap_fraction"]:
        soft.append("the registered source covers only %.0f%% of the reference frame, "
                    "so the transform is extrapolated over the remainder; with "
                    "correspondences confined to part of the frame, rotation and "
                    "translation trade off against each other" % (100 * ovl))
    if mask.is_synthetic_terrain:
        soft.append("a SYNTHETIC terrain model contributed to this result; it is not "
                    "evidence about the real surface")
    if opts.mask_mode != "none" and m.get("usable_fraction", 1.0) < T["warn_usable_fraction"]:
        soft.append("only %.1f%% of the source window was judged matchable"
                    % (100 * m["usable_fraction"]))

    if hard:
        return "refused", 0.0, hard + soft

    # Confidence: a bounded, monotone combination of the quantities the decision
    # already uses. It is a summary of those numbers, not a calibrated probability.
    ratio_term = min(m["inlier_ratio"] / 0.6, 1.0)
    cover_term = min(m["spatial_coverage"] / 0.7, 1.0)
    rmse_term = 1.0 / (1.0 + (r or 0.0) / max(opts.ransac_threshold, 1e-6))
    ho_term = 1.0 if ho is None else 1.0 / (1.0 + ho / T["warn_heldout_rmse_px"])
    conf = float(round(0.30 * ratio_term + 0.25 * cover_term
                       + 0.25 * rmse_term + 0.20 * ho_term, 4))
    if soft:
        return "warning", conf, soft
    return "accepted", conf, ["all acceptance checks passed"]


def _metric_note(m: dict) -> str:
    bits = ["inlier ratio %.1f%%" % (100 * m["inlier_ratio"])]
    r = (m.get("reprojection") or {}).get("rmse_px")
    if r is not None:
        bits.append("reprojection RMSE %.3f px" % r)
    ho = (m.get("heldout") or {}).get("rmse_px")
    if ho is not None:
        bits.append("held-out RMSE %.3f px" % ho)
    gt = m.get("ground_truth") or {}
    if gt.get("corner_error_px") is not None:
        bits.append("ground-truth corner error %.3f px" % gt["corner_error_px"])
    dis = m.get("geometry_prior_disagreement") or {}
    if dis.get("corner_disagreement_m") is not None:
        bits.append("prior disagreement %.0f m" % dis["corner_disagreement_m"])
    bits.append("coverage %.0f%%" % (100 * m["spatial_coverage"]))
    return ", ".join(bits)


def _refused(pair, opts, stages, timer, reasons, images, warnings):
    m = metrics.summarise(candidates=0, verified=0, selected=0, timings=timer.t,
                          coverage={"spatial_coverage": 0.0,
                                    "spatial_coverage_before": 0.0,
                                    "spatial_coverage_all_inliers": 0.0,
                                    "spatial_dispersion": 0.0,
                                    "spatial_dispersion_before": 0.0},
                          extra={"reprojection": {}, "heldout": None,
                                 "ground_truth": None,
                                 "ground_truth_available": pair.has_ground_truth,
                                 "geometry_prior_disagreement": None})
    return RegistrationResult(
        pair_id=pair.pair_id, status="refused", transform=None,
        model_type=opts.model_type, confidence=0.0, reasons=reasons,
        metrics=m, stages=[s.__dict__ for s in stages], options=opts.__dict__,
        pair=pair.meta(), warnings=warnings, images=images)


def viz_mask(img: np.ndarray, mask) -> np.ndarray:
    """Source window with the predicted-unusable region tinted."""
    base = cv2.cvtColor(preprocess.to_gray_u8(img), cv2.COLOR_GRAY2BGR)
    if mask.source == "none":
        cv2.putText(base, "no masking (baseline)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 210), 1, cv2.LINE_AA)
        return base
    bad = ~mask.usable
    tint = base.copy()
    tint[bad] = viz.OUTLIER
    out = cv2.addWeighted(base, 0.62, tint, 0.38, 0)
    cv2.putText(out, "%s: %.0f%% matchable" % (mask.source, 100 * mask.usable_fraction),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 200, 255), 1, cv2.LINE_AA)
    if mask.is_synthetic_terrain:
        cv2.putText(out, "SYNTHETIC TERRAIN - not science", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 130, 233), 1, cv2.LINE_AA)
    return out
