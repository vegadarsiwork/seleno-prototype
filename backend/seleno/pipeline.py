"""The Seleno pipeline.

    source + reference
        -> preprocessing
        -> feature / correspondence extraction
        -> geometric verification (robust estimator)
        -> spatially distributed match selection
        -> transformation refit + registration
        -> metrics

Every stage records its own runtime and its own outputs. Nothing is precomputed
or cached between runs, so the numbers the UI shows are the numbers this run
produced.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import matchers, metrics, preprocess, register, spatial, verify, viz


@dataclass
class Options:
    matcher: str = "sift"
    preprocess_enabled: bool = True
    use_clahe: bool = True
    use_flatten: bool = True
    use_denoise: bool = False
    harmonise_gsd: bool = True          # resample the finer image toward the coarser GSD
    verify_enabled: bool = True
    model_type: str = "homography"
    ransac_threshold: float = 3.0
    spatial_enabled: bool = True
    grid: int = 6
    per_cell: int = 3
    ratio: float = 0.80
    mutual_check: bool = True
    max_features: int = 8000

    @classmethod
    def from_dict(cls, d: dict) -> "Options":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class StageLog:
    name: str
    status: str = "ok"                  # ok | skipped | failed
    note: str = ""
    seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        self._k = key
        return self

    def __enter__(self):
        self._s = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.t[self._k] = self.t.get(self._k, 0.0) + (time.perf_counter() - self._s)
        return False


def run(pair: dict, img_src: np.ndarray, img_ref: np.ndarray, opts: Options) -> dict:
    """Execute the pipeline on one pair. Returns a JSON-safe dict plus images."""
    timer = Timer()
    stages: list[StageLog] = []
    images: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    gsd_s = float(pair.get("gsd_source_m") or 0) or None
    gsd_r = float(pair.get("gsd_reference_m") or 0) or None

    # ------------------------------------------------- 1. preprocessing
    with timer("preprocess"):
        target = None
        if opts.harmonise_gsd and gsd_s and gsd_r and gsd_r > gsd_s * 1.05:
            target = gsd_r                     # bring the finer source down to the coarser GSD
        pre_s = preprocess.preprocess(
            img_src, enabled=opts.preprocess_enabled, use_clahe=opts.use_clahe,
            use_flatten=opts.use_flatten, use_denoise=opts.use_denoise,
            src_gsd=gsd_s, target_gsd=target)
        pre_r = preprocess.preprocess(
            img_ref, enabled=opts.preprocess_enabled, use_clahe=opts.use_clahe,
            use_flatten=opts.use_flatten, use_denoise=opts.use_denoise)
    a, b = pre_s.image, pre_r.image
    stages.append(StageLog(
        "Preprocessing", "ok" if opts.preprocess_enabled else "skipped",
        " | ".join(pre_s.steps), timer.t["preprocess"],
        {"source": pre_s.stats, "reference": pre_r.stats,
         "source_steps": pre_s.steps, "reference_steps": pre_r.steps,
         "source_coordinate_scale": pre_s.scale}))

    # Correspondences are found in preprocessed coordinates; this maps them back
    # to original source pixels so every reported number is in real image space.
    inv_scale = 1.0 / pre_s.scale if pre_s.scale else 1.0

    # ------------------------------------------------- 2. correspondence
    mres = None
    try:
        with timer("match"):
            kw = {"ratio": opts.ratio, "n_features": opts.max_features,
                  "mutual_check": opts.mutual_check}
            if opts.matcher == "loftr":
                kw = {}
            mres = matchers.run_matcher(opts.matcher, a, b, **kw)
        stages.append(StageLog(
            "Feature extraction and candidate matching", "ok",
            "%d keypoints (source), %d (reference) -> %d candidate correspondences"
            % (mres.n_features_src, mres.n_features_ref, len(mres.kp_src)),
            timer.t["match"], mres.detail))
    except Exception as exc:
        stages.append(StageLog("Feature extraction and candidate matching", "failed",
                               str(exc), timer.t.get("match", 0.0)))
        return _fail(pair, opts, stages, timer, str(exc), images)

    p_src = mres.kp_src * inv_scale             # original source pixel coordinates
    p_ref = mres.kp_ref
    n_cand = len(p_src)

    images["candidates"] = viz.draw_matches(a, b, mres.kp_src, p_ref,
                                            np.ones(n_cand, bool), max_lines=300,
                                            color_in=(200, 200, 210), draw_outliers=False)

    # ------------------------------------------------- 3. geometric verification
    with timer("verify"):
        vres = verify.verify(p_src, p_ref, model_type=opts.model_type,
                             threshold=opts.ransac_threshold, enabled=opts.verify_enabled)
    expected_scale = 1.0
    if gsd_s and gsd_r:
        # after harmonisation the source already sits at the reference GSD
        expected_scale = (gsd_s / gsd_r) if target is None else 1.0
    if vres.ok:
        warnings.extend(verify.plausibility(vres.model, opts.model_type, expected_scale))
    stages.append(StageLog(
        "Geometric verification", "ok" if vres.ok else ("skipped" if not opts.verify_enabled else "failed"),
        ("%d candidates -> %d inliers (%.1f%%), %d rejected"
         % (vres.n_candidates, vres.n_inliers, 100 * vres.inlier_ratio, vres.n_outliers))
        if vres.ok else vres.reason,
        timer.t["verify"], vres.detail))

    images["verified"] = viz.draw_matches(a, b, mres.kp_src, p_ref, vres.inlier_mask,
                                          max_lines=350)

    # ------------------------------------------------- 4. spatial selection
    inl = vres.inlier_mask
    p_src_in, p_ref_in = p_src[inl], p_ref[inl]
    sc_in = mres.score[inl] if len(mres.score) == n_cand else np.ones(len(p_src_in), np.float32)
    with timer("spatial"):
        sres = spatial.select(p_src_in, sc_in, img_src.shape[:2], enabled=opts.spatial_enabled,
                              grid=(opts.grid, opts.grid), per_cell=opts.per_cell)
    p_src_sel, p_ref_sel = p_src_in[sres.keep], p_ref_in[sres.keep]
    stages.append(StageLog(
        "Spatial match selection", "ok" if opts.spatial_enabled else "skipped",
        ("%d inliers -> %d retained; grid coverage %.1f%% (top-%d by confidence) "
         "vs %.1f%% (spatially selected)"
         % (len(p_src_in), len(p_src_sel), 100 * sres.coverage_before,
            len(p_src_sel), 100 * sres.coverage_after)),
        timer.t["spatial"], sres.detail))

    grid = (opts.grid, opts.grid)
    p_top = p_src_in[sres.keep_topk] if sres.keep_topk is not None else p_src_in
    images["spatial_before"] = viz.draw_points_grid(
        a, p_top * pre_s.scale, grid, color=viz.INLIER,
        label="top-%d by confidence  |  cell coverage %.0f%%"
              % (len(p_top), 100 * sres.coverage_before))
    images["spatial_after"] = viz.draw_points_grid(
        a, p_src_sel * pre_s.scale, grid,
        color=viz.SELECTED if opts.spatial_enabled else viz.INLIER,
        label="%s: %d  |  cell coverage %.0f%%"
              % ("retained" if opts.spatial_enabled else "top-K used (spatial selection off)",
                 len(p_src_sel), 100 * sres.coverage_after))

    # ------------------------------------------------- 5. registration
    H_final = None
    reg = None
    with timer("register"):
        min_needed = {"similarity": 2, "affine": 3, "homography": 4}[opts.model_type]
        if len(p_src_sel) >= min_needed:
            H_final = register.refit(p_src_sel, p_ref_sel, opts.model_type)
        if H_final is None and vres.model is not None:
            H_final = vres.model
            if len(p_src_sel) >= min_needed:
                warnings.append("least-squares refit failed; using the robust estimate")
        if H_final is not None:
            reg = register.register(img_src, img_ref, H_final, opts.model_type)
    if reg is not None:
        images["overlay"] = reg["anaglyph"]
        images["checker"] = reg["checker"]
        images["difference"] = reg["diff"]
        images["footprint"] = viz.draw_footprint(b, H_final, img_src.shape[:2])
        stages.append(StageLog("Registration", "ok",
                               "%s refit on %d correspondences; %.1f%% of the reference frame covered"
                               % (verify.MODELS[opts.model_type], len(p_src_sel),
                                  100 * reg["overlap_fraction"]),
                               timer.t["register"],
                               {"model": verify.MODELS[opts.model_type],
                                "matrix": [[float(v) for v in r] for r in H_final]}))
    else:
        stages.append(StageLog("Registration", "failed",
                               "no transform could be estimated", timer.t["register"]))

    # ------------------------------------------------- 6. metrics
    with timer("metrics"):
        repro = metrics.reprojection_metrics(H_final, p_src_sel, p_ref_sel)
        gt = None
        H_gt = pair.get("gt_homography")
        if H_gt is not None and H_final is not None:
            H_gt = np.asarray(H_gt, np.float64)
            gt = {}
            gt.update(metrics.corner_error(H_final, H_gt, img_src.shape[:2]))
            gt.update(metrics.ground_truth_residuals(H_final, H_gt, p_src_sel))
        ncc_before = register.identity_baseline_ncc(preprocess.to_gray_u8(img_src),
                                                    preprocess.to_gray_u8(img_ref))
        cov = {
            "spatial_coverage_before": round(sres.coverage_before, 4),
            "spatial_coverage_all_inliers": round(sres.coverage_all_inliers, 4),
            "spatial_coverage": round(sres.coverage_after, 4),
            "spatial_dispersion_before": round(sres.dispersion_before, 4),
            "spatial_dispersion": round(sres.dispersion_after, 4),
        }
        gsd_out = gsd_r or gsd_s      # errors live in reference-frame pixels
        if gsd_out:
            if repro.get("rmse_px") is not None:
                repro["rmse_m"] = round(repro["rmse_px"] * gsd_out, 4)
            if gt and gt.get("corner_error_px") is not None:
                gt["corner_error_m"] = round(gt["corner_error_px"] * gsd_out, 4)
                gt["gt_rmse_m"] = round(gt["gt_rmse_px"] * gsd_out, 4)
        extra = {
            "reprojection": repro,
            "error_gsd_m_per_px": gsd_out,
            "ground_truth": gt,
            "ground_truth_available": bool(pair.get("ground_truth_available")),
            "ncc_before": None if np.isnan(ncc_before) else round(float(ncc_before), 4),
            "ncc_after": None if (reg is None or np.isnan(reg["ncc_after"]))
                         else round(float(reg["ncc_after"]), 4),
            "overlap_fraction": None if reg is None else round(reg["overlap_fraction"], 4),
            "keypoints_source": mres.n_features_src,
            "keypoints_reference": mres.n_features_ref,
        }
    summary = metrics.summarise(candidates=n_cand, verified=int(inl.sum()),
                                selected=len(p_src_sel), timings=timer.t,
                                coverage=cov, extra=extra)
    stages.append(StageLog("Metrics", "ok", _metrics_note(summary), timer.t["metrics"]))

    verdict = _verdict(summary, vres, opts, pair)
    return {
        # "a transform came out of the estimator" - NOT "the registration is
        # good". The accept/reject decision lives in `verdict.registered`.
        "ok": H_final is not None,
        "transform_estimated": H_final is not None,
        "pair_id": pair.get("id"),
        "options": opts.__dict__,
        "stages": [s.__dict__ for s in stages],
        "metrics": summary,
        "warnings": warnings,
        "verdict": verdict,
        "_images": images,
    }


def _metrics_note(s: dict) -> str:
    r = s["reprojection"].get("rmse_px")
    parts = ["inlier ratio %.1f%%" % (100 * s["inlier_ratio"])]
    if r is not None:
        parts.append("reprojection RMSE %.3f px" % r)
    gt = s.get("ground_truth") or {}
    if gt.get("corner_error_px") is not None:
        parts.append("ground-truth corner error %.3f px" % gt["corner_error_px"])
    parts.append("coverage %.0f%%" % (100 * s["spatial_coverage"]))
    return ", ".join(parts)


def _verdict(s: dict, vres, opts: Options, pair: dict) -> dict:
    """A conservative accept/reject decision with the reasons stated."""
    reasons = []
    accept = True
    if s["inliers"] < 12:
        accept = False
        reasons.append("only %d geometrically consistent matches (need >= 12)" % s["inliers"])
    if s["inlier_ratio"] < 0.15:
        accept = False
        reasons.append("inlier ratio %.1f%% is below the 15%% floor" % (100 * s["inlier_ratio"]))
    if s["spatial_coverage"] < 0.15:
        accept = False
        reasons.append("matches cover only %.0f%% of the grid" % (100 * s["spatial_coverage"]))
    r = s["reprojection"].get("rmse_px")
    if r is not None and r > 4 * opts.ransac_threshold:
        accept = False
        reasons.append("reprojection RMSE %.2f px exceeds 4x the RANSAC threshold" % r)
    if not vres.ok:
        accept = False
        reasons.append(vres.reason or "geometric verification did not produce a model")
    return {
        "registered": accept,
        "reasons": reasons or ["all acceptance checks passed"],
        "expected_outcome": pair.get("expected_outcome"),
    }


def _fail(pair, opts, stages, timer, msg, images):
    return {
        "ok": False,
        "transform_estimated": False,
        "pair_id": pair.get("id"),
        "options": opts.__dict__,
        "stages": [s.__dict__ for s in stages],
        "metrics": metrics.summarise(candidates=0, verified=0, selected=0, timings=timer.t,
                                     coverage={"spatial_coverage": 0.0,
                                               "spatial_coverage_before": 0.0,
                                               "spatial_dispersion": 0.0,
                                               "spatial_dispersion_before": 0.0},
                                     extra={"reprojection": {}, "ground_truth": None,
                                            "ground_truth_available": bool(pair.get("ground_truth_available"))}),
        "warnings": [],
        "verdict": {"registered": False, "reasons": [msg],
                    "expected_outcome": pair.get("expected_outcome")},
        "error": msg,
        "_images": images,
    }
