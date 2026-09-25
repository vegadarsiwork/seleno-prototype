"""Frozen spatial folds and final, reproducible evaluation of an export model.

Fold membership depends only on the initial source working-grid coordinates,
image shape and seed. It never depends on a match, its residual, or a model.
Correspondence extraction is not ground truth: retain all test correspondences,
including wrong ones, and describe the resulting score accordingly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import warp_model as WM
from .methods import Correspondences
from .coordinates import grid_to_reference, project


class SpatialSplit:
    FIT, VALIDATION, TEST = 0, 1, 2

    def __init__(self, shape, holdout=0.35, seed=0, grid=8, frame=None):
        """`grid` is a cell count per side, or (rows, cols) for non-square cells.

        `frame`, when given, lays the cells along the overlap instead of the
        whole working grid: {"origin": (x, y), "axes": [[ux, uy], [vx, vy]],
        "extent": (u0, v0, u1, v1)}. A strip crossing a square frame then gets
        cells along its own length, so held-out and fit cells alternate down
        the strip rather than one cell holding out a whole end of it.
        """
        if not 0 < holdout < 0.8:
            raise ValueError("holdout must be between 0 and 0.8")
        self.shape = tuple(shape)
        self.frame = frame
        self.grid_shape = ((int(grid), int(grid)) if np.isscalar(grid)
                           else (int(grid[0]), int(grid[1])))
        self.grid = grid
        self.seed = seed
        n = self.grid_shape[0] * self.grid_shape[1]
        order = np.random.default_rng(seed).permutation(n)
        n_test = max(1, round(holdout * len(order)))
        n_val = max(1, round(0.15 * (len(order) - n_test)))
        self.folds = np.zeros(n, np.uint8)
        self.folds[order[:n_test]] = self.TEST
        self.folds[order[n_test:n_test + n_val]] = self.VALIDATION

    def cells(self, points):
        h, w = self.shape
        gy, gx = self.grid_shape
        points = np.asarray(points, np.float64).reshape(-1, 2)
        if self.frame is not None:
            o = np.asarray(self.frame["origin"], np.float64)
            ax = np.asarray(self.frame["axes"], np.float64)
            u0, v0, u1, v1 = self.frame["extent"]
            d = points - o
            u, v = d @ ax[0], d @ ax[1]
            x = np.clip(np.floor((u - u0) * gx / max(u1 - u0, 1e-9)).astype(int), 0, gx - 1)
            y = np.clip(np.floor((v - v0) * gy / max(v1 - v0, 1e-9)).astype(int), 0, gy - 1)
            return y * gx + x
        x = np.clip((points[:, 0] * gx / w).astype(int), 0, gx - 1)
        y = np.clip((points[:, 1] * gy / h).astype(int), 0, gy - 1)
        return y * gx + x

    def labels(self, points):
        return self.folds[self.cells(points)]

    def fit_mask(self):
        y, x = np.indices(self.shape)
        return self.labels(np.column_stack([x.ravel(), y.ravel()])).reshape(self.shape) == self.FIT

    def summary(self):
        return {"kind": "fixed spatial cells in initial source working coordinates",
                "shape": list(self.shape), "grid": list(self.grid_shape), "seed": self.seed,
                "frame": self.frame,
                "cell_folds": self.folds.tolist(),
                "labels": {"0": "fit", "1": "validation", "2": "test"}}


def subset(corr, keep):
    detail = dict(corr.detail)
    for name in ("back_x", "back_y"):
        if name in detail:
            detail[name] = np.asarray(detail[name])[keep]
    return Correspondences(corr.src[keep], corr.ref[keep], corr.confidence[keep],
                           corr.method, corr.kind, detail)


def partition(corr, split):
    labels = split.labels(corr.src)
    return tuple(subset(corr, labels == fold) for fold in range(3))


def matrix_digest(matrix):
    return hashlib.sha256(np.asarray(matrix, dtype="<f8").tobytes()).hexdigest()


def error_statistics(vectors):
    """Summarize every measured error; invalid observations fail the assessment.

    Keeping the two components also exposes a systematic offset which a count
    of RANSAC inliers, or a median alone, can hide.
    """
    vectors = np.asarray(vectors, np.float64).reshape(-1, 2)
    valid = np.isfinite(vectors).all(axis=1)
    invalid = int((~valid).sum())
    stats = {"n": len(vectors), "valid_n": int(valid.sum()), "invalid_n": invalid, "rmse_px": None,
             "median_px": None, "p90_px": None, "max_px": None,
             "within_1_px": None, "bias_xy_px": None, "bias_px": None}
    # Report the available evidence without concealing invalid predictions.
    # Acceptance still fails on invalid_n, even when the valid subset is exact.
    if not valid.any():
        return stats
    vectors = vectors[valid]
    distances = np.linalg.norm(vectors, axis=1)
    bias = np.mean(vectors, axis=0)
    return stats | {"rmse_px": float(np.sqrt(np.mean(distances ** 2))),
                    "median_px": float(np.median(distances)),
                    "p90_px": float(np.percentile(distances, 90)),
                    "max_px": float(np.max(distances)),
                    "within_1_px": float(np.mean(distances < 1.0)),
                    "bias_xy_px": bias.tolist(), "bias_px": float(np.linalg.norm(bias))}


def score_export(job_dir, test, split, *, working_to_source=None, check_points=None,
                 check_point_rule=None):
    """Only call after ALL model decisions and transform.json are finalized.

    No inlier filtering, RANSAC, selection, or refitting is allowed here. Reload
    the serialized matrix rather than evaluating a parallel in-memory estimate.

    Every held-out correspondence is scored (`held_out_*`). `check_points`, when
    given, marks the held-out correspondences that passed a screen decided
    WITHOUT the exported model - a mismatch that disagrees with its own
    held-out neighbours says nothing about the transform, while a model error
    moves a whole neighbourhood and survives the screen. Those are scored again
    as `check_point_*`, and how many the screen removed is reported with them.
    """
    with open(os.path.join(job_dir, "transform.json")) as fh:
        transform = json.load(fh)
    matrix = np.asarray(transform["matrix"], dtype=np.float64)
    evidence = {"split": split.summary(), "coordinates": "working-grid pixel centres",
                "source": test.src.astype(np.float64).tolist(),
                "reference": test.ref.astype(np.float64).tolist(),
                "matrix_sha256": matrix_digest(matrix),
                "transform_sha256": WM.digest(transform),
                "applied_model": transform["application"],
                "residual_definition": "symmetric forward/backward transfer error of applied model",
                "basis": "all held-out matcher correspondences; not external ground truth",
                "model_selected_without_test": True}
    acc = {"held_out_n": len(test), "held_out_rmse_px": None,
           "held_out_median_px": None, "held_out_p90_px": None,
           "held_out_exactly_on_model_n": None,
           "evaluation_file": "evaluation.json", "matrix_sha256": evidence["matrix_sha256"],
           "evaluation_basis": evidence["basis"],
           "transform_sha256": evidence["transform_sha256"],
           "applied_model": evidence["applied_model"]}
    if len(test) >= 3:
        residual = WM.residuals(transform, test.src.astype(np.float64), test.ref.astype(np.float64))
        working_stats = error_statistics(np.column_stack([residual, np.zeros(len(residual))]))
        acc.update(held_out_rmse_px=working_stats["rmse_px"],
                   held_out_median_px=working_stats["median_px"],
                   held_out_p90_px=working_stats["p90_px"],
                   held_out_invalid_n=working_stats["invalid_n"],
                   # Points exactly on the scored model. Expected only when the model
                   # really is exact (a same-file pair); otherwise a point was built
                   # through the model rather than measured against it.
                   held_out_exactly_on_model_n=int((residual < 1e-6).sum()))
        reference_errors = ((WM.forward_points(transform, test.src) - test.ref)
                            @ grid_to_reference(transform.get("frame", {}))[:2, :2].T)
        reference_stats = error_statistics(reference_errors)
        acc.update({"held_out_reference_" + key: value for key, value in reference_stats.items()})
        evidence["reference_errors"] = reference_stats
        # A symmetric transfer distance mixes the two grids' units. Evaluate
        # backward transfer in actual source detector coordinates instead of
        # multiplying that mixed distance by a nominal GSD ratio.
        if working_to_source is not None:
            if all(key in test.detail for key in ("back_x", "back_y")):
                observed = np.column_stack([test.detail["back_x"], test.detail["back_y"]])
            else:
                observed = np.asarray(working_to_source(test.src), np.float64)
            predicted = np.asarray(working_to_source(WM.inverse_points(transform, test.ref)), np.float64)
            errors = predicted - observed
            stats = error_statistics(errors)
            acc.update({"held_out_source_" + key: value for key, value in stats.items()})
            acc["source_error_definition"] = "backward transfer in native source pixel centres"
            # JSON null records invalid predictions; they are never dropped.
            evidence["native_source"] = np.where(np.isfinite(observed), observed, None).tolist()
            evidence["predicted_native_source"] = np.where(np.isfinite(predicted), predicted, None).tolist()
            evidence["source_errors"] = stats
        if check_points is not None:
            cp = np.asarray(check_points, bool)
            evidence["check_point"] = cp.tolist()
            evidence["check_point_rule"] = check_point_rule
            acc.update(check_point_n=int(cp.sum()), check_point_rejected_n=int((~cp).sum()),
                       check_point_fraction=float(cp.mean()) if len(cp) else None,
                       check_point_rule=check_point_rule)
            if cp.sum() >= 3:
                r = residual[cp]
                cwork = error_statistics(np.column_stack([r, np.zeros(len(r))]))
                acc.update(check_point_rmse_px=cwork["rmse_px"],
                           check_point_median_px=cwork["median_px"],
                           check_point_p90_px=cwork["p90_px"],
                           check_point_invalid_n=cwork["invalid_n"])
                cref = error_statistics(reference_errors[cp])
                acc.update({"check_point_reference_" + key: value for key, value in cref.items()})
                evidence["check_point_reference_errors"] = cref
                if working_to_source is not None:
                    cstats = error_statistics(errors[cp])
                    acc.update({"check_point_source_" + key: value
                                for key, value in cstats.items()})
                    evidence["check_point_source_errors"] = cstats
    with open(os.path.join(job_dir, "evaluation.json"), "w") as fh:
        json.dump(evidence, fh, indent=1, allow_nan=False)
    return acc


def _precision_failures(stats, prefix="source"):
    failures = []
    if stats.get("n", 0) < 12:
        failures.append("fewer than 12 %s check points" % prefix)
    if stats.get("invalid_n", 0):
        failures.append("%s check points include invalid model predictions" % prefix)
    for key, limit, below in (("rmse_px", 1., True), ("p90_px", 1., True),
                              ("within_1_px", .95, False), ("bias_px", .5, True)):
        value = stats.get(key)
        if value is None or not np.isfinite(value):
            failures.append("%s %s is unknown" % (prefix, key))
        elif (value >= limit if below else value < limit):
            failures.append("%s %s %.4g fails %s %.4g" %
                            (prefix, key, value, "<" if below else ">=", limit))
    return failures


def acceptance_report(accuracy, distribution, independent=None):
    """Keep execution, matcher consistency and independent accuracy separate.

    Thresholds are fixed before a run. A result cannot pass by increasing its
    tolerance, removing difficult points, or quoting only a working-grid RMSE.
    Passing internal checks never certifies accuracy on real images.
    """
    prefix = ("check_point_source_" if "check_point_source_n" in accuracy
              else "held_out_source_")
    stats = {key.removeprefix(prefix): value for key, value in accuracy.items()
             if key.startswith(prefix)}
    reasons = _precision_failures(stats, "check-point source" if prefix.startswith("check")
                                  else "held-out source")
    if (accuracy.get("held_out_source_invalid_n", 0) or accuracy.get("held_out_invalid_n", 0)
            or accuracy.get("held_out_reference_invalid_n", 0)):
        reasons.append("held-out correspondences include invalid model predictions")
    fraction = accuracy.get("check_point_fraction")
    if fraction is not None and fraction < .8:
        reasons.append("only %.0f%% of held-out correspondences are usable check points"
                       % (100 * fraction))
    coverage, extrap = (distribution.get("coverage_fraction"),
                       distribution.get("extrapolation_fraction"))
    if coverage is None or not np.isfinite(coverage) or coverage < .5:
        reasons.append("fit points cover less than 50% of eligible grid cells")
    if extrap is None or not np.isfinite(extrap) or extrap > .25:
        reasons.append("more than 25% of eligible area is outside fit-point support")
    external = ["independent reference points were not supplied"]
    if independent is not None:
        external = list(independent.get("reasons", []))
        if not independent.get("passed", False) and not external:
            external = ["independent reference assessment did not pass"]
    verified = not reasons and not external
    return {"passed": not reasons, "independently_verified": verified,
            "status": "verified" if verified else "unverified" if not reasons else "failed",
            "reasons": reasons, "independent_reasons": external,
            "basis": ("held-out check points (model-independent neighbour screen)"
                      if prefix.startswith("check") else "every held-out correspondence"),
            "thresholds": {"minimum_check_points": 12, "check_point_fraction_at_least": .8,
                           "source_rmse_px_below": 1.,
                           "source_p90_px_below": 1., "within_1_source_px_at_least": .95,
                           "source_bias_px_below": .5, "coverage_at_least": .5,
                           "extrapolation_at_most": .25},
            "meaning": "internal consistency and spatial support; independent check points required for verified accuracy"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ground_truth(path, *, source_path, reference_path, source_shape, reference_shape):
    """Validate image-bound, independently acquired detector control points.

    Independence and uncertainty are supplied by the reference author, not
    inferred from small residuals. The manifest digest records that assertion;
    file digests ensure points cannot silently be applied to another image.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema_version") != 1 or document.get("coordinates") != "native_pixel_centres":
        raise ValueError("ground truth requires schema_version 1 and native_pixel_centres coordinates")
    provenance = document.get("provenance", {})
    if provenance.get("kind") not in ("manual_control_points", "surveyed_control_points", "synthetic_transform"):
        raise ValueError("ground truth provenance kind must identify independently acquired controls")
    if provenance.get("independent_of_registration") is not True or not str(provenance.get("reference", "")).strip():
        raise ValueError("ground truth requires an independent acquisition assertion and reference")
    uncertainty = provenance.get("uncertainty_source_px")
    if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)) or not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("ground truth requires a finite nonnegative uncertainty_source_px")
    for key, expected, shape in (("source", source_path, source_shape),
                                  ("reference", reference_path, reference_shape)):
        entry = document.get(key, {})
        supplied = (path.parent / entry.get("path", "")).resolve()
        if supplied != Path(expected).resolve():
            raise ValueError("ground truth %s path does not identify the registered image" % key)
        if entry.get("shape") != list(shape):
            raise ValueError("ground truth %s shape does not match the registered image" % key)
        if entry.get("sha256") != file_sha256(expected):
            raise ValueError("ground truth %s SHA-256 does not match the registered image" % key)
    points = document.get("points", [])
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("ground truth requires at least three paired points")
    ids = [p.get("id") for p in points]
    if any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("ground truth point IDs must be unique nonempty strings")
    for key, shape in (("source", source_shape), ("reference", reference_shape)):
        xy = np.asarray([point.get(key) for point in points], np.float64)
        if xy.shape != (len(points), 2) or not np.isfinite(xy).all():
            raise ValueError("ground truth %s points must be finite x,y pairs" % key)
        if np.any(xy < 0) or np.any(xy[:, 0] > shape[1] - 1) or np.any(xy[:, 1] > shape[0] - 1):
            raise ValueError("ground truth %s points lie outside the native image" % key)
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError("ground truth contains duplicate %s points" % key)
    return document, hashlib.sha256(raw).hexdigest()


def score_ground_truth(job_dir, manifest_path, *, working_to_source, source_path,
                       reference_path, source_shape, reference_shape):
    """Score the finalized export against native independent reference points.

    This entry point is deliberately separate from fitting and model selection.
    No RANSAC or error-dependent filtering is permitted here. A synthetic
    control experiment is explicitly distinguished from observed-image truth.
    """
    from .. import spatial

    document, manifest_digest = load_ground_truth(manifest_path, source_path=source_path,
        reference_path=reference_path, source_shape=source_shape, reference_shape=reference_shape)
    with open(os.path.join(job_dir, "transform.json")) as fh:
        transform = json.load(fh)
    source = np.asarray([p["source"] for p in document["points"]], np.float64)
    reference = np.asarray([p["reference"] for p in document["points"]], np.float64)
    reference_working = project(np.linalg.inv(grid_to_reference(transform["frame"])), reference)
    predicted = np.asarray(working_to_source(WM.inverse_points(transform, reference_working)), np.float64)
    stats = error_statistics(predicted - source)
    coverage = spatial.cell_coverage(source, source_shape, (4, 4))
    extrapolation = spatial.extrapolation_fraction(source, source_shape, (4, 4))
    reasons = _precision_failures(stats, "independent source")
    uncertainty = document["provenance"]["uncertainty_source_px"]
    if uncertainty > .25:
        reasons.append("reference uncertainty exceeds 0.25 source pixels")
    if coverage < .5 or extrapolation > .25:
        reasons.append("independent controls do not cover enough of the native source image")
    report = {"passed": not reasons, "reasons": reasons,
              "kind": document["provenance"]["kind"],
              "independence": "asserted by reference author; not inferred from residuals",
              "provenance": document["provenance"], "manifest_sha256": manifest_digest,
              "transform_sha256": WM.digest(transform), "statistics": stats,
              "distribution": {"grid": [4, 4], "coverage_fraction": coverage,
                               "extrapolation_fraction": extrapolation},
              "point_ids": [p["id"] for p in document["points"]],
              "source": source.tolist(), "reference": reference.tolist(),
              "predicted_source": np.where(np.isfinite(predicted), predicted, None).tolist(),
              "residual_definition": "backward transfer in native source pixel centres; all supplied points"}
    with open(os.path.join(job_dir, "ground_truth_evaluation.json"), "w") as fh:
        json.dump(report, fh, indent=1, allow_nan=False)
    return report
