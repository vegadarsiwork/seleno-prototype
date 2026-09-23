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

import numpy as np

from . import warp_model as WM
from .methods import Correspondences


class SpatialSplit:
    FIT, VALIDATION, TEST = 0, 1, 2

    def __init__(self, shape, holdout=0.35, seed=0, grid=8):
        if not 0 < holdout < 0.8:
            raise ValueError("holdout must be between 0 and 0.8")
        self.shape = tuple(shape)
        self.grid = grid
        self.seed = seed
        order = np.random.default_rng(seed).permutation(grid * grid)
        n_test = max(1, round(holdout * len(order)))
        n_val = max(1, round(0.15 * (len(order) - n_test)))
        self.folds = np.zeros(grid * grid, np.uint8)
        self.folds[order[:n_test]] = self.TEST
        self.folds[order[n_test:n_test + n_val]] = self.VALIDATION

    def cells(self, points):
        h, w = self.shape
        x = np.clip((points[:, 0] * self.grid / w).astype(int), 0, self.grid - 1)
        y = np.clip((points[:, 1] * self.grid / h).astype(int), 0, self.grid - 1)
        return y * self.grid + x

    def labels(self, points):
        return self.folds[self.cells(points)]

    def fit_mask(self):
        y, x = np.indices(self.shape)
        return self.labels(np.column_stack([x.ravel(), y.ravel()])).reshape(self.shape) == self.FIT

    def summary(self):
        return {"kind": "fixed spatial cells in initial source working coordinates",
                "shape": list(self.shape), "grid": self.grid, "seed": self.seed,
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


def score_export(job_dir, test, split):
    """Only call after ALL model decisions and transform.json are finalized.

    No inlier filtering, RANSAC, selection, or refitting is allowed here. Reload
    the serialized matrix rather than evaluating a parallel in-memory estimate.
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
    with open(os.path.join(job_dir, "evaluation.json"), "w") as fh:
        json.dump(evidence, fh, indent=1, allow_nan=False)
    acc = {"held_out_n": len(test), "held_out_rmse_px": None,
           "held_out_median_px": None, "held_out_p90_px": None,
           "evaluation_file": "evaluation.json", "matrix_sha256": evidence["matrix_sha256"],
           "evaluation_basis": evidence["basis"],
           "transform_sha256": evidence["transform_sha256"],
           "applied_model": evidence["applied_model"]}
    if len(test) >= 3:
        residual = WM.residuals(transform, test.src.astype(np.float64), test.ref.astype(np.float64))
        acc.update(held_out_rmse_px=float(np.sqrt(np.mean(residual ** 2))),
                   held_out_median_px=float(np.median(residual)),
                   held_out_p90_px=float(np.percentile(residual, 90)))
    return acc
