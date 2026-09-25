"""The serialized export model, shared by raster sampling and final evaluation.

Segment inverse maps are blended with a C1 smoothstep between band centres.
Blending coordinates before sampling avoids brightness seams/double exposure.
An optional `local_field` (see `local_model`) adds a smooth correction to that
inverse map; everything here goes through `inverse_points`, so the raster, the
preview and the held-out score all apply the same complete model.
"""
import copy
import hashlib
import json

import cv2
import numpy as np

from .coordinates import project
from . import local_model
from . import terrain
from ..verify import transfer_error


def digest(model):
    return hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inverse_points(model, points):
    """Reference -> source through the complete model (base plus local field)."""
    out = _base_inverse(model, points)
    parallax = model.get("parallax")
    if parallax is not None:
        out = out + terrain.term(parallax, points)
    field = model.get("local_field")
    if field is not None:
        out = out + local_model.evaluate(field, points)
    return out


def _nonlinear(model):
    return bool(model.get("segments") or model.get("local_field") is not None
                or model.get("parallax") is not None)


def _base_inverse(model, points):
    points = np.asarray(points, np.float64)
    segments = model.get("segments") or []
    global_h = np.asarray(model["matrix"], np.float64)
    if not segments:
        return project(np.linalg.inv(global_h), points)
    axis = 1 if segments[0]["axis"] == "row" else 0
    centres = np.array([(s["from"] + s["to"]) / 2 for s in segments])
    right = np.clip(np.searchsorted(centres, points[:, axis]), 1, len(centres) - 1)
    left = right - 1
    t = np.clip((points[:, axis] - centres[left]) / (centres[right] - centres[left]), 0, 1)
    weight = t * t * (3 - 2 * t)
    out = np.zeros_like(points)
    for k, segment in enumerate(segments):
        w = np.where(left == k, 1 - weight, 0) + np.where(right == k, weight, 0)
        active = w > 0
        if np.any(active):
            h = np.asarray(segment.get("matrix", global_h), np.float64)
            out[active] += project(np.linalg.inv(h), points[active]) * w[active, None]
    return out


def forward_points(model, points):
    """Invert the actual blended sampling field using a numerical 2D Jacobian."""
    points = np.asarray(points, np.float64)
    q = project(np.asarray(model["matrix"], np.float64), points)
    if not _nonlinear(model):
        return q
    eps = 0.01
    for _ in range(30):
        predicted = inverse_points(model, q)
        error = predicted - points
        if np.max(np.abs(error), initial=0) < 1e-10:
            break
        jx = (inverse_points(model, q + [eps, 0]) - predicted) / eps
        jy = (inverse_points(model, q + [0, eps]) - predicted) / eps
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        safe = np.abs(det) > 1e-12
        delta = error.copy()
        delta[safe, 0] = (jy[safe, 1] * error[safe, 0] - jy[safe, 0] * error[safe, 1]) / det[safe]
        delta[safe, 1] = (-jx[safe, 1] * error[safe, 0] + jx[safe, 0] * error[safe, 1]) / det[safe]
        q -= np.clip(delta, -100, 100)
    return q


def residuals(model, source, reference):
    if not _nonlinear(model):
        return transfer_error(np.asarray(model["matrix"], np.float64), source, reference)
    forward = forward_points(model, source) - reference
    backward = inverse_points(model, reference) - source
    return 0.5 * (np.linalg.norm(forward, axis=1) + np.linalg.norm(backward, axis=1))


def warp(src, valid, model, shape):
    out, mask = np.empty(shape, np.float32), np.empty(shape, bool)
    data = np.nan_to_num(src).astype(np.float32)
    for start in range(0, shape[0], 256):
        end = min(start + 256, shape[0])
        y, x = np.mgrid[start:end, :shape[1]]
        xy = inverse_points(model, np.column_stack([x.ravel(), y.ravel()]))
        mx, my = [xy[:, k].reshape(y.shape).astype(np.float32) for k in (0, 1)]
        out[start:end] = cv2.remap(data, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask[start:end] = cv2.remap(valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT).astype(bool)
    return out, mask


def conjugate(model, matrix):
    """Carry the complete model onto another axis-aligned display grid."""
    result = copy.deepcopy(model)
    inv = np.linalg.inv(matrix)
    result["matrix"] = (matrix @ np.asarray(model["matrix"]) @ inv).tolist()
    for s in result.get("segments") or []:
        axis = 1 if s["axis"] == "row" else 0
        for bound in ("from", "to"):
            s[bound] = s[bound] * matrix[axis, axis] + matrix[axis, 2]
        if "matrix" in s:
            s["matrix"] = (matrix @ np.asarray(s["matrix"]) @ inv).tolist()
    for key in ("local_field", "parallax"):
        term = result.get(key)
        if term is not None:
            # The term stays on the grid it was fitted on; record how this grid
            # relates to that one (composing with any earlier carriage).
            prior = np.asarray(term.get("frame", np.eye(3)), np.float64)
            term["frame"] = (matrix @ prior).tolist()
    return result
