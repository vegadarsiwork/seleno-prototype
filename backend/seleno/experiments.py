"""Reproducible experiment configurations and the ablation arms.

Presets are named so an experiment can be quoted by name in the docs and rerun
verbatim. Each arm differs from its stated parent in exactly one respect, which
is what makes the comparison mean anything.

Two arms exist to record **negative** results measured on real OHRC repeat-pass
windows (five windows, 1024 px, `20241115T1326` -> `20241115T1525`, mean
held-out RMSE):

    E  observed mask, all inliers, no refinement      1.41 px   <- best
    H  + spatial grid selection                       1.71 px
    I  + sub-pixel refinement                         2.36 px
    J  + both                                         2.92 px

On this data neither spatially-uniform selection nor patch-correlation sub-pixel
refinement helps; both make the transform generalise worse. Both stages stay
implemented and selectable, and both are **off** in the `seleno` preset, because
a preset named after the project should be the best configuration we can defend
rather than an aspirational one. `ASSUMPTIONS.md` explains why each fails here.

Arm E is also the **control** for arms F and G: excluding dark pixels helps a
matcher whether or not a shadow *prediction* was any good, so a predicted mask
that does not beat the observed mask has demonstrated nothing.
"""
from __future__ import annotations

from copy import deepcopy

from .registration import Options

# Phase-4 baseline: as plain as it can be while still producing a transform.
_BASE = dict(
    matcher="sift", ratio=0.80, mutual_check=True,
    preprocess_enabled=False, use_clahe=False, use_flatten=False,
    harmonise_gsd=False,
    mask_mode="none", terrain_model="none", remove_shading=False,
    verify_enabled=True, ransac_threshold=3.0,
    selection_mode="all", grid=6, per_cell=3,
    subpixel_enabled=False, holdout_fraction=0.30, seed=0,
)


def _arm(**over):
    d = deepcopy(_BASE)
    d.update(over)
    return d


_PREPROCESSED = dict(harmonise_gsd=True, preprocess_enabled=True,
                     use_clahe=True, use_flatten=True)
_MASKED = dict(_PREPROCESSED, mask_mode="observed")

PRESETS: dict[str, dict] = {
    # ---------------------------------------------------------- ablation arms
    "baseline": _arm(verify_enabled=False),
    "baseline_verified": _arm(),
    "resolution_harmonised": _arm(harmonise_gsd=True),
    "preprocessed": _arm(**_PREPROCESSED),
    "observed_mask": _arm(**_MASKED),
    "predicted_mask": _arm(**dict(_PREPROCESSED, mask_mode="predicted",
                                  terrain_model="mock")),
    "predicted_mask_shading": _arm(**dict(_PREPROCESSED, mask_mode="predicted",
                                          terrain_model="mock",
                                          remove_shading=True)),
    "predicted_and_observed": _arm(**dict(_PREPROCESSED, mask_mode="both",
                                          terrain_model="mock")),
    # the two measured negatives, isolated
    "spatial_grid": _arm(**dict(_MASKED, selection_mode="grid")),
    "spatial_topk": _arm(**dict(_MASKED, selection_mode="topk")),
    "subpixel": _arm(**dict(_MASKED, subpixel_enabled=True)),
    "spatial_and_subpixel": _arm(**dict(_MASKED, selection_mode="grid",
                                        subpixel_enabled=True)),
    # ------------------------------------------------------ the recommendation
    # Best measured configuration: geometry-first coarse alignment (applied when
    # the pair is built), preprocessing, the DEM-free usability mask, robust
    # verification, and every verified inlier used in the fit.
    "seleno": _arm(**_MASKED),
    # ------------------------------------------------ a deliberately naive arm
    "naive": _arm(verify_enabled=False, ratio=0.95, mutual_check=False),
}

# Ordered arms for the ablation table.
ABLATION = [
    ("A", "baseline", "SIFT + ratio test, no verification"),
    ("B", "baseline_verified", "+ robust geometric verification"),
    ("C", "resolution_harmonised", "+ GSD harmonisation"),
    ("D", "preprocessed", "+ illumination flattening and CLAHE"),
    ("E", "observed_mask", "+ observed usability mask, no DEM  <- control & best"),
    ("F", "predicted_mask", "+ predicted shadow mask (SYNTHETIC terrain)"),
    ("G", "predicted_mask_shading", "+ broad shading removal (SYNTHETIC terrain)"),
    ("H", "spatial_grid", "E + spatial grid selection"),
    ("I", "subpixel", "E + sub-pixel refinement"),
    ("J", "spatial_and_subpixel", "E + both selection and refinement"),
]


def build_options(preset: str, pair=None, overrides: dict | None = None) -> Options:
    """Materialise a preset, defaulting the transform model to the pair's own."""
    if preset not in PRESETS:
        raise ValueError("unknown preset %r (have: %s)"
                         % (preset, ", ".join(sorted(PRESETS))))
    d = deepcopy(PRESETS[preset])
    # The transform model is a property of the PAIR, not of the ablation arm: a
    # synthetic pair built with an 11 deg rotation and a perspective term needs a
    # homography, and forcing a similarity onto it makes every arm look bad for a
    # reason unrelated to what the arm changes. Presets leave `model_type` unset
    # unless they are specifically about it.
    if "model_type" not in d:
        d["model_type"] = (getattr(pair, "recommended_model", None)
                           or Options().model_type)
    if overrides:
        d.update({k: v for k, v in overrides.items() if v is not None})
    return Options.from_dict(d)


def describe_presets() -> list[dict]:
    return [{"id": k, "options": v} for k, v in sorted(PRESETS.items())]
