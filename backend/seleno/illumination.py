"""Deciding what can be matched, before trying to match it.

This is where Seleno's proposed contribution lives, and where its limits are
stated. Read `ASSUMPTIONS.md` section on illumination before quoting anything
from here.

The premise
-----------
Sun-angle variation on this archive is not a brightness-normalisation problem.
Between the two 2024-11-15 strips the solar azimuth changes by ~67 deg and the
elevation by ~0.96 deg, at a grazing incidence of 89-91 deg. Cast shadows move
bodily and shading gradients reverse sign. No intensity transform recovers a
correspondence across that, because the two images are not two exposures of one
scene - they are two different light fields on the same terrain.

The proposal is therefore to predict *which regions cannot be matched* from
delivered Sun geometry plus a terrain model, exclude them, and let the matcher
work on the usable residual.

The hard constraint that shapes the whole design
------------------------------------------------
At a solar elevation of 0.79 deg, one metre of relief casts a **73 metre**
shadow. At 0.2 deg it casts 286 m. So a DEM with +/-2 m of vertical error places
a shadow boundary only to within ~150-600 m, which at 0.24 m/px is 600-2500
pixels. A predicted shadow mask built on any presently available lunar DEM
**cannot** have a crisp boundary at OHRC resolution.

Consequences, all enforced below:

* Masks are probabilistic, never binary. `ShadowPrediction` carries a
  `boundary_uncertainty_m` derived from the DEM's stated vertical error and the
  actual solar elevation, and the confidence field is feathered by it.
* Terrain is behind an interface. No external DEM is hard-coded, and no DEM is
  present in this repository.
* A DEM-free **observed** usability mask is implemented alongside, and is the
  control the predicted mask must be measured against. Without it, a
  shadow-prediction claim is untestable: excluding dark pixels helps a matcher
  whether or not the prediction was any good.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class UsabilityMask:
    """Per-pixel judgement of whether a region can support correspondence.

    `usable` is the boolean the pipeline acts on. `confidence` is the graded
    field it came from, kept so the UI can show how soft the decision was.
    """

    usable: np.ndarray                       # bool, True = matchable
    confidence: np.ndarray                   # float32 0..1, higher = more matchable
    source: str                              # observed-signal | predicted-shadow | none | combined
    is_prediction: bool                      # False for observed masks
    is_synthetic_terrain: bool = False       # True when a mock DEM produced it
    detail: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_fraction(self) -> float:
        return float(self.usable.mean()) if self.usable.size else 0.0

    def summary(self) -> dict:
        d = {
            "source": self.source,
            "is_prediction": self.is_prediction,
            "is_synthetic_terrain": self.is_synthetic_terrain,
            "usable_fraction": round(self.usable_fraction, 4),
            "mean_confidence": round(float(self.confidence.mean()), 4)
            if self.confidence.size else 0.0,
            "warnings": list(self.warnings),
        }
        d.update(self.detail)
        return d

    def combine(self, other: "UsabilityMask") -> "UsabilityMask":
        """Intersect two masks; confidence is the elementwise minimum."""
        return UsabilityMask(
            usable=self.usable & other.usable,
            confidence=np.minimum(self.confidence, other.confidence),
            source="combined(%s + %s)" % (self.source, other.source),
            is_prediction=self.is_prediction or other.is_prediction,
            is_synthetic_terrain=self.is_synthetic_terrain or other.is_synthetic_terrain,
            detail={"parts": [self.summary(), other.summary()]},
            warnings=self.warnings + other.warnings)


def all_usable(shape) -> UsabilityMask:
    """The no-op mask, for the baseline arm of an ablation."""
    return UsabilityMask(np.ones(shape, bool), np.ones(shape, np.float32),
                         source="none", is_prediction=False,
                         detail={"note": "masking disabled"})


@dataclass
class ShadowPrediction:
    """Output of a terrain model's shadow computation."""

    shadow_prob: np.ndarray                  # float32 0..1, 1 = certainly shadowed
    boundary_uncertainty_m: float
    solar_elevation_deg: float
    solar_azimuth_deg: float
    shadow_length_per_metre: float
    terrain_name: str
    is_synthetic: bool
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# observed usability - the DEM-free control
# --------------------------------------------------------------------------- #

def observed_signal_mask(tile: np.ndarray, *, window: int = 33,
                         std_floor: float = 3.0, dn_floor: int = 6,
                         levels_floor: int = 6) -> UsabilityMask:
    """Where does this image actually carry gradient information?

    Three local tests, all measured on the tile itself:

    * local standard deviation above `std_floor`,
    * local mean above `dn_floor` (a region pinned to the detector floor has no
      recoverable structure however it is stretched),
    * enough distinct DN levels in the neighbourhood - the specific pathology of
      this archive, where a shadowed patch can hold five values in total.

    This makes **no** prediction and needs no terrain. It is the honest control
    for the predicted-shadow experiment, and on its own it is a perfectly
    reasonable engineering answer to the problem.
    """
    a = tile.astype(np.float32)
    k = max(3, int(window) | 1)
    mean = cv2.blur(a, (k, k))
    sq = cv2.blur(a * a, (k, k))
    var = np.maximum(sq - mean * mean, 0.0)
    std = np.sqrt(var)

    # Local level diversity: dilation minus erosion is a cheap proxy for the
    # number of distinct values present in the neighbourhood.
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    spread = (cv2.dilate(tile, ker).astype(np.int16)
              - cv2.erode(tile, ker).astype(np.int16)).astype(np.float32)

    c_std = np.clip(std / (2.0 * std_floor), 0, 1)
    c_dn = np.clip(mean / (2.0 * dn_floor), 0, 1)
    c_lv = np.clip(spread / (2.0 * levels_floor), 0, 1)
    confidence = (c_std * c_dn * c_lv).astype(np.float32)

    usable = (std >= std_floor) & (mean >= dn_floor) & (spread >= levels_floor)
    return UsabilityMask(
        usable=usable, confidence=confidence,
        source="observed-signal", is_prediction=False,
        detail={"window_px": k, "std_floor": std_floor, "dn_floor": dn_floor,
                "levels_floor": levels_floor,
                "note": "measured from the image; no terrain model involved"})


# --------------------------------------------------------------------------- #
# terrain interface
# --------------------------------------------------------------------------- #

class TerrainModel(ABC):
    """Pluggable terrain source.

    Coordinates are **south polar stereographic metres**, matching
    `seleno.ohrc.geometry`, because longitude is degenerate at 89.9 S.

    A real implementation would wrap LOLA/LDEM or a TMC-2 derived DEM. None is
    present in this repository, and `available` reports that honestly rather
    than substituting something.
    """

    name: str = "abstract"
    resolution_m: float = float("nan")
    vertical_error_m: float = float("nan")
    available: bool = False
    is_synthetic: bool = False
    unavailable_reason: str = ""

    @abstractmethod
    def elevation_at(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        """Elevation in metres at projected coordinates."""

    def normal_at(self, x_m: np.ndarray, y_m: np.ndarray, h: float | None = None):
        """Unit surface normal, by central differences on the elevation field."""
        h = h or max(self.resolution_m if self.resolution_m == self.resolution_m else 1.0, 1.0)
        zx = (self.elevation_at(x_m + h, y_m) - self.elevation_at(x_m - h, y_m)) / (2 * h)
        zy = (self.elevation_at(x_m, y_m + h) - self.elevation_at(x_m, y_m - h)) / (2 * h)
        n = np.stack([-zx, -zy, np.ones_like(zx)], axis=-1)
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)

    # ------------------------------------------------------------------ shadow
    def shadow_mask(self, x_m: np.ndarray, y_m: np.ndarray,
                    sun_azimuth_deg: float, sun_elevation_deg: float,
                    *, max_ray_m: float = 4000.0,
                    samples: int = 64) -> ShadowPrediction:
        """Ray-cast towards the Sun to decide which points are occluded.

        Marches along the solar azimuth and tests whether any point along the
        ray rises above the straight line to the Sun. This is the standard
        horizon test, and at grazing elevation it needs a long ray - hence
        `max_ray_m`, which at 0.79 deg elevation covers only ~55 m of relief.

        The returned probability is the binary test feathered by
        `boundary_uncertainty_m`, because a crisp boundary would misrepresent
        what a coarse DEM can support.
        """
        if not self.available:
            raise RuntimeError("terrain model %r unavailable: %s"
                               % (self.name, self.unavailable_reason))
        el = math.radians(max(sun_elevation_deg, 1e-4))
        # Azimuth is measured clockwise from north; the projected plane has y
        # towards the prime meridian, so this is the direction TOWARDS the Sun.
        az = math.radians(sun_azimuth_deg)
        dx, dy = math.sin(az), math.cos(az)
        tan_el = math.tan(el)

        z0 = self.elevation_at(x_m, y_m)
        blocked = np.zeros(np.shape(z0), bool)
        margin = np.full(np.shape(z0), -np.inf, np.float32)
        for i in range(1, samples + 1):
            s = max_ray_m * i / samples
            zi = self.elevation_at(x_m + dx * s, y_m + dy * s)
            need = z0 + s * tan_el                  # ray height at distance s
            excess = (zi - need).astype(np.float32)
            blocked |= excess > 0
            margin = np.maximum(margin, excess)

        unc = self.shadow_boundary_uncertainty_m(sun_elevation_deg)
        # Feather: convert the vertical margin into a soft probability whose
        # transition width matches the DEM's vertical error.
        v_err = self.vertical_error_m if self.vertical_error_m == self.vertical_error_m else 1.0
        prob = 1.0 / (1.0 + np.exp(-margin / max(v_err, 1e-3)))
        return ShadowPrediction(
            shadow_prob=prob.astype(np.float32),
            boundary_uncertainty_m=unc,
            solar_elevation_deg=sun_elevation_deg,
            solar_azimuth_deg=sun_azimuth_deg,
            shadow_length_per_metre=(1.0 / tan_el) if tan_el > 0 else float("inf"),
            terrain_name=self.name,
            is_synthetic=self.is_synthetic,
            detail={"max_ray_m": max_ray_m, "ray_samples": samples,
                    "dem_resolution_m": self.resolution_m,
                    "dem_vertical_error_m": self.vertical_error_m})

    def shadow_boundary_uncertainty_m(self, sun_elevation_deg: float) -> float:
        """How far a shadow edge can move given the DEM's vertical error.

        `vertical_error / tan(elevation)`. This is the number that decides
        whether a predicted mask is worth anything at a given Sun elevation.
        """
        if sun_elevation_deg <= 0:
            return float("inf")
        v = self.vertical_error_m
        if v != v:
            return float("nan")
        return float(v / math.tan(math.radians(sun_elevation_deg)))

    def slope_shading(self, x_m: np.ndarray, y_m: np.ndarray,
                      sun_azimuth_deg: float, sun_elevation_deg: float) -> np.ndarray:
        """Lambertian cos(incidence) from the terrain normal and Sun direction.

        A first-order estimate of *broad* terrain shading, for removing the
        predictable low-frequency component. It is not a photometric model of the
        lunar surface - no Hapke or Lommel-Seeliger opposition or roughness terms
        - and it is not used to synthesise an image.
        """
        if not self.available:
            raise RuntimeError("terrain model %r unavailable: %s"
                               % (self.name, self.unavailable_reason))
        el = math.radians(sun_elevation_deg)
        az = math.radians(sun_azimuth_deg)
        sun = np.array([math.cos(el) * math.sin(az),
                        math.cos(el) * math.cos(az),
                        math.sin(el)], np.float64)
        n = self.normal_at(x_m, y_m)
        return np.clip((n * sun).sum(axis=-1), 0.0, 1.0).astype(np.float32)


class NullTerrain(TerrainModel):
    """No terrain available. Every call raises; `available` is False.

    This is the default, and it is what makes the missing-DEM state visible in
    the UI instead of silently degrading to something that looks like a result.
    """

    name = "none"
    available = False
    unavailable_reason = ("no digital elevation model is present in this repository; "
                          "real shadow ray-casting needs an external DEM "
                          "(LOLA/LDEM or a TMC-2 derived product)")

    def elevation_at(self, x_m, y_m):
        raise RuntimeError("NullTerrain has no elevation data: " + self.unavailable_reason)


class MockTerrain(TerrainModel):
    """Deterministic synthetic terrain, for developing and testing the plumbing.

    **Not science.** It is band-limited value noise with a few superimposed
    crater-like depressions. It has the right *character* - metre-scale relief
    over hundreds of metres, which at grazing Sun produces long shadows - so the
    mask, feathering, preprocessing and metric plumbing can be exercised
    end-to-end. It bears no relation to the actual terrain under any OHRC strip.

    Every product derived from it is stamped `is_synthetic=True`, and the
    pipeline refuses to report a predicted-shadow result as evidence when that
    flag is set.
    """

    name = "mock-synthetic"
    available = True
    is_synthetic = True
    resolution_m = 5.0
    vertical_error_m = 2.0

    def __init__(self, seed: int = 7, relief_m: float = 12.0,
                 feature_scale_m: float = 400.0, craters: int = 14):
        self.seed = int(seed)
        self.relief_m = float(relief_m)
        self.feature_scale_m = float(feature_scale_m)
        rng = np.random.default_rng(self.seed)
        # A few octaves of value noise, evaluated analytically so any coordinate
        # can be queried without allocating a grid.
        self._waves = []
        for octave in range(4):
            k = (2 ** octave) * 2.0 * math.pi / self.feature_scale_m
            for _ in range(3):
                theta = float(rng.uniform(0, 2 * math.pi))
                self._waves.append((k * math.cos(theta), k * math.sin(theta),
                                    float(rng.uniform(0, 2 * math.pi)),
                                    self.relief_m / (2 ** octave) / 3.0))
        self._craters = [(float(rng.uniform(-15000, 15000)),
                          float(rng.uniform(-15000, 15000)),
                          float(rng.uniform(60, 400)),
                          float(rng.uniform(3, 25))) for _ in range(craters)]

    def elevation_at(self, x_m, y_m):
        x = np.asarray(x_m, np.float64)
        y = np.asarray(y_m, np.float64)
        z = np.zeros(np.broadcast(x, y).shape, np.float64)
        for kx, ky, phase, amp in self._waves:
            z = z + amp * np.sin(kx * x + ky * y + phase)
        for cx, cy, radius, depth in self._craters:
            r = np.hypot(x - cx, y - cy)
            bowl = np.clip(1.0 - (r / radius) ** 2, 0.0, 1.0)
            rim = np.exp(-((r - radius) / (0.25 * radius)) ** 2)
            z = z - depth * bowl + 0.35 * depth * rim
        return z


# --------------------------------------------------------------------------- #
# turning a shadow prediction into a usability mask
# --------------------------------------------------------------------------- #

def predicted_shadow_mask(terrain: TerrainModel, x_m: np.ndarray, y_m: np.ndarray,
                          sun_a: dict, sun_b: dict, *,
                          shadow_prob_threshold: float = 0.5,
                          gsd_m: float = 0.24,
                          coarse_px: int | None = None) -> UsabilityMask:
    """Regions matchable under BOTH illuminations, per the terrain model.

    A pixel is unusable if it is predicted shadowed in either image: shadowed in
    one means no signal there, shadowed in the other means no counterpart. The
    two predictions use each image's own interpolated Sun geometry.

    **The prediction is computed on a decimated grid and upsampled**, at a
    spacing derived from the DEM's own resolution. That is not a shortcut: a
    terrain model sampled at 5 m has no information at 0.24 m, so evaluating the
    shadow test per full-resolution pixel would manufacture detail the input
    cannot support - and it costs ~64x more ray casts for it. `coarse_px`
    overrides the derived spacing when a caller needs to.

    The boolean is then eroded by the predicted boundary uncertainty converted
    to pixels, because a boundary this model cannot locate should not be trusted
    to a pixel.
    """
    full_shape = np.shape(x_m)

    # ---- applicability check, before any computation -----------------------
    # A local ray cast against a flat horizon is only meaningful when the Sun is
    # ABOVE the local horizon. Three of the four products in this archive have a
    # mid-strip solar elevation at or below zero, yet plainly contain lit
    # terrain: near the pole, high ground sees the Sun over a depressed horizon
    # while the sub-satellite point does not. Predicting that needs a
    # horizon-angle computation over tens of kilometres of DEM, which is not
    # what this local test does. Rather than return a mask that declares
    # everything shadowed - technically the output of the model, and completely
    # wrong - the prediction declares itself inapplicable and the pipeline falls
    # back to no masking with the reason stated.
    el_a = float(sun_a.get("elevation_deg", 0.0))
    el_b = float(sun_b.get("elevation_deg", 0.0))
    MIN_EL = 0.05
    if min(el_a, el_b) <= MIN_EL:
        m = all_usable(full_shape)
        m.source = "predicted-shadow (inapplicable)"
        m.is_prediction = True
        m.is_synthetic_terrain = terrain.is_synthetic
        m.detail = {
            "terrain": terrain.name,
            "sun_a": sun_a, "sun_b": sun_b,
            "min_elevation_deg": round(min(el_a, el_b), 4),
            "min_applicable_elevation_deg": MIN_EL,
        }
        m.warnings.append(
            "predicted-shadow masking is INAPPLICABLE here: the solar elevation is "
            "%.3f deg, at or below the local horizon. A local horizon ray cast would "
            "declare the entire window shadowed, which contradicts the imagery - near "
            "the pole high ground is lit over a depressed horizon. Deciding that "
            "requires horizon angles computed over tens of km of DEM, which this test "
            "does not do. No masking was applied." % min(el_a, el_b))
        return m

    if coarse_px is None:
        # one prediction sample per DEM post, floored at 4 px to keep the cost sane
        dem_res = terrain.resolution_m if terrain.resolution_m == terrain.resolution_m else 5.0
        coarse_px = int(max(4, round(dem_res / max(gsd_m, 1e-6))))
    coarse_px = max(1, min(coarse_px, max(1, min(full_shape) // 8)))

    xs = np.asarray(x_m)[::coarse_px, ::coarse_px]
    ys = np.asarray(y_m)[::coarse_px, ::coarse_px]

    pa = terrain.shadow_mask(xs, ys, sun_a["azimuth_deg"], sun_a["elevation_deg"])
    pb = terrain.shadow_mask(xs, ys, sun_b["azimuth_deg"], sun_b["elevation_deg"])

    lit_coarse = ((1.0 - pa.shadow_prob) * (1.0 - pb.shadow_prob)).astype(np.float32)
    lit_prob = cv2.resize(lit_coarse, (full_shape[1], full_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    usable = lit_prob >= shadow_prob_threshold

    unc_m = max(pa.boundary_uncertainty_m, pb.boundary_uncertainty_m)
    warnings = []
    if terrain.is_synthetic:
        warnings.append(
            "terrain model is SYNTHETIC (%s): this mask exercises the pipeline and is "
            "not evidence about the real surface" % terrain.name)
    if not np.isfinite(unc_m):
        warnings.append(
            "solar elevation is at or below the horizon in at least one image, so the "
            "shadow boundary is unbounded - the prediction carries no spatial precision")
    else:
        unc_px = unc_m / max(gsd_m, 1e-6)
        if unc_px > 1.0:
            r = int(min(max(1, round(unc_px / 8.0)), 25))   # capped for cost
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            usable = cv2.erode(usable.astype(np.uint8), ker).astype(bool)
            warnings.append(
                "shadow boundary uncertain to %.0f m (%.0f px) at elevation %.3f deg; "
                "mask eroded by %d px (uncertainty capped for cost)"
                % (unc_m, unc_px, min(sun_a["elevation_deg"], sun_b["elevation_deg"]),
                   r))

    return UsabilityMask(
        usable=usable, confidence=lit_prob.astype(np.float32),
        source="predicted-shadow", is_prediction=True,
        is_synthetic_terrain=terrain.is_synthetic,
        detail={"terrain": terrain.name,
                "dem_resolution_m": terrain.resolution_m,
                "dem_vertical_error_m": terrain.vertical_error_m,
                "boundary_uncertainty_m": None if not np.isfinite(unc_m) else round(unc_m, 1),
                "shadow_length_per_metre_a": pa.shadow_length_per_metre,
                "shadow_length_per_metre_b": pb.shadow_length_per_metre,
                "sun_a": sun_a, "sun_b": sun_b,
                "threshold": shadow_prob_threshold,
                "prediction_grid_px": coarse_px,
                "prediction_grid_m": round(coarse_px * gsd_m, 2)},
        warnings=warnings)


def remove_predicted_shading(tile: np.ndarray, shading: np.ndarray,
                             strength: float = 1.0) -> np.ndarray:
    """Divide out an estimated broad shading field.

    Only the low-frequency part of the estimate is used: a coarse terrain model
    has no business claiming to predict crater-scale shading, and dividing by a
    high-frequency estimate would inject its errors straight into the texture
    the matcher depends on.
    """
    s = shading.astype(np.float32)
    s = cv2.GaussianBlur(s, (0, 0), 25.0)
    s = 1.0 + strength * (s - float(s.mean()))
    out = tile.astype(np.float32) / np.maximum(s, 0.15)
    lo, hi = np.percentile(out, (0.5, 99.5))
    if hi <= lo:
        return tile.copy()
    return np.clip((out - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

TERRAIN_MODELS: dict[str, dict] = {
    "none": {"factory": NullTerrain,
             "label": "No terrain (masking from the image only)",
             "status": "available"},
    "mock": {"factory": MockTerrain,
             "label": "Mock synthetic terrain (development only, NOT science)",
             "status": "synthetic"},
    # A real DEM lands here as {"factory": LolaTerrain, "status": "available"}
    # once an external product is fetched. Nothing else in the pipeline changes.
}


def get_terrain(name: str) -> TerrainModel:
    spec = TERRAIN_MODELS.get(name)
    if spec is None:
        raise ValueError("unknown terrain model %r (have: %s)"
                         % (name, ", ".join(TERRAIN_MODELS)))
    return spec["factory"]()


MASK_MODES = {
    "none": "No masking (baseline)",
    "observed": "Observed signal mask (measured from the image, no DEM)",
    "predicted": "Predicted shadow mask (needs a terrain model)",
    "both": "Predicted shadow mask intersected with the observed signal mask",
}
