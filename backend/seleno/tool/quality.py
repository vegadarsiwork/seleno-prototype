"""Read-only diagnostics of frozen evidence, also usable on older saved runs.

Nothing here fits, screens or selects a transform. Native predictions are read
from evaluation.json so inspection never needs a DEM, source cube or matcher.

Beyond the distance summaries this adds what one global number hides: error
profiles along and across the strip, spatial-block bootstrap intervals, the
final warp's local validity, and the same errors restated in the forms NASA
(ASP), USGS (ISIS) and JAXA (Kaguya TC) publish theirs in. Restating a number
in an agency's form makes it readable beside theirs; it does not make the two
measurements the same, and every row says how they differ.
"""
import json
from pathlib import Path

import numpy as np

from .evaluation import SpatialSplit, error_statistics
from . import warp_model as WM

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 0

# Published reference points, checked against the primary sources on
# 2026-09-26. Values are context for the statistic's form, never a target.
AGENCY_SOURCES = {
    "asp": {"agency": "NASA Ames Stereo Pipeline", "tool": "bundle_adjust",
            "url": "https://stereopipeline.readthedocs.io/en/stable/tools/bundle_adjust.html",
            "reports": "mean and median pixel reprojection error per camera, with the count",
            "guidance": "errors should be under 1 pixel, ideally under 0.5 pixels; the count "
                        "must be at least a dozen"},
    "isis": {"agency": "USGS ISIS", "tool": "jigsaw",
             "url": "https://isis.astrogeology.usgs.gov/dev/Application/presentation/Tabbed/jigsaw/jigsaw.html",
             "reports": "sample, line and overall residual of every control measure, in pixels"},
    "kaguya": {"agency": "JAXA SELENE (Kaguya) Terrain Camera", "tool": "Haruyama et al., LPSC 43 (2012) #1200",
               "url": "https://www.lpi.usra.edu/meetings/lpsc2012/pdf/1200.pdf",
               "reports": "mean and 1-sigma of longitude, latitude and elevation differences in "
                          "metres at nine identical locations in TC DTMs from different "
                          "observation times",
               "published": {"longitude_m": [5.4, 8.0], "latitude_m": [3.6, 7.2],
                             "elevation_m": [2.6, 4.3], "locations": 9, "gsd_m": 10.0}},
    "sldem": {"agency": "NASA GSFC / JAXA SLDEM2015", "tool": "TC DEM tiles co-registered to LOLA",
              "url": "https://pgda.gsfc.nasa.gov/products/54",
              "reports": "residual distributions per tile and their spatial variation "
                         "(vertical, against altimetry)"},
}


def from_evidence(metrics, evidence, transform):
    expected = evidence.get("transform_sha256")
    if not expected or expected != WM.digest(transform) or expected != metrics.get("accuracy", {}).get("transform_sha256"):
        raise ValueError("evaluation and metrics do not identify the current exported transform")
    points = np.asarray(evidence.get("source", []), float).reshape(-1, 2)
    n = len(points)
    cp = evidence.get("check_point")
    screened = None if cp is None else np.asarray(cp, bool)
    if screened is not None and screened.shape != (n,):
        raise ValueError("check-point flags do not match held-out observations")
    source_meta = metrics.get("source", {})
    result = {"schema_version": 2, "basis": "held-out matcher consistency; not independent ground truth",
              "transform_sha256": expected, "n": n,
              "screened_n": None if screened is None else int(screened.sum()),
              "excluded_n": None if screened is None else int((~screened).sum()),
              "source": None, "reference": None, "ground": None, "ground_frame": evidence.get("ground_frame"),
              "residuals": [], "profiles": None, "intervals": None, "warp": None, "conventions": [],
              "source_shape": source_meta.get("shape"), "source_gsd_m": source_meta.get("gsd_m"),
              "support": metrics.get("distribution", {}).get("valid_pixel_support"),
              "acceptance": metrics.get("acceptance"),
              "independent": metrics.get("accuracy", {}).get("independent_ground_truth"),
              "threshold_definition": "strictly below threshold; denominator includes invalid predictions as failures",
              "distance_definition": "finite observations only; invalid counts are reported separately",
              "reference_metres_per_pixel": metrics.get("accuracy", {}).get("units", {}).get("metres_per_reference_px"),
              "metre_definition": "nominal reference-scale distance; not absolute geolocation accuracy",
              "ground_definition": ("forward error in local east/north metres on the reference datum; "
                                    "relative to the reference, not absolute geolocation")}

    def summarize(vectors, **units):
        return {"all": error_statistics(vectors, **units),
                "screened": None if screened is None else error_statistics(vectors[screened], **units)}

    observed = errors = None
    if "native_source" in evidence and "predicted_native_source" in evidence:
        observed = np.asarray(evidence["native_source"], float).reshape(-1, 2)
        predicted = np.asarray(evidence["predicted_native_source"], float).reshape(-1, 2)
        if observed.shape != (n, 2) or predicted.shape != observed.shape:
            raise ValueError("native predictions do not match held-out observations")
        errors = predicted - observed
        result["source"] = summarize(errors)
    if "reference_error_vectors" in evidence:
        vectors = np.asarray(evidence["reference_error_vectors"], float).reshape(-1, 2)
        if vectors.shape != (n, 2):
            raise ValueError("reference errors do not match held-out observations")
        result["reference"] = summarize(vectors)
    elif "reference_errors" in evidence:
        # Older evidence stores the correct directional summary, but not the
        # vectors needed for new quantiles. Do not invent them from the RMSE.
        result["reference"] = {"all": evidence["reference_errors"],
                               "screened": evidence.get("check_point_reference_errors")}
    ground = None
    if "ground_error_vectors" in evidence:
        ground = np.asarray(evidence["ground_error_vectors"], float).reshape(-1, 2)
        if ground.shape != (n, 2):
            raise ValueError("ground errors do not match held-out observations")
        result["ground"] = summarize(ground, unit="m", axes="en", thresholds=())

    if errors is not None:
        rows = np.column_stack([observed, errors] + ([ground] if ground is not None else []))
        clean = np.where(np.isfinite(rows), rows, None).tolist()
        result["residuals"] = [{"x": row[0], "y": row[1], "dx": row[2], "dy": row[3],
                                **({"de": row[4], "dn": row[5]} if ground is not None else {}),
                                "screened": None if screened is None else bool(screened[i])}
                               for i, row in enumerate(clean)]
        if result["source_shape"]:
            result["profiles"] = profiles(observed, errors, result["source_shape"], ground)
    blocks = _blocks(evidence, points)
    if blocks is not None:
        result["intervals"] = {
            "method": "spatial block bootstrap: held-out split cells resampled with replacement",
            "note": "sampling uncertainty of these matcher observations only; it excludes any "
                    "bias shared with the reference or the matcher",
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "level": 0.95,
            "source": None if errors is None else _intervals_pair(errors, blocks, screened, "px"),
            "ground": None if ground is None else _intervals_pair(ground, blocks, screened, "m")}
    try:
        result["warp"] = warp_validity(transform, evidence.get("split", {}).get("shape"),
                                       _native_per_working(points, observed))
    except Exception as exc:                                          # noqa: BLE001
        # A terrain term needs its DEM; a missing file is not a model defect.
        result["warp"] = {"available": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
    result["conventions"] = conventions(result)
    return result


# --------------------------------------------------------------------------- #
# profiles along and across the strip
# --------------------------------------------------------------------------- #

def _bin_stats(errors, ground, keep):
    stats = error_statistics(errors[keep], thresholds=())
    out = {"n": stats["n"], "invalid_n": stats["invalid_n"], "rmse_px": stats["rmse_px"],
           "median_px": stats["median_px"],
           # A 95th percentile of four numbers is the maximum; do not call it p95.
           "p95_px": stats["p95_px"] if stats["valid_n"] >= 8 else None,
           "bias_xy_px": stats["bias_xy_px"]}
    if ground is not None:
        g = error_statistics(ground[keep], unit="m", axes="en", thresholds=())
        out.update(rmse_m=g["rmse_m"], bias_en_m=g["bias_en_m"])
    return out


def profiles(observed, errors, shape, ground=None):
    """Binned error along the strip's long axis and across its short one.

    Bins span the whole native source image, so stretches with no held-out
    observation appear as empty bins instead of being silently skipped. All
    held-out observations are used: screening could hide a difficult region.
    """
    h, w = int(shape[0]), int(shape[1])
    along = 1 if h >= w else 0
    finite = np.isfinite(errors).all(axis=1)
    usable = np.isfinite(observed).all(axis=1)
    n_valid = int((finite & usable).sum())
    report = {"basis": "all held-out observations, native source pixels",
              "along_axis": "line" if along == 1 else "sample",
              "across_axis": "sample" if along == 1 else "line"}
    for name, axis, count in (("along", along, int(np.clip(n_valid // 12, 3, 12))),
                              ("across", 1 - along, int(np.clip(n_valid // 25, 2, 5)))):
        extent = (h if axis == 1 else w)
        edges = np.linspace(-0.5, extent - 0.5, count + 1)
        position = observed[:, axis]
        bins = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            keep = usable & (position >= lo) & ((position < hi) if i < count - 1 else (position <= hi))
            bins.append({"from": float(lo), "to": float(hi), **_bin_stats(errors, ground, keep)})
        report[name] = bins
    # Linear drift of each error component along the strip, per 1000 source px.
    k = finite & usable
    if k.sum() >= 8 and np.ptp(observed[k, along]) > 0:
        t = (observed[k, along] - observed[k, along].mean()) / 1000.0
        slope = [float(np.dot(t, errors[k, c] - errors[k, c].mean()) / np.dot(t, t)) for c in (0, 1)]
        report["drift_xy_px_per_1000"] = slope
    else:
        report["drift_xy_px_per_1000"] = None
    return report


# --------------------------------------------------------------------------- #
# spatial-block bootstrap
# --------------------------------------------------------------------------- #

def _blocks(evidence, points):
    """Held-out cell of each observation, from the sealed split definition."""
    spec = evidence.get("split") or {}
    if not spec.get("shape") or not spec.get("grid") or not len(points):
        return None
    split = SpatialSplit(spec["shape"], seed=spec.get("seed", 0), grid=tuple(spec["grid"]),
                         frame=spec.get("frame"))
    return split.cells(points)


def _intervals_pair(vectors, blocks, screened, unit):
    out = {"all": block_bootstrap(vectors, blocks, unit)}
    out["screened"] = None if screened is None else block_bootstrap(vectors[screened], blocks[screened], unit)
    return out


def block_bootstrap(vectors, blocks, unit="px", replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """95% percentile intervals, resampling whole spatial cells.

    Neighbouring matches share texture and model error, so resampling single
    points would treat them as independent and make the interval too narrow.
    """
    vectors = np.asarray(vectors, float).reshape(-1, 2)
    labels, index = np.unique(np.asarray(blocks), return_inverse=True)
    if len(labels) < 5:
        return {"blocks": int(len(labels)), "reason": "fewer than five spatial blocks"}
    distance = np.linalg.norm(vectors, axis=1)            # NaN for invalid predictions
    members = [np.flatnonzero(index == b) for b in range(len(labels))]
    rng = np.random.default_rng(seed)
    names = ("rmse", "median", "p90", "p95") + (("below_1",) if unit == "px" else ())
    draws = {name: [] for name in names}
    for _ in range(replicates):
        take = np.concatenate([members[b] for b in rng.integers(0, len(members), len(members))])
        d = distance[take]
        f = d[np.isfinite(d)]
        if not len(f):
            continue
        draws["rmse"].append(np.sqrt(np.mean(f ** 2)))
        draws["median"].append(np.median(f))
        draws["p90"].append(np.percentile(f, 90))
        draws["p95"].append(np.percentile(f, 95))
        if unit == "px":
            draws["below_1"].append(np.sum(f < 1.0) / len(d))   # invalid counts as failure
    out = {"blocks": int(len(labels)), "n": int(len(vectors))}
    for name, values in draws.items():
        key = "below_1_px_fraction" if name == "below_1" else "%s_%s" % (name, unit)
        out[key] = ([float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                    if values else None)
    return out


# --------------------------------------------------------------------------- #
# final-warp validity
# --------------------------------------------------------------------------- #

def _native_per_working(points, observed):
    """Native source px per source working px, from the held-out pairs themselves."""
    if observed is None or len(points) < 3:
        return None
    ok = np.isfinite(observed).all(axis=1) & np.isfinite(points).all(axis=1)
    if ok.sum() < 3:
        return None
    design = np.column_stack([points[ok], np.ones(ok.sum())])
    coef = np.linalg.lstsq(design, observed[ok], rcond=None)[0]
    scale = float(np.sqrt(abs(np.linalg.det(coef[:2]))))
    return scale if np.isfinite(scale) and scale > 0 else None


def warp_validity(transform, source_shape, native_per_working=None, max_points=40_000):
    """Local Jacobian of the complete inverse map over its delivered footprint.

    Relative to the global/segment base model, so strip geometry the base
    already explains is not flagged: det(J J_base^-1) is the local field's
    change of area, its singular-value ratio the shear it adds, and a
    non-positive det means the exported warp folds over itself.
    """
    frame = transform.get("frame", {})
    if not frame.get("target_shape") or not source_shape:
        return {"available": False, "reason": "saved transform has no grid shapes"}
    H, W = (int(v) for v in frame["target_shape"])
    sh, sw = (int(v) for v in source_shape)
    stride = max(1, int(np.ceil(np.sqrt(H * W / max_points))))
    y, x = np.mgrid[stride / 2:H:stride, stride / 2:W:stride]
    q = np.column_stack([x.ravel(), y.ravel()])
    s = WM.inverse_points(transform, q)
    inside = (s[:, 0] >= 0) & (s[:, 0] <= sw - 1) & (s[:, 1] >= 0) & (s[:, 1] <= sh - 1)
    q = q[inside]
    if len(q) < 16:
        return {"available": False, "reason": "the model maps almost no reference grid into the source"}

    def jacobian(fn):
        e = 0.5
        jx = (fn(q + [e, 0]) - fn(q - [e, 0])) / (2 * e)
        jy = (fn(q + [0, e]) - fn(q - [0, e])) / (2 * e)
        return np.stack([jx, jy], axis=-1)                   # (n, 2 source, 2 reference)

    full = jacobian(lambda p: WM.inverse_points(transform, p))
    base = jacobian(lambda p: WM._base_inverse(transform, p))
    relative = full @ np.linalg.inv(base)
    det = np.linalg.det(relative)
    sv = np.linalg.svd(relative, compute_uv=False)
    shear = sv[:, 0] / np.maximum(sv[:, 1], 1e-12)
    correction = np.linalg.norm(WM.inverse_points(transform, q) - WM._base_inverse(transform, q), axis=1)
    # A mirrored pair has a negative Jacobian everywhere; a fold is a sign flip.
    full_det = np.linalg.det(full)
    orientation = np.sign(np.median(full_det)) or 1.0
    folded = float(np.mean(full_det * orientation <= 0))
    report = {"available": True, "points": int(len(q)), "stride_working_px": stride,
              "domain": "reference working-grid lattice mapped inside the source working grid",
              "folded_fraction": folded,
              "relative_area": {"p1": float(np.percentile(det, 1)), "median": float(np.median(det)),
                                "p99": float(np.percentile(det, 99)),
                                "min": float(det.min()), "max": float(det.max())},
              "relative_shear": {"median": float(np.median(shear)), "p99": float(np.percentile(shear, 99)),
                                 "max": float(shear.max())},
              "local_correction": {"unit": "source working px", "median": float(np.median(correction)),
                                   "p95": float(np.percentile(correction, 95)),
                                   "max": float(correction.max())},
              "nonlinear_terms": [k for k in ("segments", "local_field", "parallax") if transform.get(k)]}
    if native_per_working:
        report["local_correction_native_px"] = {
            key: report["local_correction"][key] * native_per_working for key in ("median", "p95", "max")}
        report["native_per_working_px"] = native_per_working
    flags = []
    if folded > 0:
        flags.append("the warp folds over itself on %.2f%% of the footprint" % (100 * folded))
    if det.min() < 0.8 or det.max() > 1.25:
        flags.append("the local correction changes area by more than 20%% somewhere "
                     "(%.2f to %.2f of the base model)" % (det.min(), det.max()))
    if shear.max() > 1.25:
        flags.append("the local correction adds shear up to %.2f:1" % shear.max())
    report["flags"] = flags
    report["flag_basis"] = "diagnostic thresholds, not acceptance gates"
    return report


# --------------------------------------------------------------------------- #
# the same errors in the forms agencies publish
# --------------------------------------------------------------------------- #

def _pick(groups):
    """Screened set when it exists (the headline), else every held-out one."""
    if not groups:
        return None, None
    if groups.get("screened"):
        return groups["screened"], "screened held-out"
    return groups.get("all"), "all held-out"


def conventions(q):
    rows = []
    src, basis = _pick(q.get("source"))
    raw = (q.get("source") or {}).get("all")
    residuals = q.get("residuals") or []

    def mean_distance(only_screened):
        d = [np.hypot(r["dx"], r["dy"]) for r in residuals
             if r["dx"] is not None and r["dy"] is not None
             and (not only_screened or r["screened"] is not False)]
        return float(np.mean(d)) if d else None

    if src:
        mean_px = mean_distance(basis.startswith("screened"))
        median = src.get("median_px")
        count = src.get("valid_n", 0)
        met = (None if mean_px is None or median is None else
               "under 0.5 px" if max(mean_px, median) < .5 else
               "under 1 px" if max(mean_px, median) < 1 else "not met")
        rows.append({"id": "asp", **AGENCY_SOURCES["asp"],
                     "ours": {"basis": basis, "mean_px": mean_px, "median_px": median, "count": count,
                              "all_held_out_mean_px": mean_distance(False),
                              "all_held_out_median_px": raw.get("median_px") if raw else None},
                     "guidance_met": met, "count_met": count >= 12,
                     "same": "image-space residual in the adjusted image's own pixels, mean and median",
                     "different": "ASP reprojects triangulated points through adjusted cameras; ours is "
                                  "backward transfer of held-out matcher correspondences through a 2-D warp"})
        rows.append({"id": "isis", **AGENCY_SOURCES["isis"],
                     "ours": {"basis": basis, "rmse_sample_px": (src.get("rmse_xy_px") or [None, None])[0],
                              "rmse_line_px": (src.get("rmse_xy_px") or [None, None])[1],
                              "rmse_px": src.get("rmse_px"), "count": count},
                     "same": "residuals split into sample and line components, in pixels",
                     "different": "jigsaw residuals are measure minus back-projected adjusted ground point; "
                                  "ours have no camera model or ground point"})
    ground, gbasis = _pick(q.get("ground"))
    if ground and ground.get("bias_en_m") is not None:
        published = AGENCY_SOURCES["kaguya"]["published"]
        gsd = q.get("source_gsd_m")
        mean, std = ground["bias_en_m"], ground.get("std_en_m") or [None, None]
        per_gsd = (None if not gsd else
                   {"east": [mean[0] / gsd, None if std[0] is None else std[0] / gsd],
                    "north": [mean[1] / gsd, None if std[1] is None else std[1] / gsd]})
        rows.append({"id": "kaguya", **AGENCY_SOURCES["kaguya"],
                     "ours": {"basis": gbasis, "east_m": [mean[0], std[0]], "north_m": [mean[1], std[1]],
                              "count": ground.get("valid_n"), "source_gsd_m": gsd, "per_source_gsd": per_gsd},
                     "published_per_gsd": {"longitude": [v / published["gsd_m"] for v in published["longitude_m"]],
                                           "latitude": [v / published["gsd_m"] for v in published["latitude_m"]]},
                     "same": "mean and 1-sigma of east (longitude) and north (latitude) differences in metres",
                     "different": "Kaguya compares TC terrain models of the same sites from different orbits "
                                  "after camera calibration; ours compares a cross-sensor strip with a mosaic "
                                  "through matcher correspondences. Published values are context, not a target"})
        rows.append({"id": "circular_error", "agency": "Horizontal circular error", "tool": "CE90 / CE95",
                     "reports": "radius containing 90% / 95% of horizontal errors in ground units",
                     "ours": {"basis": gbasis, "ce90_m": ground.get("p90_m"), "ce95_m": ground.get("p95_m"),
                              "count": ground.get("valid_n")},
                     "same": "empirical radial percentile in surface metres",
                     "different": "empirical over matcher observations relative to the reference mosaic, "
                                  "not a statistical accuracy bound on absolute position"})
    prof = q.get("profiles")
    if prof and prof.get("along"):
        filled = [b for b in prof["along"] if b["median_px"] is not None]
        if filled:
            worst = max(filled, key=lambda b: b["median_px"])
            best = min(filled, key=lambda b: b["median_px"])
            rows.append({"id": "sldem", **AGENCY_SOURCES["sldem"],
                         "ours": {"along_axis": prof.get("along_axis"),
                                  "bins": len(prof["along"]), "bins_with_data": len(filled),
                                  "worst_median_px": worst["median_px"], "worst_from": worst["from"],
                                  "worst_to": worst["to"], "best_median_px": best["median_px"],
                                  "drift_xy_px_per_1000": prof.get("drift_xy_px_per_1000")},
                         "same": "error reported per region along the product, not one global number",
                         "different": "SLDEM2015 residuals are vertical against LOLA altimetry; ours are "
                                      "horizontal image errors. Only the reporting practice carries over"})
    return rows


def load(job_dir, metrics=None):
    root = Path(job_dir)
    if metrics is None:
        metrics = json.loads((root / "metrics.json").read_text())
    return from_evidence(metrics, json.loads((root / "evaluation.json").read_text()),
                         json.loads((root / "transform.json").read_text()))
