"""Seleno demo server.

    python run.py            # http://127.0.0.1:8000

Serves the built frontend plus a JSON API. Two families of endpoints:

* ``/api/ohrc/*`` and ``/api/register`` - the current architecture, running on
  real Chandrayaan-2 OHRC products.
* ``/api/config``, ``/api/run``, ``/api/compare`` - retained from the earlier
  prototype so the six legacy manifest pairs still work and their numbers stay
  reproducible.

Pipeline runs happen on request. Result images live in a small in-memory cache
keyed by run id and are streamed as PNG; nothing is precomputed, so what the UI
shows is what the run produced.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from seleno import (alignment, illumination, matchers, ohrc, pairs,   # noqa: E402
                    pipeline, registration, store, verify, viz)
from seleno.experiments import ABLATION, PRESETS, build_options       # noqa: E402
from seleno.ohrc import tiles as T                                    # noqa: E402

DATA = os.path.join(ROOT, "data")
DIST = os.path.join(ROOT, "frontend", "dist")
CACHE = os.path.join(ROOT, "results", "dataset")
ALIGN_CACHE = os.path.join(ROOT, "results", "illumination")

app = FastAPI(title="Seleno", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

try:
    STORE = store.Store(DATA)
except FileNotFoundError:
    STORE = None

_PRODUCTS: list = []
_PRODUCTS_ERR = ""
try:
    _PRODUCTS = ohrc.discover()
    if not _PRODUCTS:
        _PRODUCTS_ERR = ("no OHRC products found; set SELENO_OHRC_ROOT or place the "
                         "archive at ../datasettesting/dataset")
except Exception as exc:                                              # noqa: BLE001
    _PRODUCTS_ERR = "OHRC discovery failed: %s" % exc

_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MAX_RUNS = 40


def _stash(images: dict) -> str:
    rid = uuid.uuid4().hex[:12]
    with _LOCK:
        _RUNS[rid] = {"images": images, "ts": time.time()}
        if len(_RUNS) > _MAX_RUNS:
            for k in sorted(_RUNS, key=lambda k: _RUNS[k]["ts"])[: len(_RUNS) - _MAX_RUNS]:
                _RUNS.pop(k, None)
    return rid


def _product(ts: str):
    if not _PRODUCTS:
        raise HTTPException(503, _PRODUCTS_ERR or "no OHRC products loaded")
    try:
        return ohrc.by_timestamp(_PRODUCTS, ts)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


def _png(img, max_side: int = 900) -> Response:
    return Response(viz.to_png_bytes(viz.fit(img, max_side)), media_type="image/png")


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #

class RegisterRequest(BaseModel):
    # OHRC window pair
    source: str | None = None
    reference: str | None = None
    sample0: int | None = None
    line0: int | None = None
    size: int = 1024
    apply_coarse_offset: bool = True
    # or a legacy manifest pair
    legacy: str | None = None
    # configuration
    preset: str = "seleno"
    options: dict = {}


class CompareRequest(RegisterRequest):
    arms: list[str] | None = None


class RunRequest(BaseModel):
    """Legacy endpoint payload, unchanged."""

    pair: str
    options: dict = {}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    return {"ok": True, "version": "0.2.0",
            "ohrc_products": len(_PRODUCTS), "ohrc_error": _PRODUCTS_ERR}


@app.get("/api/config")
def config():
    legacy_pairs = STORE.catalogue() if STORE else []
    return {
        "version": "0.2.0",
        "ohrc": {
            "available": bool(_PRODUCTS),
            "error": _PRODUCTS_ERR,
            "root": _PRODUCTS[0].root if _PRODUCTS else None,
            "products": [
                {"timestamp": p.timestamp, "product_id": p.product_id,
                 "lines": p.lines, "samples": p.samples, "gsd_m": p.gsd_m,
                 "start_time": p.label.start_time.isoformat() if p.label.start_time else None,
                 "imaging_orbit": p.label.imaging_orbit,
                 "sun_azimuth_deg": p.label.sun_azimuth_deg,
                 "sun_elevation_deg": p.label.sun_elevation_deg,
                 "solar_incidence_deg": p.label.solar_incidence_deg,
                 "spacecraft_altitude_km": p.label.spacecraft_altitude_km,
                 "roll_deg": p.label.roll_deg,
                 "limb_direction": p.label.limb_direction,
                 "corners_refined": p.label.corners_refined,
                 "reference_data_used": p.label.reference_data_used,
                 "label_notes": p.label.notes,
                 "has_geometry": p.has_geometry, "has_sun_series": p.has_sun_series}
                for p in _PRODUCTS],
        },
        "legacy_pairs": legacy_pairs,
        "matchers": [
            {"id": k, "label": v["label"], "kind": v["kind"],
             "available": bool(v["available"]), "reason": v.get("reason")}
            for k, v in matchers.MATCHERS.items()],
        "models": [{"id": k, "label": v} for k, v in verify.MODELS.items()],
        "mask_modes": [{"id": k, "label": v} for k, v in illumination.MASK_MODES.items()],
        "terrain_models": [
            {"id": k, "label": v["label"], "status": v["status"]}
            for k, v in illumination.TERRAIN_MODELS.items()],
        "presets": sorted(PRESETS),
        "ablation": [{"code": c, "preset": p, "label": l} for c, p, l in ABLATION],
        "defaults": registration.Options().__dict__,
        "thresholds": registration.THRESHOLDS,
        "thresholds_calibrated": False,
        "selection_modes": [
            {"id": "grid", "label": "Spatial grid quota"},
            {"id": "topk", "label": "Top-K by confidence (same budget)"},
            {"id": "all", "label": "All verified inliers"}],
        "stage_names": [
            "Metadata and geometry", "Reference and overlap estimation",
            "Illumination-aware preprocessing", "Correspondence",
            "Robust geometric verification", "Correspondence selection",
            "Sub-pixel refinement", "Metrics", "Trust and refusal decision"],
    }


# --------------------------------------------------------------------------- #
# OHRC browsing
# --------------------------------------------------------------------------- #

@app.get("/api/ohrc/products")
def ohrc_products():
    if not _PRODUCTS:
        raise HTTPException(503, _PRODUCTS_ERR or "no OHRC products loaded")
    return {"products": [p.describe() for p in _PRODUCTS]}


@app.get("/api/ohrc/product/{ts}")
def ohrc_product(ts: str):
    p = _product(ts)
    d = p.describe()
    try:
        idx = T.StripIndex.cached(p, CACHE, tile=512, stride=2048, step=8)
        d["tile_index"] = idx.summary()
    except Exception as exc:                                          # noqa: BLE001
        d["tile_index_error"] = str(exc)
    return d


@app.get("/api/ohrc/overlaps")
def ohrc_overlaps():
    if len(_PRODUCTS) < 2:
        return {"overlaps": []}
    return {"overlaps": ohrc.overlap_matrix(_PRODUCTS, cell_m=100.0)}


@app.get("/api/ohrc/thumbnail/{ts}")
def ohrc_thumbnail(ts: str, max_side: int = 1000, rotate: bool = True):
    p = _product(ts)
    th = T.normalise_for_display(p.thumbnail(max_side))
    if rotate:
        th = cv2.rotate(th, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return _png(th, max_side)


@app.get("/api/ohrc/tile/{ts}")
def ohrc_tile(ts: str, x: int = 0, y: int = 0, w: int = 512, h: int = 512,
              step: int = 1, stretch: bool = True, max_side: int = 700):
    p = _product(ts)
    tile = p.read_tile(x, y, w, h, step=step)
    if tile.size == 0:
        raise HTTPException(400, "empty window")
    return _png(T.normalise_for_display(tile) if stretch else tile, max_side)


@app.get("/api/ohrc/tile-stats/{ts}")
def ohrc_tile_stats(ts: str, x: int = 0, y: int = 0, w: int = 512, h: int = 512):
    p = _product(ts)
    tile = p.read_tile(x, y, w, h, step=max(1, min(w, h) // 256))
    st = T.tile_stat(tile, x, y, w, h)
    out = st.to_dict()
    if p.has_geometry:
        out["footprint"] = p.window_footprint(x, y, w, h)
    if p.has_sun_series:
        out["sun"] = p.sun_at_window(y, h)
    return out


@app.get("/api/ohrc/windows/{a_ts}/{b_ts}")
def ohrc_windows(a_ts: str, b_ts: str, size: int = 1024, limit: int = 8,
                 apply_coarse_offset: bool = True):
    """Candidate window pairs over mutually usable ground."""
    a, b = _product(a_ts), _product(b_ts)
    align = alignment.get_offset(a, b, ALIGN_CACHE)
    off = alignment.offset_or_zero(align) if apply_coarse_offset else (0.0, 0.0)
    idx = T.StripIndex.cached(a, CACHE, tile=512, stride=1024, step=8)
    cand, used = [], []
    for st in sorted(idx.usable(), key=lambda t: -t.std):
        if len(cand) >= limit * 3:
            break
        s0 = int(np.clip(st.sample0 - (size - 512) // 2, 0, a.samples - size))
        l0 = int(np.clip(st.line0 - (size - 512) // 2, 0, a.lines - size))
        if any(abs(l0 - u) < 3 * size for u in used):
            continue
        try:
            pr = pairs.ohrc_window_pair(a, b, s0, l0, size, offset_stereo_m=off)
        except ValueError:
            continue
        ref_std = float(pr.reference.std())
        if ref_std < 4.0:
            continue
        used.append(l0)
        cand.append({"sample0": s0, "line0": l0, "size": size,
                     "source_mean": st.mean, "source_std": st.std,
                     "reference_mean": round(float(pr.reference.mean()), 2),
                     "reference_std": round(ref_std, 2),
                     "min_std": round(min(st.std, ref_std), 2),
                     "illumination": st.illumination})
    # Rank by the WEAKER of the two sides. A correspondence needs texture in both
    # images, and on this pair the reference side is the binding constraint:
    # windows that registered had reference DN sd >= 42, those refused 24-30.
    # Sorting on the source alone puts a bright-but-unmatchable pair first.
    cand.sort(key=lambda w: -w["min_std"])
    return {"source": a.timestamp, "reference": b.timestamp,
            "coarse_alignment": align, "applied_offset_stereo_m": list(off),
            "ranking": "descending min(source_std, reference_std)",
            "windows": cand[:limit]}


@app.get("/api/ohrc/alignment/{a_ts}/{b_ts}")
def ohrc_alignment(a_ts: str, b_ts: str, rebuild: bool = False):
    a, b = _product(a_ts), _product(b_ts)
    return alignment.get_offset(a, b, ALIGN_CACHE, rebuild=rebuild)


# --------------------------------------------------------------------------- #
# the staged pipeline
# --------------------------------------------------------------------------- #

def _build_pair(req: RegisterRequest):
    if req.legacy:
        if STORE is None:
            raise HTTPException(503, "legacy manifest not available")
        try:
            return pairs.legacy_pair(STORE, req.legacy), None
        except KeyError as exc:
            raise HTTPException(404, str(exc))
    if not (req.source and req.reference and req.sample0 is not None
            and req.line0 is not None):
        raise HTTPException(400, "need source, reference, sample0 and line0, or legacy")
    a, b = _product(req.source), _product(req.reference)
    align = alignment.get_offset(a, b, ALIGN_CACHE)
    off = alignment.offset_or_zero(align) if req.apply_coarse_offset else (0.0, 0.0)
    try:
        pr = pairs.ohrc_window_pair(a, b, req.sample0, req.line0, req.size,
                                    offset_stereo_m=off, coarse_alignment=align)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return pr, align


@app.post("/api/register")
def api_register(req: RegisterRequest):
    pair, align = _build_pair(req)
    try:
        opts = build_options(req.preset, pair, req.options or {})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    res = registration.run(pair, opts)
    out = res.json()
    out["run_id"] = _stash(res.images)
    out["coarse_alignment"] = align
    return out


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """Run several ablation arms on one pair. Every row is a real run."""
    if req.legacy is None and req.source is None:
        # legacy signature: {"pair": ..., "configs": [...]} handled by /api/run
        raise HTTPException(400, "need a pair specification")
    pair, align = _build_pair(req)
    codes = req.arms
    arms = ABLATION if not codes else [
        (c, p, l) for c, p, l in ABLATION if c in codes or p in codes]
    if not arms:
        arms = [(p[:3].upper(), p, p) for p in (codes or [])]
    rows = []
    for code, preset, label in arms:
        try:
            opts = build_options(preset, pair, req.options or {})
            res = registration.run(pair, opts)
            rows.append({
                "code": code, "preset": preset, "label": label,
                "status": res.status, "confidence": res.confidence,
                "metrics": res.metrics, "reasons": res.reasons,
                "warnings": res.warnings, "options": res.options,
                "run_id": _stash(res.images),
                "available_images": sorted(res.images.keys()),
            })
        except Exception as exc:                                      # noqa: BLE001
            rows.append({"code": code, "preset": preset, "label": label,
                         "status": "error", "error": str(exc), "metrics": None})
    return {"pair": pair.meta(), "coarse_alignment": align, "rows": rows}


@app.get("/api/pair-preview")
def pair_preview(source: str, reference: str, sample0: int, line0: int,
                 size: int = 1024, which: str = "source",
                 apply_coarse_offset: bool = True, max_side: int = 640):
    a, b = _product(source), _product(reference)
    align = alignment.get_offset(a, b, ALIGN_CACHE)
    off = alignment.offset_or_zero(align) if apply_coarse_offset else (0.0, 0.0)
    try:
        pr = pairs.ohrc_window_pair(a, b, sample0, line0, size, offset_stereo_m=off)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    img = pr.source if which == "source" else pr.reference
    return _png(T.normalise_for_display(img), max_side)


@app.get("/api/image/{run_id}/{name}")
def image(run_id: str, name: str, max_side: int = 900):
    rec = _RUNS.get(run_id)
    if rec is None:
        raise HTTPException(404, "run expired; re-run the pipeline")
    img = rec["images"].get(name)
    if img is None:
        raise HTTPException(404, "no image %r in this run" % name)
    return _png(img, max_side)


# --------------------------------------------------------------------------- #
# legacy endpoints, preserved
# --------------------------------------------------------------------------- #

@app.post("/api/run")
def api_run(req: RunRequest):
    """The earlier prototype's pipeline on the six manifest pairs."""
    if STORE is None:
        raise HTTPException(503, "legacy manifest not available")
    try:
        pair = STORE.pair(req.pair)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    base = pipeline.Options()
    base.model_type = pair.get("recommended_model", "homography")
    merged = dict(base.__dict__)
    merged.update({k: v for k, v in (req.options or {}).items() if k in merged})
    src, ref = STORE.images(req.pair)
    res = pipeline.run(pair, src, ref, pipeline.Options.from_dict(merged))
    images = res.pop("_images")
    res["run_id"] = _stash(images)
    res["available_images"] = sorted(images.keys())
    res["pair"] = {k: v for k, v in pair.items() if k != "gt_homography"}
    return res


@app.get("/api/pair-image/{pair_id}/{which}")
def pair_image(pair_id: str, which: str, max_side: int = 900):
    if STORE is None:
        raise HTTPException(503, "legacy manifest not available")
    if which not in ("source", "reference"):
        raise HTTPException(400, "which must be 'source' or 'reference'")
    try:
        pair = STORE.pair(pair_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    img = cv2.imread(os.path.join(STORE.pairs_dir, pair[which + "_file"]),
                     cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(404, "missing file")
    return _png(img, max_side)


@app.get("/api/docs-md/{name}")
def docs_md(name: str):
    allowed = {"README": "README.md", "RESEARCH": "RESEARCH.md",
               "ASSUMPTIONS": "ASSUMPTIONS.md", "HANDOFF": "HANDOFF.md",
               "DATA": os.path.join("data", "README.md"),
               "ILLUMINATION": os.path.join("results", "illumination", "ILLUMINATION.md"),
               "DATASET": os.path.join("results", "dataset", "dataset_summary.json")}
    if name not in allowed:
        raise HTTPException(404, "unknown document")
    path = os.path.join(ROOT, allowed[name])
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    with open(path, encoding="utf-8") as fh:
        return {"name": name, "markdown": fh.read()}


# --------------------------------------------------------------------------- #

if os.path.isdir(DIST):
    assets = os.path.join(DIST, "assets")
    if os.path.isdir(assets):
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(DIST, "index.html"))
else:
    @app.get("/")
    def index_missing():
        return Response(
            "<h1>Seleno</h1><p>Frontend is not built. Run:</p>"
            "<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>"
            "<p>The JSON API is live at <code>/api/config</code>.</p>",
            media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)),
                log_level="info")
