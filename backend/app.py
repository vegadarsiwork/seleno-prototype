"""Seleno demo server.

    python backend/app.py            # http://127.0.0.1:8000

Serves the built frontend and a small JSON API. Pipeline runs happen on request;
result images are held in an in-memory cache keyed by run id and streamed as PNG.
"""
from __future__ import annotations

import io
import os
import sys
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from seleno import matchers, pipeline, store, verify, viz  # noqa: E402

DATA = os.path.join(ROOT, "data")
DIST = os.path.join(ROOT, "frontend", "dist")

app = FastAPI(title="Seleno", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STORE = store.Store(DATA)

# run_id -> {"images": {...}, "ts": float}
_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MAX_RUNS = 24


def _stash(images: dict) -> str:
    rid = uuid.uuid4().hex[:12]
    with _LOCK:
        _RUNS[rid] = {"images": images, "ts": time.time()}
        if len(_RUNS) > _MAX_RUNS:
            for k in sorted(_RUNS, key=lambda k: _RUNS[k]["ts"])[: len(_RUNS) - _MAX_RUNS]:
                _RUNS.pop(k, None)
    return rid


class RunRequest(BaseModel):
    pair: str
    options: dict = {}


class CompareRequest(BaseModel):
    pair: str
    configs: list[dict] = []


def _options_for(pair: dict, raw: dict) -> pipeline.Options:
    base = pipeline.Options()
    base.model_type = pair.get("recommended_model", "homography")
    merged = dict(base.__dict__)
    merged.update({k: v for k, v in (raw or {}).items() if k in merged})
    return pipeline.Options.from_dict(merged)


@app.get("/api/health")
def health():
    return {"ok": True, "version": "0.1.0"}


@app.get("/api/config")
def config():
    """Everything the UI needs to render its controls, driven by the backend."""
    return {
        "pairs": STORE.catalogue(),
        "products": STORE.manifest.get("products", {}),
        "matchers": [
            {"id": k, "label": v["label"], "kind": v["kind"],
             "available": bool(v["available"]), "reason": v.get("reason")}
            for k, v in matchers.MATCHERS.items()
        ],
        "models": [{"id": k, "label": v} for k, v in verify.MODELS.items()],
        "defaults": pipeline.Options().__dict__,
        "learned_matcher_status": {
            "available": matchers.LEARNED_AVAILABLE,
            "reason": matchers.LEARNED_REASON,
        },
    }


@app.post("/api/run")
def run(req: RunRequest):
    try:
        pair = STORE.pair(req.pair)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    opts = _options_for(pair, req.options)
    src, ref = STORE.images(req.pair)
    res = pipeline.run(pair, src, ref, opts)
    images = res.pop("_images")
    res["run_id"] = _stash(images)
    res["available_images"] = sorted(images.keys())
    res["pair"] = {k: v for k, v in pair.items() if k != "gt_homography"}
    return res


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Run several configurations on one pair and return them side by side.

    Used for the baseline-vs-variant table. Every row is a real run; nothing is
    copied from a previous request.
    """
    try:
        pair = STORE.pair(req.pair)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    src, ref = STORE.images(req.pair)
    rows = []
    for cfg in req.configs or []:
        label = cfg.pop("label", None) or cfg.get("matcher", "run")
        opts = _options_for(pair, cfg)
        try:
            res = pipeline.run(pair, src, ref, opts)
            imgs = res.pop("_images")
            err = res.get("error")
            rows.append({
                "label": label, "ok": res["ok"], "options": res["options"],
                # A stage that could not run has no numbers to report; sending
                # zeros would put a fabricated row in the comparison table.
                "metrics": None if err else res["metrics"],
                "verdict": res["verdict"], "warnings": res["warnings"],
                "error": err, "run_id": _stash(imgs),
                "available_images": sorted(imgs.keys()),
            })
        except Exception as exc:
            rows.append({"label": label, "ok": False, "error": str(exc),
                         "metrics": None, "verdict": None})
    return {"pair": req.pair, "rows": rows}


@app.get("/api/image/{run_id}/{name}")
def image(run_id: str, name: str, max_side: int = 900):
    rec = _RUNS.get(run_id)
    if rec is None:
        raise HTTPException(404, "run expired; re-run the pipeline")
    img = rec["images"].get(name)
    if img is None:
        raise HTTPException(404, "no image '%s' in this run" % name)
    return Response(viz.to_png_bytes(viz.fit(img, max_side)), media_type="image/png")


@app.get("/api/pair-image/{pair_id}/{which}")
def pair_image(pair_id: str, which: str, max_side: int = 900):
    if which not in ("source", "reference"):
        raise HTTPException(400, "which must be 'source' or 'reference'")
    try:
        pair = STORE.pair(pair_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    path = os.path.join(STORE.pairs_dir, pair[which + "_file"])
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(404, "missing file")
    return Response(viz.to_png_bytes(viz.fit(img, max_side)), media_type="image/png")


@app.get("/api/docs-md/{name}")
def docs_md(name: str):
    """Serve the project's own markdown so the UI can show assumptions in place."""
    allowed = {"README": "README.md", "RESEARCH": "RESEARCH.md",
               "ASSUMPTIONS": "ASSUMPTIONS.md", "DATA": os.path.join("data", "README.md")}
    if name not in allowed:
        raise HTTPException(404, "unknown document")
    path = os.path.join(ROOT, allowed[name])
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    with open(path, encoding="utf-8") as fh:
        return {"name": name, "markdown": fh.read()}


if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

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
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="info")
