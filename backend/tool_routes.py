"""HTTP surface for the Phase 7 registration tool.

Separate from the endpoints in `app.py`, which wrap the earlier OHRC-specific
pipeline. These drive `seleno.tool.register` on arbitrary file pairs: the same
entry point the CLI uses, with the same artifact set landing on disk. The
server adds nothing to the result and hides nothing from it - every number the
UI shows is read back out of the job's own `metrics.json`.

Runs happen in a worker thread and stream their real stage transitions through
`register(progress=...)`, so the progress the UI draws is the pipeline's own
log rather than a timer pretending to be one.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from seleno.tool import FAILURE_CODES, register
from seleno.tool.profiles import Profiles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUTS = os.path.join(ROOT, "outputs")

router = APIRouter(prefix="/api/tool", tags=["tool"])

# Extensions the tool will attempt. It degrades gracefully on anything it can
# open, so this list is about not showing the user a directory full of .oat and
# .spm files, not about what the reader can handle.
READABLE = (".xml", ".lbl", ".img", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".cub")

# Where real archive products live. Anything selectable from outside these roots
# is labelled a fixture in the listing and in the UI, because a demo that shows
# synthetic data without saying so is the failure mode this project is meant to
# avoid.
PRODUCT_ROOTS = (os.path.join(ROOT, "data", "raw"),)
FIXTURE_ROOTS = (os.path.join(ROOT, "data", "fixtures"),
                 os.path.join(ROOT, "tests", "fixtures"),
                 os.path.join(ROOT, "Dataset"))

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# input discovery
# --------------------------------------------------------------------------- #

def _classify(path: str) -> str:
    ap = os.path.abspath(path)
    for r in PRODUCT_ROOTS:
        if ap.startswith(os.path.abspath(r) + os.sep):
            return "product"
    for r in FIXTURE_ROOTS:
        if ap.startswith(os.path.abspath(r) + os.sep):
            return "fixture"
    return "fixture"


def _pairable(name: str) -> bool:
    low = name.lower()
    if not low.endswith(READABLE):
        return False
    # A PDS product is one logical file with several physical ones. Offer the
    # label, which carries the geometry, not the bare .img beside it.
    return True


def _scan(root: str, limit: int = 400) -> list[dict]:
    out: list[dict] = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in sorted(filenames):
            if not _pairable(f):
                continue
            full = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            out.append({"path": os.path.relpath(full, ROOT),
                        "name": f, "bytes": size,
                        "kind": _classify(full)})
            if len(out) >= limit:
                return out
    return out


def _dedupe_pds(items: list[dict]) -> list[dict]:
    """Collapse a PDS product's label and data file into one selectable entry.

    A .img next to a .xml or .lbl of the same stem is the detached data file of
    that label. Listing both invites the user to pick the one without geometry
    and then wonder why the run degraded.
    """
    labels = {os.path.splitext(i["path"])[0] for i in items
              if i["name"].lower().endswith((".xml", ".lbl"))}
    keep = []
    for i in items:
        stem = os.path.splitext(i["path"])[0]
        if i["name"].lower().endswith(".img") and stem in labels:
            continue
        keep.append(i)
    return keep


@router.get("/files")
def list_files():
    """Selectable inputs, each labelled product or fixture."""
    items: list[dict] = []
    for r in PRODUCT_ROOTS + FIXTURE_ROOTS:
        items += _scan(r)
    items = _dedupe_pds(items)
    prof = Profiles.load()
    for i in items:
        p = prof.match("", i["path"])
        i["instrument"] = p.get("name", "unknown")
        i["gsd_m"] = p.get("gsd_m")
    items.sort(key=lambda i: (i["kind"] != "product", i["instrument"], i["name"]))
    return {"root": ROOT, "count": len(items), "files": items,
            "note": "entries marked 'fixture' are not archive products"}


@router.get("/profiles")
def list_profiles():
    prof = Profiles.load()
    return {"profiles": [dict(prof.get(n), key=n) for n in prof.names()]}


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #

class RunRequest(BaseModel):
    source: str
    reference: str
    model: str = "auto"
    max_side: int = 2048
    grid: int = 8
    segments: int = 0
    subpixel: bool = True


def _resolve(rel: str) -> str:
    """Resolve a client-supplied path, refusing anything outside the repo.

    The picker only ever sends paths this server handed out, but it is the
    server's job to enforce that rather than to trust it.
    """
    full = os.path.abspath(os.path.join(ROOT, rel))
    if not (full == ROOT or full.startswith(ROOT + os.sep)):
        raise HTTPException(400, "path outside the project: %r" % rel)
    if not os.path.exists(full):
        raise HTTPException(404, "no such file: %r" % rel)
    return full


def _worker(jid: str, req: RunRequest, src: str, ref: str):
    job = _JOBS[jid]

    def progress(line: str):
        with _LOCK:
            job["log"].append({"t": round(time.time() - job["started"], 2),
                               "line": line})
            job["stage"] = line.split(":")[0].strip() if ":" in line else line[:40]

    try:
        res = register(src, ref, OUTPUTS, model=req.model, max_side=req.max_side,
                       grid=(req.grid, req.grid), segments=req.segments,
                       subpixel=req.subpixel, progress=progress, verbose=False)
        with _LOCK:
            job.update(state="done", job_dir_id=res.job_id, status=res.status,
                       reason=res.reason, metrics=res.metrics,
                       finished=time.time())
    except Exception as exc:                                          # noqa: BLE001
        # A crash here is a bug in the tool, not a registration failure - the
        # tool reports those through metrics.json. Keep them distinguishable.
        with _LOCK:
            job.update(state="crashed", error="%s: %s" % (type(exc).__name__, exc),
                       finished=time.time())


@router.post("/register")
def start(req: RunRequest):
    src, ref = _resolve(req.source), _resolve(req.reference)
    jid = uuid.uuid4().hex[:12]
    _JOBS[jid] = {"id": jid, "state": "running", "stage": "starting",
                  "source": req.source, "reference": req.reference,
                  "options": req.model_dump(), "log": [],
                  "started": time.time(), "finished": None,
                  "metrics": None, "status": None, "reason": None,
                  "job_dir_id": None, "error": None}
    threading.Thread(target=_worker, args=(jid, req, src, ref), daemon=True).start()
    return {"job": jid}


@router.get("/jobs")
def jobs():
    with _LOCK:
        return {"jobs": [{k: v for k, v in j.items() if k not in ("metrics", "log")}
                         for j in sorted(_JOBS.values(),
                                         key=lambda x: -x["started"])][:40]}


@router.get("/jobs/{jid}")
def job(jid: str, since: int = 0):
    j = _JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "no such job")
    with _LOCK:
        out = {k: v for k, v in j.items() if k != "log"}
        out["log"] = j["log"][since:]
        out["log_len"] = len(j["log"])
        out["elapsed_s"] = round((j["finished"] or time.time()) - j["started"], 2)
    out["failure_codes"] = list(FAILURE_CODES)
    return out


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #

def _job_dir(jid: str) -> str:
    j = _JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "no such job")
    if not j.get("job_dir_id"):
        raise HTTPException(409, "job has not produced artifacts yet")
    d = os.path.join(OUTPUTS, j["job_dir_id"])
    if not os.path.isdir(d):
        raise HTTPException(404, "artifact directory is gone")
    return d


ARTIFACTS = ("metrics.json", "transform.json", "matches.csv", "report.md",
             "preview.json", "overlay.png", "source.png", "reference.png",
             "registered.png", "registered.tif")


@router.get("/jobs/{jid}/artifacts")
def artifacts(jid: str):
    d = _job_dir(jid)
    have = []
    for n in ARTIFACTS:
        p = os.path.join(d, n)
        if os.path.exists(p):
            have.append({"name": n, "bytes": os.path.getsize(p)})
    return {"job": jid, "artifacts": have}


@router.get("/jobs/{jid}/file/{name}")
def artifact(jid: str, name: str):
    if name not in ARTIFACTS:
        raise HTTPException(404, "not an artifact of this tool: %r" % name)
    p = os.path.join(_job_dir(jid), name)
    if not os.path.exists(p):
        raise HTTPException(404, "%s was not written for this job" % name)
    if name.endswith(".json"):
        return Response(open(p, "rb").read(), media_type="application/json")
    if name.endswith(".png"):
        return Response(open(p, "rb").read(), media_type="image/png")
    if name.endswith(".md"):
        return Response(open(p, "rb").read(), media_type="text/markdown")
    return FileResponse(p, filename=name)


@router.get("/jobs/{jid}/download")
def download(jid: str):
    """Every artifact of one run, as a zip. This is the deliverable."""
    d = _job_dir(jid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                z.write(p, arcname=os.path.join(_JOBS[jid]["job_dir_id"], n))
    buf.seek(0)
    return Response(buf.read(), media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="seleno_%s.zip"'
                               % _JOBS[jid]["job_dir_id"]})
