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
import re
import shutil
import threading
import time
import uuid
import zipfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from seleno.tool import FAILURE_CODES, register
from seleno.tool import view as tview
from seleno.tool.profiles import Profiles
from seleno.tool.scene import UnreadableInput

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUTS = os.path.join(ROOT, "outputs")

router = APIRouter(prefix="/api/tool", tags=["tool"])

# Extensions the tool will attempt. It degrades gracefully on anything it can
# open, so this list is about not showing the user a directory full of .oat and
# .spm files, not about what the reader can handle.
READABLE = (".xml", ".lbl", ".img", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".cub",
            ".jp2", ".j2k", ".bmp", ".gif", ".webp", ".vrt", ".fits", ".fit")

# Where real archive products live. Anything selectable from outside these roots
# is labelled a fixture in the listing and in the UI, because a demo that shows
# synthetic data without saying so is the failure mode this project is meant to
# avoid.
PRODUCT_ROOTS = (os.path.join(ROOT, "data", "raw"),)
FIXTURE_ROOTS = (os.path.join(ROOT, "data", "fixtures"),
                 os.path.join(ROOT, "tests", "fixtures"),
                 os.path.join(ROOT, "Dataset"))
# Files a user uploaded through the UI. Neither archive products nor fixtures:
# their provenance is whatever the user says it is, so they get their own label.
UPLOADS = os.path.join(ROOT, "data", "uploads")
VIEW_CACHE = os.path.join(OUTPUTS, ".viewcache")

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# input discovery
# --------------------------------------------------------------------------- #

def _classify(path: str) -> str:
    ap = os.path.abspath(path)
    if ap.startswith(os.path.abspath(UPLOADS) + os.sep):
        return "upload"
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
    for r in (UPLOADS,) + PRODUCT_ROOTS + FIXTURE_ROOTS:
        items += _scan(r)
    items = _dedupe_pds(items)
    prof = Profiles.load()
    for i in items:
        p = prof.match("", i["path"])
        i["instrument"] = p.get("name", "unknown")
        i["gsd_m"] = p.get("gsd_m")
    rank = {"upload": 0, "product": 1, "fixture": 2}
    items.sort(key=lambda i: (rank.get(i["kind"], 3), i["instrument"], i["name"]))
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
    max_side: int | None = None
    grid: int | None = None
    segments: int = 0
    subpixel: bool = True
    locate: str = "auto"


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
                       grid=(req.grid, req.grid) if req.grid is not None else None, segments=req.segments,
                       subpixel=req.subpixel, locate=req.locate,
                       progress=progress, verbose=False)
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


ARTIFACTS = ("evaluation.json", "metrics.json", "transform.json", "matches.csv", "report.md",
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


# --------------------------------------------------------------------------- #
# viewing: any input, any zoom
# --------------------------------------------------------------------------- #

def _view(rel: str) -> tview.Viewable:
    full = _resolve(rel)
    if os.path.isdir(full):
        raise HTTPException(400, "%r is a directory" % rel)
    try:
        return tview.open_view(full, VIEW_CACHE)
    except UnreadableInput as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:                                          # noqa: BLE001
        raise HTTPException(422, "could not open %s for viewing (%s: %s)"
                            % (os.path.basename(full), type(exc).__name__, exc))


@router.get("/view/info")
def view_info(path: str):
    """Size, georeferencing and stretch of one input.

    The first call on a large file builds its overview, one pass over the file;
    the result is cached on disk, so later calls return at once.
    """
    v = _view(path)
    return dict(v.info(), path=path, name=os.path.basename(v.path))


@router.get("/view/tile")
def view_tile(path: str, s: int, x: int, y: int):
    """One 256 px tile at 1/s scale. The client adds `v=<mtime>` to the URL, so
    a replaced file never serves a stale cached tile."""
    v = _view(path)
    try:
        a = tview.tile(v, s, x, y)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return Response(tview.png(a), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/view/region")
def view_region(path: str, x0: float, y0: float, x1: float, y1: float):
    """A window of native pixels as a PNG download, at the finest scale that
    keeps it under the size and read limits in `seleno.tool.view`."""
    v = _view(path)
    try:
        a, (bx0, by0, bx1, by1), s = tview.region(v, x0, y0, x1, y1)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    stem = os.path.splitext(os.path.basename(v.path))[0]
    fn = "%s_x%d-%d_y%d-%d_1to%d.png" % (stem, bx0, bx1, by0, by1, s)
    return Response(tview.png(a, 3), media_type="image/png", headers={
        "Content-Disposition": 'attachment; filename="%s"' % fn,
        "X-Scale": str(s)})


# --------------------------------------------------------------------------- #
# uploads
# --------------------------------------------------------------------------- #

_BATCH = re.compile(r"[a-z0-9]{6,32}")
_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def _upload_target(batch: str, name: str) -> str:
    """Where an uploaded file lands.

    Each upload batch gets its own directory, so a PDS label and its data file
    land side by side and two uploads with the same name do not collide.
    Relative folders inside a dropped folder are kept, because a PDS4 product
    finds its geometry CSV by walking up from the image. Every segment is
    sanitised, and the result must stay inside the batch directory.
    """
    if not _BATCH.fullmatch(batch):
        raise HTTPException(400, "bad upload batch id")
    segs = []
    for seg in name.replace("\\", "/").split("/"):
        seg = _UNSAFE.sub("_", seg).lstrip(".")
        if seg:
            segs.append(seg)
    if not segs or len(segs) > 8:
        raise HTTPException(400, "bad file name: %r" % name)
    base = os.path.join(UPLOADS, batch)
    dest = os.path.abspath(os.path.join(base, *segs))
    if not dest.startswith(base + os.sep):
        raise HTTPException(400, "bad file name: %r" % name)
    return dest


@router.put("/upload")
async def upload(request: Request, batch: str, name: str):
    """Stream one file to disk. The body is the raw file, not a multipart form,
    so a 6 GB mosaic goes straight to disk instead of being buffered first."""
    dest = _upload_target(batch, name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    declared = request.headers.get("content-length")
    declared = int(declared) if declared and declared.isdigit() else None
    if declared is not None:
        free = shutil.disk_usage(os.path.dirname(dest)).free
        if declared > free - (1 << 30):
            raise HTTPException(507, "not enough disk space: %.1f GB needed, %.1f GB free"
                                % (declared / 1e9, free / 1e9))
    part = dest + ".part"
    n = 0
    fh = open(part, "wb")
    try:
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) >= 8 << 20:
                data, buf = bytes(buf), bytearray()
                await run_in_threadpool(fh.write, data)
                n += len(data)
        if buf:
            await run_in_threadpool(fh.write, bytes(buf))
            n += len(buf)
        fh.close()
        if declared is not None and n != declared:
            raise HTTPException(400, "upload truncated: %d of %d bytes" % (n, declared))
        os.replace(part, dest)
    except BaseException:
        fh.close()
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    return {"path": os.path.relpath(dest, ROOT), "name": os.path.basename(dest),
            "bytes": n}
