"""Exercise HTTP handlers through ASGI without a server or extra dependency."""
import asyncio
import io
import json
from pathlib import Path
import sys
import zipfile
import faulthandler
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import app
import tool_routes

HERE = Path(__file__).resolve().parent
tool_routes.OUTPUTS = str(ROOT / "outputs/validation_20260923/api")
torch.set_num_threads(2)


async def request(path, body=None):
    print("REQUEST", path, flush=True)
    data = b"" if body is None else json.dumps(body).encode()
    sent = []
    consumed = False

    async def receive():
        nonlocal consumed
        if consumed:
            await asyncio.Event().wait()
        consumed = True
        return {"type": "http.request", "body": data, "more_body": False}

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "method": "GET" if body is None else "POST", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "scheme": "http", "http_version": "1.1", "server": ("test", 80),
             "client": ("test", 123), "headers": [(b"content-type", b"application/json")]}
    await app(scope, receive, send)
    print("RESPONSE", path, flush=True)
    status = next(x["status"] for x in sent if x["type"] == "http.response.start")
    payload = b"".join(x.get("body", b"") for x in sent if x["type"] == "http.response.body")
    assert status == 200, (path, status, payload[:500])
    return payload


async def main():
    # A timer also keeps the event loop responsive in restricted execution
    # environments where a thread's self-pipe wakeup may not be delivered.
    async def heartbeat():
        while True:
            await asyncio.sleep(.1)
    asyncio.create_task(heartbeat())
    files = json.loads(await request("/api/tool/files"))
    path = "outputs/validation_20260923/contract_inputs/native_identity.tif"
    response = json.loads(await request("/api/tool/register", {
        "source": path, "reference": path, "max_side": 512}))
    jid = response["job"]
    while True:
        job = json.loads(await request("/api/tool/jobs/" + jid))
        if job["state"] != "running":
            break
        await asyncio.sleep(.25)
    assert job["state"] == "done", job
    metrics = json.loads(await request(f"/api/tool/jobs/{jid}/file/metrics.json"))
    assert metrics == job["metrics"]
    tif = await request(f"/api/tool/jobs/{jid}/file/registered.tif")
    archive = await request(f"/api/tool/jobs/{jid}/download")
    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert len(names) == 10, names
    report = {"status": "passed", "job_status": job["status"], "file_listing_keys": list(files),
              "registered_tif_bytes": len(tif), "zip_entries": names,
              "metrics_equal_job_response": True, "log_lines": job["log_len"]}
    (HERE / "api.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


faulthandler.dump_traceback_later(30, repeat=True)
asyncio.run(main())
faulthandler.cancel_dump_traceback_later()
