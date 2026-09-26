"""Run one registration for the web app and report over a pipe.

    python -m seleno.tool.jobrun <fd>   < {"source":..., "reference":..., "out":..., "options":{...}}

The server starts this as its own command, inside its own memory scope when
systemd can make one, so a job never shares memory with the server: whatever
the viewer or earlier requests left in the server's heap, a registration sizes
its working grid exactly as a command-line run under the same cap does.

Messages are JSON lines `[kind, payload]` on file descriptor `fd`, not stdout,
which libraries also print to. The file is line-buffered, so every line is in
the pipe before the next one is produced - including the last line before an
out-of-memory kill.
"""
import json
import os
import sys


def main():
    spec = json.load(sys.stdin)
    channel = os.fdopen(int(sys.argv[1]), "w", buffering=1)

    def send(kind, payload):
        channel.write(json.dumps([kind, payload]) + "\n")

    try:
        from seleno.tool import register
        res = register(spec["source"], spec["reference"], spec["out"],
                       progress=lambda line: send("log", line), verbose=False, **spec["options"])
        send("done", {"job_dir_id": res.job_id, "status": res.status,
                      "reason": res.reason, "metrics": res.metrics})
    except Exception as exc:                                          # noqa: BLE001
        send("crashed", "%s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
