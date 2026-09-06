#!/usr/bin/env python
"""Single entry point.

    python run.py

Checks the demo pairs exist, then serves the UI and API on
http://127.0.0.1:8000. Pass --port to change the port.
"""
import argparse
import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(ROOT, "data", "manifest.json")):
        sys.exit(
            "data/manifest.json is missing - the demo pairs have not been built.\n"
            "  python scripts/fetch_products.py --out data/raw\n"
            "  python scripts/prepare_data.py --raw data/raw")

    if not os.path.isdir(os.path.join(ROOT, "frontend", "dist")):
        print("note: frontend/dist is missing; only the JSON API will be served.\n"
              "      build it with:  cd frontend && npm install && npm run build\n")

    import uvicorn
    from app import app

    url = "http://127.0.0.1:%d" % args.port
    print("\n  SELENO  ->  %s\n" % url)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
