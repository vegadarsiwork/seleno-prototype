"""After the fix: the same instrumented probe as ../ohrc_offset/probe_fine.py.

It reports what the fine stage's correspondences are made of, and how the
coarse, final and true models compare.

    OPENBLAS_NUM_THREADS=2 .venv/bin/python -u \
      reports/validation_fixes_20260923/heldout_independence/probe_after.py
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("probe_fine", HERE.parent / "ohrc_offset" / "probe_fine.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def main():
    controls = {c["name"]: c for c in json.loads((P.DIAG / "controls.json").read_text())}
    out = []
    for n in ("control_ohrc_nac", "control_ohrc_nac_km", "control_tmc2_selene"):
        print("PROBE", n, flush=True)
        out.append(P.probe(controls[n]))
        print(json.dumps(out[-1]), flush=True)
    (HERE / "probe_after.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
