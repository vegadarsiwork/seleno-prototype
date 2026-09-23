"""Small targeted probes of exported coordinates, validation and metadata."""
import contextlib
import importlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import Affine
import torch

from run_audit import HERE, OUT, ROOT, CRS, write_tif, run

torch.set_num_threads(2)
sys.path.insert(0, str(ROOT / "tests"))
from test_tool import terrain
from seleno.tool.scene import load
from seleno.tool.register import _target_grid, prealign

RG = importlib.import_module("seleno.tool.register")
results = {}
inputs = OUT / "contract_inputs"
inputs.mkdir(parents=True, exist_ok=True)
a = .1 + .8 * terrain(152, 256)
src = write_tif(inputs / "native_identity.tif", a, Affine(1, 0, 1000, 0, -1, 1000))
results["native_identity"] = run("contract_native_identity", src, src,
                                 truth=np.eye(3), shape=a.shape, max_side=512)

# Trace which fitted matrix is actually written when ECC is disabled.
original_refit = RG.__dict__.get("unused", None)
import seleno.register as fits
refits = []
orig = fits.refit

def capture(*args, **kw):
    H = orig(*args, **kw)
    refits.append(None if H is None else H.copy())
    return H

fixture = OUT / "inputs"
with patch.object(fits, "refit", capture):
    r = run("contract_holdout", fixture / "translation.png", fixture / "reference.png",
            max_side=512, fine=False, subpixel=False)
tj = json.loads((Path(r["out_dir"]) / "transform.json").read_text())
results["holdout_export"] = {
    "heldout_count": r["accuracy"]["held_out_n"], "refit_calls": len(refits),
    "export_equals_holdout_fit": bool(np.allclose(tj["matrix"], refits[0])),
    "max_matrix_difference": float(np.max(np.abs(np.asarray(tj["matrix"]) - refits[0]))),
    "run": r,
}

# Native reference validity must exclude finite nodata, just as the source does.
masked = a.copy()
masked[:64, :] = -9999
maskfile = inputs / "finite_nodata.tif"
with rasterio.open(maskfile, "w", driver="GTiff", height=256, width=256, count=1,
                   dtype="float32", crs=CRS, transform=Affine(1, 0, 1000, 0, -1, 1000),
                   nodata=-9999) as ds:
    ds.write(masked, 1)
scene = load(str(maskfile))
step, origin, target, valid, centre = _target_grid(scene, 512)
results["native_reference_nodata"] = {
    "nodata": scene.nodata, "expected_invalid_pixels": 64 * 256,
    "observed_invalid_pixels": int((~valid).sum()),
    "finite_nodata_marked_valid": bool(valid[:64].all()),
}

# A geographic GeoTIFF gives angular sampling, not equal metre widths in x/y.
gfile = inputs / "geographic.tif"
with rasterio.open(gfile, "w", driver="GTiff", height=256, width=256, count=1,
                   dtype="float32", crs="+proj=longlat +R=1737400",
                   transform=Affine(.001, 0, 38, 0, -.001, -60)) as ds:
    ds.write(a, 1)
gs = load(str(gfile))
results["geographic_units"] = {"reported_scalar_gsd_m": gs.gsd_m,
    "approx_x_m_at_60south": np.pi / 180 * 1737400 * .001 * .5,
    "approx_y_m": np.pi / 180 * 1737400 * .001}

(HERE / "contracts.json").write_text(json.dumps(results, indent=2, default=str))
print(json.dumps({k: v for k, v in results.items() if k != "native_identity"},
                 indent=2, default=str))
