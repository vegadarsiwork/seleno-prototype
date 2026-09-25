"""Recheck the IIRS case with the earlier report's requested grid ceiling."""
import json
import torch
from run_audit import HERE, ROOT, run

torch.set_num_threads(2)
row = run("real_iirs_wac_large", ROOT / "data/raw/ch2/iirs/extracted/ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.xml",
          ROOT / "data/raw/wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
          max_side=6144, grid=(12, 12), segments=6)
(HERE / "extra_iirs.json").write_text(json.dumps(row, indent=2))
