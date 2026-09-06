"""Dataset access: reads the manifest written by scripts/prepare_data.py."""
from __future__ import annotations

import json
import os

import cv2


class Store:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.pairs_dir = os.path.join(data_dir, "pairs")
        path = os.path.join(data_dir, "manifest.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "data/manifest.json is missing. Build the demo pairs first:\n"
                "    python scripts/prepare_data.py --raw <dir-with-ISRO-products>")
        with open(path) as fh:
            self.manifest = json.load(fh)
        self.pairs = self.manifest["pairs"]
        self._by_id = {p["id"]: p for p in self.pairs}

    def pair(self, pid: str) -> dict:
        if pid not in self._by_id:
            raise KeyError("unknown pair '%s'" % pid)
        return self._by_id[pid]

    def images(self, pid: str):
        p = self.pair(pid)
        src = cv2.imread(os.path.join(self.pairs_dir, p["source_file"]), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(self.pairs_dir, p["reference_file"]), cv2.IMREAD_GRAYSCALE)
        if src is None or ref is None:
            raise FileNotFoundError("image files missing for pair '%s'" % pid)
        return src, ref

    def catalogue(self) -> list[dict]:
        """Pair metadata for the UI, without the ground-truth matrix."""
        out = []
        for p in self.pairs:
            q = {k: v for k, v in p.items() if k != "gt_homography"}
            out.append(q)
        return out
