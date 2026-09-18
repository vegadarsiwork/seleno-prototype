"""Real illumination-change pairs with zero labelling cost.

Two sub-solar-longitude bins of the same LROC controlled tile are the same ground
on the same grid, so the transform relating them is the **identity, exact by
construction**. That is the whole reason this dataset exists: it supplies real
shadow reversal without a synthetic renderer and without a single annotation.

Three rules are enforced here rather than left to the caller, because each one is
a way the experiment could quietly become meaningless:

1. **Splits are by tile, never by pair.** Two pairs from the same tile share
   ground; putting one in train and one in validation leaks the answer.
   `split_by_tile` asserts the partition is disjoint and non-empty.
2. **Sampling is weighted towards large Sun-azimuth differences.** Uniform
   sampling over bin pairs buries the 90-180 degree band - the band where every
   matcher currently scores zero - under easy near-duplicate pairs.
3. **Chips must carry data in both bins.** A polar mosaic is mostly nodata and
   shadow; a pair where one side is blank trains the model to predict nothing.

Nothing in this module knows about OHRC. Cross-sensor pairs are test-only and
live in the evaluation path, so they cannot leak into training by accident.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np


def d_az(a: str, b: str) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def split_by_tile(tiles, fractions=(0.7, 0.15, 0.15), seed=0):
    """Partition tiles into train/val/test. Disjointness is asserted, not assumed."""
    tiles = sorted(set(tiles))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(tiles))
    n = len(order)
    n_tr = max(1, int(round(fractions[0] * n)))
    n_va = max(1, int(round(fractions[1] * n)))
    if n_tr + n_va >= n:
        n_tr = max(1, n - 2)
        n_va = 1
    out = {"train": sorted(order[:n_tr]),
           "val": sorted(order[n_tr:n_tr + n_va]),
           "test": sorted(order[n_tr + n_va:])}
    seen = set()
    for k, v in out.items():
        assert v, "split %r is empty; need at least %d tiles" % (k, 3)
        assert not (seen & set(v)), "tile leak between splits: %s" % (seen & set(v))
        seen |= set(v)
    assert seen == set(tiles), "split does not cover every tile"
    return out


@dataclass
class Chip:
    tile: str
    bin_a: str
    bin_b: str
    row: int
    col: int
    d_az: float


@dataclass
class NACIlluminationPairs:
    """Illumination pairs cut from the LROC controlled mosaics.

    `__getitem__` returns ``(a, b, valid_a, valid_b, meta)`` with `a` and `b`
    uint8 chips on the same grid; the ground-truth transform between them is the
    identity.
    """

    index_path: str
    tiles: list[str]
    chip: int = 512
    stride: int | None = None
    min_valid: float = 0.5
    daz_bins: tuple = ((0, 45, 0.5), (45, 90, 1.0), (90, 135, 2.0), (135, 181, 2.0))
    seed: int = 0
    samples: int = 4000
    root: str = ""
    _by_tile: dict = field(default_factory=dict, repr=False)
    _plan: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        meta = json.load(open(self.index_path))
        # index paths are repo-relative; the corpus lives at <root>/data/processed/
        # nac_illum/, so the root is four levels up from index.json.
        self.root = self.root or os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(self.index_path)), "..", "..", ".."))
        keep = set(self.tiles)
        for r in meta["products"]:
            if r["tile"] in keep:
                self._by_tile.setdefault(r["tile"], {})[r["bin"]] = r["path"]
        missing = keep - set(self._by_tile)
        assert not missing, "tiles absent from the corpus: %s" % sorted(missing)
        self.stride = self.stride or self.chip // 2
        self._build_plan()

    # ---------------------------------------------------------------- planning
    def _weight(self, d):
        for lo, hi, w in self.daz_bins:
            if lo <= d < hi:
                return w
        return 0.0

    def _build_plan(self):
        """Enumerate (tile, bin pair) candidates and draw a weighted sample.

        Chip locations are resolved lazily in `__getitem__`, so planning does not
        need to open 862 rasters.
        """
        rng = np.random.default_rng(self.seed)
        cand, wts = [], []
        for tile, bins in sorted(self._by_tile.items()):
            bs = sorted(bins)
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    d = d_az(bs[i], bs[j])
                    w = self._weight(d)
                    if w <= 0:
                        continue
                    cand.append((tile, bs[i], bs[j], d))
                    wts.append(w)
        assert cand, "no bin pairs survived weighting"
        wts = np.asarray(wts, np.float64)
        wts /= wts.sum()
        idx = rng.choice(len(cand), size=min(self.samples, len(cand) * 8),
                         replace=True, p=wts)
        self._plan = [cand[k] for k in idx]

    def daz_histogram(self):
        h = {}
        for lo, hi, _ in self.daz_bins:
            key = "%d-%d" % (lo, min(hi, 180))
            h[key] = sum(1 for _, _, _, d in self._plan if lo <= d < hi)
        return h

    # ----------------------------------------------------------------- access
    def __len__(self):
        return len(self._plan)

    def _load(self, tile, b):
        import cv2
        path = os.path.join(self.root, self._by_tile[tile][b])
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError("unreadable %s" % path)
        return raw[::-1]                        # north-up -> +y with row, as seleno.nac does

    def __getitem__(self, i):
        tile, ba, bb, d = self._plan[i]
        A = self._load(tile, ba)
        B = self._load(tile, bb)
        h, w = A.shape
        k = self.chip
        rng = np.random.default_rng((self.seed * 1000003 + i) & 0xFFFFFFFF)
        both = (A > 0) & (B > 0)
        for _ in range(24):
            r = int(rng.integers(0, max(1, h - k)))
            c = int(rng.integers(0, max(1, w - k)))
            if both[r:r + k, c:c + k].mean() >= self.min_valid:
                break
        else:
            # no chip in this pair carries data in both bins; report it rather
            # than silently returning a blank the model would learn from
            return None
        a = A[r:r + k, c:c + k]
        b = B[r:r + k, c:c + k]
        return (a, b, a > 0, b > 0,
                Chip(tile=tile, bin_a=ba, bin_b=bb, row=r, col=c, d_az=d))


def describe(ds: NACIlluminationPairs) -> str:
    return ("%d samples over %d tiles; d_az histogram %s"
            % (len(ds), len(ds._by_tile), ds.daz_histogram()))
