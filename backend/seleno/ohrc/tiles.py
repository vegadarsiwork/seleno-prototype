"""Tile indexing and illumination classification for OHRC strips.

Why this exists: on this archive two thirds to four fifths of every strip sits
at or below DN 10 out of 255, and a 256x2048 patch from the middle of a strip can
contain as few as **five distinct DN values**. Cutting windows at random and
feeding them to a matcher is therefore mostly a test of what a detector does with
noise. Every experiment needs to know, before it starts, which windows carry
usable signal and which do not.

The index is built by striding the memmap, so scanning a whole 1.2 GB strip reads
only a few tens of megabytes and holds a few hundred kilobytes.

Two independent axes are recorded per tile, and they are deliberately kept apart:

* **illumination class** - ``lit`` / ``transition`` / ``shadow``, from the
  fraction of pixels at or below DN 10. This describes the scene.
* **usability** - whether the tile carries enough radiometric texture for a
  gradient-based detector to work at all, from the standard deviation and the
  count of distinct DN levels. This describes what a matcher can do with it.

The usability flag is an **observed** measurement, not a prediction. It is the
DEM-free control that any predicted-shadow mask has to be measured against;
without it, a shadow-prediction claim is untestable.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

# Thresholds. Chosen from the measured DN distribution of this archive and
# stated here rather than buried, because every downstream classification
# inherits them. They are not calibrated against any external truth.
DARK_DN = 10                 # "at or below DN 10" - the archive's shadow floor
SHADOW_FRACTION = 0.75       # >= this fraction dark  -> shadow
LIT_FRACTION = 0.25          # <= this fraction dark  -> lit
USABLE_STD = 6.0             # DN standard deviation floor for a detector
USABLE_DISTINCT = 16         # distinct DN levels floor


@dataclass
class TileStat:
    """Radiometric summary of one window. Coordinates are full-resolution pixels."""

    sample0: int
    line0: int
    width: int
    height: int
    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    max: int
    frac_dark: float             # fraction <= DARK_DN
    frac_bright: float           # fraction >= 200
    distinct: int
    illumination: str            # lit | transition | shadow
    usable: bool
    step: int = 1

    @property
    def centre(self) -> tuple[float, float]:
        return (self.sample0 + self.width / 2.0, self.line0 + self.height / 2.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usable"] = bool(self.usable)
        return d


def classify(frac_dark: float) -> str:
    if frac_dark >= SHADOW_FRACTION:
        return "shadow"
    if frac_dark <= LIT_FRACTION:
        return "lit"
    return "transition"


def tile_stat(tile: np.ndarray, sample0: int, line0: int, width: int, height: int,
              step: int = 1) -> TileStat:
    """Summarise one already-read window. No I/O."""
    a = tile.reshape(-1)
    if a.size == 0:
        return TileStat(sample0, line0, width, height, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
                        1.0, 0.0, 0, "shadow", False, step)
    counts = np.bincount(a, minlength=256)
    total = float(a.size)
    cdf = np.cumsum(counts) / total
    frac_dark = float(cdf[DARK_DN])
    frac_bright = float(1.0 - cdf[199]) if len(cdf) > 199 else 0.0
    distinct = int((counts > 0).sum())
    std = float(a.std())
    illum = classify(frac_dark)
    return TileStat(
        sample0=int(sample0), line0=int(line0), width=int(width), height=int(height),
        mean=round(float(a.mean()), 4), std=round(std, 4),
        p01=float(np.searchsorted(cdf, 0.01)), p50=float(np.searchsorted(cdf, 0.50)),
        p99=float(np.searchsorted(cdf, 0.99)), max=int(a.max()),
        frac_dark=round(frac_dark, 5), frac_bright=round(frac_bright, 5),
        distinct=distinct, illumination=illum,
        usable=bool(std >= USABLE_STD and distinct >= USABLE_DISTINCT),
        step=int(step))


def scan_strip(product, tile: int = 512, stride: int | None = None,
               step: int = 4, limit: int | None = None) -> list[TileStat]:
    """Index a whole strip by striding the memmap.

    `step` decimates inside each window: at the default 4 a 512 px window is read
    as 128x128 (16 kB), so a full 101 074 x 12 000 strip costs a few tens of MB of
    reads. Radiometric fractions are barely affected by decimation; positions and
    sizes stay in full-resolution coordinates.
    """
    stride = stride or tile
    out: list[TileStat] = []
    for line0 in range(0, product.lines - tile + 1, stride):
        for sample0 in range(0, product.samples - tile + 1, stride):
            w = product.read_tile(sample0, line0, tile, tile, step=step)
            out.append(tile_stat(w, sample0, line0, tile, tile, step=step))
            if limit and len(out) >= limit:
                return out
    return out


@dataclass
class StripIndex:
    """A cached tile index for one product."""

    product_id: str
    timestamp: str
    lines: int
    samples: int
    tile: int
    stride: int
    step: int
    tiles: list[TileStat] = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, product, tile: int = 512, stride: int | None = None,
              step: int = 4) -> "StripIndex":
        stride = stride or tile
        return cls(product.product_id, product.timestamp, product.lines,
                   product.samples, tile, stride, step,
                   scan_strip(product, tile=tile, stride=stride, step=step))

    # ------------------------------------------------------------------- cache
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "product_id": self.product_id, "timestamp": self.timestamp,
            "lines": self.lines, "samples": self.samples,
            "tile": self.tile, "stride": self.stride, "step": self.step,
            "thresholds": {"dark_dn": DARK_DN, "shadow_fraction": SHADOW_FRACTION,
                           "lit_fraction": LIT_FRACTION, "usable_std": USABLE_STD,
                           "usable_distinct": USABLE_DISTINCT},
            "tiles": [t.to_dict() for t in self.tiles],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    @classmethod
    def load(cls, path: str) -> "StripIndex":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        idx = cls(d["product_id"], d["timestamp"], d["lines"], d["samples"],
                  d["tile"], d["stride"], d["step"])
        idx.tiles = [TileStat(**t) for t in d["tiles"]]
        return idx

    @classmethod
    def cached(cls, product, cache_dir: str, tile: int = 512,
               stride: int | None = None, step: int = 4,
               rebuild: bool = False) -> "StripIndex":
        stride = stride or tile
        path = os.path.join(cache_dir, "tileindex_%s_t%d_s%d_d%d.json"
                            % (product.timestamp, tile, stride, step))
        if os.path.exists(path) and not rebuild:
            try:
                return cls.load(path)
            except Exception:
                pass
        idx = cls.build(product, tile=tile, stride=stride, step=step)
        idx.save(path)
        return idx

    # ------------------------------------------------------------------ query
    def of_class(self, illumination: str) -> list[TileStat]:
        return [t for t in self.tiles if t.illumination == illumination]

    def usable(self) -> list[TileStat]:
        return [t for t in self.tiles if t.usable]

    def best(self, illumination: str | None = None, n: int = 1,
             key=lambda t: t.std) -> list[TileStat]:
        pool = self.of_class(illumination) if illumination else list(self.tiles)
        return sorted(pool, key=key, reverse=True)[:n]

    def summary(self) -> dict:
        n = len(self.tiles) or 1
        by = {k: len(self.of_class(k)) for k in ("lit", "transition", "shadow")}
        means = np.array([t.mean for t in self.tiles], float)
        return {
            "product_id": self.product_id,
            "timestamp": self.timestamp,
            "tiles": len(self.tiles),
            "tile_px": self.tile,
            "stride_px": self.stride,
            "read_step": self.step,
            "counts": by,
            "fractions": {k: round(v / n, 4) for k, v in by.items()},
            "usable_tiles": len(self.usable()),
            "usable_fraction": round(len(self.usable()) / n, 4),
            "tile_mean_dn": {"min": round(float(means.min()), 3),
                             "max": round(float(means.max()), 3),
                             "median": round(float(np.median(means)), 3)},
        }


# --------------------------------------------------------------------------- #
# strip-level radiometric profiles
# --------------------------------------------------------------------------- #

def cross_track_profile(product, rows: int = 800) -> np.ndarray:
    """Mean DN per sample (column), from `rows` evenly spaced full-width lines.

    Detects the swath brightness gradient. On this archive it is large - one
    strip runs 9.5 DN at sample 0 to 54.2 DN at sample 11999 - which is why
    normalisation has to be per tile, not per strip.
    """
    step = max(1, product.lines // max(1, rows))
    acc = np.zeros(product.samples, np.float64)
    n = 0
    for l0 in range(0, product.lines, step):
        acc += product.read_tile(0, l0, product.samples, 1)[0].astype(np.float64)
        n += 1
    return acc / max(n, 1)


def along_track_profile(product, samples_per_row: int = 1200,
                        rows: int = 1200) -> np.ndarray:
    """Mean DN per sampled line, over a decimated set of columns."""
    lstep = max(1, product.lines // max(1, rows))
    sstep = max(1, product.samples // max(1, samples_per_row))
    lines = list(range(0, product.lines, lstep))
    out = np.empty(len(lines), np.float64)
    for i, l0 in enumerate(lines):
        row = product.read_tile(0, l0, product.samples, 1, step=1)[0]
        out[i] = row[::sstep].mean()
    return out


def dn_histogram(product, rows: int = 1000) -> np.ndarray:
    """256-bin DN histogram from evenly spaced full-width lines."""
    step = max(1, product.lines // max(1, rows))
    hist = np.zeros(256, np.int64)
    for l0 in range(0, product.lines, step):
        row = product.read_tile(0, l0, product.samples, 1)[0]
        hist += np.bincount(row, minlength=256)
    return hist


def histogram_stats(hist: np.ndarray) -> dict:
    total = float(hist.sum()) or 1.0
    cdf = np.cumsum(hist) / total
    dn = np.arange(256)
    mean = float((dn * hist).sum() / total)
    var = float((hist * (dn - mean) ** 2).sum() / total)
    return {
        "pixels_sampled": int(hist.sum()),
        "mean": round(mean, 3),
        "std": round(var ** 0.5, 3),
        "median": int(np.searchsorted(cdf, 0.5)),
        "p01": int(np.searchsorted(cdf, 0.01)),
        "p99": int(np.searchsorted(cdf, 0.99)),
        "frac_le_5": round(float(cdf[5]), 5),
        "frac_le_10": round(float(cdf[10]), 5),
        "frac_ge_200": round(float(1.0 - cdf[199]), 5),
        "distinct_levels": int((hist > 0).sum()),
        "max_dn": int(np.max(np.nonzero(hist)[0])) if hist.any() else 0,
    }


def normalise_for_display(tile: np.ndarray, low: float = 1.0,
                          high: float = 99.0) -> np.ndarray:
    """Per-tile percentile stretch. **Visualisation only.**

    Global normalisation is useless on these strips (row means span 0.4 to 167
    DN), and this stretch is never applied before matching - preprocessing owns
    that decision, and the metrics must reflect the real radiometry.
    """
    a = tile.astype(np.float32)
    lo, hi = np.percentile(a, (low, high))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(tile, np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def pick_representative(index: StripIndex, wanted: Iterable[str] = ("lit", "transition", "shadow"),
                        per_class: int = 2) -> dict[str, list[TileStat]]:
    """Highest-texture examples of each illumination class."""
    out = {}
    for k in wanted:
        out[k] = index.best(k, n=per_class, key=lambda t: t.std)
    return out
