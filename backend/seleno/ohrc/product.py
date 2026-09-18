"""OHRC product discovery and memory-safe tile access.

Design rules, all enforced here:

* **The raster is never fully loaded.** `.img` files are 0.96-1.21 GB each.
  Access goes through `numpy.memmap(mode="r")` and every public accessor returns
  a bounded window. `read_tile` copies the window out of the map so callers
  cannot accidentally hold a view onto the whole file.
* **The archive is immutable.** Every file handle is opened read-only, and
  nothing in this package writes into the dataset tree.
* **Dimensions come from the label**, not from a hard-coded constant: the four
  products have three different line counts.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .geometry import GeometryGrid, footprint_overlap
from .label import OhrcLabel, parse_label, verify_raster_size
from .sun import SunSeries, line_time_seconds

# ch2_ohr_ncp_20241115T1326321339_d_img_d18.img   (OHRC)
# ch2_tmc_ncn_20260813T0627378557_d_img_d18.img   (TMC-2)
# The instrument field was hardcoded to `ohr`; TMC-2 products are byte-identical
# in layout and parse with the same code, so it is a group rather than a literal.
_STEM = re.compile(r"^(?P<mission>ch2)_(?P<inst>ohr|tmc|iir)_(?P<mode>\w+?)_"
                   r"(?P<timestamp>\d{8}T\d{10,16})_(?P<p>\w)_(?P<prd>\w{3})_(?P<stn>\w+)$")
INSTRUMENTS = {"ohr": "OHRC", "tmc": "TMC-2", "iir": "IIRS"}


def default_dataset_root() -> str | None:
    """Locate the OHRC archive.

    Honours ``SELENO_OHRC_ROOT`` first, then ``data/raw/ch2/ohrc`` inside the
    repository (where Phase 1 extracts the ISSDC bundle), then the legacy
    ``datasettesting/dataset`` layout beside the prototype directory.
    """
    env = os.environ.get("SELENO_OHRC_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    proto = os.path.abspath(os.path.join(here, "..", "..", ".."))     # prototype/
    for cand in (os.path.join(proto, "data", "raw", "ch2", "ohrc"),
                 os.path.join(proto, "..", "datasettesting", "dataset"),
                 os.path.join(proto, "datasettesting", "dataset")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


@dataclass
class OhrcProduct:
    """One OHRC product: raster, label, geometry grid and ancillary series."""

    product_id: str                       # full stem, e.g. ch2_ohr_ncp_2024...d18
    timestamp: str                        # 20241115T1326321339
    img_path: str
    label_path: str
    instrument: str = "OHRC"              # OHRC | TMC-2 | IIRS, from the stem
    geometry_csv: str | None = None
    spm_path: str | None = None
    oat_path: str | None = None
    lbr_path: str | None = None
    browse_png: str | None = None
    root: str = ""
    _missing: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ label
    @cached_property
    def label(self) -> OhrcLabel:
        return parse_label(self.label_path)

    @cached_property
    def shape(self) -> tuple[int, int]:
        return self.label.shape

    @property
    def lines(self) -> int:
        return self.label.lines

    @property
    def samples(self) -> int:
        return self.label.samples

    @property
    def gsd_m(self) -> float | None:
        return self.label.pixel_resolution_m

    def verify(self) -> tuple[bool, str]:
        """Label vs raster on disk. Read-only."""
        return verify_raster_size(self.label, self.img_path)

    # ----------------------------------------------------------------- raster
    @cached_property
    def _map(self) -> np.memmap:
        ok, why = self.verify()
        if not ok:
            raise ValueError("refusing to memmap %s: %s" % (self.img_path, why))
        return np.memmap(self.img_path, dtype=np.dtype(self.label.numpy_dtype),
                         mode="r", shape=self.shape)

    def read_tile(self, sample0: int, line0: int, width: int, height: int,
                  step: int = 1) -> np.ndarray:
        """A copied (height, width) window in the label's own dtype. Clipped to
        the raster.

        The dtype is whatever the label declares - uint8 for OHRC, uint16 for
        TMC-2 - never a hardcoded assumption.

        `step` decimates by simple striding, which is what makes whole-strip
        thumbnails cheap: the OS pages in only the rows actually touched.
        """
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        s0 = int(max(0, min(sample0, self.samples - 1)))
        l0 = int(max(0, min(line0, self.lines - 1)))
        s1 = int(min(self.samples, s0 + width))
        l1 = int(min(self.lines, l0 + height))
        step = max(1, int(step))
        return np.array(self._map[l0:l1:step, s0:s1:step], copy=True)

    def read_rows(self, line0: int, count: int, step: int = 1) -> np.ndarray:
        """Full-width rows. Use sparingly - one row is 12 000 bytes."""
        return self.read_tile(0, line0, self.samples, count, step=step)

    def thumbnail(self, max_side: int = 1400) -> np.ndarray:
        """Whole-strip preview by striding the memmap.

        Cost is bounded by `max_side**2` bytes read, not by the file size.
        """
        step = max(1, int(np.ceil(max(self.shape) / float(max_side))))
        return np.array(self._map[::step, ::step], copy=True)

    def close(self) -> None:
        """Drop the memmap. Safe to call repeatedly."""
        if "_map" in self.__dict__:
            m = self.__dict__.pop("_map")
            try:
                m._mmap.close()
            except Exception:
                pass

    # --------------------------------------------------------------- geometry
    @cached_property
    def geometry(self) -> GeometryGrid:
        if not self.geometry_csv:
            raise FileNotFoundError("no geometry CSV found for %s" % self.product_id)
        return GeometryGrid.from_csv(self.geometry_csv)

    @property
    def has_geometry(self) -> bool:
        return bool(self.geometry_csv) and os.path.exists(self.geometry_csv)

    def window_footprint(self, sample0, line0, width, height) -> dict:
        return self.geometry.window_footprint(sample0, line0, width, height)

    # -------------------------------------------------------------------- sun
    @cached_property
    def sun(self) -> SunSeries:
        if not self.spm_path:
            raise FileNotFoundError("no .spm found for %s" % self.product_id)
        return SunSeries.from_spm(self.spm_path)

    @property
    def has_sun_series(self) -> bool:
        return bool(self.spm_path) and os.path.exists(self.spm_path)

    def sun_at_line(self, line: int) -> dict:
        """Solar geometry interpolated to the acquisition time of one image line.

        The `.spm` series starts before the image and ends after it, so the line
        time is offset onto the series epoch using the label's start time.
        """
        lab = self.label
        dwell = lab.dwell_seconds or 0.0
        t_line = line_time_seconds(line, lab.lines, dwell)
        series = self.sun
        if lab.start_time and series.epoch:
            offset = (lab.start_time - series.epoch).total_seconds()
        else:
            offset = 0.0
        out = series.at_seconds(offset + t_line)
        out["line"] = int(line)
        out["t_since_start_s"] = round(t_line, 4)
        return out

    def sun_at_window(self, line0: int, height: int) -> dict:
        """Solar geometry at the top, middle and bottom of a line range."""
        return {
            "top": self.sun_at_line(line0),
            "middle": self.sun_at_line(line0 + height // 2),
            "bottom": self.sun_at_line(min(line0 + height, self.lines - 1)),
        }

    # ------------------------------------------------------------------ report
    def describe(self) -> dict:
        lab = self.label
        ok, why = self.verify()
        d = {
            "product_id": self.product_id,
            "timestamp": self.timestamp,
            "lines": lab.lines,
            "samples": lab.samples,
            "bytes_on_disk": os.path.getsize(self.img_path) if os.path.exists(self.img_path) else None,
            "raster_check": {"ok": ok, "detail": why},
            "gsd_m": lab.pixel_resolution_m,
            "start_time": lab.start_time.isoformat() if lab.start_time else None,
            "stop_time": lab.stop_time.isoformat() if lab.stop_time else None,
            "dwell_s": lab.dwell_seconds,
            "line_period_us": round(lab.line_period_us, 4) if lab.line_period_us else None,
            "imaging_orbit": lab.imaging_orbit,
            "spacecraft_altitude_km": lab.spacecraft_altitude_km,
            "tdi_stages": lab.tdi_stages,
            "limb_direction": lab.limb_direction,
            "roll_deg": lab.roll_deg,
            "pitch_deg": lab.pitch_deg,
            "yaw_deg": lab.yaw_deg,
            "sun_azimuth_deg_label": lab.sun_azimuth_deg,
            "sun_elevation_deg_label": lab.sun_elevation_deg,
            "solar_incidence_deg_label": lab.solar_incidence_deg,
            "projection": lab.projection,
            "area": lab.area,
            "corners_lon_lat": lab.corner_lonlat(),
            "corners_refined": lab.corners_refined,
            "reference_data_used": lab.reference_data_used,
            "label_notes": lab.notes,
            "has_geometry_csv": self.has_geometry,
            "has_sun_series": self.has_sun_series,
            "has_browse_png": bool(self.browse_png),
            "missing_files": list(self._missing),
        }
        if self.has_sun_series:
            d["sun_series"] = self.sun.summary()
        return d


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def _find_sibling(root: str, subdir: str, timestamp: str, suffixes) -> str | None:
    base = os.path.join(root, subdir)
    if not os.path.isdir(base):
        return None
    for dirpath, _dirs, files in os.walk(base):
        for f in files:
            if timestamp in f and f.endswith(tuple(suffixes)):
                return os.path.join(dirpath, f)
    return None


def discover(root: str | None = None, instrument: str | None = None) -> list[OhrcProduct]:
    """Find every Chandrayaan-2 imaging product under `root`, newest timestamp last.

    `instrument` filters to one of "OHRC", "TMC-2", "IIRS"; the default returns
    everything found, which for a root holding only OHRC is the previous
    behaviour unchanged.

    A product is defined by a `data/**/*.img` plus its sibling `.xml`. Missing
    optional companions (geometry, sun, browse) are recorded on the product
    rather than raising, so an incompletely extracted archive still loads.
    """
    root = root or default_dataset_root()
    if not root or not os.path.isdir(root):
        return []
    out: list[OhrcProduct] = []
    for img in sorted(glob.glob(os.path.join(root, "data", "**", "*.img"), recursive=True)):
        stem = os.path.splitext(os.path.basename(img))[0]
        m = _STEM.match(stem)
        if not m:
            continue
        ts = m.group("timestamp")
        inst = INSTRUMENTS.get(m.group("inst"), m.group("inst").upper())
        if instrument and inst != instrument:
            continue
        label = os.path.splitext(img)[0] + ".xml"
        if not os.path.exists(label):
            continue
        p = OhrcProduct(product_id=stem, timestamp=ts, instrument=inst, img_path=img,
                        label_path=label, root=root)
        p.geometry_csv = _find_sibling(root, "geometry", ts, (".csv",))
        p.spm_path = _find_sibling(root, "miscellaneous", ts, (".spm",))
        p.oat_path = _find_sibling(root, "miscellaneous", ts, (".oat",))
        p.lbr_path = _find_sibling(root, "miscellaneous", ts, (".lbr",))
        p.browse_png = _find_sibling(root, "browse", ts, (".png",))
        for name, val in (("geometry CSV", p.geometry_csv), ("sun .spm", p.spm_path),
                          ("browse PNG", p.browse_png)):
            if not val:
                p._missing.append(name)
        out.append(p)
    return out


def by_timestamp(products, key: str) -> OhrcProduct:
    """Look a product up by timestamp, product id, or unique prefix."""
    for p in products:
        if key in (p.timestamp, p.product_id):
            return p
    hits = [p for p in products if p.timestamp.startswith(key) or key in p.product_id]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError("no OHRC product matching %r (have: %s)"
                       % (key, ", ".join(p.timestamp for p in products)))
    raise KeyError("%r matches %d products: %s"
                   % (key, len(hits), ", ".join(p.timestamp for p in hits)))


def overlap_matrix(products, cell_m: float = 100.0, step: int = 1) -> list[dict]:
    """Pairwise footprint overlap for every product pair, best overlap first."""
    rows = []
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            a, b = products[i], products[j]
            if not (a.has_geometry and b.has_geometry):
                continue
            ov = footprint_overlap(a.geometry, b.geometry, cell_m=cell_m, step=step)
            ov["a"] = a.timestamp
            ov["b"] = b.timestamp
            rows.append(ov)
    rows.sort(key=lambda r: -r["fraction_of_smaller"])
    return rows
