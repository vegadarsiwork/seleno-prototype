"""Read an arbitrary image into one normalised form, and say what was missing.

The brief's hard requirement is that the tool works on unseen input with no
assumption that ISRO geometry files, `.spm` Sun parameters or refined corners are
present. So every optional thing is genuinely optional here, and each absence is
recorded in `Scene.degraded` rather than raising. A caller can always answer
"what did we not know about this image?" from the Scene itself.

Formats handled, in the order they are tried:

1. **PDS4** - a Chandrayaan-2 `.xml` label beside a `.img`. Gives dtype, shape,
   GSD, acquisition time, Sun angles and (if the sibling geometry CSV exists) a
   per-pixel lon/lat lattice.
2. **PDS3 detached** - a `.lbl` beside a `.img`. LOLA GDR and SELENE TC.
3. **PDS3 attached** - a `.IMG` whose own header is the label. LRO NAC EDR.
4. **GeoTIFF / anything GDAL reads** - carries CRS and transform directly.
5. **Plain raster** - PNG, JPG. No geometry at all; the tool then works in pixel
   space and says so.

Nothing here loads a whole 1.2 GB raster unless asked: `Scene.data` is lazy and
`read_decimated` exists for previews.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# the normalised form
# --------------------------------------------------------------------------- #

@dataclass
class Scene:
    """One image plus whatever was known about it."""

    path: str
    array: np.ndarray                      # memmap or ndarray, NOT force-copied
    valid: np.ndarray                      # bool, False where nodata
    reader: str
    nodata: float | None = 0.0             # so validity can be evaluated per sample                            # which backend read it
    profile: str = "unknown"               # sensor profile name
    instrument: str = "unknown"
    gsd_m: float | None = None
    crs: object | None = None              # rasterio CRS, if any
    transform: object | None = None        # rasterio Affine, if any
    lonlat: object | None = None           # per-pixel lattice, if any
    acquired_utc: str | None = None
    sun_azimuth_deg: float | None = None
    sun_incidence_deg: float | None = None
    sub_solar_lon_deg: float | None = None
    bit_depth: int | None = None
    meta_scale: float = 1.0                # value = raw * meta_scale + meta_offset
    meta_offset: float = 0.0
    degraded: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.array.shape

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None

    def valid_fraction(self, max_samples: int = 4_000_000) -> float:
        """Estimated from a decimated read.

        Materialising a whole-raster boolean mask costs 592 MB on a TMC-2 strip
        and was one of the allocations that made this tool exhaust memory.
        """
        h, w = self.array.shape
        step = max(1, int(math.ceil(math.sqrt(h * w / float(max_samples)))))
        sub = np.asarray(self.array[::step, ::step])
        m = np.isfinite(sub)
        if self.nodata is not None:
            m &= sub != self.nodata
        return float(m.mean())

    def summary(self) -> dict:
        return {"path": os.path.basename(self.path), "reader": self.reader,
                "profile": self.profile, "instrument": self.instrument,
                "shape": list(self.array.shape), "gsd_m": self.gsd_m,
                "bit_depth": self.bit_depth,
                "georeferenced": self.georeferenced,
                "crs": str(self.crs) if self.crs is not None else None,
                "has_lonlat_lattice": self.lonlat is not None,
                "acquired_utc": self.acquired_utc,
                "sun_azimuth_deg": self.sun_azimuth_deg,
                "sun_incidence_deg": self.sun_incidence_deg,
                "sub_solar_lon_deg": self.sub_solar_lon_deg,
                "valid_fraction": round(self.valid_fraction(), 4),
                "degraded": self.degraded}


class UnreadableInput(Exception):
    """Raised when nothing can read the file. Maps to the `unreadable_input` code."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_PDS3_DTYPE = {
    ("LSB_INTEGER", 16): "<i2", ("MSB_INTEGER", 16): ">i2",
    ("LSB_UNSIGNED_INTEGER", 16): "<u2", ("MSB_UNSIGNED_INTEGER", 16): ">u2",
    ("UNSIGNED_INTEGER", 8): "u1", ("LSB_UNSIGNED_INTEGER", 8): "u1",
    ("MSB_UNSIGNED_INTEGER", 8): "u1", ("INTEGER", 8): "i1",
    ("PC_REAL", 32): "<f4", ("IEEE_REAL", 32): ">f4",
    ("LSB_INTEGER", 32): "<i4", ("MSB_INTEGER", 32): ">i4",
}


def _parse_pds3(path: str, limit_bytes: int = 1 << 20) -> dict:
    """Scrape PDS3 keywords. Works for a detached .lbl or an attached header."""
    out, depth = {}, []
    with open(path, "rb") as fh:
        raw = fh.read(limit_bytes)
    text = raw.decode("latin-1", "replace")
    if "PDS_VERSION_ID" not in text[:400] and not path.lower().endswith(".lbl"):
        # an attached label always starts with it; a detached one usually does
        if "OBJECT" not in text[:4000]:
            return {}
    for line in text.splitlines():
        line = line.strip()
        if line == "END":
            break
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().strip('"')
        raw_v = v.strip()
        v = v.split("<")[0].strip().strip('"').strip()
        if k == "OBJECT":
            depth.append(v)
        elif k == "END_OBJECT":
            if depth:
                depth.pop()
        elif k and k not in out:
            out[k] = v
            # PDS3 carries the unit in angle brackets and a stripped value is
            # ambiguous: SELENE's MAP_SCALE is 0.0074 <km/pixel>, which read as
            # metres makes a 7.4 m product look like a 7.4 mm one and turns the
            # scale ratio into 740.
            um = re.search(r"<([^>]+)>", raw_v)
            if um:
                out[k + "__UNIT"] = um.group(1).strip().lower()
    return out


def _percentile_norm(a: np.ndarray, valid: np.ndarray, lo=2.0, hi=98.0) -> np.ndarray:
    out = np.zeros(a.shape, np.float32)
    if valid.sum() < 16:
        return out
    p0, p1 = np.percentile(a[valid].astype(np.float64), (lo, hi))
    if p1 - p0 < 1e-9:
        return out
    out[valid] = np.clip((a[valid] - p0) / (p1 - p0), 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

def _try_pds4(path: str, profiles) -> Scene | None:
    """Chandrayaan-2 PDS4: a .xml label beside a .img."""
    if path.lower().endswith(".xml"):
        lbl, img = path, os.path.splitext(path)[0] + ".img"
    elif path.lower().endswith(".img") and os.path.exists(os.path.splitext(path)[0] + ".xml"):
        img, lbl = path, os.path.splitext(path)[0] + ".xml"
    else:
        return None
    if not os.path.exists(img):
        return None
    try:
        from ..ohrc.label import parse_label
        L = parse_label(lbl)
    except Exception:
        return None
    if not L.lines or not L.samples or L.numpy_dtype is None:
        return None

    prof = profiles.match(instrument=L.logical_id or "", path=path)
    # Deliberately NOT np.asarray(..., float32): that copies the entire raster
    # into RAM (2.37 GB for a TMC-2 strip) before a single pixel is needed. The
    # memmap supports the fancy indexing `_sample` does, and pages in only what
    # is touched.
    a = np.memmap(img, dtype=np.dtype(L.numpy_dtype), mode="r",
                  shape=(L.lines, L.samples))
    valid = None
    degraded = []

    lonlat = None
    csv = _find_geometry_csv(img, L)
    if csv:
        try:
            from ..ohrc.geometry import GeometryGrid
            lonlat = GeometryGrid.from_csv(csv)
        except Exception as exc:
            degraded.append("geometry CSV present but unreadable (%s)" % type(exc).__name__)
    else:
        degraded.append("no geometry lattice: registration will be in pixel space "
                        "unless the reference supplies a frame")

    trusted_sun = True
    if L.roll_deg is not None and abs(L.roll_deg) > 1.0:
        trusted_sun = False
        degraded.append("label Sun angles are confounded by %.1f deg of roll; "
                        "treated as untrusted" % L.roll_deg)
    if L.reference_data_used and str(L.reference_data_used).lower() == "system":
        degraded.append("reference_data_used = System: geolocation is unrefined "
                        "ephemeris, expect kilometre-scale error")

    return Scene(path=img, array=a, valid=valid, reader="pds4",
                 nodata=prof.get("nodata", 0),
                 profile=prof["name"], instrument=prof["instrument"],
                 gsd_m=L.pixel_resolution_m or prof.get("gsd_m"),
                 lonlat=lonlat,
                 acquired_utc=L.start_time.isoformat() if L.start_time else None,
                 sun_azimuth_deg=L.sun_azimuth_deg if trusted_sun else None,
                 sun_incidence_deg=L.solar_incidence_deg if trusted_sun else None,
                 bit_depth=8 * L.item_bytes, degraded=degraded,
                 meta={"logical_id": L.logical_id, "lines": L.lines,
                       "samples": L.samples, "data_type": L.data_type,
                       "reference_data_used": L.reference_data_used,
                       "label_sun_trusted": trusted_sun})


def _find_geometry_csv(img_path: str, label) -> str | None:
    """The sibling `geometry/**/<stem>_g_grd_*.csv`, if the archive tree is intact."""
    stem = os.path.basename(img_path)
    m = re.search(r"(\d{8}T\d{10,16})", stem)
    if not m:
        return None
    ts = m.group(1)
    root = os.path.dirname(img_path)
    for _ in range(5):
        cand = os.path.join(root, "geometry")
        if os.path.isdir(cand):
            for dirpath, _dirs, files in os.walk(cand):
                for f in files:
                    if ts in f and f.endswith(".csv"):
                        return os.path.join(dirpath, f)
        root = os.path.dirname(root)
    return None


def _try_pds3(path: str, profiles) -> Scene | None:
    """PDS3, detached (.lbl + .img) or attached (.IMG with its own header)."""
    lbl = img = None
    if path.lower().endswith(".lbl"):
        lbl = path
        for ext in (".img", ".IMG", ".dat"):
            if os.path.exists(path[:-4] + ext):
                img = path[:-4] + ext
                break
    elif path.lower().endswith((".img", ".dat")):
        for ext in (".lbl", ".LBL"):
            if os.path.exists(os.path.splitext(path)[0] + ext):
                lbl, img = os.path.splitext(path)[0] + ext, path
                break
        else:
            lbl = img = path                       # attached label
    if img is None:
        return None
    L = _parse_pds3(lbl)
    if not L or "LINES" not in L or "LINE_SAMPLES" not in L:
        return None
    try:
        lines, samples = int(L["LINES"]), int(L["LINE_SAMPLES"])
        bits = int(L.get("SAMPLE_BITS", 8))
        key = (L.get("SAMPLE_TYPE", "UNSIGNED_INTEGER"), bits)
        dt = _PDS3_DTYPE.get(key)
        if dt is None:
            return None
    except Exception:
        return None

    offset = 0
    degraded = []
    if lbl == img:                                  # attached: skip the header
        rb = int(L.get("RECORD_BYTES", 0))
        ptr = L.get("^IMAGE", "")
        m = re.search(r"(\d+)", str(ptr))
        if rb and m:
            offset = (int(m.group(1)) - 1) * rb
    need = lines * samples * (bits // 8)
    actual = os.path.getsize(img)
    if actual < offset + need:
        raise UnreadableInput(
            "%s is %d bytes; the label declares %d lines x %d samples x %d bit "
            "= %d bytes from offset %d" % (os.path.basename(img), actual, lines,
                                           samples, bits, need, offset))

    prof = profiles.match(instrument=L.get("INSTRUMENT_ID", "") or L.get("DATA_SET_ID", ""),
                          path=path)
    a = np.memmap(img, dtype=np.dtype(dt), mode="r", shape=(lines, samples),
                  offset=offset)
    scale = float(L.get("SCALING_FACTOR", 1.0) or 1.0)
    off = float(L.get("OFFSET", 0.0) or 0.0)
    # Scaling is applied to sampled windows, not to the whole raster, for the
    # same reason: a 12 288^2 float32 copy is 604 MB.
    valid = None

    crs = transform = None
    proj = L.get("MAP_PROJECTION_TYPE", "")
    if proj:
        crs, transform, why = _pds3_projection(L, lines, samples)
        if crs is None:
            degraded.append("map projection %r not reconstructed (%s)" % (proj, why))
    else:
        degraded.append("no map projection in the label; pixel-space registration only")

    return Scene(path=img, array=a, valid=valid, reader="pds3",
                 nodata=prof.get("nodata", 0),
                 profile=prof["name"], instrument=prof["instrument"],
                 meta_scale=scale, meta_offset=off,
                 gsd_m=_pds3_gsd(L) or prof.get("gsd_m"),
                 crs=crs, transform=transform,
                 acquired_utc=L.get("START_TIME"),
                 bit_depth=bits, degraded=degraded,
                 meta={k: L[k] for k in ("DATA_SET_ID", "PRODUCT_ID", "MAP_PROJECTION_TYPE",
                                         "TARGET_NAME") if k in L})


def _pds3_gsd(L) -> float | None:
    if "MAP_SCALE" in L:
        try:
            v = float(str(L["MAP_SCALE"]).split()[0])
            unit = L.get("MAP_SCALE__UNIT", "")
            if "km" in unit:
                v *= 1000.0
            elif unit and "m" not in unit:
                v = None                      # an unexpected unit is not guessed at
            if v:
                return v
        except Exception:
            pass
    if "MAP_RESOLUTION" in L:
        try:
            ppd = float(str(L["MAP_RESOLUTION"]).split()[0])
            r = float(str(L.get("A_AXIS_RADIUS", 1737.4)).split()[0]) * 1000.0
            return (math.pi / 180.0) * r / ppd
        except Exception:
            pass
    return None


def _pds3_projection(L, lines, samples):
    """Rebuild CRS + transform for the projections this project actually meets."""
    try:
        from rasterio.crs import CRS
        from rasterio.transform import Affine
    except ImportError:
        return None, None, "rasterio unavailable"
    proj = str(L.get("MAP_PROJECTION_TYPE", "")).upper()
    try:
        r = float(str(L.get("A_AXIS_RADIUS", 1737.4)).split()[0]) * 1000.0
        if proj.startswith("POLAR"):
            scale = float(str(L["MAP_SCALE"]).split()[0])
            clat = float(str(L.get("CENTER_LATITUDE", -90)).split()[0])
            lo = float(str(L.get("LINE_PROJECTION_OFFSET", 0)).split()[0])
            so = float(str(L.get("SAMPLE_PROJECTION_OFFSET", 0)).split()[0])
            crs = CRS.from_proj4("+proj=stere +lat_0=%g +lon_0=0 +k=1 +x_0=0 +y_0=0 "
                                 "+R=%g +units=m +no_defs" % (clat, r))
            # PDS3 offsets are (line, sample) of the projection origin
            x0 = -(so + 0.5) * scale
            y0 = (lo + 0.5) * scale
            return crs, Affine(scale, 0, x0, 0, -scale, y0), ""
        if proj.startswith("SIMPLE") or proj.startswith("EQUIRECT"):
            ppd = float(str(L["MAP_RESOLUTION"]).split()[0])
            ul_lon = float(str(L["UPPER_LEFT_LONGITUDE"]).split()[0])
            ul_lat = float(str(L["UPPER_LEFT_LATITUDE"]).split()[0])
            crs = CRS.from_proj4("+proj=longlat +R=%g +no_defs" % r)
            d = 1.0 / ppd
            return crs, Affine(d, 0, ul_lon, 0, -d, ul_lat), ""
    except Exception as exc:
        return None, None, type(exc).__name__
    return None, None, "unsupported projection"


def _try_gdal(path: str, profiles) -> Scene | None:
    try:
        import rasterio
    except ImportError:
        return None
    try:
        with rasterio.open(path) as ds:
            a = ds.read(1).astype(np.float32)
            crs, transform, nodata = ds.crs, ds.transform, ds.nodata
            # A plain PNG/JPG opens through GDAL with an IDENTITY transform, so
            # transform.a is 1.0 and looks like a 1 m pixel. Taking it would make
            # the tool report a metre-scale RMSE for an image that has no scale at
            # all. Only trust the transform when there is a CRS to go with it.
            gsd = abs(transform.a) if (transform is not None and crs is not None) else None
            bits = {"uint8": 8, "int8": 8, "uint16": 16, "int16": 16,
                    "float32": 32}.get(ds.dtypes[0])
    except Exception:
        return None
    prof = profiles.match(instrument="", path=path)
    valid = np.isfinite(a)
    if nodata is not None:
        valid &= a != nodata
    else:
        valid &= a != prof.get("nodata", 0)
    degraded = []
    if crs is None:
        degraded.append("GeoTIFF carries no CRS; pixel-space registration only")
    if crs is not None and crs.is_geographic and gsd is not None:
        gsd = gsd * math.pi / 180.0 * 1737400.0     # degrees -> metres on the Moon
    if gsd is None and prof.get("gsd_m"):
        gsd = float(prof["gsd_m"])
        degraded.append("GSD not in the file; using the %s profile's nominal %.3f m"
                        % (prof["name"], gsd))
    return Scene(path=path, array=a, valid=valid, reader="gdal",
                 nodata=nodata if nodata is not None else prof.get("nodata", 0),
                 profile=prof["name"], instrument=prof["instrument"],
                 gsd_m=gsd, crs=crs, transform=transform,
                 bit_depth=bits, degraded=degraded)


def _try_plain(path: str, profiles) -> Scene | None:
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    prof = profiles.match(instrument="", path=path)
    arr = a.astype(np.float32)
    return Scene(path=path, array=arr, valid=arr != prof.get("nodata", 0),
                 nodata=prof.get("nodata", 0),
                 reader="plain", profile=prof["name"], instrument=prof["instrument"],
                 gsd_m=prof.get("gsd_m"), bit_depth=8 * a.dtype.itemsize,
                 degraded=["plain raster: no geometry, no CRS, no Sun angles; "
                           "registration is in pixel space and no metre-scale "
                           "accuracy can be reported"])


BACKENDS = (_try_pds4, _try_pds3, _try_gdal, _try_plain)


def load(path: str, profiles=None) -> Scene:
    """Read any supported file. Raises `UnreadableInput` if none can."""
    if profiles is None:
        from .profiles import Profiles
        profiles = Profiles.load()
    if not os.path.exists(path):
        raise UnreadableInput("no such file: %s" % path)
    if os.path.getsize(path) == 0:
        raise UnreadableInput("empty file: %s" % path)
    errors = []
    for fn in BACKENDS:
        try:
            sc = fn(path, profiles)
        except UnreadableInput:
            raise
        except Exception as exc:
            errors.append("%s: %s" % (fn.__name__, type(exc).__name__))
            continue
        if sc is not None:
            if sc.valid_fraction() * sc.array.size < 64:
                raise UnreadableInput("%s read but holds almost no valid data"
                                      % os.path.basename(path))
            return sc
    raise UnreadableInput("no reader accepted %s%s"
                          % (os.path.basename(path),
                             (" (" + "; ".join(errors) + ")") if errors else ""))


def normalised(scene: Scene, profile: dict) -> np.ndarray:
    """Radiometric normalisation per the sensor profile, into [0, 1]."""
    n = profile.get("normalise", {}) or {}
    a = _percentile_norm(scene.array, scene.valid,
                         n.get("low", 2.0), n.get("high", 98.0))
    c = profile.get("clahe", {}) or {}
    if c.get("enabled"):
        import cv2
        u8 = np.clip(a * 255, 0, 255).astype(np.uint8)
        a = cv2.createCLAHE(float(c.get("clip", 2.5)),
                            (int(c.get("tiles", 8)),) * 2).apply(u8).astype(np.float32) / 255.0
        a[~scene.valid] = 0.0
    return a
