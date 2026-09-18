"""PDS4 label parsing for Chandrayaan-2 OHRC products.

The labels declare the `isda:` prefix in a way that trips
`xml.etree.ElementTree` ("unbound prefix"), so values are extracted with
targeted regular expressions instead. That is deliberate: the fields we need are
flat scalars in a stable, ISRO-generated layout, and a tolerant reader is more
useful here than a strict one that refuses the whole file.

Two label defects are known and handled explicitly (see ASSUMPTIONS.md):

* ``isda:line_exposure_duration`` is tagged ``unit="ms"`` but the value is
  microseconds. 101 074 lines x 162.1 ms would be 4.5 hours against a 16.383 s
  acquisition. We expose the raw value, the corrected microsecond value, and the
  period implied by dividing the actual dwell by the line count.
* ``Refined_Corner_Coordinates`` is byte-identical to
  ``System_Level_Coordinates`` in every product, i.e. no photogrammetric
  refinement has been applied. ``corners_refined`` records whether the two
  differ, so downstream code never treats system-predicted geometry as refined.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _scalar(text: str, tag: str) -> str | None:
    """Value of the first <tag ...>value</tag>, ignoring any attributes."""
    m = re.search(r"<%s(?:\s[^>]*)?>([^<>]*)</%s>" % (re.escape(tag), re.escape(tag)), text)
    return m.group(1).strip() if m else None


def _float(text: str, tag: str) -> float | None:
    v = _scalar(text, tag)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(text: str, tag: str) -> int | None:
    v = _scalar(text, tag)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _iso(v: str | None) -> datetime | None:
    if not v:
        return None
    s = v.strip().rstrip("Z")
    # ISRO writes fractional seconds with 4 digits; datetime wants <= 6.
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?$", s)
    if not m:
        return None
    frac = (m.group(2) or "0")[:6].ljust(6, "0")
    return datetime.strptime(m.group(1) + "." + frac, "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=timezone.utc)


def _corner_block(text: str, block: str) -> dict[str, float]:
    m = re.search(r"<isda:%s>(.*?)</isda:%s>" % (block, block), text, re.S)
    if not m:
        return {}
    out = {}
    # The numeric text may carry surrounding whitespace: 20241115T1525 writes
    # lower_right_longitude as " 4.686608" (single-digit degrees padded to
    # width). A pattern anchored straight onto the digits silently drops that
    # corner, so whitespace is matched explicitly.
    for k, v in re.findall(
            r"<isda:(\w+_(?:latitude|longitude))(?:\s[^>]*)?>\s*([-+]?[\d.]+)\s*<",
            m.group(1)):
        out[k] = float(v)
    return out


@dataclass
class OhrcLabel:
    """Everything we read from one ``data/**/*.img`` PDS4 label."""

    path: str
    logical_id: str | None = None
    lines: int = 0
    samples: int = 0
    data_type: str = ""
    offset: int = 0
    axis_index_order: str = ""
    declared_file_size: int | None = None
    md5_checksum: str | None = None
    image_file_name: str | None = None

    start_time: datetime | None = None
    stop_time: datetime | None = None

    imaging_orbit: int | None = None
    dumping_orbit: int | None = None
    pixel_resolution_m: float | None = None
    spacecraft_altitude_km: float | None = None
    focal_length_mm: float | None = None
    detector_pixel_width_um: float | None = None
    tdi_stages: str | None = None
    bits_selection: str | None = None
    limb_direction: str | None = None
    reference_data_used: str | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None

    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    solar_incidence_deg: float | None = None

    projection: str | None = None
    area: str | None = None

    corners_system: dict[str, float] = field(default_factory=dict)
    corners_refined_block: dict[str, float] = field(default_factory=dict)

    # label defects, surfaced rather than silently patched
    line_exposure_raw: float | None = None
    line_exposure_declared_unit: str | None = None
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ derived
    @property
    def shape(self) -> tuple[int, int]:
        return (self.lines, self.samples)

    @property
    def expected_bytes(self) -> int:
        return self.lines * self.samples * self.item_bytes

    # PDS4 sample types this loader can memory-map. OHRC delivers UnsignedByte;
    # TMC-2 delivers UnsignedLSB2, and reading that as uint8 would not fail
    # loudly - it would return an image of half the declared line count made of
    # interleaved high and low bytes. So the mapping is explicit and anything
    # absent from it is refused rather than guessed.
    DTYPES = {"UnsignedByte": "u1", "SignedByte": "i1",
              "UnsignedLSB2": "<u2", "SignedLSB2": "<i2",
              "UnsignedMSB2": ">u2", "SignedMSB2": ">i2",
              "UnsignedLSB4": "<u4", "SignedLSB4": "<i4",
              "IEEE754LSBSingle": "<f4", "IEEE754MSBSingle": ">f4"}

    @property
    def numpy_dtype(self) -> str | None:
        return self.DTYPES.get(self.data_type)

    @property
    def item_bytes(self) -> int:
        d = self.numpy_dtype
        return 1 if d is None else int(d[-1])

    @property
    def dwell_seconds(self) -> float | None:
        if not (self.start_time and self.stop_time):
            return None
        return (self.stop_time - self.start_time).total_seconds()

    @property
    def line_period_us(self) -> float | None:
        """Line period in microseconds, derived from the actual dwell.

        Preferred over the label value, whose unit attribute is wrong.
        """
        d = self.dwell_seconds
        if not d or not self.lines:
            return None
        return d / self.lines * 1e6

    @property
    def corners_refined(self) -> bool:
        """True only if the refined block genuinely differs from the system block."""
        if not self.corners_system or not self.corners_refined_block:
            return False
        return any(abs(self.corners_system.get(k, 0.0) - v) > 1e-9
                   for k, v in self.corners_refined_block.items())

    def corner_lonlat(self) -> list[tuple[float, float]]:
        """UL, UR, LR, LL as (lon, lat), from the system-level block."""
        c = self.corners_system
        order = [("upper_left_longitude", "upper_left_latitude"),
                 ("upper_right_longitude", "upper_right_latitude"),
                 ("lower_right_longitude", "lower_right_latitude"),
                 ("lower_left_longitude", "lower_left_latitude")]
        return [(c[a], c[b]) for a, b in order if a in c and b in c]

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["start_time"] = self.start_time.isoformat() if self.start_time else None
        d["stop_time"] = self.stop_time.isoformat() if self.stop_time else None
        d["dwell_seconds"] = self.dwell_seconds
        d["line_period_us"] = self.line_period_us
        d["corners_refined"] = self.corners_refined
        d["expected_bytes"] = self.expected_bytes
        return d


def parse_label(path: str) -> OhrcLabel:
    with open(path, encoding="utf-8", errors="replace") as fh:
        t = fh.read()

    lab = OhrcLabel(path=path)
    lab.logical_id = _scalar(t, "logical_identifier")
    lab.data_type = _scalar(t, "data_type") or ""
    lab.offset = _int(t, "offset") or 0
    lab.axis_index_order = _scalar(t, "axis_index_order") or ""
    lab.declared_file_size = _int(t, "file_size")
    lab.md5_checksum = _scalar(t, "md5_checksum")
    lab.image_file_name = _scalar(t, "file_name")

    # Axis_Array order is Line then Sample ("Last Index Fastest").
    els = [int(x) for x in re.findall(r"<elements>(\d+)</elements>", t)]
    if len(els) >= 2:
        lab.lines, lab.samples = els[0], els[1]

    lab.start_time = _iso(_scalar(t, "start_date_time"))
    lab.stop_time = _iso(_scalar(t, "stop_date_time"))

    lab.imaging_orbit = _int(t, "isda:imaging_orbit_number")
    lab.dumping_orbit = _int(t, "isda:dumping_orbit_number")
    lab.pixel_resolution_m = _float(t, "isda:pixel_resolution")
    lab.spacecraft_altitude_km = _float(t, "isda:spacecraft_altitude")
    lab.focal_length_mm = _float(t, "isda:focal_length")
    lab.detector_pixel_width_um = _float(t, "isda:detector_pixel_width")
    lab.tdi_stages = _scalar(t, "isda:tdi_stages")
    lab.bits_selection = _scalar(t, "isda:bits_selection")
    lab.limb_direction = _scalar(t, "isda:orbit_limb_direction")
    lab.reference_data_used = _scalar(t, "isda:reference_data_used")
    lab.roll_deg = _float(t, "isda:roll")
    lab.pitch_deg = _float(t, "isda:pitch")
    lab.yaw_deg = _float(t, "isda:yaw")

    lab.sun_azimuth_deg = _float(t, "isda:sun_azimuth")
    lab.sun_elevation_deg = _float(t, "isda:sun_elevation")
    lab.solar_incidence_deg = _float(t, "isda:solar_incidence")

    lab.projection = _scalar(t, "isda:projection")
    lab.area = _scalar(t, "isda:area")

    lab.corners_system = _corner_block(t, "System_Level_Coordinates")
    lab.corners_refined_block = _corner_block(t, "Refined_Corner_Coordinates")

    lab.line_exposure_raw = _float(t, "isda:line_exposure_duration")
    m = re.search(r"<isda:line_exposure_duration\s+unit=\"([^\"]+)\"", t)
    lab.line_exposure_declared_unit = m.group(1) if m else None

    # ---- defects worth carrying into the UI rather than hiding -------------
    if lab.line_exposure_raw and lab.line_period_us:
        if abs(lab.line_exposure_raw - lab.line_period_us) < 1.0 and \
                lab.line_exposure_declared_unit == "ms":
            lab.notes.append(
                "line_exposure_duration is tagged unit=\"ms\" but the value (%.3f) matches "
                "the line period in MICROSECONDS derived from the acquisition dwell (%.2f us); "
                "the unit attribute is wrong."
                % (lab.line_exposure_raw, lab.line_period_us))
    if lab.corners_system and not lab.corners_refined:
        lab.notes.append(
            "Refined_Corner_Coordinates is identical to System_Level_Coordinates: no "
            "photogrammetric refinement has been applied, so corner geolocation is "
            "system-predicted and must not be treated as sub-pixel truth.")
    want = {"upper_left", "upper_right", "lower_left", "lower_right"}
    for blk_name, blk in (("System_Level_Coordinates", lab.corners_system),
                          ("Refined_Corner_Coordinates", lab.corners_refined_block)):
        if blk:
            missing = sorted(w + suf for w in want for suf in ("_latitude", "_longitude")
                             if (w + suf) not in blk)
            if missing:
                lab.notes.append("%s is incomplete: missing %s"
                                 % (blk_name, ", ".join(missing)))
    if lab.reference_data_used and lab.reference_data_used.strip().lower() == "system":
        lab.notes.append(
            "isda:reference_data_used = 'System' - geolocation derives from predicted "
            "orbit/attitude, not from a control network.")
    return lab


def verify_raster_size(lab: OhrcLabel, img_path: str) -> tuple[bool, str]:
    """Check the raster on disk against the label. Read-only."""
    if not os.path.exists(img_path):
        return False, "image file missing: %s" % img_path
    actual = os.path.getsize(img_path)
    if lab.numpy_dtype is None:
        return False, ("unsupported data_type %r; known: %s"
                       % (lab.data_type, ", ".join(sorted(lab.DTYPES))))
    if lab.offset != 0:
        return False, "unexpected offset %d (loader assumes headerless)" % lab.offset
    if actual != lab.expected_bytes:
        return False, ("size mismatch: %d bytes on disk, label implies %d x %d = %d"
                       % (actual, lab.lines, lab.samples, lab.expected_bytes))
    if lab.declared_file_size is not None and lab.declared_file_size != actual:
        return False, ("label file_size %d disagrees with %d bytes on disk"
                       % (lab.declared_file_size, actual))
    return True, ("ok: %d bytes == %d lines x %d samples x %d B (%s)"
                  % (actual, lab.lines, lab.samples, lab.item_bytes, lab.data_type))
