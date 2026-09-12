"""Sun and orbit geometry from the ancillary ASCII files (.spm, .oat, .oath).

The `.spm` file is the one that matters for illumination: one record per 40 ms
carrying phase angle, sun aspect, sun azimuth and sun elevation, plus spacecraft
position and velocity in a Moon-centred J2000 frame.

Why we read it instead of the single scalar in the PDS4 label: **sun azimuth
sweeps tens of degrees along one 16 s strip.** Measured on this archive,
20241115T1525 runs 203.1 deg -> 275.9 deg across the acquisition. That happens
because azimuth is referenced to local north and these strips pass within 0.05
deg of the south pole, where local north rotates fast. The label's scalar is
just the mid-strip sample. Any per-tile illumination reasoning has to interpolate
the series to that tile's line range.

Two documented defects are handled:

* `readme.txt` claims the record cadence is 512 ms. The records are 40 ms apart.
  The cadence is derived from the timestamps, never assumed.
* `readme.txt` labels the `.spm` section "Liberation angle Data File" (a
  copy-paste from the `.lbr` section) and misnumbers a field. The declared
  fixed-width column layout also does not match the emitted field widths, so we
  split on whitespace and take fields by position from the end, which is stable
  across all four products.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


# Fixed byte offsets for the record header. These MUST be used for the
# timestamp: the I4 block-length field (" 249") abuts the 4-digit year, so
# whitespace splitting yields a single merged token like "2492024" and the whole
# time field shifts by one. Only the trailing float columns are whitespace-safe.
_HDR_END = 46            # A8 + I6 + I4 + 7I4
_TIME_SLICES = [(18, 22), (22, 26), (26, 30), (30, 34), (34, 38), (38, 42), (42, 46)]


def _record_time(line: str) -> datetime | None:
    """Parse the 7-integer UTC stamp at its fixed byte offsets."""
    if len(line) < _HDR_END:
        return None
    try:
        y, mo, d, h, mi, s, ms = [int(line[a:b]) for a, b in _TIME_SLICES]
    except ValueError:
        return None
    try:
        return datetime(y, mo, d, h, mi, s, ms * 1000, tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass
class SunSeries:
    """Time series of solar geometry over one acquisition."""

    times: np.ndarray             # (n,) float, seconds since the first record
    epoch: datetime | None
    phase_deg: np.ndarray         # (n,) label calls this phase; == 90 - elevation here
    aspect_deg: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    sat_pos_km: np.ndarray        # (n, 3) Moon-centred J2000
    source: str = ""

    # ------------------------------------------------------------------ load
    @classmethod
    def from_spm(cls, path: str) -> "SunSeries":
        t0 = None
        ts, ph, asp, az, el, pos = [], [], [], [], [], []
        skipped = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line.startswith("ORBTATTD"):
                    continue
                when = _record_time(line)
                if when is None:
                    skipped += 1
                    continue
                # Everything after the fixed header is whitespace-separated
                # floats: X Y Z Xdot Ydot Zdot phase aspect azimuth elevation.
                # The emitted float widths do not match the widths declared in
                # readme.txt, so we split rather than slice here.
                try:
                    tail = [float(x) for x in line[_HDR_END:].split()]
                except ValueError:
                    skipped += 1
                    continue
                if len(tail) < 10:
                    skipped += 1
                    continue
                phase, aspect, azimuth, elevation = tail[6], tail[7], tail[8], tail[9]
                # Self-check: on this archive the "phase angle" column is
                # 90 - elevation. If that identity fails the column order is not
                # what we assume, and guessing would corrupt the illumination
                # reasoning downstream.
                if abs((90.0 - elevation) - phase) > 0.05:
                    raise ValueError(
                        "unexpected .spm column layout in %s: phase=%.6f but "
                        "90-elevation=%.6f (record at %s)"
                        % (path, phase, 90.0 - elevation, when.isoformat()))
                if t0 is None:
                    t0 = when
                ts.append((when - t0).total_seconds())
                ph.append(phase); asp.append(aspect)
                az.append(azimuth); el.append(elevation)
                pos.append(tail[0:3])

        if not ts:
            raise ValueError("no usable ORBTATTD records parsed from %s "
                             "(%d malformed)" % (path, skipped))
        return cls(np.asarray(ts, np.float64), t0,
                   np.asarray(ph, np.float64), np.asarray(asp, np.float64),
                   np.asarray(az, np.float64), np.asarray(el, np.float64),
                   np.asarray(pos, np.float64), source=path)

    # --------------------------------------------------------------- properties
    @property
    def cadence_s(self) -> float:
        """Median record spacing, derived - never taken from readme.txt."""
        if len(self.times) < 2:
            return float("nan")
        return float(np.median(np.diff(self.times)))

    @property
    def span_s(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

    def _interp(self, arr, t):
        return np.interp(np.asarray(t, np.float64), self.times, arr)

    def at_seconds(self, t: float) -> dict:
        """Solar geometry `t` seconds after the first ancillary record.

        Azimuth is unwrapped before interpolation so a 360 deg crossing inside
        the strip cannot produce a spurious mid-value.
        """
        az_unwrapped = np.degrees(np.unwrap(np.radians(self.azimuth_deg)))
        return {
            "azimuth_deg": float(self._interp(az_unwrapped, t)) % 360.0,
            "elevation_deg": float(self._interp(self.elevation_deg, t)),
            "phase_deg": float(self._interp(self.phase_deg, t)),
            "aspect_deg": float(self._interp(self.aspect_deg, t)),
        }

    def at_time(self, when: datetime) -> dict:
        if self.epoch is None:
            raise ValueError("series has no epoch")
        return self.at_seconds((when - self.epoch).total_seconds())

    def summary(self) -> dict:
        az = np.degrees(np.unwrap(np.radians(self.azimuth_deg)))
        return {
            "records": int(len(self.times)),
            "cadence_s": round(self.cadence_s, 6),
            "span_s": round(self.span_s, 4),
            "epoch": self.epoch.isoformat() if self.epoch else None,
            "azimuth_deg": [round(float(az.min()) % 360, 4), round(float(az.max()) % 360, 4)],
            "azimuth_sweep_deg": round(float(az.max() - az.min()), 4),
            "elevation_deg": [round(float(self.elevation_deg.min()), 4),
                              round(float(self.elevation_deg.max()), 4)],
            "source": os.path.basename(self.source),
        }


def line_time_seconds(line: int, lines_total: int, dwell_s: float) -> float:
    """Seconds from acquisition start to the given image line.

    The pushbroom samples one line per line period, so time is linear in line
    index. `dwell_s` comes from the label's start/stop times, which is more
    reliable than the label's line_exposure_duration (wrong unit).
    """
    if lines_total <= 1:
        return 0.0
    return dwell_s * (float(line) / float(lines_total - 1))


def shadow_length_per_metre(elevation_deg: float) -> float:
    """Horizontal shadow length cast by 1 m of relief, in metres.

    The single most important number for this dataset. At a solar elevation of
    0.2 deg one metre of relief casts a 286 m shadow, so shadow geometry is
    hyper-sensitive to terrain error: a DEM good to +/-2 m vertically places
    shadow boundaries only to within several hundred metres. Any predicted
    shadow mask built on a coarse DEM must carry that uncertainty rather than
    pretend to a crisp boundary.

    Returns inf for elevation <= 0 (sun at or below the local horizon).
    """
    import math
    if elevation_deg <= 0:
        return float("inf")
    return 1.0 / math.tan(math.radians(elevation_deg))
