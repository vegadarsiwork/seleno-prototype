"""Declarative sensor profiles, loaded from config/sensors/*.yaml.

Per-instrument behaviour lives in data files rather than in `if` branches, so
adding a sensor is a new YAML file and not a patch to the pipeline. Matching is
deliberately loose: an arbitrary file may carry no instrument identifier at all,
in which case `_default` applies and the tool reports reduced capability instead
of guessing.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

CONFIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config", "sensors"))

def _token(t: str) -> str:
    """Match `t` as a standalone token, treating `_`, `-` and `.` as separators.

    `\\b` is the obvious thing to reach for and it is wrong here: `_` is a word
    character, so `\\bnac\\b` does not match `NAC_CM355_...`, which is exactly how
    these products are named. That silently dropped a real NAC mosaic to the
    `_default` profile.
    """
    return r"(?<![a-z0-9])%s(?![a-z0-9])" % t


# Substrings that identify an instrument in a logical id, a DATA_SET_ID or a
# filename. Ordered: the first hit wins, so more specific patterns come first.
PATTERNS = [
    ("ohrc", [r"ch2_ohr", _token("ohrc"), r"cho\.ohr"]),
    ("tmc2", [r"ch2_tmc", _token("tmc"), r"cho\.tmc", _token("tmc2")]),
    ("iirs", [r"ch2_iir", _token("iirs"), r"cho\.iir"]),
    ("selene_tc", [r"tco_map", r"sln-l-tc", r"selene", _token("kaguya")]),
    ("nac", [r"nac_pole", r"lroc.*nac", _token("nac"), r"lro-l-lroc"]),
    ("wac", [_token("wac"), r"lroc-wac"]),
]


@dataclass
class Profiles:
    profiles: dict = field(default_factory=dict)

    @classmethod
    def load(cls, config_dir: str | None = None) -> "Profiles":
        import yaml
        d = config_dir or CONFIG_DIR
        out = {}
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if not f.endswith((".yaml", ".yml")):
                    continue
                key = os.path.splitext(f)[0]
                with open(os.path.join(d, f)) as fh:
                    out[key] = yaml.safe_load(fh) or {}
                out[key].setdefault("name", key)
                out[key].setdefault("instrument", key)
        if "_default" not in out:
            out["_default"] = {"name": "unknown", "instrument": "unknown",
                               "nodata": 0, "shadow_threshold": 0.08,
                               "texture_threshold": 0.015,
                               "normalise": {"method": "percentile",
                                             "low": 2.0, "high": 98.0}}
        return cls(out)

    def get(self, name: str) -> dict:
        return self.profiles.get(name, self.profiles["_default"])

    def match(self, instrument: str = "", path: str = "") -> dict:
        """Identify the sensor from a label identifier and/or a filename."""
        hay = ("%s %s" % (instrument or "", os.path.basename(path or ""))).lower()
        for key, pats in PATTERNS:
            if key not in self.profiles:
                continue
            for p in pats:
                if re.search(p, hay):
                    return self.profiles[key]
        return self.profiles["_default"]

    def names(self):
        return sorted(k for k in self.profiles if not k.startswith("_"))
