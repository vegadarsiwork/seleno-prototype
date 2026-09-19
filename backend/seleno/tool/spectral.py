"""Collapsing a hyperspectral cube into something a matcher can use.

IIRS is an imaging spectrometer: one scene arrives as a few hundred bands. Every
matcher in this project takes a single 2D raster, so the cube has to become one
image first, and *which* bands go into it is not a detail.

Two things rule bands out:

* **Thermal contamination.** Past roughly 2.5 um a lunar daytime spectrum is no
  longer reflected sunlight but the surface's own thermal emission. That signal
  tracks temperature, not albedo or topography, so it does not resemble the
  reflected-light reference it would be matched against, and near the terminator
  it is anti-correlated with the shadowing that carries the shape information.
* **Atmosphere-free does not mean artefact-free.** The ends of a spectrometer's
  range have the worst SNR, and detector order-sorting filter joins sit at fixed
  wavelengths. Averaging them in adds noise without adding structure.

So the default window is the reflected-light shortwave range, 0.8-1.6 um.

**This module has never been run against a real IIRS product.** No IIRS data has
been delivered to this project. It is written against the PDS4 spec and tested
against a synthetic cube, and the band-centre reader in `ohrc.label` tries
several spellings rather than committing to one. When a real product lands,
check `pseudo_pan_report` before trusting the result.
"""
from __future__ import annotations

import numpy as np

# Reflected-light shortwave window, micrometres.
DEFAULT_LO_UM = 0.8
DEFAULT_HI_UM = 1.6


def select_bands(centres_um, lo=DEFAULT_LO_UM, hi=DEFAULT_HI_UM) -> list:
    """Indices of bands whose centres fall inside the window."""
    return [i for i, c in enumerate(centres_um)
            if c is not None and np.isfinite(c) and lo <= float(c) <= hi]


def _band(cube, i, order):
    """One band as a 2D array, whatever the interleave."""
    if order == "bip":                       # line, sample, band
        return cube[:, :, i]
    if order == "bil":                       # line, band, sample
        return cube[:, i, :]
    return cube[i]                           # bsq: band, line, sample


def pseudo_pan(cube, centres_um=None, *, order="bsq", lo=DEFAULT_LO_UM,
               hi=DEFAULT_HI_UM, nodata=None, max_bands=48):
    """Average the reflected-light bands into one raster.

    Returns ``(image, report)``. Bands are standardised before averaging: a
    spectrometer's response varies by an order of magnitude across its range,
    and a straight mean would be whichever band happens to be brightest rather
    than a combination of all of them.

    `max_bands` caps the work - averaging 48 bands already buries the per-band
    noise, and reading 200 off a memmap for no further gain is just slow.
    """
    report = {"order": order, "window_um": [lo, hi], "n_bands_total": 0,
              "bands_used": 0, "band_indices": [], "dropped_flat": 0,
              "selection": None, "degraded": []}

    order = (order or "bsq").lower()
    nb = {"bip": cube.shape[2], "bil": cube.shape[1]}.get(order, cube.shape[0])
    report["n_bands_total"] = int(nb)

    if centres_um:
        idx = select_bands(centres_um, lo, hi)
        report["selection"] = "band centres from the label"
        if not idx:
            report["degraded"].append(
                "no band centre falls in %.2f-%.2f um (label range %.2f-%.2f um); "
                "using the middle third of the cube instead"
                % (lo, hi, min(centres_um), max(centres_um)))
            idx = list(range(nb // 3, max(nb // 3 + 1, 2 * nb // 3)))
            report["selection"] = "middle third (no band centre in window)"
    else:
        # No wavelengths at all. The middle of a spectrometer's range is the
        # least bad guess, but it IS a guess and has to be reported as one.
        idx = list(range(nb // 3, max(nb // 3 + 1, 2 * nb // 3)))
        report["selection"] = "middle third (label carried no band centres)"
        report["degraded"].append(
            "the label carried no band centre wavelengths, so the reflected-light "
            "window could not be applied; the middle third of the cube was used "
            "and the result is not a calibrated pseudo-pan")

    if len(idx) > max_bands:                 # even stride keeps the window's shape
        idx = [idx[i] for i in
               np.linspace(0, len(idx) - 1, max_bands).round().astype(int)]
        idx = sorted(set(idx))

    acc = None
    used = []
    for i in idx:
        b = np.asarray(_band(cube, i, order), np.float32)
        good = np.isfinite(b)
        if nodata is not None:
            good &= b != nodata
        if good.sum() < 16:
            continue
        v = b[good]
        sd = float(v.std())
        if sd < 1e-6:                        # dead or saturated band
            report["dropped_flat"] += 1
            continue
        z = np.zeros_like(b)
        z[good] = (v - float(v.mean())) / sd
        acc = z if acc is None else acc + z
        used.append(int(i))

    if acc is None:
        report["degraded"].append("no band in the window carried any signal")
        return None, report

    acc /= len(used)
    report["bands_used"] = len(used)
    report["band_indices"] = used
    if centres_um and used:
        report["window_used_um"] = [round(float(centres_um[used[0]]), 4),
                                    round(float(centres_um[used[-1]]), 4)]
    return acc.astype(np.float32), report


def pseudo_pan_report(report: dict) -> str:
    """One line for a log or a `degraded` entry."""
    w = report.get("window_used_um")
    return ("pseudo-pan from %d of %d bands (%s%s)"
            % (report["bands_used"], report["n_bands_total"],
               report["selection"],
               (", %.2f-%.2f um" % (w[0], w[1])) if w else ""))
