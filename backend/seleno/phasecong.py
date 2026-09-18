"""Phase congruency and a RIFT2-style descriptor.

Why this and not another gradient descriptor
--------------------------------------------
Every classical matcher in this repository keys on intensity gradient, and
gradient is exactly what a Sun-azimuth change destroys: move the Sun 90 degrees
and a crater's bright limb and dark shadow swap sides, so the gradient reverses
while the *structure* does not.

Phase congruency (Kovesi 1999) measures where the Fourier components of an image
are maximally in phase. That is a dimensionless property of local structure, so
it is invariant to brightness and contrast, and it responds to a step edge, a
shadow boundary and a ridge alike. RIFT (Li et al. 2020) builds a descriptor not
on gradient orientation but on the **maximum index map** - which log-Gabor
orientation channel is strongest at each pixel - which inherits that invariance.

The research summary in `research_by_space.md` rates this the highest
credibility-to-effort option available: no GPU, no training, and it has already
been benchmarked on Chandrayaan-2 data by its authors.

Scope, stated plainly
---------------------
* This is a faithful but compact implementation, not a port of the authors' code.
  It is a baseline to be measured, not a claim to reproduce their numbers.
* Rotation invariance is **off by default**. Both images in this project's polar
  work are resampled onto one map projection first, so the residual rotation is
  zero by construction, and RIFT's cyclic-shift trick costs `norient` times the
  matching work. `rotation_invariant=True` enables it for the general case.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:                                       # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------- #
# phase congruency
# --------------------------------------------------------------------------- #

def _lowpass(shape, cutoff=0.45, n=15):
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    x = (x - cols // 2) / cols
    y = (y - rows // 2) / rows
    radius = np.sqrt(x * x + y * y)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))


def phase_congruency(img, nscale=4, norient=6, min_wavelength=3.0, mult=2.1,
                     sigma_onf=0.55, k=2.0, cut_off=0.5, g=10.0, eps=1e-4):
    """Kovesi phase congruency.

    Returns ``(pc, mim, amplitude)``:

    * ``pc``  - phase congruency summed over orientations, in [0, ~norient].
      Bright wherever there is structure, regardless of its contrast.
    * ``mim`` - maximum index map: the index of the strongest orientation
      channel at each pixel, 0..norient-1. This is what the descriptor bins.
    * ``amplitude`` - per-orientation summed log-Gabor amplitude, kept so a
      caller can weight or threshold on response strength.
    """
    img = np.asarray(img, np.float64)
    rows, cols = img.shape
    IMG = np.fft.fft2(img)

    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    x = (x - cols // 2) / cols
    y = (y - rows // 2) / rows
    radius = np.fft.ifftshift(np.sqrt(x * x + y * y))
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    radius[0, 0] = 1.0
    sintheta, costheta = np.sin(theta), np.cos(theta)
    lp = _lowpass((rows, cols))

    # radial log-Gabor components, shared by every orientation
    log_gabor = []
    for s in range(nscale):
        wavelength = min_wavelength * mult ** s
        f0 = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / f0) ** 2) / (2 * np.log(sigma_onf) ** 2)) * lp
        lg[0, 0] = 0.0
        log_gabor.append(lg)

    pc_sum = np.zeros((rows, cols))
    amplitude = np.zeros((norient, rows, cols))

    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.minimum(np.abs(np.arctan2(ds, dc)) * norient / 2.0, np.pi)
        spread = (np.cos(dtheta) + 1.0) / 2.0

        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        sum_an = np.zeros((rows, cols))
        max_an = np.zeros((rows, cols))
        tau = None
        eo = []
        for s in range(nscale):
            filt = log_gabor[s] * spread
            resp = np.fft.ifft2(IMG * filt)
            eo.append(resp)
            an = np.abs(resp)
            sum_an += an
            sum_e += resp.real
            sum_o += resp.imag
            if s == 0:
                # Rayleigh noise estimate from the smallest scale, as Kovesi does:
                # the median of the smallest-scale response is dominated by noise,
                # so it calibrates the threshold from the image instead of a constant.
                tau = np.median(sum_an) / np.sqrt(np.log(4.0))
                max_an = an
            else:
                max_an = np.maximum(max_an, an)

        amplitude[o] = sum_an
        x_energy = np.sqrt(sum_e ** 2 + sum_o ** 2) + eps
        mean_e, mean_o = sum_e / x_energy, sum_o / x_energy
        energy = np.zeros((rows, cols))
        for resp in eo:
            energy += resp.real * mean_e + resp.imag * mean_o \
                - np.abs(resp.real * mean_o - resp.imag * mean_e)

        # expected noise energy, then a soft threshold
        total_tau = tau * (1 - (1 / mult) ** nscale) / (1 - (1 / mult))
        m_noise = total_tau * np.sqrt(np.pi / 2.0)
        s_noise = total_tau * np.sqrt((4 - np.pi) / 2.0)
        t = m_noise + k * s_noise
        energy = np.maximum(energy - t, 0.0)

        width = (sum_an / (max_an + eps) - 1.0) / (nscale - 1)
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * g))
        pc_sum += weight * energy / (sum_an + eps)

    mim = np.argmax(amplitude, axis=0).astype(np.uint8)
    return pc_sum, mim, amplitude


# --------------------------------------------------------------------------- #
# RIFT-style keypoints and descriptor
# --------------------------------------------------------------------------- #

def detect_on_pc(pc, mask=None, n_features=3000, min_distance=4):
    """FAST corners on the phase-congruency map.

    Detecting on `pc` rather than on intensity is the whole point: a shadow
    boundary and a lit ridge both raise phase congruency, so the same physical
    structure is found whichever way the Sun happens to be pointing.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    p = pc / (pc.max() + 1e-9)
    u8 = np.clip(p * 255.0, 0, 255).astype(np.uint8)
    if mask is not None:
        u8[~mask] = 0
    corners = cv2.goodFeaturesToTrack(u8, maxCorners=n_features, qualityLevel=0.01,
                                      minDistance=min_distance,
                                      mask=None if mask is None else mask.astype(np.uint8))
    if corners is None:
        return np.zeros((0, 2), np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def describe_mim(mim, pts, norient=6, patch=96, grid=6, shift=0, weight=None):
    """Histogram-of-MIM descriptor, SIFT-shaped but orientation-index valued.

    Two things make the difference between this working and not:

    * **Contributions are weighted by phase congruency**, not counted. Most of a
      polar patch is flat shadow, where the winning orientation channel is
      arbitrary; counting it equally swamps the structure that actually
      distinguishes one keypoint from another. Unweighted, nearest-neighbour
      distances among a patch's own descriptors averaged 0.52 and a 0.8 ratio
      test returned zero matches.
    * **A Gaussian spatial envelope**, as in SIFT, so a keypoint's descriptor is
      not dominated by whatever happens to sit at the edge of its window.

    `shift` cyclically rotates the orientation labels, which is how RIFT obtains
    rotation invariance without recomputing anything: a rotation of the image
    permutes which log-Gabor channel wins, so the descriptor of a rotated patch
    is a cyclic shift of the original.
    """
    h, w = mim.shape
    half = patch // 2
    cell = patch // grid
    m = ((mim.astype(np.int16) + shift) % norient)
    if weight is None:
        weight = np.ones(mim.shape, np.float32)
    weight = np.asarray(weight, np.float32)

    yy, xx = np.mgrid[0:patch, 0:patch].astype(np.float32)
    sigma = patch / 2.0
    envelope = np.exp(-(((xx - half) ** 2 + (yy - half) ** 2) / (2 * sigma * sigma)))

    out = np.zeros((len(pts), grid * grid * norient), np.float32)
    keep = np.zeros(len(pts), bool)
    for i, (x, y) in enumerate(pts):
        cx, cy = int(round(x)), int(round(y))
        if cx - half < 0 or cy - half < 0 or cx + half >= w or cy + half >= h:
            continue
        win = m[cy - half:cy + half, cx - half:cx + half]
        wgt = weight[cy - half:cy + half, cx - half:cx + half] * envelope
        d = np.zeros((grid, grid, norient), np.float32)
        for gy in range(grid):
            for gx in range(grid):
                sl = (slice(gy * cell, (gy + 1) * cell), slice(gx * cell, (gx + 1) * cell))
                d[gy, gx] = np.bincount(win[sl].ravel(), weights=wgt[sl].ravel(),
                                        minlength=norient)[:norient]
        v = d.ravel()
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        v = v / n
        v = np.clip(v, 0, 0.2)                       # SIFT-style clip, then renormalise
        v /= max(np.linalg.norm(v), 1e-6)
        out[i] = v
        keep[i] = True
    return out[keep], pts[keep]


def rift_features(img, mask=None, *, nscale=4, norient=6, n_features=3000,
                  patch=96, grid=6):
    """Keypoints plus descriptors for one image."""
    z = np.asarray(img, np.float64)
    if z.max() > z.min():
        z = (z - z.min()) / (z.max() - z.min())
    pc, mim, _ = phase_congruency(z, nscale=nscale, norient=norient)
    pts = detect_on_pc(pc, mask=mask, n_features=n_features)
    if len(pts) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, grid * grid * norient), np.float32), pc, mim
    desc, pts = describe_mim(mim, pts, norient=norient, patch=patch, grid=grid,
                             weight=pc)
    return pts, desc, pc, mim
