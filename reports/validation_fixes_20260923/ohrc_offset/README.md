# OHRC control offset: the block-corner hypothesis is rejected

**Hypothesis.** The source block-averaging path samples at the block corner rather
than its centre. That would shift every sample by (f − 1)/2 = 1.58 source px, which
is 0.38 reference px.

**Result: rejected.** Source sampling is centred. The offset is already absent
immediately after placement, and first appears in the fine stage's matching. No fix
was made in this step; the cause is fixed under step 4
(`../heldout_independence/`).

## Test A: sampling alone (`sampling_test.py`, ramp)

The real source geometry was used with the pixel values replaced by a linear ramp.
`prealign()` placed it on native reference windows, and each output value was
compared with the source coordinate recorded for that pixel in `back_x` / `back_y`.

| Control | Windows × axes | Measured reduction f | Taps | Mean bias, source px | Corner hypothesis predicts |
|---|---:|---:|---:|---:|---:|
| OHRC (small offset) | 5 × 2 | 2.533–2.568 | 3 | −0.0005 to +0.0001 | +1.58 |
| OHRC (4.6 km offset) | 4 × 2 | 2.549–2.569 | 3 | 0.0000 to +0.0071 | +1.58 |
| TMC-2 | 5 × 2 | 1.496–1.498 | 1 | −0.0001 to +0.0001 | +0.18 |

**Side finding: the 4.17x path is not what runs in OHRC native windows.** `_sample`
measures the reduction along each image axis. The OHRC strip is rotated against the
NAC grid, so it measures 2.53–2.57 rather than 4.17 and uses 3 taps (−1, 0, +1).
That is symmetric, so it adds no bias, but it is narrower than the true 4.17 source
px footprint, so the samples are under-filtered. This is not fixed here.

## Test B: placement before any matching (`sampling_test.py`, placement)

The real source was prealigned through the TRUE control transform (prewarp = T) and
its shift against the synthetic reference measured over 64 × 64 px subwindows of
gradient magnitude. The estimator is calibrated and resolves 0.02 px.

- In every window where phase correlation and NCC agree in at least 50 of 64
  subwindows, the median shift is **(0.000, 0.000)** ref px. That holds for 3 of 5
  OHRC windows, 2 of 5 OHRC 4.6 km windows, and 5 of 5 TMC-2 windows (the largest
  TMC-2 reading is −0.010).
- The remaining windows cannot be measured: the two estimators agree in only 1–7
  of their 64 subwindows.

## Where the offset does come from (`probe_fine.py`)

This was an instrumented control run; `register.py` was not modified. It captures
the coarse model the fine stage prewarps with, and the native correspondences the
fine stage returns.

| Control | Fine matcher | Fine points | Measured displacement exactly 0 | Whole-pixel | Coarse vs truth, median ref px | Final vs truth, median ref px |
|---|---|---:|---:|---:|---:|---:|
| OHRC (small offset) | ORB | 44141 | 63.2% | 73.7% | 0.594 | 0.210 |
| OHRC (4.6 km offset) | ORB | 48084 | 68.8% | 78.0% | 4.702 | 1.369 |
| TMC-2 | SIFT | 45146 | 0.0% | 0.0% | 0.228 | 0.013 |

ORB keypoints lie on whole pixels. Inside a window that is already prewarped by the
coarse model, most matches therefore report a displacement of exactly zero, and a
zero-displacement correspondence lies exactly on the coarse model. A fit dominated
by such points keeps part of the coarse model's error. That is the offset.

It is also why held-out points score exactly 0 (step 4). Correction to
`../diagnostics/FINDINGS.md`: those exactly-zero points are in the **fitted**
segments, not the fallback ones. `_segment_fit` refits each segment with `refit()`,
which uses LMEDS for affine. When more than half of a segment's points lie exactly
on the prewarp, least median of squares returns the prewarp itself.

| Control | Segment | Fit inliers | Held-out points | Exactly on the model |
|---|---:|---:|---:|---:|
| OHRC (small offset) | 0 | 9824 | 3607 | 498 |
| OHRC (small offset) | 1 | 15674 | 10632 | 2738 |
| OHRC (4.6 km offset) | 3 | 5771 | 2400 | 499 |
| OHRC (4.6 km offset) | 4 | 33902 | 3269 | 824 |

The fallback segments hold no zeros.
