# Why the post-fix held-out error is large: (a), (b) or (c)?

Measurement only; `register.py` is unchanged. The numbers are in
[diagnostics.md](diagnostics.md) and [diagnostics.json](diagnostics.json), and the
vector plots are in [residuals/](residuals/). Reproduce with `control.py` and then
`analyse.py`.

- (a) the transform is genuinely off
- (b) the held-out set is mostly bad matcher output and the transform is fine
- (c) a coordinate or frame bug from the seven fixes

## Verdict

**(a) explains the large numbers. (b) is real but inflates only RMSE and adds about
1 reference px of scatter. (c) is ruled out at the scale of the reported errors.
Two sub-reference-pixel issues remain open; they are listed below.**

| Pair | Held-out median | What dominates | Evidence |
|---|---:|---|---|
| OHRC → NAC | 6.6 ref px | (a) | Systematic part 6.6 ref px; random part 0.95, the same as the ideal control's 1.00. An independent local-shift measurement in the same windows reads 6.0 against 6.1 held-out, and the two differ by 0.6 ref px. The plot shows a linear field that is zero along one line and reaches ±15 ref px on either side. |
| TMC-2 → SELENE | 4.5 ref px | (a), plus cross-sensor matcher noise | Systematic part 3.7 ref px; random part 1.0, against the control's 0.19. Residuals are cross-track and consistent within each patch. The exported model has a 1.3% cross-track scale; the control recovers scale to 1e-5. |
| IIRS 2025-07-29 → WAC | 10.2 ref px | (a), probably | Mean residual (−6.4, 0.1) ref px, direction coherence 0.54; 10 of 13 test-cell means point −x, at 3–16 ref px. There are too few points for the neighbour split, and the local-shift estimators never agree. |
| IIRS 2024-01-20 → WAC | 11.0 ref px | (a), probably | Test-cell means move from −19 ref px at one end of the strip to between +12 and +23 at the other: a cross-track error that grows along the track (rotation or shear). There are only 22 fit inliers. The local shift is not measurable here (the estimators agree in 0% of windows). |

## What each measurement showed

**1. Positive controls.** The synthetic reference was rendered from the real product
through its own geometry and a known affine, on the real reference's grid, with
the same options.

| Control | Transform error vs truth, median | p90 |
|---|---:|---:|
| TMC-2, 1.35x | 0.018 ref px (0.025 src px) | 0.13 ref px |
| OHRC, 4.17x | 0.57 ref px (2.4 src px) | 1.05 ref px |
| OHRC, 4.17x, 4.6 km offset (the real run's size) | 0.78 ref px (3.2 src px) | 1.61 ref px |

- **Recovered parameters:** rotation and scale come back to within 0.004° and 2e-4.
- **Export georeference:** registered.tif is offset 0.000 working px from the grid
  the model was fitted on.
- **What this excludes:** the chain does not produce errors of 5 to 15 ref px, even
  through a km-scale prewarp. That rules out (c) as the cause of the reported
  errors.

**2. Residual field.** The real OHRC and TMC-2 residuals are spatially systematic.
Held-out neighbours 8 to 40 working px away predict nearly all of the typical
residual, and what remains is about 1 ref px. In every control the systematic part
is at most 0.02 ref px.

The direction-coherence statistic on its own understates OHRC (0.29). A linear
error field points opposite ways on either side of its zero line, so coherence
reads low even though the field is entirely systematic.

**3. Matcher-independent local shift.** This is phase correlation and NCC on
gradient magnitude, in frozen test cells only.

- **Estimator error:** on 200 known shifts, 0.05 working px. In the controls, 0.2
  ref px (TMC-2) and 0.5–0.6 ref px (OHRC).
- **Real pairs:** the two estimators agree in only 0–50% of windows, against
  68–100% in the controls. Cross-sensor illumination changes gradient structure, so
  medians over all windows (10–18 ref px) mix real shifts with failures and are not
  a usable accuracy figure.
- **Where it works:** in the OHRC windows that contain held-out points, it confirms
  the held-out residuals independently.

**4. Reference-px thresholds.**

| Pair | ≤1 ref px | ≤2 ref px | ≤3 ref px |
|---|---:|---:|---:|
| OHRC | 4.0% | 15.6% | 27.8% |
| TMC-2 | 8.6% | 21.3% | 36.1% |
| IIRS 2025 | 1.2% | 5.4% | 10.2% |
| IIRS 2024 | 2.4% | 7.9% | 15.0% |

For OHRC, 1 ref px is 4.17 source px, so even the ≤1 ref px band is outside
source-pixel accuracy.

## Where (b) does matter

The TMC-2 control's transform is correct to 0.018 ref px. Yet its held-out RMSE is
42.9 source px, because the SIFT fine stage kept only 44% of its points as inliers
and every test-cell point is scored. Its held-out median, 0.25 source px, is
unaffected.

So on the real runs, RMSE mostly measures the matcher's tail. Medians, and the
systematic/random split, are what describe the transform.

## Open issues (not fixed in this pass)

1. **Held-out points lying exactly on the exported model.** In both OHRC controls,
   22–23% of held-out correspondences have residual below 1e-6. Their reference
   coordinates are integer native pixels. In both runs ORB was chosen as the
   fine-stage matcher. (Corrected later: the points are in the **fitted** segments,
   not the fallback ones, and the mechanism is now verified; see
   `../ohrc_offset/README.md`.)

   Mechanism: `_fine_stage` sets each source position to `prewarp(window point)`. A
   match with zero integer displacement therefore lands exactly on the prewarp
   model, and the LMEDS segment refit returns that model when such points are the
   majority of a segment. Held-out points built this way make
   held-out accuracy look better than it is: the 4.6 km control reports a held-out
   median of 0.00 while its true transform error is 0.78 ref px.

   None of the four real runs has any such point.
2. **A consistent offset of about 0.5–0.8 ref px in both OHRC controls.** Translation
   error at the centre is (0.44, 0.37) and (0.58, 0.63) ref px, and the residual bias
   points the same way. The TMC-2 control has none (0.004 px). At 4.17x this alone is
   2–3 source px, so OHRC cannot reach source-pixel accuracy even on ideal content.
   Its cause, a 4.17x-path convention or matcher localisation, is not isolated here.
3. **The two SELENE georeferences disagree by (−0.04, −0.06) working px (about 0.3 ref
   px, 2–3 m).** These are the pipeline's PDS3 reader and GDAL's reading of the same
   label. This does not affect held-out scoring, which uses a single grid, but does
   affect the absolute placement of the exported product. Which reader is right is
   not determined.
4. **The control runs depend on their path.** Rendering the TMC-2 control over the
   whole grid instead of the footprint box moved dense-NCC coverage from 0.50 to
   0.48. That fell under the early-stop threshold, so SIFT ran and the held-out RMSE
   went from 1.6 to 42.9 source px, while the transform error stayed below 0.1 ref px.
