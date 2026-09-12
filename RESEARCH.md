# Research notes

Compiled while building this prototype. Everything here was checked against the
linked source or measured on the data; nothing is recalled from memory.

---

## 1. Where the project now stands

**Framing:** *geometry-first lunar image registration with predicted unmatchable
regions and calibrated refusal.*

The argument in one paragraph. Matcher comparisons on Chandrayaan-2 data are
already published; spatially uniform feature selection is prior art; lunar
registration systems routinely use geometry, DEMs, control networks and robust
estimation. What is under-served is the step *before* correspondence: deciding
which parts of an image can be matched at all, and deciding when a registration
should be refused. On south-polar OHRC data that question dominates everything
else, because two thirds to four fifths of every strip carries no usable signal
and the Sun sits on the horizon.

**What we set out to test:** that delivered Sun geometry plus a coarse terrain
model can predict unmatchable regions well enough to improve registration, and
that a calibrated refusal decision can be built on top.

**What the data said instead** — the two findings that actually matter, both
measured in this repository:

1. **The delivered geolocation of two products disagrees by ~538 m** (≈2 240
   pixels at 0.24 m/px). Correcting that bulk offset moved the baseline matcher
   from 2–3 inliers at a 4–8 % inlier ratio to over a thousand inliers at
   56–60 %.
   **This single geometry-first step is worth more than every other stage in the
   pipeline combined**, and without it any statement about illumination
   robustness on this data is an artefact.
2. **A local horizon ray-cast is inapplicable to most of this archive.** Three
   of the four products have a mid-strip solar elevation at or below zero yet
   plainly contain lit terrain — near the pole, high ground is lit over a
   depressed horizon. So the shadow-prediction idea, as specified, cannot be
   tested on three quarters of these products without a wide-area horizon-angle
   computation over a real DEM, which is a different algorithm.

Both are reported in full in §6 rather than buried.

---

## 2. Official ISRO sources

| Source | URL | What we took from it |
|---|---|---|
| ISRO Payloads Data & Science Handbook | <https://www.isro.gov.in/media_isro/pdf/science/hand_book_payloads_data_and_science.pdf> | Nominal payload specifications |
| PRADAN — Chandrayaan-2 data dissemination | <https://pradan.issdc.gov.in/ch2/> | The authoritative archive (registration required) |
| Chandrayaan Map Browse | <https://chmapbrowse.issdc.gov.in/> | Footprint browse and per-product download |
| ISSDC acknowledgement / usage terms | <https://pradan.issdc.gov.in/ch2/ack.xhtml> | Licensing and attribution |
| PDS4 data archive for CH-2 payloads (TMC2, OHRC, IIRS) | <https://www.researchgate.net/publication/375922277_PDS4_DATA_ARCHIVE_FOR_CHANDRAYAAN-2_MISSION_PAYLOADS_TMC2_OHRC_and_IIRS> | Archive structure, product naming |
| Terrain Mapping Camera-2, *Current Science* 118(4) | <https://www.currentscience.ac.in/Volumes/118/04/0566.pdf> | TMC-2 design and its ~5 m/px GSD |
| OHRC data products user guide | shipped in the archive (`OtherDownloads/OHRC/`) | `ncp`/`nrp` levels, ancillary file layout |

### Payload specifications

| Payload | Spatial resolution | Spectral | In this prototype |
|---|---|---|---|
| **OHRC** | 0.25 m nadir nominal | Panchromatic | **Yes — four real Level-2 products** |
| **TMC-2** | ~5 m | Panchromatic stereo | No real data (a legacy pair simulates its GSD only) |
| **IIRS** | ~80 m, 0.8–5.0 µm | Hyperspectral | No. Not implemented. |

The four products report `isda:pixel_resolution` of 0.22–0.28 m/px, finer than
the nominal 0.25 m in two cases because they were taken from 85.7–111.97 km
rather than a fixed reference altitude. Per-product labels beat brochure numbers.

---

## 3. The paper that defines what is *not* novel here

**Makharia, R., Singla, J. G., Amitabh, Dube, N., Sharma, H. (2025).
"Comparative Evaluation of Traditional and Deep Learning Feature Matching
Algorithms using Chandrayaan-2 Lunar Data."** arXiv:2509.04775 —
<https://arxiv.org/abs/2509.04775>

It evaluates **SIFT, ASIFT, AKAZE, RIFT2 and SuperGlue** on cross-modality lunar
pairs from equatorial *and* polar regions using Chandrayaan-2 data, behind a
preprocessing pipeline of georeferencing, resolution alignment, intensity
normalisation, adaptive histogram equalisation, PCA and shadow correction.
Findings we treat as established: SuperGlue gives the lowest RMSE and the fastest
runtimes; SIFT and AKAZE do well near the equator and degrade under polar
lighting; on SAR sets the classical methods failed to register at all where
SuperGlue succeeded; preprocessing and learned matching both matter.

**Consequence: comparing feature matchers on Chandrayaan-2 data is done work, and
we do not present it as a contribution.** Our matcher rows exist to characterise
our own pipeline on our own windows. Our polar degradation being consistent with
theirs is a sanity check on the implementation, not a discovery.

---

## 4. Related work and what it means for our positioning

| Work | Link | Why it constrains us |
|---|---|---|
| MoonMetaSync: lunar image registration analysis | <https://arxiv.org/pdf/2410.11118> | Registration across OHRC/TMC-2/IIRS resolutions; the cross-resolution framing is not new |
| Deep local feature extraction for Yutu-2 PCAM images | <https://www.sciencedirect.com/science/article/abs/pii/S0924271623002964> | Learned features already applied to real lunar surface imagery |
| Automated registration of full-Moon images with triangulated network constraints | <https://www.researchgate.net/publication/381355146> | Geometric-constraint registration is established |
| Comprehensive review of remote sensing image registration | <https://www.researchgate.net/publication/351709629> | The taxonomy and metric conventions we follow |
| SuperGlue | <https://www.emergentmind.com/topics/superglue> | The reference learned matcher in the CH-2 paper |
| SIH problem statement SIH1732 | <https://vedas.sac.gov.in/static/pdf/SIH_2024/SIH1732_CH2_PS.pdf> | Confirms OHRC delivery as PDS-4 `.IMG` via chmapbrowse |

Recurring themes, all confirmed on our data:

1. Lunar imagery is low contrast with sparse texture — measured here: 22–42 % of
   512 px tiles carry usable gradient information.
2. Illumination is the dominant nuisance variable, worst near the poles.
3. Cross-sensor pairs differ in resolution, illumination and distortion at once.
4. Detector-free and attention-based matchers hold up better where texture is
   sparse.

**The gap we aim at:** none of this literature reports *deciding not to
register*, or predicting matchability before correspondence. That is where the
proposed contribution sits — and it is a proposal under test, not a result.

---

## 5. The dataset, as actually measured

Four OHRC Level-2 south-polar strips, 4.6 GB of unique pixels, **no annotations
of any kind**. Full inventory and provenance in `data/README.md`; assumptions and
defects in `ASSUMPTIONS.md`.

| | 20211228T2209 | 20241115T1326 | 20241115T1525 | 20251010T0942 |
|---|--:|--:|--:|--:|
| Lines × samples | 79 796 × 12 000 | 101 074 × 12 000 | 101 074 × 12 000 | 101 075 × 12 000 |
| GSD (m/px) | 0.28 | 0.24 | 0.24 | 0.22 |
| Orbit | 10445 | 23328 | 23329 | 27338 |
| Roll (deg) | −7.86 | 14.57 | 15.19 | 26.67 |
| Sun elevation mid-strip (deg) | −0.043 | −0.169 | +0.786 | −0.771 |
| Pixels ≤ DN 10 | 80.2 % | 71.4 % | 68.6 % | 57.6 % |
| Usable 512 px tiles | 22 % | 31 % | 33 % | 42 % |

Pairwise footprint overlap, computed independently in a south polar
stereographic plane at 100 m cells:

| pair | overlap | % of smaller |
|---|--:|--:|
| 20241115T1326 ↔ 20241115T1525 | 79.76 km² | **95.3 %** |
| 20211228 ↔ 20241115T1326 | 69.85 km² | 83.5 % |
| 20211228 ↔ 20241115T1525 | 67.06 km² | 79.7 % |
| 20241115T1525 ↔ 20251010 | 64.97 km² | 77.2 % |
| 20241115T1326 ↔ 20251010 | 62.11 km² | 74.2 % |
| 20211228 ↔ 20251010 | 53.27 km² | 57.6 % |

**Primary illumination-stress pair:** `20241115T1326` → `20241115T1525`.
Consecutive orbits two hours apart, 95.3 % mutual footprint, and at mid-strip
**Δazimuth 67.3°, Δelevation 0.955°** at 89–91° solar incidence. Cast shadows
move bodily between them; a shadow that dominates one window is absent from the
other over the same ground.

---

## 6. What we measured

All numbers regenerated by `python scripts/illumination_experiment.py`; the
generated tables live in `results/illumination/ILLUMINATION.md`.

### 6.1 Coarse geometric alignment is the dominant effect

Reprojecting both 2024 strips onto a common projected grid and correlating
gradient magnitude over the mutually lit ground:

| quantity | value |
|---|--:|
| Measured offset between the two products' geolocation | **538 m** |
| The same, in source pixels | **2 244 px** |
| Correlation peak / runner-up / margin | 0.180 / 0.070 / 0.110 |

Effect on the baseline matcher, per window:

| window | without the correction | with it |
|---|--:|--:|
| s10240 l52224 | 46 cand, 2 inliers, 4.3 % | 366 cand, 79 inliers, 21.6 % |
| s9216 l52224 | 58 cand, 3 inliers, 5.2 % | 73 cand, 22 inliers, 30.1 % |
| s10976 l41984 | 33 cand, 2 inliers, 6.1 % | 206 cand, 59 inliers, 28.6 % |

**This is the finding that changes how the project should be pitched.** The
easily-published-and-wrong conclusion was sitting right there: "SIFT collapses to
a 4 % inlier ratio under a 67° solar azimuth change." It collapses because the
windows were 2 200 pixels apart.

### 6.2 Ablation on the primary pair

Five 1024 px windows, affine model, mean over windows. **No ground truth exists**
for any of them, so held-out RMSE is the strongest honest figure.

| arm | configuration | cand. | inliers | inlier ratio | reproj RMSE | held-out RMSE | coverage | accepted |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| A | no verification | 2554 | 2554 | 100.0 % | 81.47 px | 85.55 px | 80 % | 0/5 |
| B | + robust verification | 2554 | 1347 | 55.8 % | 1.46 px | 1.46 px | 44 % | 5/5 |
| C | + GSD harmonisation | 2554 | 1347 | 55.8 % | 1.46 px | 1.46 px | 44 % | 5/5 |
| D | + flattening and CLAHE | 2364 | 1345 | 59.7 % | 1.50 px | 1.53 px | 46 % | 5/5 |
| E | + observed usability mask | 2362 | 1303 | 58.2 % | 1.40 px | **1.41 px** | 50 % | 5/5 |
| F | + predicted shadow mask *(SYNTHETIC terrain)* | 986 | 487 | 21.0 % | 1.61 px | 1.57 px | 16 % | 2/5 |
| G | + shading removal *(SYNTHETIC terrain)* | 934 | 453 | 40.8 % | 0.96 px | 1.49 px | 20 % | 2/5 |

Readings:

* **Robust verification is the second-biggest effect after coarse alignment.**
  Arm A reports a 100 % "inlier ratio" alongside an 81 px reprojection RMSE and
  0/5 acceptances, which is the clearest possible demonstration that an inlier
  ratio without verification is meaningless.
* **Arm C is identical to B**, because both products are 0.24 m/px — there is
  nothing to harmonise. Reported as a no-op rather than dressed up.
* **Arms F and G are worse on every figure that matters.** The synthetic mask
  cuts candidates from 2 362 to ~950 and coverage from 50 % to 16–20 %, and
  acceptance falls from 5/5 to 2/5. Arm G's low reprojection RMSE (0.96 px) comes
  from fitting a much smaller, better-behaved region — a selection effect, not an
  improvement. **These arms are not evidence about the real surface** — the
  terrain model is synthetic.

### 6.3 Two negative results about our own stages

The same five windows as §6.2, arm E as the parent (rows H, I, J):

| arm | configuration | retained | reproj RMSE | held-out RMSE | coverage |
|---|---|--:|--:|--:|--:|
| E | all inliers, no refinement | 1303 | 1.40 px | **1.41 px** |
| H | E + spatial grid selection | 44 | 1.43 px | 1.71 px |
| I | E + sub-pixel refinement | 1303 | 2.36 px | 2.36 px |
| J | E + both | 44 | 2.41 px | 2.92 px |

Reprojection RMSE barely moves under grid selection (1.40 → 1.43 px) while
held-out error rises by a fifth (1.41 → 1.71 px). That divergence is the
signature of a worse-conditioned fit, and it is another concrete demonstration
that reprojection RMSE must not be read as accuracy.

An independent three-window sweep gave the same ordering
(1.346 px unrefined, 2.331 px with grid selection, 2.438 px with refinement),
which is why the conclusion is not an artefact of one window sample.

**Spatially uniform selection — prior art we adopted — hurts here.** With ~1300
inliers already confined to the lit minority of the window, thinning to ~44 by
grid quota removes real information without improving conditioning. On the
legacy *synthetic* pair the same stage helps (ground-truth corner error 0.163 px
→ 0.035 px), so this is a regime difference, not a bug. The selector now skips
thinning below 60 inliers, and `seleno` leaves it off.

**Patch-correlation sub-pixel refinement hurts here, and tightening it does not
rescue it:**

Measured on the three-window sweep:

| acceptance criterion | refined | held-out RMSE |
|---|--:|--:|
| none | – | **1.346 px** |
| NCC ≥ 0.35, shift ≤ 4 px | 786 | 2.438 px |
| NCC ≥ 0.60, shift ≤ 2 px | 699 | 2.087 px |
| NCC ≥ 0.80, shift ≤ 1 px | 417 | 1.521 px |

Monotone toward *not refining*. The physical reason is clean: patch correlation
assumes the two patches are the same scene under a photometric transform. Under a
67° change of solar azimuth at grazing incidence they are two different light
fields, so the correlation peak is biased toward whatever the changed shading
favours. Off by default, with the finding recorded.

### 6.4 Sensitivity worth disclosing

* Transform model matters more than any matcher choice: on one window from
  identical correspondences, similarity gave 52 inliers at 22.5 %, affine gave
  133 at 57.6 %, homography 140 at 60.6 % but with worse held-out error
  (4.93 px vs 1.70 px) — it absorbs relief into perspective terms.
* Window size matters: 512 px yields ~14 candidates and is always refused;
  2048 px with affine gives ~1 024 inliers and 50 % coverage.
* Coverage depends on the grid: the same run scored 11 % on 6×6 and 19 % on 4×4.

---

## 7. What Seleno claims, and what it does not

### Claims, supported by numbers in this repository

* A working pipeline on **real Chandrayaan-2 OHRC Level-2 products**, from PDS4
  label and memory-mapped raster through to a registration decision, with a
  read-only, memory-safe dataset layer and 28 passing tests.
* **The measurement that the delivered inter-product geolocation is ~538 m out**,
  and the demonstration that correcting it is worth more than every other
  pipeline stage combined.
* A result object that **refuses** rather than returning a confident transform,
  with the failing checks enumerated.
* Metrics that keep reprojection RMSE, held-out RMSE, ground-truth error and
  disagreement-with-delivered-geometry as four separate concepts.
* Two honest negative results about stages we implemented (§6.3).

### Not claimed

* **Not validated.** The proposed contribution — predicted unmatchable regions
  from Sun geometry plus terrain — is implemented end to end and has **not been
  shown to work**. No DEM ships with this repository; the only working terrain
  model is synthetic; and the local horizon test is inapplicable at the negative
  solar elevations that dominate this archive.
* **Not calibrated.** The refusal thresholds are hand-picked. No precision or
  recall for the decision has been measured. `thresholds_calibrated` is `false`
  everywhere.
* **Not novel feature matching.** See §3.
* **Not sub-pixel geodetic accuracy.** The sub-pixel figures in this repository
  belong to synthetic legacy pairs. On real repeat-pass windows the honest figure
  is a ~1.2–1.3 px held-out RMSE with no ground truth at all.
* **Not cross-sensor, not cross-modal.** No real TMC-2 or IIRS data is used.
* **Not ISRO validated, endorsed or approved. Not production ready.**

---

## 8. Where a real contribution could still come from

Ranked by what would most change the answer.

1. **Wide-area horizon angles from a real polar DEM.** The shadow-prediction idea
   is not refuted — it is untested, because the local ray-cast is the wrong
   algorithm for a Sun on the horizon (§6, finding 2). Computing horizon
   elevation per azimuth over tens of kilometres of LOLA/LDEM is the correct
   formulation and is the single most valuable next step.
2. **Calibrate the refusal decision.** Build labelled positive pairs (windows
   with a verified correspondence) and negative pairs (windows known not to
   overlap, which the disjoint-orbit geometry provides for free), then fit the
   decision and report precision/recall. This turns "calibrated refusal" from a
   name into a result.
3. **Along-strip alignment instead of one bulk translation.** The residual
   disagreement with the corrected prior is still 160–290 m and varies along the
   strip because of the roll difference. A per-segment or low-order along-track
   model is the obvious next refinement, and it is measurable.
4. **A photometric-invariant descriptor or a learned matcher.** Given §6.3, the
   remaining appearance gap is exactly where SuperGlue/LoFTR-class methods are
   reported to help. The interface exists; the weights do not.
5. **Sensor-model registration** with SPICE and the `.oat`/`.lbr`/`.spm` series,
   replacing the global 2D transform, which is what a rigorous pushbroom solution
   requires.
6. **Multi-frame use of all four strips** over the 53–80 km² they share, rather
   than pairwise registration.
