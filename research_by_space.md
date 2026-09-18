# How the Space Agencies Actually Solve Lunar Image Registration

**Research compiled September 2026 · for SIH26166 / Seleno-prototype**

Covers NASA/USGS, JAXA, CNSA (China), Roscosmos and ESA, plus the cross-cutting techniques,
and ends with what Seleno could realistically take from it before 30 September.

Confidence is flagged throughout. Where a source was paywalled or robots-blocked, that is
stated rather than papered over.

---

## 0. The headline

Read this section even if you read nothing else.

> **No agency solves this problem by matching image A directly to image B.**

Every mature pipeline in the world avoids the cross-illumination matching problem rather than
attacking it head-on. There are exactly **three devices** in use, and effectively all published
work is one of them, or a combination:

| # | Pattern | The idea | Who uses it |
|---|---------|----------|-------------|
| **1** | **Render the reference into the query's appearance** | Don't compare two illuminations — *generate* the matching one from a DEM, then match like-to-like | NASA (LOLA hillshade matching), China (CE-2 DEM hillshades; CE-6 descent-image simulation), ESA (PANGU), JAXA (EIGENCRATER) |
| **2** | **Match geometry, not radiometry** | Shadows move; crater rims don't. Build the descriptor from inter-crater geometry or from 3-D surface shape | JAXA (SLIM ETSM — flight-proven), China (CNSFM, LGCN2025 polar), DLR/ESA (CNav/LION), NASA-adjacent academia (Christian conic invariants) |
| **3** | **Cascade the scale/illumination gap** | Never make one big jump. Chain many small ones | NASA (sub-solar-longitude ordering), China (CE-5 coarse-to-fine descent cascade) |

Plus two **absolute-control devices** that recur everywhere:

- **Laser altimetry (LOLA / LALT)** as the geodetic datum — active, so illumination-independent.
- **Laser retroreflectors** (Apollo/Lunokhod) as centimetre-accurate absolute control points.

And one **structural fact** that reframes the whole problem:

> **Registration, as agencies practise it, is not a 2-D warp between two images. It is a global
> least-squares solve for camera position and pointing across *all* images at once, anchored to a
> geodetic frame. Pairwise matching is a measurement *input* to that solve, not the answer.**

---

## 1. The physics — what can and cannot be fixed

Before any method, be precise about what actually differs between two images under different sun.

| Effect | Physical cause | Removable by photometric normalisation? |
|---|---|---|
| **Smooth BRDF variation** — limb darkening, phase brightening, opposition surge | Reflectance is a function of (i, e, g) | **Yes.** This is exactly what Hapke / Lunar-Lambert / Minnaert model |
| **Shading from resolved topography** | Local slope changes local i and e | **Only if you have a DEM at or below image resolution** |
| **Cast shadows and shadow-direction reversal** | Occlusion geometry | **No.** A shadow is missing signal. Nothing recovers it |

Where:
- **i** = incidence angle (sun to local surface normal)
- **e** = emission angle (surface normal to camera)
- **g** = phase angle (sun–surface–camera separation)

**The consequence, stated bluntly:** when the sun azimuth flips by 180°, every crater's bright and
dark crescents swap sides. The gradient field literally reverses. **No BRDF model repairs that, and
no intensity-gradient descriptor survives it** — which is why SIFT dies at the poles and why the
field split into the three patterns above.

This is the single most important framing point in the entire document.

---

## 2. NASA / USGS — the reference implementation

### 2.1 The conceptual core: control networks and bundle adjustment

NASA and USGS **register the cameras to a geodetic frame, and the images follow.**

A **Control Network** is a three-level structure:

```
ControlNetwork
 └── ControlPoint      — one physical place on the Moon
      └── ControlMeasure — that place's (sample, line) position in ONE image
```

A point seen in 8 images has 8 measures. Point types carry the geodesy:

| Type | Behaviour in the solve |
|---|---|
| **Free** | Coordinates are unknowns, solved for. Gives *relative* consistency only |
| **Constrained** | A-priori coordinates *with sigmas*; pulled toward them with weight 1/σ² |
| **Fixed** | Held immovable. Hard ground truth |

> **A network of only Free points has a datum defect.** It can be internally seamless while being
> globally translated, rotated or scaled off the Moon entirely. Absolute accuracy requires Fixed or
> Constrained points — in lunar practice, points whose coordinates come from LOLA.

That is, in a sentence, the geodetic version of Seleno's own "self-consistency is not accuracy"
argument. It has been institutional knowledge at USGS for decades.

**`jigsaw`** is the bundle adjustment: a simultaneous least-squares solve over camera position and
pointing (and optionally point coordinates and camera intrinsics) minimising reprojection residuals.
Every ControlPoint seen in *k* images contributes *2k* observation equations for 3 unknowns, so the
system is massively overdetermined and errors average down. Critically, **errors propagate
transitively** — images A and Z that never overlap are made mutually consistent through the chain of
shared points.

Key `jigsaw` parameters worth knowing:

| Parameter | Meaning |
|---|---|
| `CAMSOLVE` | `ANGLES` / `VELOCITIES` / `ACCELERATIONS` — polynomial degree for pointing |
| `OVEREXISTING` / `OVERHERMITE` | Fit the solved polynomial *as a correction over* existing SPICE, **preserving high-frequency jitter** rather than smoothing it away |
| `MODEL1/2/3` | Robust M-estimators: HUBER, HUBER_MODIFIED, WELSCH, CHEN |
| `OUTLIER_REJECTION` + `REJECTION_MULTIPLIER` | MAD-based residual rejection (default 3.0) |
| `ERRORPROPAGATION` | Compute the full variance-covariance matrix → real uncertainty estimates |
| `SIGMA0` | Convergence threshold on σ₀, the std. dev. of unit weight |

**σ₀ is their headline quality number.** Values near 1.0 mean residuals match the a-priori
observation weights. The 2026 production paper reports mean σ₀ = 0.41 for NAC controlled mosaics.

### 2.2 Pairwise matching vs. control network — the comparison that matters to Seleno

| | Pairwise matching (Seleno today) | Control network + bundle adjustment |
|---|---|---|
| **Solves for** | a 2-D transform between two images | 3-D camera pose for *N* images + 3-D ground coordinates |
| **Consistency** | pairwise only; chains **drift** | globally consistent, **transitive** |
| **Terrain** | ignored — a 2-D warp cannot express parallax under a moving camera | explicit: rays intersect a shape model |
| **Absolute frame** | none | LOLA-anchored via Fixed/Constrained points |
| **Outliers** | RANSAC, per pair | MAD rejection + Huber/Welsch M-estimators, network-wide |
| **Uncertainty** | not modelled | full variance-covariance |
| **Pushbroom validity** | **invalid** — no single exterior orientation exists | valid — per-line pose from a polynomial/spline |
| **Output** | a warped image | corrected *camera geometry*, reprojectable at will |

That last row is the practical punchline: bundle adjustment produces a **reusable geometric
correction**, not a one-off resampled raster.

### 2.3 The pushbroom problem — why a single homography is the wrong *model class*

A **framing camera** exposes the whole focal plane at one instant: 6 exterior-orientation parameters
total. For flat terrain a homography between two framing images is *exactly* correct.

A **pushbroom camera** builds the image one line at a time. **Each line has its own 6 parameters.** A
52,224-line LROC NAC image has, in principle, 52,224 independent exterior orientations acquired over
~17.6 seconds.

Consequences:
1. There is no single exterior orientation to solve for. A global 2-D transform is not merely
   inaccurate — **it is the wrong model class.**
2. Pose errors map into image-space *distortion*, not displacement. A roll oscillation gives a wavy
   cross-track warp; pitch gives along-track stretch.
3. Terrain amplifies it — the ground projection of a pose error scales with slant range and relief.
4. The errors are high-frequency, driven by mechanical resonances above what the attitude record samples.

**Measured LROC NAC jitter:** amplitude 0.2–2 px (problematic above ~1.5 px), ~6 Hz, caused by solar
array resonance. Commissioning-phase DTMs showed "ripple patterns of tens of metres in elevation."

**Elegant detail worth stealing conceptually:** LROC measures its own jitter using the ~135-pixel
overlap strip between NAC-L and NAC-R, running correlation down it to get offset vs. line number.
**The instrument is its own jitter sensor.**

**Fixes:** `jigsaw` polynomials for low-order drift; ASP **`jitter_solve`** for actual oscillation —
it resamples camera position and orientation along the track (`--num-lines-per-position`,
`--num-lines-per-orientation`), giving enough variables to represent a 6 Hz wobble by a Nyquist
argument that a low-order polynomial structurally cannot.

### 2.4 LOLA — why altimetry is the datum

| Property | Value |
|---|---|
| Wavelength | 1064.4 nm, 5 spots in an X pattern |
| Shot rate | 28 Hz (140 measurements/s) |
| Spot diameter | ~5 m at 50 km |
| Range precision | ~10 cm |
| Cross-track spacing | ~1.2 km at equator, ~200 m at 80°N/S |
| Total shots | ~4.5 × 10⁹ used in SLDEM2015 |

**Four reasons it is the reference and not imagery:**

1. **It measures range directly.** Time-of-flight gives absolute distance. An image gives only an
   angle — converting to position requires knowing where the camera was, which is the thing you are
   trying to determine. **Altimetry breaks the circularity.**
2. **It is illumination-independent.** Active instrument. Works identically in permanent shadow, at
   midnight, at grazing incidence. On the Moon — where the poles are both the most interesting and
   the worst-lit region — this is decisive.
3. **Crossover adjustment.** Where ascending and descending tracks intersect, elevation must agree.
   Any disagreement is orbit/pointing error. Solving globally to minimise crossover discrepancies
   reduced south-pole orbital positioning errors by **>10×**, reaching 10–20 cm horizontal.
4. **It defines the frame** — MOON_ME (Mean Earth / Polar Axis), tied to JPL DE421, sphere radius
   1,737,400 m.

### 2.5 The cheapest registration win in the whole document

**Smithed SPICE kernels.** LRO ships two tiers of ephemeris: routine reconstructed navigation
solutions, and **"smithed" SPKs** derived by fitting LOLA altimetric crossovers and GRAIL gravity.
Invoked as `spiceinit ... spksmithed=true`.

**The LROC NAC Processing Guide states these improve positional accuracy by 5–10×.**

That is a 5–10× accuracy improvement from *one boolean flag*, before any image processing happens at
all. The general lesson: **use the best available ancillary data before you write a single line of
matching code.**

### 2.6 Photometric normalisation — the model ladder

All predict reflectance `r(i, e, g)`. Write μ₀ = cos i, μ = cos e.

**Lambert** — `r ∝ μ₀`
Perfectly diffuse. Depends on incidence only. Physically wrong for regolith (predicts limb darkening
the Moon does not show), but it is the baseline.

**Lommel–Seeliger** — `r ∝ μ₀ / (μ₀ + μ)`
Falls straight out of the radiative-transfer integral for **single scattering in a semi-infinite
particulate medium**. Lunar regolith is dark, so single scattering dominates and multiple scattering
is negligible — **this is physically the right first-order law for the Moon.** ISIS implements it with
*no free parameters*.

**Lunar-Lambert (McEwen 1991)** — `r = A[ 2·L(g)·μ₀/(μ₀+μ) + (1 − L(g))·μ₀ ]`
A phase-dependent blend of the two. **The practical workhorse.** Cheap, differentiable, captures the
dominant behaviour with one phase-indexed coefficient. ISIS ships `LunarLambertMcEwen` with the lunar
L(g) curve **baked in — no parameters required**. Also ASP's `--reflectance-type 1`, the recommended
default for lunar shape-from-shading.

**Minnaert** — `r ∝ μ₀^k · μ^(k−1)`. Empirical limb-darkening law, one exponent. k=1 → Lambert.

**Hapke** — the full physical BRDF, modelling single scattering, multiple scattering, macroscopic
roughness and the opposition effect separately. Best fit, but 5–6 free parameters that must be
*retrieved* from repeat observations at many geometries.

**You do not need to fit Hapke for the Moon.** Sato et al. (2014) already did it: ~66,000 LROC WAC
observations over 21 months, producing **resolved Hapke parameter maps at 1°×1°, 70°N–70°S** — the
`SDWHAP` product — normalised to the standard geometry i=60°, e=0°, g=60°. Free to download.

**The mechanics:**

```
I_normalised = I_observed × r_model(i_ref, e_ref, g_ref) / r_model(i_obs, e_obs, g_obs)
```

Divide out the model's prediction at the pixel's actual geometry; multiply by its prediction at a
chosen standard geometry. ISIS `photomet` does exactly this.

**The crucial dependency:** per-pixel i, e, g come from **SPICE + a shape model**. Photometric
normalisation *requires topography*, because i and e are defined against the **local** surface normal,
not the sphere normal. Using an ellipsoid shape model removes only the broad across-strip gradient.

**ISIS normalisation modes — and which to pick for registration:**

| Mode | What it does | For matching? |
|---|---|---|
| `Albedo` | Divides out the full photometric function → an albedo map | **Usually wrong on the Moon.** Most lunar contrast *is* topographic, so this can leave you a nearly featureless image |
| `Topo` | Equalises topographic shading contrast | **Usually the right choice** — preserves the texture matchers need while equalising its strength |
| `Shade` | Produces simulated shaded relief | The *synthetic* half of a real-to-synthetic match |
| `Mixed` | Albedo at low incidence, Topo at high; `INCMAT` sets the transition | Practical for strips spanning a wide incidence range |

**How practitioners actually use it (NAC, 2026 production paper):**
- incidence < 60° → `lronacpho`, "Moon Mean Photometric" correction (valid phase 15–65°)
- incidence > 60° → `photomet` with the full **Hapke** model
- **then still manual histogram normalisation** to match median pixel values across overlaps

That 60° switchpoint is a real dividing line: at high incidence the empirical corrections break down
and you need the full BRDF. And note the third bullet — **the field's own admission that the
physics-based path runs out.** For ShadowCam they applied *no* photometric correction at all, because
secondary illumination in permanently shadowed regions is unpredictable.

### 2.7 Illumination chaining — the highest-value trick in the document

**Used in both the LROC NAC South Pole Controlled Mosaic and ASP's shape-from-shading workflow.**

> **Don't try to match a dawn image to a dusk image. Sort all images by sub-solar longitude, match
> only *adjacent* pairs in that ordering, and let the network's transitivity chain the
> correspondences across the full illumination range.**

**Why it works:** feature matching degrades *continuously* with illumination difference. If A↔Z is
impossible but A↔B, B↔C, … Y↔Z are each easy, transitivity delivers A↔Z consistency for free.
**This converts an impossible matching problem into a graph-connectivity problem.**

It is available *only because* the registration is a global network solve rather than pairwise
alignment. ASP's docs state the requirement plainly: *"Images must be sorted by solar azimuth angle
to ensure gradual illumination changes and enable successful interest point matching."*

**Worked example — LROC NAC South Pole Controlled Mosaic (Wagner et al., LPSC 2022):**
- **~39,948 images aligned** out of ~50,000 below 85°S
- 637 reference images, 5,875 alignment chains
- Method: brute-force automated 2-D alignment in map space using ISIS `findfeatures` with **SIFT**
- Iterative expansion outward from a single well-aligned seed image
- **Illumination handled purely by sub-solar-longitude ordering**
- Anchored to a NAC DTM itself tied to LOLA within ~3.3 m → **absolute accuracy better than 5 m**
- Result: image-to-image seams **<10 m**, versus **~100 m** uncontrolled

Note what this proves: **plain SIFT, correctly sequenced, aligned 40,000 polar images.** The
sequencing did the work, not the descriptor.

### 2.8 Validation without ground truth — directly applicable to Seleno

NAC mosaics lack enough precisely-known reference points for direct absolute assessment. The 2026
production team's workaround:

Assemble a set of **similarly illuminated** NAC images meeting explicit criteria:
- sub-solar longitude difference **< 60°**
- incidence angle difference **< 25°**
- **< 80% shadowed**
- ≥ 200×200 px overlap around each test point

Align that series to the mosaic, and take the **median location across the set as ground truth.**

For ShadowCam they instead compare level-2 cubes directly against **hillshades rendered from LOLA
DEMs** — a more direct check, possible precisely because hillshading lets you synthesise a
geodetically-true image under *any* illumination you like.

**Achieved:** NAC controlled mosaics, median positional offsets <12 m latitude, <5 m longitude.
Seams <15 m. σ₀ mean 0.41.

### 2.9 Prior art for Seleno's `spatial.py`

**ISIS `findfeatures` already has point-distribution algorithms:** a **grid** algorithm that spreads
matches over large images with iteratively refined spacing, and a **radial** algorithm (the default)
emitting concentric rings from the image centre.

This matters two ways. It **validates** the coverage idea as established practice — and it means
Seleno should **cite it rather than claim novelty**. The novel part of Seleno's contribution is the
*measurement* (showing RMSE rewards clustering), not the grid quota itself.

`findfeatures` also offers:
- Detectors: FAST, AGAST, GFTT, MSD, MSER, Star, Blob
- Detector+extractor: SIFT, ORB, BRISK, AKAZE, KAZE
- Four-stage outlier removal: ratio test → symmetry test → fundamental-matrix RANSAC → homography refinement
- **`FASTGEOM`** — projects the trainer image into the query's camera geometry *before* matching,
  removing scale and rotation so that non-invariant detectors become usable

That last one is important and appears again in §6.

---

## 3. JAXA — Kaguya, SLDEM2015, and flight-proven crater matching

### 3.1 Terrain Camera geometry

| Parameter | Value |
|---|---|
| Type | Monochrome pushbroom, **two optical heads** (TC1 forward, TC2 backward) |
| Detector | 1-D CCD, 4096 px per head |
| Band | Panchromatic 430–850 nm |
| Resolution | **10 m/px** from 100 km |
| Slant angles | **±15° from nadir** |
| Convergence angle | **30°** |
| Base-to-height ratio | **0.57** |
| Swath | 40 km full / 35 km nominal |
| Coverage | **>99% of the Moon in stereo** |

**The design insight worth noting:** because both heads fire *simultaneously*, a TC stereo pair has
**identical illumination and albedo in both looks** — the only difference is parallax. That removes
the single biggest confounder in multi-pass stereo, by construction. Compare Seleno's problem, where
the two images are separated in time.

### 3.2 Kaguya deliberately does *not* photometrically flatten TC

An important and often-missed point:

- `TCOrtho` products convert radiance to radiance factor using **only the incidence/emission/phase
  angles at the scene centre**, *"while preserving local topographic features."* JAXA wants the local
  shading kept, because TC's purpose is morphology.
- The **Morning and Evening MAP mosaics** are built from **mono, low-sun** images specifically
  *because* long shadows maximise morphological legibility. **Illumination is treated as a feature,
  not as noise.**
- Morning gives right-to-left low-angle illumination; evening gives left-to-right. Because the
  shading is *reversed*, the pair together disambiguates crater-vs-dome relief that a single
  illumination makes ambiguous.

**Practical consequence:** you cannot reliably match a TC morning mosaic to a TC evening mosaic with
intensity correlation. JAXA's own answer is to go through geometry (DEM↔DEM), or use a
photometrically normalised product, or render.

### 3.3 SLDEM2015 — the canonical cross-instrument registration success story

*Barker et al. (2016), Icarus 273:346–355.* This is the template for the entire discipline and worth
reading closely.

**The setup — two datasets with opposite error characteristics:**

| | LOLA | SELENE TC stereo |
|---|---|---|
| Nature | ~4.5 × 10⁹ laser heights | 43,200 1°×1° stereo DTM tiles |
| Vertical | precision ~10 cm, accuracy ~1 m | few-metre, with tile-scale systematic tilts |
| Horizontal | ~10 m | dense, but geodetically floating |
| Sampling | cross-track gaps ~500 m to several km | continuous areal coverage |

**LOLA is geodetically accurate but spatially sparse; TC is spatially dense but geodetically
floating.** The merge is not about resolution — it is about transferring LOLA's absolute frame onto
TC's dense surface.

**Step 1 — per-tile 5-parameter transformation.**
For each of 43,200 TC tiles, solve a transform fitting the tile to the ~100,000 unbinned LOLA points
inside it.
- **Parameters (5):** three translations Δx, Δy, Δz plus **two planar tilts** about the tile centre.
  No rotation about vertical, no scale.
- **Cost:** minimise RMS vertical residual with **Huber weighting**, `w(r) = min(1.0, 3σ/|r|)`. Huber
  is what makes it robust — gross blunders (bad matches in shadow, boulders, unresolved craters) get
  down-weighted *linearly* instead of dominating a least-squares fit *quadratically*.
- **Optimiser:** bounded downhill simplex (Nelder–Mead) from **multiple random starting points**
  within ±120 m horizontal, ±10 m vertical, ±15 m/deg tilt. Multi-start because the cost surface over
  rough planetary terrain is non-convex — it has local minima where the surface locks onto the wrong
  crater.
- **Why tilts:** a stereo tile's principal systematic error is not a pure shift but a *plane*, induced
  by camera attachment-angle and orbit error accumulating along the strip. Solving Δz alone leaves
  that unmodelled.

**Step 2 — per-LOLA-profile 3-D offsets.**
Having fixed the TC surface, now adjust the *altimetry*. Each ~1° segment of each individual LOLA
profile gets its own 3-D translational offset.
- **Why this is the clever half:** LOLA's residual error is dominated by *per-track* orbit/pointing
  error, near-constant over ~1° but varying between tracks. The TC surface, once tile-registered, is a
  continuous dense reference against which each sparse track can be slid into place.

> **In effect: TC removes LOLA's cross-track geolocation noise, while LOLA supplies TC's absolute
> datum. The information flows both directions — which is why the merged product beats both inputs.**

**Quantified:**

| Stage | Median RMS vertical residual | 90th percentile |
|---|---|---|
| Before co-registration | 4.9 m | ~10 m |
| After Step 1 | 3.4 m | ~5 m |
| After Step 2 | **2.6 m** | **3.4 m** |

**An honest choice worth noting:** TC tiles were **downsampled to 512 ppd (~59 m)** to match TC's
*effective* post-matching resolution, not its nominal 10 m. They did not claim resolution they had
not achieved.

**The transferable lesson:** SLDEM2015 succeeds because it **never tries to match an image to an image
across illumination**. It converts camera data to geometry first, then registers geometry to geometry
with a robust, physically-motivated, low-dimensional transformation.

### 3.4 Why DEM alignment works where image alignment fails

USGS reprocessed Kaguya TC with ASP, and the cross-instrument registration step is `pc_align` to
LOLA in four passes: extract LOLA point cloud → gross alignment at 3× median offset → optional
revised pass → fine pass with **Fast Global Registration (FGR)**.

> **ICP/FGR-family alignment operates on 3-D geometry, which is illumination-invariant by
> construction. Converting each sensor's imagery to a DEM first, and only then registering, is the
> standard way NASA/JAXA/USGS bridge sensors with wildly different lighting and resolution.**

**`pc_align` practical notes:**
- **The denser cloud must be given first, as the reference.** Aligning an ASP DEM to LOLA means
  passing `DEM.tif LOLA.csv`, then using `--save-inv-transformed-reference-points`.
- **`--max-displacement` is the critical, "very sensitive" parameter.** It functions as the outlier
  gate: correspondences beyond it are discarded. Too small → true correspondences rejected. Too large
  → outliers dominate. Recipe: run `geodiff` first, take a multiple of the standard deviation.
- **The transform is applied to the cameras**, not just the DEM — fed back via
  `bundle_adjust --initial-transform`, so all downstream products inherit the anchoring.

**Alignment methods:** `point-to-plane` ICP (default — more robust to large translations because
points can slide along the surface), `point-to-point`, similarity variants that also solve scale,
**`nuth`** (Nuth & Kääb — exploits the fact that elevation bias between misaligned DEMs varies
*cosinusoidally with terrain aspect*, so the shift can be solved analytically; achieves sub-grid
accuracy), `fgr`, plus feature-based and correlation-based initialisers on hillshades for the
large-offset case.

### 3.5 SLIM — crater matching, flight-proven

SLIM ("Moon Sniper") landed January 2024 near Shioli crater, targeting **<100 m** pinpoint accuracy.

**Three-part architecture:**

1. **Onboard crater database**, built on the ground from **Kaguya and LRO** orbital data.
2. **Crater extraction** using **PCA over historical lunar crater image patches** to build template
   vectors ("eigencraters"), correlated against the descent image. *Why PCA:* a crater under
   directional light presents a characteristic bright-limb/dark-limb dipole; the leading principal
   components of a large crater corpus capture depth, rim-to-floor contrast and rim asymmetry, so a
   handful of templates spans most real appearances far more cheaply than a CNN.
3. **Crater matching** comparing extracted crater *positions* — **not image data** — against the
   database.

> **The key design decision: by reducing both the descent image and the map to a point set of crater
> centres, the matcher becomes invariant to absolute brightness, contrast, and largely to
> illumination direction. It becomes a geometric point-pattern problem.**

The published matcher is **Evolutional Triangle Similarity Matching (ETSM)**: form triangles among
crater centres, match on **triangle similarity invariants** (invariant to translation, rotation,
scale). The improved version adds **elimination of line-symmetric triangles** (which otherwise alias),
**comparison of triangle rotation relationships** to kill false positives, and **Delaunay
triangulation** to choose which triangles to form rather than all C(n,3), which explodes
combinatorially.

**Flight results:** image matching over seven areas, **14 of 14 matching operations successful**.
Navigation accuracy better than ~10 m at obstacle-detection altitude, ~3–4 m before the engine
anomaly. Image processing ran **on an FPGA**, required to complete **within 5 seconds** from capture.

**Relevant to India:** the post-landing verification used **Chandrayaan-2 high-resolution imagery**.
A Japanese lander used Indian orbital imagery as reference.

**The frontier version — EIGENCRATER (arXiv 2605.17125)** argues SLIM's image-space PCA templates
couple geometry with lighting, and instead runs PCA over crater **DEM patches** (1,780 craters from
GLD100 + the Robbins catalogue), clusters them, then **renders** cluster-mean templates under a
Lunar-Lambert model at the expected illumination before correlating. Same principle: **put the
invariant in the geometry, synthesise the appearance on demand.**

---

## 4. China / CNSA — the best-documented cross-mission registration work anywhere

### 4.1 Chang'e-2 self-calibration bundle adjustment

**Camera:** single lens, two look angles in the same track, pushbroom TDI CCD. Forward +8°, backward
−17.2°, two line arrays of 6,144 px, f = 144.3 mm, 10.1 µm pixels. **7 m/px** from 100 km; ~1.3–1.5 m
from the lowered ~15 km perilune.

The authors are explicit that intra-track intersection (~25.2°) and inter-track (~17.6°) are both
*relatively poor* stereo geometries. Weak geometry is exactly what forces the self-calibration
approach:

1. **Problem:** attitude telemetry RMSE 0.0124° (worse than the 0.01° requirement); orbit uncertainty
   ~100 m. Raw geolocation gave **~20 px inter-track** and **>5 px intra-track** residuals.
2. **Interior orientation self-calibration:** four extra parameters per line array
   (`x_offset, x_scale, y_offset, y_scale`), 8 additional IO parameters per stereo pair, absorbing
   residual camera-model error.
3. **Exterior orientation by polynomial fitting:** rather than per-scanline EO, model it as
   **third-order polynomials in time** for Xs, Ys, Zs, φ, ω, κ. Long tracks segmented with
   **first-derivative continuity constraints** at the joins.
4. **Tie points:** SIFT + RANSAC, hundreds evenly distributed, split into **intra-track** and
   **inter-track** classes. The inter-track points stitch the global mosaic together.
5. **Regularisation — the interesting part.** The normal equations are ill-posed without ground
   control, so:
   - **pseudo-observations** on selected original EO parameters at regular intervals, preventing rank
     deficiency
   - **variance-based weighting** (tie point σ ≈ 0.5 px ≈ 3.5 m; orbit σ = 100 m; pointing σ = 0.01°)
     with **Huber robust estimation**
   - **truncated SVD** on the normal-equation coefficient matrix, keeping only the *t* largest
     singular values
6. **Result:** residuals from ~20 px and >5 px down to **sub-pixel**; elevation differences against
   LOLA reduced by 9–10 m.

**Why it works:** systematic error in a pushbroom lunar mapper is dominated by slowly varying
orbit/attitude drift plus a static camera-model bias. Modelling the first as a low-order time
polynomial and the second as a small affine IO correction gives a compact parameterisation that a few
hundred tie points can actually determine — with TSVD and pseudo-observations preventing the
near-singular directions (the ones the weak geometry cannot constrain) from blowing up.

### 4.2 The hillshade trick — DEM-to-DEM registration via rendering

*Geometric Quality Assessment of Chang'E-2 Global DEM Product, Remote Sensing 12(3):526 (2020).*

**The key move:** the 3-D DEM-to-DEM matching problem is **converted into a 2-D image matching
problem by rendering simulated hillshade images from both DEMs under a common illumination.** You
then match *images*, not point clouds.

**Chain:** **ASIFT** (Affine-SIFT) tie points between hillshades → **RANSAC** → **Local Outlier
Factor (LOF)** for residual gross errors → **Fourier analysis** of the residual field to extract
periodic, orbit-correlated error.

**Why it works:** a DEM has no radiometry — nothing for a feature detector to grab. Synthesising
shaded relief from both DEMs with **identical** sun geometry removes the illumination difference *by
construction* and turns topography into texture. ASIFT then handles the residual affine difference
that plain SIFT cannot. And Fourier analysis of the residuals is diagnostic: a periodic signature at
the orbital period points at ephemeris error rather than terrain error.

**Result:** CE2TMap2015 vs SLDEM2015 — area-weighted mean **horizontal displacement 183.1 m
(σ 101.2 m)**, vertical 2.3 m (σ 15.4 m).

### 4.3 The frame-disagreement warning

Three independent measurements of CE-2 vs LRO frame disagreement:

| Measurement | Value | Source |
|---|---|---|
| Global DEM horizontal | **183 m** mean | RS 12:526 (2020) |
| CE-4 far-side landing site | **415 m** | Nature Comms 10:4229 (2019) |
| CE-5 / CE-6 landing sites | ~50–80 m | RS 13:590; ISPRS 2025 |

**Anyone fusing products from two missions must handle this explicitly rather than assuming a common
frame.** This is a concrete, citable illustration that "registered" does not mean "in the same frame."

### 4.4 The descent-imagery evolution — four missions, one lesson

**CE-3 (2013):** descent image sequence registered to the CE-2 DOM; local products to **0.05 m**.
DOM matching gave the highest-accuracy rover traverse (111.2 m vs 114.8 m from cross-site visual),
independently checked against LRO NAC where the tracks are directly visible.

**CE-4 (2019):** 164 descent images; **SIFT applied not only between adjacent frames but across
multi-sequential frames**, with higher weights on multi-image tie points to strengthen geometry.
Fundamental-matrix RANSAC. Self-calibration bundle adjustment. **Absolute control from 12 evenly
distributed GCPs picked manually from LRO DOM and SLDEM.** Accuracy: mean 5.02 m 3-D, 0.33 px
reprojection. *Honest note: this paper does not address illumination compensation at all.*

**CE-5 (2020) — the scale cascade:** ~240 descent images from 9 km to touchdown. Two base maps: CE-2
DOM at 7 m/px and an LROC NAC DOM at 1.5 m/px built from 765 images.
> **Match the lowest-resolution (highest-altitude) descent frames to the base map first, then
> progressively register each higher-resolution frame to the *previously rectified* frame, walking the
> scale gap down to touchdown.**
This is how a 7 m ↔ 0.05 m mismatch is bridged **without ever attempting a direct match across three
orders of magnitude in GSD.** Delivered ~30 minutes after landing.

**CE-6 (2024) — two approaches in parallel:**

*(a) Descent image simulation.* Rather than matching a real descent image to a real orthophoto
(different illumination, different viewpoint), **synthesise the expected descent image from the
pre-landing orbital basemap + terrain model**, then match real-to-synthetic. This **fully automates**
the matching. Same logical device as the hillshade trick: **make the reference look like the query
before you match.**

*(b) Learned matching, sub-metre, in 30 minutes* (Bai et al., Comms Earth & Environment 2025;
code at github.com/BroenLin/moon_location):
- 630 CE-6 descent frames against **LROC DOM at 1.12 m/px**, tiled into 516 × 2048² tiles
- **SuperPoint → SuperGlue** "Matching Cascading Network", per-tile confidence to rank candidates
- **Illumination handling: histogram specification of each keyframe using the DOM as reference
  template** — force the query's intensity distribution onto the reference's
- **Scale handling:** descent images resampled to the DOM's GSD using intrinsics and altitude
- **Rotation handling:** small-area DOMs rotated in **45° increments**
- Coarse localisation by homography propagation → **10.21 m**; fine by bundle adjustment
  aerotriangulation → **0.90 m**
- **Whole pipeline <30 minutes on one RTX 4090**

> **The lesson consistent across all four missions: normalise the nuisance variables (illumination,
> scale, rotation) analytically wherever you can, and only then hand the residual problem to the
> matcher.**

### 4.5 LGCN2025 — where crater matching became operational

*Di K. et al., LPSC 2025 abstract #1380.* A new lunar global control network:
- **>450,000 LROC NAC images** (0.5–2.5 m/px), supplemented by Chang'e-2 and laser altimetry
- **Matching: SIFT for low/mid latitudes; the crater-neighbourhood-structure method for polar
  regions**, where grazing illumination defeats intensity matching
- Two-stage adjustment: adaptive adjustment within 188 sub-regions using an affine correction model
  in image space, then global adjustment in object space
- **>1.5 million control points**; horizontal accuracy ~20 m mean, assessed against the five laser
  retroreflectors; vertical 3–4 m

**This is the direct operational justification for crater-structure matching** — not a research
curiosity, but the production choice where SIFT stops working.

### 4.6 CNSFM — the most on-topic paper for Seleno

*Zhang, Xie, Liu, Di et al., "Robust Feature Matching of Multi-Illumination Lunar Orbiter Images
Based on Crater Neighborhood Structure," Remote Sensing 17(13):2302 (2025).*

1. **Detection:** YOLOv9, transfer-learned on a custom crater dataset spanning multiple illuminations
   and geological units
2. **Descriptor (CNSF):** a crater plus its **K nearest neighbouring craters**, described **not** by
   grey-level statistics but by **angles and distance ratios** within that constellation
3. **Matching:** similarity comparison on those geometric invariants, then **outlier removal by local
   geometric transformation consistency**; explicitly tolerates *missed* detections by pruning
   non-corresponding craters from a neighbourhood before scoring
4. **Why it works:** angles and distance ratios are invariant under similarity transforms. A change
   in sun azimuth moves shadows — destroying intensity descriptors — but does **not** move crater
   centres. **Illumination robustness is structural, not learned.**
5. **Data:** **MiLOIs** — 321 LROC image pairs across latitudes and illumination conditions
6. **Beat** SIFT, HAPCG, ML-HLMO and WSSF on RMSE and correct-match rate
7. **Inputs:** two images, a trained crater detector, K, and a consistency threshold. **No intensity
   normalisation and no a-priori pose required.**

---

## 5. Russia and ESA — honest assessment

### 5.1 Roscosmos: essentially no published registration methodology

**Luna-25 (2023).** The imaging system was **STS-L**, eight CMOS cameras with multi-exposure HDR
acquisition, described in *Solar System Research* 55(6):588–604 (2021). It is a **documentation and
operations** camera set, not a mapping camera. The published description covers automated *panorama
formation* — intra-camera mosaicking from a known rotating head — **not** registration to an orbital
basemap.

> **I found no published Russian description of onboard terrain-relative navigation, crater matching,
> or descent-image-to-orbital-map registration for Luna-25.** Its landing was designed around Doppler
> radar altimetry. Anyone claiming Luna-25 flew image-based absolute localisation is going beyond the
> published record.

STS-L returned orbital images in August 2023, notably of Zeeman crater. The one substantive science
analysis (LPSC 2024 #1982) uses the STS-L image *qualitatively* and does its quantitative work on
**LOLA 60 m DTM** — with no discussion of georeferencing the STS-L image at all.

**Luna-26** has exactly one peer-reviewed concept paper (Polyansky et al., *Planetary and Space
Science* 162:216, 2018). **I could not open it** — ADS, ScienceDirect and ResearchGate all blocked the
fetch — so I cite its existence and title only. Secondary sources give a stereo topographic camera
targeting 2–3 m maps, but **treat that figure as unconfirmed by a primary source.**

**Where Russian methodology *is* published: MIIGAiK / MExLab — applied to American data.**

The strongest paper is Karachevtseva et al., *Icarus* 283:104–121 (2017), on the Luna-21 /
Lunokhod-2 site — a genuine cross-epoch, cross-sensor registration problem (1973 Soviet surface TV
panoramas against 2010s orbital imagery):
- 59 LROC NAC images, **342 tie points** at ~10 measurements each, least-squares bundle adjustment on
  an RPC model; residuals RMS ±1.8 m / ±3.9 m / ±4.3 m
- DTM by **semi-global matching** with iterative deformation, 5 stereo pairs at 25.5°–39.3°
  convergence → 2.5 m/px
- **Absolute control: the lunar laser ranging retroreflector position, correcting an ~118 m absolute
  positional error** → final residuals ±2 m horizontal
- **Illumination exploited rather than fought:** rover tracks digitised preferentially from
  **high-incidence (low-sun) images**, where shadow enhances the low-relief wheel tracks

### 5.2 ESA: no cartographic methodology, strong navigation methodology

ESA's own page states the approach plainly: an optical navigation system processes descent imagery
onboard, **identifies craters and matches them against a stored database built from NASA's LRO and
Japan's Kaguya data.** Hazard avoidance uses a **scanning lidar** that works in poor lighting.

> **This is the clearest public statement that ESA's absolute lunar navigation reference frame is
> American and Japanese data.** Reporting following an ESA council meeting states ESA has no
> proprietary lunar topographic data for Argonaut and must obtain mapping data from other nations —
> LRO, Chandrayaan-2, Chang'e. *(secondary-source reporting; treat specific numbers as such)*

**What ESA does have, documented:**
- **LION** (Landing with Inertial and Optical Navigation) — matches observed landmarks against
  onboard reference material. **Inputs: high-resolution orbital imagery (LRO) plus a DEM** — the DEM
  is needed so relief-induced parallax does not corrupt landmark geometry. Validated
  hardware-in-the-loop at ESTEC on a scale lunar model built by DLR. **Better than 50 m at 3 km
  altitude, at scale.**
- **PANGU** — Planet and Asteroid Natural scene Generation Utility. Renders synthetic planetary
  scenes from DEMs, crater/boulder size-frequency distributions, with **dynamic shadows**. *Why it
  matters here:* PANGU is how ESA generates the **illumination-varied image sets needed to test
  whether a matcher survives a sun-angle change** — the same synthetic-reference logic as CE-6's
  descent simulation, applied to *testing* rather than to matching.
- **DLR's CNav**, which ESA draws on: an **"edge-free, scale-, pose- and illumination-invariant"**
  crater detector built on **MSER (Maximally Stable Extremal Regions)**, fused loosely with a
  tactical-grade IMU through a Kalman filter. Four steps: image acquisition → crater detection →
  crater identification via unique constellations → pose estimation. "Theoretically requires only a
  single image" for full position and attitude. **TRL 4.**

**Why MSER-based crater detection is illumination-robust:** MSER finds regions whose *shape* is
stable across a wide range of intensity thresholds. A crater under oblique light presents a
bright/dark lobe pair whose **boundary** is threshold-stable even as absolute brightness changes; the
detector keys on that stability rather than on any particular grey level or gradient magnitude.

---

## 6. The technique catalogue

### 6.1 Illumination-invariant matching without photometric correction

**Mutual Information (MI / NMI).** Maximises `MI(A,B) = H(A) + H(B) − H(A,B)` from a joint 2-D
histogram. Survives illumination change because it assumes only that knowing a pixel's value in A
*reduces uncertainty* about its value in B — any consistent statistical mapping leaves MI at its
maximum.
**Where it breaks, and this matters:** the illumination relation on a shaded surface is **not a
function of intensity alone**. Two pixels equally bright under sun A (one on a north slope, one on a
south slope) map to *different* brightnesses under sun B. That is one-to-many, and it degrades MI the
way it degrades everything else. **Treat MI as "modality-robust," not "sun-azimuth-robust."**
Also: MI needs ≫ n_bins² samples, so it is a global or large-window metric, never a per-keypoint
descriptor. *Difficulty: low — SimpleITK/elastix in ~30 lines.*

**Phase correlation / Fourier–Mellin.** Normalised cross-power spectrum, inverse FFT, delta peak.
Amplitude normalisation discards all magnitude and keeps only phase, so any **linear** brightness
change `I → aI + b` is annihilated. Fourier–Mellin extends to similarity transforms via log-polar
resampling. Sub-pixel to 0.01–0.1 px on clean data.
**Limit:** invariance is to *linear* photometric change only. A shadow appearing in one image is a
structural change and does shift the phase. **Excellent for residual sub-pixel alignment once you're
close; unreliable from scratch across illumination.** *Difficulty: very low —
`skimage.registration.phase_cross_correlation` is one call.* Windowing (Hann/Blackman) is mandatory.

**Census / rank transform.** Replace each pixel by the bit-string of comparisons `I(p') > I(p)` over
a window; Hamming distance as cost. **Invariant to any strictly monotonic intensity mapping** — gain,
gamma, exposure. Default cost in most SGM implementations. **Breaks completely on shadow reversal,
which is not monotonic.** *Difficulty: trivial.*

**CFOG (Channel Features of Orientated Gradients).** Dense per-pixel generalisation of HOG: build an
orientation-channel vector per pixel, convolve in the orientation dimension, match the 3-D feature
volumes with **FFT-accelerated NCC**. One of the strongest classical baselines for multimodal remote
sensing. *Difficulty: medium — the FFT-NCC over channel volumes is the fiddly part.*

**Local Self-Similarity family** — LSS, **DOBSS**, **OSS** ("Illumination-Robust remote sensing image
matching based on oriented self-similarity"), **RI-SS**. Describes a point by the correlation surface
of its own neighbourhood *against itself*. Because the descriptor is computed entirely within one
image, whatever intensity mapping that image underwent cancels out.

**MIND (Modality-Independent Neighbourhood Descriptor).** For pixel x, compute patch SSD between x
and several neighbouring offsets r; estimate local variance V(x); descriptor component is
`MIND(I,x,r) = (1/n)·exp(−D_p(I,x,x+r) / V(x))`. Every quantity is a *within-image* patch distance,
so whatever function maps A's intensities to B's acts on both patches in the difference, and division
by V(x) removes first-order scaling. Dense and differentiable, so it drops into any optimisation-based
registration as a replacement for SSD. **No planetary MIND paper found** — flagged as an opportunity.
*Difficulty: low-medium, ~50 lines with box filters.*

### 6.2 Phase congruency and RIFT2 — the strongest hand-crafted option

**Phase congruency (Kovesi).** Convolve with a bank of **2-D log-Gabor** filters at multiple scales
and orientations, giving even- and odd-symmetric responses. PC measures how *aligned in phase* those
responses are across scales, **normalised by amplitude**. Because it is a ratio of amplitudes, PC is a
**dimensionless quantity in [0,1] invariant to contrast and brightness** — a step edge and a faint
step edge give the same PC value. Kovesi's own framing: "a low-level image invariant."

**RIFT (Radiation-invariant Feature Transform).** Two ideas:

1. **Detection on PC, not intensity.** Moment analysis of the PC map: the *minimum moment* gives
   corners, the *maximum moment* gives edges. FAST on both, combining corners (high repeatability)
   with edge points (high cardinality).
2. **Description — the Maximum Index Map (MIM).** For each of N_o orientations, sum the log-Gabor
   amplitude across all scales. At each pixel, MIM stores the **index (1…N_o) of whichever orientation
   had maximum amplitude.**

> **That index is the trick. An index carries no magnitude, so any monotone — or even non-monotone —
> amplitude scaling that preserves *which* orientation dominates leaves MIM unchanged.**

A SIFT/GLOH-style descriptor is then built on MIM: 6×6 sub-grids × N_o bins.

**RIFT results:** 60 pairs across six datasets, **100% success rate vs SIFT's 31.7%**; mean positional
error ~1.8 px. **One of the six datasets is day–night** — the closest public analogue to a lunar
sun-angle flip.

**RIFT2** (arXiv 2303.00319) replaces RIFT's brute-force rotation handling with a dominant-index-value
technique: **~3× faster, ~3× less memory, similar performance. Source on GitHub.**

**Why this matters for Seleno specifically: RIFT2 was one of the five methods in the Chandrayaan-2
comparative evaluation.** It is already benchmarked on your exact data type, needs no GPU and no
training. *Difficulty: medium-high from scratch; **low** to use — reference code exists in MATLAB and
Python.*

### 6.3 Render-then-match, in detail

**The idea:** if you have a DEM, you can render a synthetic image of it under *any* sun geometry.
Instead of matching image A (sun azimuth 30°) against image B (azimuth 210°) — impossible — render the
DEM at 210° and match **real B against synthetic B**, a same-illumination problem.

**Inputs required:**
- DEM covering the area (LOLA ~60–120 m/px globally; higher from stereo or SfS)
- Reflectance model (Lunar-Lambert usually suffices)
- Albedo estimate (constant is a common first approximation)
- Sun vector at the image epoch — from SPICE
- Camera model — from SPICE

**Free tooling:** ASP 3.7.0 (June 2026) split the old `--save-computed-intensity-only` into
**`--save-sim-intensity-only`** and `--save-meas-intensity-only`. The first is precisely *"render the
synthetic image this DEM + sun + reflectance model predicts, and stop."* **That is a relighting
engine, free.** ISIS `photomet` `Shade` mode does a simpler version.

**Measured performance — Mars rotorcraft (arXiv 2502.09795), the most directly relevant evaluation:**
- **Geo-LoFTR**: LoFTR extended with a parallel cross-attention branch fusing **depth** (from the DEM)
  with visual features
- Trained on 150,705 triplets from 4,500 nadir observations × 17 orthographic maps at different lighting
- **Sun elevation:** 87% accuracy @1 m at matched illumination; at extreme grazing sun (EL 2°),
  **17% @1 m — versus 0% for every other method tested**
- **Sun azimuth (0–180° offset):** Geo-LoFTR holds **54–63% @1 m across the full azimuth range**.
  **SIFT fails completely beyond 90° offset.** Pre-trained LoFTR degrades significantly
- **Scale robustness** (64–200 m altitude): Geo-LoFTR −7%, fine-tuned LoFTR −33%

**Rendering fidelity is a first-order lever.** Lunar-G2R (arXiv 2601.10449) learned a spatially-varying
BRDF via differentiable rendering and beat analytical Hapke by 38% in photometric MSE. **The
registration-relevant number: MASt3R matches within 1 px went from 1.22% (Hapke renders) to 8.38%
(learned-BRDF renders).** Note both are low in absolute terms — real-to-synthetic matching on the Moon
is genuinely hard.

**The honest weakness:** you need a DEM at or near image resolution. For LROC NAC at 0.5 m/px against
LOLA at 60 m/px, the DEM cannot predict most of the visible texture, so the synthetic image is far
smoother than the real one. **This resolution gap is the practical killer**, and it applies with full
force to OHRC at 0.25–0.32 m.

### 6.4 Learned matchers as of September 2026

| Method | Type | Cross-illumination strength | Compute (~640×480) | Notes |
|---|---|---|---|---|
| SuperPoint + SuperGlue | Sparse + GNN | Moderate — detector repeatability is the bottleneck | ~50–100 ms GPU | Superseded. **SuperPoint weights are non-commercial licence** |
| SuperPoint + **LightGlue** | Sparse + adaptive GNN | Same ceiling — limited by the detector | **150 FPS @1024 kpts (RTX 3080)**, 4–10× faster than SuperGlue | LightGlue code+weights **Apache-2.0**; SuperPoint weights are not. Works with DISK (Apache-2.0), ALIKED (BSD-3), SIFT |
| **LoFTR** | Detector-free semi-dense | Better on low-texture; **degrades badly on large azimuth change** | ~100–200 ms | The base everyone forks |
| **ELoFTR** | Semi-dense | Same as LoFTR, faster | **~2.5× faster than LoFTR** | Aggregated attention + two-stage correlation |
| **XoFTR** | Cross-modal LoFTR | **Designed for thermal↔visible.** MIM pretraining, pseudo-thermal augmentation | ~LoFTR class | Closest published analogue to "very different appearance, same scene" |
| **RoMa** | Dense, frozen DINOv2 | Among the most robust to appearance change | ~300 ms | — |
| **RoMa v2** | Dense, frozen **DINOv3** | Claims robustness to "**extreme photometric changes, including different modalities**" | Reduced memory via custom CUDA kernel | DINOv3 gives 19.0 EPE vs DINOv2's 27.1 |
| **MatchAnything** | Retrained ELoFTR + RoMa | **Purpose-built for unseen cross-modality**, single weights | **40 ms (ELoFTR), 303 ms (RoMa)** | See below |
| **MASt3R / DUSt3R** | 3-D pointmap regression | Very robust to viewpoint; illumination robustness is 3-D-mediated | Heavy | Most used in recent *lunar* work |

**MatchAnything** (TPAMI 2026) retrains ELoFTR and RoMa on ~800M synthetically generated cross-modal
pairs — CycleGAN style translation to thermal/nighttime plus DepthAnything monocular depth converted
to grayscale. They *manufacture* modality gaps rather than collecting them. Single-checkpoint results
on unseen tasks: **+255.0% (RoMa, visible–thermal remote sensing)**, +78.5% (visible–SAR).
**No planetary or lunar evaluation** — explicitly checked.

**Do pretrained weights transfer to planetary imagery? Partially, and never as well as fine-tuning.**
Every published planetary result either fine-tunes or reports degradation. The reasons are structural,
not incidental: planetary orbital imagery is grayscale, has near-zero semantic content (no buildings,
vegetation, or man-made verticals to anchor on), enormous dynamic-range variation, and textures with
no terrestrial analogue. **Foundation-model features (DINOv2/v3) transfer better than task-specific
ones** — a good argument for RoMa v2.

Evidence: MarsDINO (domain-pretrained) beat generic models with **79× more parameters** on Mars crater
retrieval. Domain pretraining dominates scale.

> **Note the counter-intuitive conclusion: SuperPoint+LightGlue is the *wrong* tool for the extreme
> illumination case despite being the best speed/accuracy tool for the easy case — sparse detectors
> simply don't fire at the same places when shadows move.**

### 6.5 Geometry priors from SPICE — six distinct uses

1. **Per-pixel illumination angles.** i, e, g for photometric normalisation come from SPICE + shape
   model. Without SPICE there is no photometric correction at all.
2. **Map-projection to a common frame.** The single biggest win — see §6.6.
3. **A quantified search radius.** After map-projection the residual is dominated by ephemeris and
   pointing error, and you can *bound* it. Post-jigsaw NAC offsets are <12 m latitude, <5 m longitude;
   at 0.5 m/px that's a ~20 px search radius — small enough that phase correlation suffices.
4. **Pre-warping so non-invariant detectors work.** ISIS **`FASTGEOM`** transforms the trainer image
   into the query's camera space using a-priori SPICE, explicitly so that scale- and
   rotation-*non*-invariant algorithms become usable.
   > **This generalises into a principle: if you can remove scale and rotation with geometry, you can
   > spend your descriptor's entire invariance budget on radiometry.**
5. **Epipolar constraint.** With both camera models known, a match must lie on the epipolar curve.
   For pushbroom sensors these are **curves, not lines** (each scanline has its own exterior
   orientation) — which is why coupled epipolar rectification for LROC NAC is its own research topic.
   Reducing the search from 2-D to 1-D roughly squares your outlier rejection power.
6. **Sun-vector-aware descriptors.** Because SPICE gives the solar azimuth of *both* images, you know
   a priori the direction shadows moved. Practical exploitation: rotate descriptor reference frames to
   the shadow azimuth, mask or down-weight gradients parallel to the illumination-change direction,
   or restrict phase-congruency orientation bins.
   > ⚠ **This is inference, not established practice — no planetary paper was found doing it
   > explicitly. Flagged as an opportunity rather than a technique.** *(This one is potentially a
   > genuine research contribution.)*

### 6.6 Tiling very large pushbroom strips

OHRC strips are ~12,000 × 101,075 px — over a billion pixels. TMC-2 is ~4,000 × 190,000 per detector.
Neither fits in memory; neither can be fed to a matcher that wants 640–1600 px inputs.

**The strategy:**

1. **Match in map space, not raw space.** Use SPICE plus a DEM to project both images into the *same*
   projection at the *same* GSD. This absorbs pushbroom geometry, terrain parallax (to DEM accuracy),
   and most rotation and scale, leaving a residual close to pure local translation.
   > **This single step is worth more than any clever matcher.** It converts a general 2-D warp search
   > into a small-radius translation search per tile.
2. **Pyramid, coarse-to-fine.** Solve a global similarity/affine at the coarsest level where the whole
   strip fits in memory. Halve the search radius at each descent.
3. **Tile with padding at the fine level.** 512–2048 px tiles with 10–25% overlap. Overlap is
   essential — matches near tile edges are unreliable and you need redundancy to blend.
4. **Seed each tile from the coarse solution**, so local search is ±few pixels.
5. **Aggregate globally.** Per-tile offsets are noisy and some tiles fail (uniform mare, full shadow).
   Fit a global model — polynomial, thin-plate spline, or a per-scanline correction for pushbroom —
   with a robust estimator, and **reject failed tiles rather than interpolating through them.** For
   pushbroom the residual is usually structured **along-track**, so a low-order along-track spline plus
   a constant cross-track term captures most of it.
6. **I/O discipline.** Never read a whole strip. Windowed block reads aligned to the file's internal
   block structure; process in along-track chunks so you stream the strip once.

**Caveat with learned matchers:** (a) a low-texture tile returns *confident garbage* — use the
matcher's certainty output to gate; (b) independently-matched adjacent tiles produce discontinuous
warps — always fit a smooth global model rather than mosaicking per-tile warps.

**Existence proof at scale:** iMars / ACRO / CASP-GO (UCL) co-registered **~400,000 images** from
Viking Orbiter, MOC, CTX and HiRISE against an HRSC map base using exactly this tiled/hierarchical/
baseline-referenced approach.

### 6.7 Crater matching — inputs, limits, difficulty

**Why craters are the right primitive:**
1. **Geometric permanence.** A crater rim is a physical 3-D curve. Its projection changes with
   viewpoint; its position on the Moon does not change with sun angle. **Shadows move; rims do not.**
2. **Projective structure.** A circular/elliptical rim projects to a **conic** under perspective
   projection. Conics are algebraically tractable — you can reason about invariants rather than pixels.
3. **Ubiquity and density-scaling.** Craters exist at every scale, so the same detector family works
   from 100 km altitude to 100 m altitude; you just change the size bin.

**The rigorous formulation — Christian, Derksen & Watkins (2021), *J. Astronaut. Sci.* 68:1056.**
Solves the **lost-in-space** crater identification problem — identify craters against a catalogue with
**no prior position or attitude estimate.** The theoretical contribution:
- no projective invariants exist for arbitrarily placed conics in 3-D
- **projective invariants *do* exist for conics constrained to a non-degenerate quadric surface** —
  exactly the case for craters on an ellipsoidal Moon
- **a crater triad admits seven algebraically independent invariants** computable directly from the
  observed image conics

Indexed over the **Robbins catalogue (~1.3 M craters)** with **HEALPix** multi-scale hierarchy,
O(log n) lookup. Pose from d ≥ 2 craters using the **full projected conic contours**, not merely
centre points — strictly more information, and it avoids the centre-offset bias oblique viewing induces.

**Detection state of the art:** U-Net++ with ResNet-50 encoder, **dual-channel output (rim mask +
centre mask)**, BCE+Dice loss, trained on the NASA Crater Detection Challenge dataset (4,150
ellipse-annotated images) plus synthetic patches from the LROC WAC mosaic. **Illumination robustness
comes from augmentation** — random gamma/brightness-contrast, CLAHE, flips, rotations, noise. Also:
YOLOv9 (CNSFM), Ellipse R-CNN (regresses ellipse parameters directly rather than segmenting).

**Limits — be clear-eyed:**
- **Terrain dependence.** Smooth mare, lava flows, heavily resurfaced terrain have too few craters.
  **This is a hard failure, not a degradation.**
- **Scale band.** Detectors work over a limited diameter range relative to GSD.
- **Degraded craters.** Ancient, subdued craters are ambiguous in both detection and centre
  estimation — and *their* appearance genuinely does depend on illumination.
- **Sparse constraint.** Craters give tens of correspondences, not thousands — enough for a global
  transform or a pose, **not for a dense warp field or fine local co-registration.**

*Difficulty: detector — medium (fine-tune U-Net++/YOLO; data and architectures exist). Constellation
matcher — medium-high (invariant design + indexing + outlier rejection is where the subtlety lives).*

---

## 7. Side-by-side comparison

| | NASA / USGS | JAXA | China / CNSA | Russia | ESA |
|---|---|---|---|---|---|
| **Core paradigm** | Control network + bundle adjustment to a geodetic frame | Same, plus DEM↔DEM alignment | Self-calibrating block adjustment + cross-mission co-registration | (mapping done on US data) | Consumer of others' maps |
| **Absolute datum** | **LOLA** | LALT, then re-tied to LOLA | SLDEM2015 + laser retroreflectors | LLR retroreflectors | LRO + Kaguya |
| **Primary matcher** | SIFT/AKAZE via ISIS `findfeatures`; area-based `pointreg` | Area-based sub-pixel correlation *(exact correlator unpublished)* | SIFT → **crater structure at poles**; SuperPoint/SuperGlue for descent | SGM, PHOTOMOD | MSER crater detection |
| **Illumination strategy** | **Sub-solar-longitude chaining**; Hapke above i=60° | Avoid it — register geometry, or use simultaneous stereo | **Render the reference** (hillshade / synthetic descent view); histogram specification | Exploit low sun for morphology | Crater geometry + lidar |
| **Learned methods in production?** | **No** — 2026 production pipeline is classical | Research only | **Yes** — CE-6 flew SuperPoint/SuperGlue | No | No |
| **Published detail** | Very high | High | **Very high** | Low | Low (cartography) / High (navigation) |

**Accuracy achieved, for calibration of expectations:**

| Product | Accuracy |
|---|---|
| LROC NAC controlled mosaic | <12 m lat, <5 m lon; seams <15 m (vs ~100 m uncontrolled) |
| SLDEM2015 | 2.6 m median RMS vertical; 3–4 m overall |
| Kaguya TC (USGS/ASP reprocessing) | ~30 m horizontal, ~3 m vertical |
| CE2TMap2015 vs SLDEM2015 | 183 m horizontal, 2.3 m vertical |
| LGCN2025 control network | ~20 m horizontal mean, 3–4 m vertical |
| SLIM landing | <10 m navigation, ~3–4 m estimated |
| CE-6 landing (learned pipeline) | 0.90 m, <30 min |

---

## 8. What this means for Seleno

### 8.1 What Seleno already gets right — and can now cite

| Seleno design choice | Independent validation from this research |
|---|---|
| **Coverage as a first-class metric** | ISIS `findfeatures` has **grid** and **radial** point-distribution algorithms; ISIS `autoseed` lays a **uniform grid** of candidate points. This is established practice — **cite it rather than claim novelty** |
| **"Self-consistency ≠ accuracy"** | This is precisely the **datum-defect** concept in control-network theory: *"a network of only Free points can be internally seamless while being globally translated off the Moon."* Decades of institutional knowledge, same insight |
| **Refusing to register bad pairs** | `jigsaw` has `OUTLIER_REJECTION`, MAD-based rejection, Huber/Welsch/Chen M-estimators; SLDEM2015 uses Huber weighting explicitly to stop blunders dominating; ASP's `--max-displacement` is an explicit outlier gate |
| **Stripping ground truth from the UI** | Matches the 2026 NAC production paper's care in constructing an *independent* validation set rather than self-marking |
| **Cross-resolution scale plausibility check** | CE-6 does GSD equalisation using camera intrinsics before matching; CE-5 does an explicit scale cascade. Context-aware scale checking is right |

### 8.2 The gaps, ranked by how much they cost you

**1. The matcher fails on exactly the challenge the PS names first.**
The Chandrayaan-2 comparative evaluation (which `HANDOFF.md` already cites) found SIFT and AKAZE
degrade under polar lighting while SuperGlue registered all pairs. The Mars rotorcraft study measured
**SIFT failing completely beyond 90° azimuth offset.** Your engine is the one the literature says
loses on sun-angle.

**2. Five of six pairs are self-matched — so your evidence doesn't contain the problem.**
An image matched to a warped copy of itself has *identical lighting*. The headline sub-pixel numbers
were produced on a problem with the sun-angle challenge removed.

**3. A single global homography is the wrong model class for pushbroom data.**
Not "slightly inaccurate" — structurally wrong. Each OHRC line has its own exterior orientation.
Over a small tile it's a fine approximation; over 101,075 lines it is not. **You do not have to fix
this in 23 days — but you should say it out loud in the report.** Naming a known limitation precisely
is a credibility gain, not a loss.

**4. Hardcoded verdict thresholds.**
≥12 inliers, 15% ratio, 15% coverage, 4× RANSAC threshold. Compare `jigsaw`'s σ₀ convergence
criterion, or SLDEM2015's `3σ` Huber cut derived from the data. **Everyone else derives their
thresholds; yours are chosen.**

**5. CLAHE hurting the low-illumination pair is explainable — and the explanation helps you.**
Your own result (0.137 px without it vs 0.288 px with) has a physical reason: CLAHE is a *local
contrast* operator with no model of the surface. The principled replacement is **Lunar-Lambert-McEwen
photometric normalisation using the incidence/emission/phase angles in your PDS4 label** — and ISIS
ships that model with **no free parameters to tune**. That upgrades "CLAHE hurt us, we don't know
why" into "we replaced an ad-hoc operator with the physically correct one, and here's the delta."

### 8.3 The single biggest unused asset

> **Your PDS4 XML label already contains solar incidence and azimuth, spacecraft state vectors, and
> imaging geometry — with SPICE kernels alongside (SPK trajectory, CK attitude, IK instrument
> geometry, PCK/FK frames).**

Almost nobody in a hackathon opens that file. Every agency pipeline in this document starts there.
Five things become available the moment you parse it:

1. **Photometric normalisation** (needs i, e, g) — replaces CLAHE with physics
2. **A bounded search radius** — turns a global search into a local one
3. **Pre-warping (FASTGEOM-style)** — spend your descriptor's invariance budget on radiometry, not
   on scale and rotation
4. **Sun-azimuth-aware matching** — you know which direction the shadows moved
5. **Illumination ordering** — you can sort any set of images by sub-solar longitude

### 8.4 Candidate directions, ranked for 23 days

**Tier 1 — do these, they are cheap and they close real gaps**

| Action | Effort | Why |
|---|---|---|
| **Parse the PDS4 XML label; report sun geometry per pair in the UI** | ~1 day | Unlocks everything below. Also just looks serious — it shows you read the data spec |
| **Add RIFT2 as a second matcher behind the existing `run_matcher` seam** | 2–3 days | No GPU, no training, reference code exists, **already benchmarked on Chandrayaan-2 data**, 100% success vs SIFT's 31.7% on the RIFT paper's six datasets including day–night. Highest ratio of credibility to effort in this document |
| **Replace CLAHE with Lunar-Lambert-McEwen normalisation** | 2–3 days | Physically correct, **zero free parameters**, uses metadata you already have, and directly addresses your own published negative result |
| **Get one genuinely real cross-instrument pair** from PRADAN | 1–2 days | The credibility fix. One real OHRC↔TMC-2 pair with real lighting difference is worth more than all six synthetic ones |
| **Steal the 2026 NAC validation methodology** | 1 day | Similarly-illuminated image series (sub-solar longitude diff <60°, incidence diff <25°, <80% shadowed), median location as ground truth. **A published answer to your "we have no real ground truth" problem** |

**Tier 2 — one real swing, pick at most one**

| Action | Effort | Risk |
|---|---|---|
| **Crater-constellation matching channel** (CNSFM-style: detector → K-NN neighbourhood → angle/distance-ratio invariants → consistency filtering) | ~1 week+ | Medium-high. **This is the strongest differentiator available.** It is what China uses in production at the poles and what JAXA flew on SLIM. But it fails hard on smooth mare, and it gives you tens of correspondences, not thousands |
| **Add a learned matcher** (LightGlue for the easy case, RoMa v2 or MatchAnything-RoMa for the hard case) | 3–5 days | Low-medium. Closes the demo gap. **Watch the SuperPoint licence** — its weights are non-commercial; LightGlue's own code and weights are Apache-2.0, and it pairs with DISK (Apache-2.0) or ALIKED (BSD-3) |
| **Sun-azimuth-conditioned descriptor weighting** | ~1 week | High risk, **but this is the one genuinely novel idea here.** No planetary paper was found doing it. You know the shadow-displacement direction from metadata; down-weight or mask gradients parallel to it. Could be a real contribution — or could not work at all |

**Tier 3 — do not attempt before 30 September**

- Render-then-match / shape-from-shading — needs a DEM at or near OHRC resolution, which does not
  exist for arbitrary sites, and the LOLA-to-OHRC resolution gap is the documented practical killer
- Bundle adjustment / control network — the right architecture, wrong timescale
- Per-line pushbroom pose modelling — same

### 8.5 The strategic point

Your evaluation framework is **matcher-agnostic**. It is a harness: matcher in one end, honest metrics
out the other. That is the thing worth defending.

So the argument to make is not "SIFT is fine." It is:

> *"We built an honest measurement harness. We ran a classical matcher, a radiation-invariant
> hand-crafted matcher, and a learned matcher through it. All three cluster their correspondences, and
> all three fool reprojection RMSE in the same direction. Here is coverage and ground-truth corner
> error showing it. The problem is not the matcher — it is how the field reports accuracy."*

That version survives contact with a judge who knows the literature. The current version — one
classical matcher plus a caveat — is at risk of reading as a rationalisation for a weak baseline.

And you now have three independent corroborations to cite:
- **The 33× swing** in the SAR-optical study, where protocol choices mattered more than matcher choice
- **The 183 m / 415 m CE-2-vs-LRO frame disagreement**, showing "registered" ≠ "in the same frame"
- **The datum-defect concept** in control-network theory — your argument, in geodetic language, 40
  years old

---

## 9. Explicit uncertainty flags

Things reported at second hand, or that could not be verified:

1. **Speyerer et al. (2024), "Where Is That Crater?"** — IOP robots.txt blocked. Cited for existence
   and topic only; **no numbers taken from it.**
2. **Laura et al. (2020) CSM paper** — Wiley 403. CSM description comes from the 2024 status paper and
   ISIS/ASP docs.
3. **Barker et al. (2016) SLDEM2015** — ADS blocked; figures from USGS Astropedia and WashU ODE, which
   are reliable secondary sources. **Confirm against the Icarus paper before quoting in a submission.**
4. **JAXA has never publicly named the correlator inside its own TC DTM production system.**
   "Sub-pixel area-based matching" is all the primary literature commits to. The SSD/NCC
   characterisation is well-supported inference from independent analysis, not a JAXA statement.
5. **Luna-26's 2–3 m resolution figure** comes from Russian-language encyclopaedic sources, not a
   primary reference. **Do not put it in a proposal** without chasing Polyansky et al. (2018).
6. **Two Chinese papers central to the cross-mission story were paywalled** — Yang et al. (2023) GRSL
   crater matching between LROC NAC and CE-2 DOM, and Kang et al. on CE-1 IIM ↔ LROC-WAC. Author,
   venue and DOI given; internals deliberately not described.
7. **"Illumination Invariant Image Matching for Lunar TRN"** (AIAA SciTech 2025, DOI
   10.2514/6.2025-2073) — paywalled, 403. **By title this is the single most on-topic reference in the
   entire document. Worth obtaining through a university library.**
8. **MatchAnything and RoMa v2 have no published planetary evaluation.** Their ranking is reasoned by
   analogy from thermal/SAR results, not measured on lunar data.
9. **Sun-vector-aware descriptor conditioning (§6.5 item 6)** is inference, not established practice.
10. **OHRC GSD** is quoted as 0.32 m by ISRO's payload page and 0.25–0.30 m by tooling docs and recent
    papers — it varies with altitude. **Don't quote a single number without saying which.**
11. **ISIS release listing:** 10.0.0 (June 2024) was the latest tagged release found; the current
    version may have advanced. ASP's cadence is clearer (3.7.0, June 2026).
12. **No evidence was found that learned matchers are used in any operational NASA/USGS lunar
    pipeline.** The 2026 production paper is classical throughout.

---

## 10. Sources

### Problem statement
- [SIH 2026 Problem Statement SIH26166](https://sih2026.vuce.in/en/ps/SIH26166)
- [SIH 2026 Problem Statements — full catalogue](https://github.com/NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements)

### Chandrayaan-2
- [Chandrayaan-2 payload specifications — ISRO](https://www.isro.gov.in/chandrayaan2-payloads.html)
- [Chandrayaan-2 PRADAN data archive (ISSDC)](https://pradan.issdc.gov.in/ch2/)
- [Ames Stereo Pipeline — Chandrayaan-2 lunar orbiter](https://stereopipeline.readthedocs.io/en/latest/examples/chandrayaan2.html)
- [Imaging Infrared Spectrometer onboard Chandrayaan-2 — Current Science](https://www.currentscience.ac.in/Volumes/118/03/0368.pdf)
- [Makharia et al., Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data (arXiv:2509.04775)](https://arxiv.org/abs/2509.04775)
- [Sub-Metre Lunar DEM Generation from Chandrayaan-2 OHRC Multi-View Imagery (arXiv 2604.01032)](https://arxiv.org/html/2604.01032)
- [DEM Refinement and Validation Using Shape-from-Shading with Chandrayaan-2 OHRC (arXiv 2604.17436)](https://arxiv.org/html/2604.17436)

### NASA / USGS
- [Working with LROC NAC Images — Processing Guide (PDF)](https://lroc.im-ldi.com/data/support/downloads/LROC_NAC_Processing_Guide.pdf)
- [ISIS Control Networks — concepts](https://astrogeology.usgs.gov/docs/concepts/control-networks/isis-control-networks/)
- [ISIS `jigsaw` documentation](https://isis.astrogeology.usgs.gov/9.0.0/Application/presentation/Tabbed/jigsaw/jigsaw.html)
- [ISIS `findfeatures` documentation](https://isis.astrogeology.usgs.gov/9.0.0/Application/presentation/Tabbed/findfeatures/findfeatures.html)
- [ISIS `photomet` documentation](https://isis.astrogeology.usgs.gov/8.1.0/Application/presentation/PrinterFriendly/photomet/photomet.html)
- [ISIS `photemplate` / `phoempglobal` (model parameter lists)](https://isis.astrogeology.usgs.gov/9.0.0/Application/presentation/Tabbed/photemplate/photemplate.html)
- [Beyer, Alexandrov & McMichael (2018), Ames Stereo Pipeline, doi:10.1029/2018EA000409](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018ea000409)
- [ASP `pc_align`](https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html) · [`jitter_solve`](https://stereopipeline.readthedocs.io/en/latest/tools/jitter_solve.html) · [`sfs`](https://stereopipeline.readthedocs.io/en/stable/tools/sfs.html) · [SfS workflow](https://stereopipeline.readthedocs.io/en/stable/sfs_usage.html)
- [Smith et al. (2010), LOLA Investigation, Space Sci. Rev. 150:209 (PDF)](https://science.nasa.gov/wp-content/uploads/2024/01/lola-smith-lola-ssr09.pdf)
- [Collins et al. (2026), Photogrammetric Processing of Regional ShadowCam and LROC NAC Controlled Mosaics, Remote Sensing 18(3):525](https://www.mdpi.com/2072-4292/18/3/525)
- [Wagner et al. (2022), LROC NAC South Pole Controlled Mosaic, LPSC #2573 (PDF)](https://www.hou.usra.edu/meetings/lpsc2022/pdf/2573.pdf)
- [Mattson et al. (2011), Continuing Analysis of Spacecraft Jitter in LROC-NAC, LPSC #2756 (PDF)](https://www.lpi.usra.edu/meetings/lpsc2011/pdf/2756.pdf)
- [Sato et al. (2014), Resolved Hapke Parameter Maps of the Moon, JGR Planets 119:1775](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JE004580)
- [LROC WAC Hapke Photometric Parameter Maps (SDWHAP)](https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/LRO/LROC/SDR/SDWHAP.htm)
- [Laura et al. (2024), Current Status of the CSM Standard, Remote Sensing 16(4):648](https://www.mdpi.com/2072-4292/16/4/648)
- [Bertone et al. (2025), Enhanced Topography Models for Lunar South Pole Regions with SfS (PGDA 104)](https://pgda.gsfc.nasa.gov/products/104)

### JAXA
- [Haruyama et al. (2008), Global lunar-surface mapping with LISM on SELENE, Earth Planets Space 60 (PDF)](https://link.springer.com/content/pdf/10.1186/BF03352788.pdf)
- [JAXA Kaguya — LISM (TC/MI/SP) instrument page](https://www.kaguya.jaxa.jp/en/equipment/tc_e.htm)
- [Barker et al. (2016), SLDEM2015, Icarus 273:346 — full text PDF](https://dspace.mit.edu/bitstream/handle/1721.1/118411/1-s2.0-S0019103515003450-main.pdf?sequence=1&isAllowed=y)
- [PGDA — SLDEM2015 product page](https://pgda.gsfc.nasa.gov/products/54)
- [Ames Stereo Pipeline — Kaguya Terrain Camera example](https://stereopipeline.readthedocs.io/en/stable/examples/kaguya.html)
- [USGS STAC — Kaguya TC DTMs (pc_align / FGR processing description)](https://stac.astrogeology.usgs.gov/docs/data/moon/kaguyatc_dtms/)
- [Goossens et al. (2020), Improving Kaguya extended-mission geometry via laser altimetry, Icarus 336](https://www.sciencedirect.com/science/article/abs/pii/S0019103519304440)
- [ISAS — SLIM's pinpoint lunar landing technology](https://www.isas.jaxa.jp/feature/forefront/220928.html)
- [JAXA/ISAS — SLIM results press conference, 25 Jan 2024 (PDF)](https://www.isas.jaxa.jp/en/outreach/announcements/files/SLIM-pressconf-20240125.pdf)
- [Kamata et al., Improvement of SLIM Location Estimation by Crater Matching Based on Similar Triangles](https://www.jstage.jst.go.jp/article/astj/17/0/17_JSASS-D-17-00011/_article/-char/en)
- [EIGENCRATER: PCA for Lunar Crater Detection (arXiv:2605.17125)](https://arxiv.org/html/2605.17125)
- [DARTS PDS3 bulk archive root](https://data.darts.isas.jaxa.jp/pub/pds3/)

### China / CNSA
- [Di K. et al. (2014), Self-calibration bundle adjustment for Chang'E-2 stereo imagery, IEEE TGRS 52(9) (PDF)](http://pmrslab.cn/publications/publications/A%20Self-calibration%20bundle%20adjustment%20method%20for%20photogrammetric%20processing%20of%20Chang'E-2%20stereo%20lunar%20imagery.pdf)
- [Geometric Quality Assessment of Chang'E-2 Global DEM Product, Remote Sensing 12(3):526](https://www.mdpi.com/2072-4292/12/3/526)
- [Zhang Y. et al. (2025), Robust Feature Matching of Multi-Illumination Lunar Orbiter Images Based on Crater Neighborhood Structure, Remote Sensing 17(13):2302](https://www.mdpi.com/2072-4292/17/13/2302)
- [Di K. et al., LGCN2025: A New Lunar Global Control Network, LPSC 2025 #1380 (PDF)](https://www.hou.usra.edu/meetings/lpsc2025/pdf/1380.pdf)
- [Liu J. et al. (2019), Descent trajectory reconstruction and landing site positioning of Chang'E-4, Nature Communications 10:4229](https://www.nature.com/articles/s41467-019-12278-3)
- [Wang Y. et al. (2021), Localization of the Chang'e-5 Lander, Remote Sensing 13(4):590](https://www.mdpi.com/2072-4292/13/4/590)
- [Di K. et al. (2025), Landing Site Mapping and Lander Localization for Chang'e-5 and Chang'e-6, ISPRS Archives XLVIII-G-2025:383 (PDF)](https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/383/2025/isprs-archives-XLVIII-G-2025-383-2025.pdf)
- [Bai X. et al. (2025), Intelligent vision-guided trajectory reconstruction for Chang'E-6, Comms Earth & Environment](https://www.nature.com/articles/s43247-025-03074-7) · [code](https://github.com/BroenLin/moon_location)
- [Wan W. et al. (2019), Descent trajectory recovery of Chang'e-4, ISPRS Archives XLII-2/W13 (PDF)](https://isprs-archives.copernicus.org/articles/XLII-2-W13/1457/2019/isprs-archives-XLII-2-W13-1457-2019.pdf)

### Russia
- [Avanesov, Polyanskii et al., Luna-25 Service Television System (STS-L), Solar System Research 55(6)](https://link.springer.com/article/10.1134/S0038094621060010)
- [Karachevtseva et al. (2017), Cartography of the Luna-21 landing site and Lunokhod-2 traverse, Icarus 283 (PDF)](http://www.geokhi.ru/Lists/List1/Attachments/7549/2017_Karachevtseva_ea_Cartography_Luna_21_Icarus.pdf)
- [Karachevtseva et al. (2015), Landing site characterisation for Luna-Glob / Luna-Resurs, Solar System Research 49(2)](https://link.springer.com/article/10.1134/S0038094615020021)
- [Polyansky et al. (2018), Stereo topographic mapping concept for Luna-Resurs-1, PSS 162:216 (ADS)](https://ui.adsabs.harvard.edu/abs/2018P&SS..162..216P/abstract) *(not read)*

### ESA / DLR
- [ESA — Next-generation landing technology (crater matching against LRO + Kaguya)](https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/Lunar_Lander/Next-generation_landing_technology)
- [ESA — Pinpoint vision-based landings on Moon, Mars and asteroids (LION)](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Pinpoint_vision-based_landings_on_Moon_Mars_and_asteroids)
- [Dubois-Matra, Parkes, Dunstan — Testing and validation of planetary VBN with PANGU, ISSFD 2009 (PDF)](https://issfd.org/ISSFD_2009/Posters/DuboisMatra.pdf)
- [Trigo, Maass, Krüger, Theil (2018), Hybrid optical navigation by crater detection, CEAS Space Journal 10](https://link.springer.com/article/10.1007/s12567-017-0188-y)
- [DLR — Optical Navigation (CNav pipeline, TRON lab, ATON project)](https://www.dlr.de/en/irs/about-us/departments/navigation-and-control-systems/optical-navigation)

### Techniques
- [RIFT: Multi-modal Image Matching Based on Radiation-invariant Feature Transform (arXiv 1804.09493)](https://arxiv.org/pdf/1804.09493)
- [RIFT2: Speeding-up RIFT with a New Rotation-Invariance Technique (arXiv 2303.00319)](https://arxiv.org/abs/2303.00319)
- [Advances and Challenges in Multimodal Remote Sensing Image Registration — survey (arXiv 2302.00912)](https://arxiv.org/pdf/2302.00912)
- [Ye et al., CFOG — Fast and Robust Matching for Multimodal Remote Sensing Registration (arXiv 1808.06194)](https://arxiv.org/abs/1808.06194)
- [Kovesi, Phase congruency: A low-level image invariant](https://www.researchgate.net/publication/12136379_Phase_congruency_A_low-level_image_invariant)
- [Heinrich et al., MIND: Modality Independent Neighbourhood Descriptor (Medical Image Analysis 2012)](https://www.sciencedirect.com/science/article/abs/pii/S1361841512000643)
- [Illumination-Robust remote sensing image matching based on oriented self-similarity (ISPRS J. 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0924271619301145)
- [MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training (arXiv 2501.07556)](https://arxiv.org/html/2501.07556v1) · [code](https://github.com/zju3dv/MatchAnything)
- [RoMa v2: Harder Better Faster Denser Feature Matching (arXiv 2511.15706)](https://arxiv.org/html/2511.15706v1)
- [LightGlue: Local Feature Matching at Light Speed — code, speeds, licensing](https://github.com/cvg/LightGlue)
- [XoFTR: Cross-modal Feature Matching Transformer (arXiv 2404.09692)](https://arxiv.org/html/2404.09692v1)
- [Geometry-aided Vision-based Localization of Future Mars Helicopters in Challenging Illumination (arXiv 2502.09795)](https://arxiv.org/html/2502.09795v3)
- [Lunar-G2R: Geometry-to-Reflectance Learning for High-Fidelity Lunar BRDF Estimation (arXiv 2601.10449)](https://arxiv.org/html/2601.10449)
- [MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data (arXiv 2604.00682)](https://arxiv.org/html/2604.00682)
- [Are Pretrained Image Matchers Good Enough for SAR–Optical Satellite Registration? (arXiv 2604.10217)](https://arxiv.org/html/2604.10217v2)
- [Christian, Derksen & Watkins (2021), Lunar Crater Identification in Digital Images, J. Astronaut. Sci. 68 (PDF)](https://link.springer.com/content/pdf/10.1007/s40295-021-00287-8.pdf)
- [Deep Learning-Based Lunar Crater Terrain Relative Navigation (arXiv 2606.14776)](https://arxiv.org/html/2606.14776v1)
- [CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale (arXiv 2604.06245)](https://arxiv.org/html/2604.06245v1)
- [Downes, Steiner, How — LunaNet: Lunar TRN Using a CNN for Visual Crater Detection (arXiv 2007.07702)](https://arxiv.org/html/2007.07702)
- [Towards Seamless Lunar Mosaics: Deep Radiometric Normalization (arXiv 2604.25208)](https://arxiv.org/html/2604.25208)
- [EU-FP7 iMars: auto-coregistration of Mars multi-resolution images (ISPRS Archives 2016)](https://isprs-archives.copernicus.org/articles/XLI-B4/453/2016/)
- [Di et al., Block adjustment and coupled epipolar rectification of LROC NAC images (PSS 2018)](https://www.sciencedirect.com/science/article/abs/pii/S0032063317304014)

### Not accessible — flagged
- [Illumination Invariant Image Matching for Lunar TRN (AIAA SciTech 2025) — paywalled](https://arc.aiaa.org/doi/10.2514/6.2025-2073) ← **highest-priority acquisition**
- [The moon's many faces: A single unified transformer for multimodal lunar reconstruction (ISPRS J. 2026) — robots-blocked](https://www.sciencedirect.com/science/article/pii/S0924271626001802)
- [Speyerer et al. (2024), Where Is That Crater? PSJ — robots-blocked](https://iopscience.iop.org/article/10.3847/PSJ/ad54c6)
