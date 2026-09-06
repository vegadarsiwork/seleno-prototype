# Research notes

Compiled while building this prototype. Everything here was checked against the
linked source; nothing is recalled from memory.

---

## 1. Official ISRO sources

| Source | URL | What we took from it |
|---|---|---|
| ISRO Payloads Data & Science Handbook | <https://www.isro.gov.in/media_isro/pdf/science/hand_book_payloads_data_and_science.pdf> | Nominal payload specifications |
| PRADAN — Chandrayaan-2 data browse and dissemination | <https://pradan.issdc.gov.in/ch2/> | The authoritative archive (registration required) |
| Chandrayaan Map Browse | <https://chmapbrowse.issdc.gov.in/> | Footprint browse and per-product download |
| ISSDC acknowledgement / usage terms | <https://pradan.issdc.gov.in/ch2/ack.xhtml> | Licensing and attribution |
| Science results from Chandrayaan-2 | <https://www.issdc.gov.in/docs/ch2/science_results_from_ch-2.pdf> | Mission science context |
| PDS4 data archive for CH-2 payloads (TMC2, OHRC, IIRS) | <https://www.researchgate.net/publication/375922277_PDS4_DATA_ARCHIVE_FOR_CHANDRAYAAN-2_MISSION_PAYLOADS_TMC2_OHRC_and_IIRS> | Archive structure |
| Terrain Mapping Camera-2 instrument paper, *Current Science* 118(4) | <https://www.currentscience.ac.in/Volumes/118/04/0566.pdf> | TMC-2 design and 5 m/px GSD |
| OHRC data products user guide (mirror) | <https://ia801806.us.archive.org/29/items/chandrayaan-2-high-resolution-images-of-the-moon/OtherDownloads/OHRC/ch2_ohrc_data_products_user_guide_hocr.html> | Product naming, `ncp`/`nrp` levels, file layout |
| OHRC release mirror (public, no credentials) | <https://archive.org/details/chandrayaan-2-high-resolution-images-of-the-moon> | The actual products used here |

### Payload specifications

| Payload | Spatial resolution | Spectral | Swath | Note |
|---|---|---|---|---|
| **OHRC** | 0.25 m nadir GSD nominal; ~0.32 m for the 12 × 3 km two-orbit product | Panchromatic | ~3 km at 100 km altitude | Built to characterise the landing site and find hazard-free zones |
| **TMC-2** | ~5 m | Panchromatic | ~20 km | Stereo triplet (fore/nadir/aft) for DEM and morphometry |
| **IIRS** | ~80 m | ~0.8–5.0 µm, ~20 nm spectral resolution | ~20 km | First lunar mapping out to 5 µm |

The products used here report `isda:pixel_resolution` of **0.22977 m/px** and
**0.23022 m/px**, taken from a spacecraft altitude near 90.4 km rather than the
nominal 100 km — which is why they are finer than the quoted 0.25 m figure. This
is a good illustration of why per-product labels beat brochure numbers.

---

## 2. The paper that defines what is *not* novel here

**Makharia, R., Singla, J. G., Amitabh, Dube, N., Sharma, H. (2025).
"Comparative Evaluation of Traditional and Deep Learning Feature Matching
Algorithms using Chandrayaan-2 Lunar Data."**
arXiv:2509.04775 — <https://arxiv.org/abs/2509.04775> · PDF <https://arxiv.org/pdf/2509.04775>

What it does:

- Evaluates **SIFT, ASIFT, AKAZE, RIFT2 and SuperGlue** on cross-modality lunar
  image pairs from equatorial *and* polar regions, using Chandrayaan-2 data.
- Proposes a preprocessing pipeline: georeferencing, resolution alignment,
  intensity normalisation, adaptive histogram equalisation, PCA, shadow correction.
- Reports RMSE and runtime as the primary metrics.

Its findings, which we treat as the state of the art to be respected, not restated:

- **SuperGlue gives the lowest RMSE and the fastest runtimes** across their sets.
- **SIFT and AKAZE do well near the equator but degrade under polar lighting.**
- On SAR (SELENE) equatorial and polar sets, **SIFT, ASIFT, RIFT2 and AKAZE all
  failed to register**, while SuperGlue succeeded.
- **Preprocessing and learned matching both matter materially.**

**Consequence for Seleno: comparing feature matchers on lunar data is done work.
We do not present it as a contribution.** Our comparison table exists to
characterise our own pipeline's behaviour on our own pairs, not to claim a
finding. Our polar result below is consistent with theirs, which is a sanity
check on our implementation, not a discovery.

---

## 3. Related work

| Work | Link | Relevance |
|---|---|---|
| MoonMetaSync: Lunar Image Registration Analysis | <https://arxiv.org/pdf/2410.11118> | Registration analysis across lunar sensors; discusses OHRC/TMC-2/IIRS resolutions |
| Deep local feature extraction for Yutu-2 PCAM images | <https://www.sciencedirect.com/science/article/abs/pii/S0924271623002964> | Learned features on real lunar surface imagery; matching and surface reconstruction |
| Automated registration of full-Moon remote sensing images with triangulated network constraints | <https://www.researchgate.net/publication/381355146> | Geometric-constraint approach to global lunar registration |
| A comprehensive review on remote sensing image registration | <https://www.researchgate.net/publication/351709629> | Standard taxonomy: area-based vs feature-based, and the metric conventions we follow |
| SuperGlue | <https://www.emergentmind.com/topics/superglue> | Attention-based learned matching; the reference learned matcher in the CH-2 paper |
| Super-Resolution ISRO Chandrayaan-2 (community code) | <https://github.com/sankn123/Super-Resolution-ISRO-Chandrayaan-2> | Prior art working directly with CH-2 OHRC products |
| SIH problem statement SIH1732 (CH-2 PS) | <https://vedas.sac.gov.in/static/pdf/SIH_2024/SIH1732_CH2_PS.pdf> | States OHRC downloads come from chmapbrowse, `.IMG`/PDS-4, read with a PDS-4 Python reader |

Recurring themes across this literature:

1. Lunar imagery is **low contrast with sparse texture**; detectors starve.
2. **Illumination is the dominant nuisance variable**, especially near the poles
   where low sun angles produce long, scene-dominating shadows that move between
   passes.
3. **Cross-sensor pairs differ in resolution, illumination and distortion
   simultaneously**, so scale handling and radiometric normalisation cannot be
   treated independently.
4. Detector-free matchers (LoFTR) and attention-based matchers (SuperGlue) hold
   up better than classical descriptors where texture is sparse.

---

## 4. Data availability, as actually tested

| Route | Credentials | Outcome |
|---|---|---|
| PRADAN `protected/browse.xhtml?id=ohrc` | Registration | Not attempted — path is marked `protected` |
| chmapbrowse.issdc.gov.in | Registration | Authoritative; interactive |
| Internet Archive OHRC mirror | **None** | **Used.** 6 OHRC scene ZIPs, 775–855 MB each |

Within the mirror, the sample `ohr.zip` and `tmc.zip` under `OtherDownloads/`
contain only calibration tables and documents — no imagery. The imagery lives in
the six large scene ZIPs. Each contains one 1.12 GB raw `.img`, an ISRO browse
PNG (10× decimated), a geometry CSV, and the PDS4 label.

**No real TMC-2 or IIRS imagery was obtainable without credentials in the time
available.** Consequently there is no real cross-sensor pair in this prototype,
and every "TMC-2" pair is explicitly a simulated GSD. This is stated in the UI,
in `data/README.md` and in `ASSUMPTIONS.md`.

---

## 5. What Seleno claims, and what it does not

### Claims

- A working end-to-end pipeline on **real Chandrayaan-2 OHRC products**, from
  PDS4 label to registered overlay.
- Metrics computed per run, with **self-consistency separated from ground-truth
  accuracy** everywhere they are reported.
- A spatial match-selection stage that measurably raises spatial coverage at a
  fixed match budget, with the fair top-K-by-confidence baseline shown alongside.
- A conservative accept/reject decision that **correctly rejects a real
  non-overlapping pair** rather than returning a confident wrong transform.

### Explicitly not claimed

- Not novel feature matching. See Makharia et al. (2025).
- Not sub-pixel *geodetic* accuracy. Reported sub-pixel figures are against a
  synthetic transform on resampled imagery, which is a far easier problem.
- Not state of the art, not a novel architecture, not better than SuperGlue —
  SuperGlue was never run here.
- Not cross-modal. IIRS is not implemented at all.
- Not validated, endorsed or approved by ISRO in any way.
- Not production ready.

### Where a real contribution could come from

The open questions this prototype is positioned to attack, none of which it has
answered yet:

1. **Illumination invariance at high latitude.** Makharia et al. show classical
   descriptors degrading under polar lighting. Whether a physically motivated
   photometric normalisation closes that gap better than CLAHE is untested here.
2. **The 21.8× OHRC↔TMC-2 scale ratio.** Our `ohrc_tmc2_extreme` pair only
   simulates the GSD gap. Real TMC-2 data adds MTF, noise and viewing-geometry
   differences that resampling does not reproduce.
3. **Spatial conditioning as an objective**, rather than the post-hoc filter
   implemented here — selecting correspondences to minimise the covariance of the
   estimated transform.
4. **Honest rejection.** Deciding when *not* to register is under-reported in the
   literature and matters operationally. Our verdict rule is a crude first cut.
