# Fine-tuning LightGlue on lunar illumination change

Optional. The shipping product is the pretrained CPU path; this produces a
checkpoint that loads if present and changes nothing if it is absent.

---

## 1. The experiment, and the rule for stopping

**Train on NAC↔NAC illumination pairs. Test on OHRC↔NAC cross-sensor pairs.**

Both are illumination-change problems; the second adds a sensor gap. The reconciled [Phase 2 benchmark](../reports/PHASE2.md#reconciled-illumination-benchmark--2026-09-23)
uses mean four-corner error ≤ 3 browse pixels. Its earlier translation-consensus
result is superseded; use the reconciled table for illumination claims.

The question this experiment answers is narrow and worth answering either way:

> Is the 90–180° failure a **domain gap in the pretrained weights**, or is it
> physical — the shadow field genuinely carrying no recoverable correspondence?

If fine-tuning lifts the >90° bins, it was a domain gap. If it does not, that is
a publishable negative and **the experiment stops**. Per the amendment: do not
iterate architectures.

---

## 2. Why the training set costs nothing to label

Every `NAC_POLE_SOUTH_CM_<bin>_<tile>` product is a subset of **one** ISIS jigsaw
solution (18 323 NAC images, LOLA-tied, 0.8 px at 2σ) resampled onto **one** 1 m
polar stereographic grid. So two bins of the same tile are:

- **the same ground at the same pixel coordinates** — the transform relating them
  is the **identity, exact by construction**, with no annotation, no synthetic
  warp, and no matcher involved in defining it; and
- **genuinely differently illuminated** — different sub-solar longitude, different
  cast shadows, different underlying NAC frames.

40 controlled tiles carry **862 single-illumination products** between them. That
is tens of thousands of real illumination-change pairs for the cost of a download.

### Resolution, stated plainly

The corpus is built from the **browse pyramids (~32 m/px)**, not full resolution:

- the full-resolution GeoTIFFs are striped, so cutting chips costs ~93 MB of
  scanline reads per tile-bin — roughly **13 GB** for this corpus versus **430 MB**
  for the pyramids;
- the reconciled Phase 2 benchmark ran at this same sampling; its corner-error
  criterion and complete results are recorded in the linked report.

**Caveat, not to be elided:** a matcher fine-tuned at 32 m/px is *not* thereby
validated at 1 m/px. Re-running on a full-resolution subset is the obvious
follow-up and is not claimed here.

---

## 3. Guardrails that are enforced in code, not just intended

| risk | guard | where |
|---|---|---|
| ground truth leaks between splits | splits are **by tile**, never by pair; disjointness and coverage are `assert`ed | `seleno.train.dataset.split_by_tile` |
| the hard band gets buried | pair sampling is **weighted toward Δaz 90–180°** (weights 2.0 vs 0.5 for 0–45°) | `config/train_lightglue.yaml: data.daz_weights` |
| test data contaminates training | OHRC pairs are **never constructed** in the training path; they live in a separate module | `seleno.train.crosssensor` |
| overfitting to NAC radiometry | **early stopping is on cross-sensor validation**, not in-domain; both curves logged every epoch | `scripts/train_lightglue.py` |
| blank chips teach nothing | a chip must carry data in **both** bins (`min_valid`) or it is rejected | `NACIlluminationPairs.__getitem__` |

The fourth is the one the amendment specifically flagged, and it is the reason
the checkpoint is selected on OHRC↔NAC solve rate rather than on training loss or
in-domain validation.

### How the cross-sensor validation set exists at all

OHRC's delivered geolocation is wrong by kilometres, so OHRC↔NAC pairs cannot be
built by co-location the way NAC↔NAC pairs can. But Phase 2 *measured* that error
for two strips, with three algorithm families agreeing to under 52 m and an
independent prior measurement agreeing to under 83 m. Applying the measured
correction puts the strip on the LOLA-tied grid, after which the truth is again
the identity.

That set therefore inherits Phase 2's caveats: **only confirmed strips are used**
(the provisional and failed ones are excluded, not guessed at), and the residual
is tens of metres rather than zero, which `eval.tolerance_px` reflects.

---

## 4. Recipe

- **DISK frozen, LightGlue trained.** Phase 2 measured that detection is not the
  problem — phase congruency repeats at 54.6% and SIFT at 43.1% across a 30°
  azimuth change, and DISK finds ~1 000 points either way. What fails is matching
  under shadow reversal. Only LightGlue's **11.88 M** parameters across 9
  transformer layers are trained.
- 512 px chips, effective batch 4 via gradient accumulation, AMP on CUDA.
- Loss: standard LightGlue negative log-likelihood over the (N+1)×(M+1)
  assignment matrix — matched pairs plus both dustbins. Adaptive depth/width
  pruning is disabled during training so the full assignment is differentiable.
- Fixed seeds, resumable (`--resume`), YAML config.

Run the sanity check before the full run:

```
python scripts/train_lightglue.py --overfit 20
```

That trains and evaluates on the same pairs. It measures **nothing about
generalisation** — it is a gradient-flow check, and the script says so in its own
output.

### Result of the check actually run (8 pairs, 6 epochs, CPU)

```
epoch  train_loss  gt_match/pair    solve_rate  med_err_px  sec
0      1.7580      165.5            0.125       0.07        27.1
1      1.3830      165.5            0.125       0.07        37.1
2      1.2405      165.5            0.125       0.09        36.1
3      1.1486      165.5            0.125       0.09        32.4
4      1.0573      165.5            0.125       0.09        36.9
5      0.9933      165.5            0.125       0.08        35.2

loss 1.7580 -> 0.9933  (down 43.5% over 6 epochs)
```

**What this establishes:** the identity transform yields 165 ground-truth
correspondences per pair with no labelling, the loss is differentiable through
LightGlue's assignment matrix, and it falls monotonically. The plumbing is
correct.

**What this does NOT establish, and must not be reported as if it did:** the
solve rate did not move (1 of 8 throughout). A proper overfit check should
eventually drive a model to fit its *own* training pairs, and 6 epochs over 8
pairs on CPU is nowhere near enough to do that with 11.88 M parameters. The one
pair that solves does so at 0.07-0.09 px; the other seven are the high-Δaz cases
the sampler deliberately over-represents, which is exactly where Phase 2
measured pretrained LightGlue at 0/12.

**So the real overfit check is still outstanding** and belongs on the GPU:
`overfit_epochs: 40` in the config, which is ~5 minutes there against ~40 on CPU.
If the solve rate still does not reach ~1.0 on its own 20 training pairs after
that, stop and debug the loss before launching any full run.

---

## 5. Running it on the second machine

The GPU is on a teammate's RTX 4060 (8 GB). Training is a self-contained
entrypoint so the whole data tree does not have to travel.

```
python scripts/train_lightglue.py --manifest
```

prints exactly what to copy. For NAC↔NAC training only it is the browse corpus
plus three small files — **~430 MB against a 13.5 GB data tree**. Cross-sensor
validation additionally wants the OHRC bundle (4.4 GB) and the NAC browse cache
(~3 MB); without them the run still trains and reports cross-sensor validation as
unavailable, which is the degraded mode, not a crash.

**Copy back the checkpoint directory only.**

### Ownership

One person should own training end to end on that machine. Shuttling data and
checkpoints between two machines for every iteration costs more than the
experiment is worth — the whole point of the manifest is that the GPU box needs
one 430 MB transfer and then runs independently.

### GPU inference is the quieter win

`scripts/benchmark_illumination.py --device cuda` runs the learned matchers on
the GPU. The 12-bin sweep is 66 bin pairs × patches × matchers; on CPU that is
tens of minutes per matcher. Faster benchmark iteration may deliver more value
than the fine-tuning itself, and it needs no training run at all.

---

## 6. What ships regardless

`seleno.matchers.disk_lightglue_match` loads a fine-tuned checkpoint **if one
exists** at `results/train/lightglue/lightglue_finetuned.pt`, and pretrained
weights otherwise. Override with `SELENO_LIGHTGLUE_CKPT=/path/to.pt`, or force
pretrained with `SELENO_LIGHTGLUE_CKPT=none`.

Every match result records which weights were used:

```python
r.detail["weights"]      # "pretrained" or "finetuned"
r.detail["checkpoint"]   # the path, or None
r.detail["device"]       # "cpu" or "cuda"
```

so no result can be reported without knowing which model produced it. A demo
cannot break because training did not finish or produced a bad model.

---

## 7. Reporting

`reports/` will carry the same Δaz table as Phase 2, **pretrained vs fine-tuned
side by side**, on the same held-out tiles. If the >90° bins do not lift, that is
the reported result and the experiment ends there.
