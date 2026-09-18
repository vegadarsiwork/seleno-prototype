"""Fine-tune LightGlue on real lunar illumination change. Self-contained entrypoint.

    python scripts/train_lightglue.py --config config/train_lightglue.yaml
    python scripts/train_lightglue.py --config ... --overfit 20     # sanity check first
    python scripts/train_lightglue.py --manifest                    # what to copy to the GPU box

The experiment
--------------
Train on NAC<->NAC illumination pairs, test on OHRC<->NAC cross-sensor pairs.
Both are illumination-change problems; the second adds a sensor gap. If the
90-180 degree Sun-azimuth band lifts, the pretrained model's failure there was a
domain gap in its weights, not a physical impossibility. If it does not lift,
that is a result and the experiment stops - **do not iterate architectures.**

Shape
-----
* **DISK is frozen.** Phase 2 measured that the detector is not the problem
  (phase-congruency and DISK both find repeatable points across shadow reversal);
  what fails is matching under it. Only LightGlue's 11.9M parameters train.
* Ground truth is the **identity transform**, exact by construction for
  NAC<->NAC, and the identity after applying Phase 2's measured geolocation
  correction for OHRC<->NAC.
* Loss is the standard LightGlue negative log-likelihood over the assignment
  matrix: matched keypoint pairs, plus both dustbins for keypoints that have no
  partner.

Guardrails that are enforced, not just documented
-------------------------------------------------
* splits are by **tile**, asserted disjoint (`seleno.train.dataset.split_by_tile`);
* OHRC pairs are never constructed in the training path;
* **early stopping is on cross-sensor validation**, not in-domain. Improving on
  held-out NAC while degrading on OHRC is the failure mode this is watching for,
  and both curves are logged every epoch;
* fixed seeds, resumable checkpoints, AMP, gradient accumulation.

Nothing here is imported by the shipping pipeline. A missing checkpoint is not an
error anywhere: `seleno.matchers` uses pretrained weights unless one is present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from seleno.train import crosssensor, dataset as DS                # noqa: E402

DEFAULT_CFG = os.path.join(ROOT, "config", "train_lightglue.yaml")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def load_config(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def set_seed(s):
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- #
# features and loss
# --------------------------------------------------------------------------- #

def to_tensor(img, device):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(img)).float()[None, None] / 255.0
    return t.repeat(1, 3, 1, 1).to(device)


def extract(disk, img, device, n_features):
    import torch
    with torch.no_grad():
        f = disk(to_tensor(img, device), n_features, pad_if_not_divisible=True)[0]
    return f.keypoints, f.descriptors


def gt_assignment(kp0, kp1, thresh_px):
    """Mutual-nearest keypoint pairs under the identity transform.

    Returns ``(matches, un0, un1)``: index pairs that should be assigned to each
    other, and the keypoints on each side that should be assigned to the dustbin.
    """
    import torch
    if len(kp0) == 0 or len(kp1) == 0:
        return (torch.zeros((0, 2), dtype=torch.long, device=kp0.device),
                torch.arange(len(kp0), device=kp0.device),
                torch.arange(len(kp1), device=kp1.device))
    d = torch.cdist(kp0, kp1)
    n0 = d.argmin(1)
    n1 = d.argmin(0)
    idx0 = torch.arange(len(kp0), device=kp0.device)
    mutual = (n1[n0] == idx0) & (d[idx0, n0] < thresh_px)
    m = torch.stack([idx0[mutual], n0[mutual]], 1)
    matched0 = torch.zeros(len(kp0), dtype=torch.bool, device=kp0.device)
    matched1 = torch.zeros(len(kp1), dtype=torch.bool, device=kp0.device)
    matched0[m[:, 0]] = True
    matched1[m[:, 1]] = True
    # A keypoint with *some* partner nearby but not mutual is ambiguous; exclude
    # it from the dustbin target rather than teach the model it is unmatchable.
    near0 = (d.min(1).values < thresh_px * 2)
    near1 = (d.min(0).values < thresh_px * 2)
    un0 = torch.nonzero(~matched0 & ~near0, as_tuple=False).squeeze(1)
    un1 = torch.nonzero(~matched1 & ~near1, as_tuple=False).squeeze(1)
    return m, un0, un1


def lightglue_loss(log_assignment, m, un0, un1, dustbin_weight=1.0):
    import torch
    la = log_assignment[0]
    terms, n = [], 0
    if len(m):
        terms.append(-la[m[:, 0], m[:, 1]].sum())
        n += len(m)
    if len(un0):
        terms.append(-dustbin_weight * la[un0, -1].sum())
        n += len(un0)
    if len(un1):
        terms.append(-dustbin_weight * la[-1, un1].sum())
        n += len(un1)
    if not terms:
        return None, 0
    return torch.stack(terms).sum() / max(n, 1), len(m)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def solve_rate(disk, lg, pairs, device, cfg, truth=(0.0, 0.0)):
    """Fraction of pairs whose implied translation lands within tolerance."""
    import torch
    tol = cfg["eval"]["tolerance_px"]
    ok, errs, per_band = 0, [], {}
    lg.eval()
    with torch.no_grad():
        for a, b, va, vb, meta in pairs:
            k0, d0 = extract(disk, a, device, cfg["model"]["n_features"])
            k1, d1 = extract(disk, b, device, cfg["model"]["n_features"])
            if len(k0) < 8 or len(k1) < 8:
                continue
            out = lg({"image0": {"keypoints": k0[None], "descriptors": d0[None],
                                 "image_size": torch.tensor([[a.shape[1], a.shape[0]]],
                                                            device=device).float()},
                      "image1": {"keypoints": k1[None], "descriptors": d1[None],
                                 "image_size": torch.tensor([[b.shape[1], b.shape[0]]],
                                                            device=device).float()}})
            idx = out["matches"][0]
            band = None
            if hasattr(meta, "d_az"):
                d = meta.d_az
                band = "0-45" if d < 45 else "45-90" if d < 90 else "90-135" if d < 135 else "135-180"
            if len(idx) < 4:
                per_band.setdefault(band, [0, 0])[1] += 1
                continue
            dd = (k1[idx[:, 1]] - k0[idx[:, 0]]).cpu().numpy()
            med = np.median(dd, axis=0)
            inl = np.linalg.norm(dd - med, axis=1) < 3.0
            slot = per_band.setdefault(band, [0, 0])
            slot[1] += 1
            if inl.sum() >= 4:
                est = dd[inl].mean(axis=0)
                e = math.hypot(est[0] - truth[0], est[1] - truth[1])
                errs.append(e)
                if e <= tol:
                    ok += 1
                    slot[0] += 1
    n = sum(v[1] for v in per_band.values()) or 1
    return {"solve_rate": ok / n, "n": n, "solved": ok,
            "median_error_px": round(float(np.median(errs)), 2) if errs else None,
            "by_band": {k: "%d/%d" % (v[0], v[1]) for k, v in per_band.items() if k}}


# --------------------------------------------------------------------------- #
# manifest for the second machine
# --------------------------------------------------------------------------- #

def print_manifest(cfg):
    """Exactly what to copy to the GPU machine, and what to copy back."""
    corpus_rel = cfg["data"]["corpus"]
    idx = os.path.join(ROOT, corpus_rel, "index.json")
    if not os.path.exists(idx):
        print("corpus index not built yet: %s" % idx)
        print("run:  python scripts/build_training_corpus.py")
        return
    meta = json.load(open(idx))
    present = [r for r in meta["products"] if os.path.exists(os.path.join(ROOT, r["path"]))]
    corpus_bytes = sum(os.path.getsize(os.path.join(ROOT, r["path"])) for r in present)
    extra = []
    for rel in ("data/index/nac_south_catalog.json",
                "results/geolocation/offsets.json",
                "config/train_lightglue.yaml"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            extra.append((rel, os.path.getsize(p)))

    print("Copy TO the GPU machine (NAC<->NAC training needs only this):\n")
    print("  %-46s %5d files %8.1f MB" % (corpus_rel + "/", len(present), corpus_bytes / 1e6))
    for rel, sz in extra:
        print("  %-46s %5s       %8.1f MB" % (rel, "1", sz / 1e6))
    total = corpus_bytes + sum(sz for _, sz in extra)
    print("\n  TOTAL %.1f MB over %d tiles  (the full data tree is 13.5 GB)"
          % (total / 1e6, len({r["tile"] for r in present})))
    print("\n  Also copy the repository itself; nothing else under data/ is needed.")
    print("\nOPTIONAL, for cross-sensor validation (recommended - it is what early")
    print("stopping watches). Without it the run still trains and reports cross-")
    print("sensor validation as unavailable:")
    print("  data/raw/ch2/ohrc/            4.4 GB   the OHRC bundle")
    print("  results/nac_cache/*.png       ~3 MB    NAC browse for the matched bins")
    print("\nCopy BACK: %s/ only (checkpoint + history.json)." % cfg["train"]["out_dir"])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overfit", type=int, default=0,
                    help="train and evaluate on the same N pairs, as a sanity check")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the config; useful for a quick CPU plumbing check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.manifest:
        print_manifest(cfg)
        return

    import torch
    import kornia.feature as KF

    device = args.device or cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["train"]["seed"])
    out_dir = os.path.join(ROOT, cfg["train"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    idx = os.path.join(ROOT, cfg["data"]["corpus"], "index.json")
    if not os.path.exists(idx):
        sys.exit("corpus index missing: %s\nrun scripts/build_training_corpus.py" % idx)
    meta = json.load(open(idx))
    tiles = sorted({r["tile"] for r in meta["products"]})
    if args.overfit:
        # An overfit check deliberately trains and evaluates on the same pairs, so
        # a split would be meaningless. It is a gradient-flow sanity test, not a
        # generalisation measurement, and is reported as such.
        splits = {"train": tiles, "val": tiles, "test": tiles}
        print("tiles: %d (OVERFIT MODE - no split; this measures nothing about "
              "generalisation)" % len(tiles))
    else:
        if len(tiles) < 3:
            sys.exit("only %d tile(s) in the corpus; a tile-level split needs >=3.\n"
                     "Let scripts/build_training_corpus.py finish, or pass --overfit N."
                     % len(tiles))
        splits = DS.split_by_tile(tiles, tuple(cfg["data"]["split"]), seed=cfg["train"]["seed"])
        print("tiles: %d train / %d val / %d test  (split by TILE, disjointness asserted)"
              % (len(splits["train"]), len(splits["val"]), len(splits["test"])))
        print("  train tiles: %s" % " ".join(splits["train"][:8]))
        print("  val   tiles: %s" % " ".join(splits["val"]))
        print("  test  tiles: %s" % " ".join(splits["test"]))

    mk = lambda names, n: DS.NACIlluminationPairs(
        index_path=idx, tiles=names, root=ROOT, chip=cfg["data"]["chip"],
        min_valid=cfg["data"]["min_valid"], seed=cfg["train"]["seed"], samples=n,
        daz_bins=tuple(tuple(x) for x in cfg["data"]["daz_weights"]))
    train_ds = mk(splits["train"], cfg["data"]["train_samples"])
    val_ds = mk(splits["val"], cfg["data"]["val_samples"])
    print("train: %s" % DS.describe(train_ds))
    print("val  : %s" % DS.describe(val_ds))

    cross = crosssensor.build_pairs(
        os.path.join(ROOT, "results/geolocation/offsets.json"),
        os.path.join(ROOT, "results/nac_cache"),
        chip=cfg["data"]["chip"], max_per_strip=cfg["eval"]["cross_per_strip"])
    cross = [(a, b, va, vb, m) for a, b, va, vb, m in cross]
    print("cross-sensor OHRC<->NAC validation pairs: %d%s"
          % (len(cross), "" if cross else "  (UNAVAILABLE - early stopping falls back to in-domain)"))

    disk = KF.DISK.from_pretrained("depth").to(device).eval()
    for p in disk.parameters():
        p.requires_grad_(False)
    lg = KF.LightGlue("disk", depth_confidence=-1, width_confidence=-1).to(device)
    n_tr = sum(p.numel() for p in lg.parameters() if p.requires_grad)
    print("DISK frozen (%.1fM params); LightGlue trainable %.2fM params; device %s"
          % (sum(p.numel() for p in disk.parameters()) / 1e6, n_tr / 1e6, device))

    opt = torch.optim.AdamW(lg.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda" and cfg["train"]["amp"]))
    start_epoch, best, history = 0, -1.0, []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        lg.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best, history = ck["epoch"] + 1, ck.get("best", -1.0), ck.get("history", [])
        print("resumed from %s at epoch %d" % (args.resume, start_epoch))

    samples = None
    if args.overfit:
        # draw only as many as are needed; materialising 4000 chips to keep 20
        # is pure waste
        samples = []
        for i in range(len(train_ds)):
            s = train_ds[i]
            if s is not None:
                samples.append(s)
            if len(samples) >= args.overfit:
                break
        print("\nOVERFIT SANITY CHECK on %d pairs (train == eval; measures plumbing, "
              "not generalisation)" % len(samples))

    accum = cfg["train"]["grad_accum"]
    epochs = args.epochs or (cfg["train"]["overfit_epochs"] if args.overfit
                             else cfg["train"]["epochs"])
    for epoch in range(start_epoch, epochs):
        lg.train()
        t0, tot, nb, nm = time.time(), 0.0, 0, 0
        order = range(len(samples)) if args.overfit else range(len(train_ds))
        opt.zero_grad(set_to_none=True)
        for step, i in enumerate(order):
            s = samples[i] if args.overfit else train_ds[i]
            if s is None:
                continue
            a, b, va, vb, cm = s
            k0, d0 = extract(disk, a, device, cfg["model"]["n_features"])
            k1, d1 = extract(disk, b, device, cfg["model"]["n_features"])
            if len(k0) < 16 or len(k1) < 16:
                continue
            m, un0, un1 = gt_assignment(k0, k1, cfg["train"]["gt_thresh_px"])
            if len(m) < cfg["train"]["min_gt_matches"]:
                continue
            with torch.amp.autocast(device, enabled=(device == "cuda" and cfg["train"]["amp"])):
                out = lg({"image0": {"keypoints": k0[None], "descriptors": d0[None],
                                     "image_size": torch.tensor([[a.shape[1], a.shape[0]]],
                                                                device=device).float()},
                          "image1": {"keypoints": k1[None], "descriptors": d1[None],
                                     "image_size": torch.tensor([[b.shape[1], b.shape[0]]],
                                                                device=device).float()}})
                loss, nmatch = lightglue_loss(out["log_assignment"], m, un0, un1,
                                              cfg["train"]["dustbin_weight"])
            if loss is None:
                continue
            scaler.scale(loss / accum).backward()
            tot += float(loss.detach()); nb += 1; nm += nmatch
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(lg.parameters(), cfg["train"]["clip_grad"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        tr_loss = tot / max(nb, 1)
        rec = {"epoch": epoch, "train_loss": round(tr_loss, 4), "batches": nb,
               "mean_gt_matches": round(nm / max(nb, 1), 1),
               "seconds": round(time.time() - t0, 1)}

        if args.overfit:
            ev = solve_rate(disk, lg, [(s[0], s[1], s[2], s[3], s[4]) for s in samples],
                            device, cfg)
            rec["overfit_solve_rate"] = round(ev["solve_rate"], 3)
            rec["overfit_median_err_px"] = ev["median_error_px"]
        else:
            in_dom = solve_rate(disk, lg, [x for x in (val_ds[i] for i in range(min(len(val_ds), cfg["eval"]["val_pairs"]))) if x], device, cfg)
            rec["val_nac_solve_rate"] = round(in_dom["solve_rate"], 3)
            rec["val_nac_by_band"] = in_dom["by_band"]
            if cross:
                xs = solve_rate(disk, lg, cross, device, cfg)
                rec["val_cross_solve_rate"] = round(xs["solve_rate"], 3)
                rec["val_cross_median_err_px"] = xs["median_error_px"]
                score = xs["solve_rate"]
            else:
                score = in_dom["solve_rate"]
                rec["val_cross_solve_rate"] = None
            if score > best:
                best = score
                torch.save({"model": lg.state_dict(), "epoch": epoch, "best": best,
                            "config": cfg, "history": history + [rec]},
                           os.path.join(out_dir, "lightglue_finetuned.pt"))
                rec["saved"] = True

        history.append(rec)
        print("  " + json.dumps(rec))
        torch.save({"model": lg.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best": best, "history": history, "config": cfg},
                   os.path.join(out_dir, "last.pt"))
        if not args.overfit and cfg["train"]["patience"] > 0:
            since = len(history) - 1 - max(
                (k for k, r in enumerate(history) if r.get("saved")), default=0)
            if since >= cfg["train"]["patience"]:
                print("  early stop: %d epochs without cross-sensor improvement" % since)
                break

    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    print("\nwrote %s" % os.path.relpath(out_dir, ROOT))


if __name__ == "__main__":
    main()
