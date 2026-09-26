"""Reproduce the report's diagnostics from saved evidence; never fits a model.

Run from the repository root:
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/seleno-report-mpl \
    .venv/bin/python reports/registration_review_20260926/analyse_saved_evidence.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
JOB = ROOT / "outputs/3acbdc7cee27"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(vectors):
    valid = np.isfinite(vectors).all(axis=1)
    v = vectors[valid]
    d = np.linalg.norm(v, axis=1)
    return {
        "n": len(vectors), "valid_n": int(valid.sum()),
        "invalid_n": int((~valid).sum()),
        "rmse_px": float(np.sqrt(np.mean(d ** 2))),
        "median_px": float(np.median(d)),
        "p90_px": float(np.percentile(d, 90)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(d.max()),
        "fraction_strictly_below_1_px_among_valid": float(np.mean(d < 1)),
        "fraction_strictly_below_1_px_among_all": float(np.sum(d < 1) / len(vectors)),
        "bias_xy_px": np.mean(v, axis=0).tolist(),
        "component_nmad_px": (1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0)).tolist(),
    }


def main():
    m = json.loads((JOB / "metrics.json").read_text())
    e = json.loads((JOB / "evaluation.json").read_text())
    observed = np.asarray(e["native_source"], float)
    predicted = np.asarray(e["predicted_native_source"], float)
    errors = predicted - observed
    valid = np.isfinite(errors).all(axis=1)
    check = np.asarray(e["check_point"], bool)
    all_stats, check_stats = statistics(errors), statistics(errors[check])
    assert e["transform_sha256"] == m["accuracy"]["transform_sha256"]
    for key in ("rmse_px", "median_px", "p90_px"):
        assert np.isclose(check_stats[key], m["accuracy"]["check_point_source_" + key])
        assert np.isclose(all_stats[key], m["accuracy"]["held_out_source_" + key])

    preview = {}
    for name in ("source.png", "reference.png", "registered.png", "overlay.png"):
        a = np.asarray(Image.open(JOB / name))
        nonzero = (a > 0).any(axis=2) if a.ndim == 3 else a > 0
        mid = a.shape[1] // 2
        preview[name] = {
            "shape": list(a.shape),
            "nonzero_fraction": float(nonzero.mean()),
            "top_1000_rows_left_half_nonzero_fraction": float(nonzero[:1000, :mid].mean()),
            "top_1000_rows_right_half_nonzero_fraction": float(nonzero[:1000, mid:].mean()),
        }

    records = []
    height = m["source"]["shape"][0]
    for index in range(10):
        in_bin = (observed[:, 1] >= height * index / 10) & (observed[:, 1] < height * (index + 1) / 10)
        rec = {"strip_decile": index + 1, "held_out_n": int(in_bin.sum())}
        if np.any(in_bin & valid):
            rec["all_held_out"] = statistics(errors[in_bin])
        records.append(rec)

    evidence = {
        "review_date": "2026-09-26", "job": str(JOB.relative_to(ROOT)),
        "generated_utc": m["generated_utc"],
        "identification": "Source filename, WAC reference and 206 preview candidates match the supplied screenshots; UI job ID was not visible.",
        "duplicate_job": "outputs/499a51341aa5 has identical reported accuracy and candidate counts",
        "basis": "Recomputed from saved matcher observations/predictions; no external truth and no registration rerun.",
        "input_sha256": {name: digest(JOB / name) for name in ("metrics.json", "evaluation.json", "transform.json", "preview.json")},
        "source": m["source"], "reference": m["reference"],
        "all_held_out_source": all_stats, "screened_held_out_source": check_stats,
        "saved_accuracy": m["accuracy"], "saved_acceptance": m["acceptance"],
        "saved_distribution": m["distribution"], "saved_matches": m["matches"],
        "preview": preview,
        "preview_caveat": "Nonzero brightness diagnoses the displayed PNG only; zero may mean nodata or dark valid terrain. These are not authoritative valid-mask coverage fractions.",
        "strip_deciles": records,
    }
    (OUT / "evidence_summary.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    d = np.linalg.norm(errors, axis=1)
    for keep, label, color in ((valid, "All finite held-out (132)", "#3a6ea5"),
                                (valid & check, "Screened held-out (122)", "#b05c00")):
        v = np.sort(d[keep])
        axes[0].step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", label=label, color=color)
    axes[0].axvline(1, color="#555", linestyle="--", linewidth=1)
    axes[0].axhline(95, color="#777", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Native source error (pixels)", ylabel="Measurements at or below error (%)", title="Error distribution", ylim=(0, 102))
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].text(.98, .45, "1 invalid prediction excluded\nfrom curves; acceptance fails", transform=axes[0].transAxes, ha="right", fontsize=8)

    axes[1].scatter(observed[valid & check, 1] / height * 100, d[valid & check], s=13, color="#b05c00", alpha=.75, label="Screened held-out")
    rejected = valid & ~check
    axes[1].scatter(observed[rejected, 1] / height * 100, d[rejected], s=27, color="#a63333", marker="x", label="Set aside by screen")
    axes[1].axhline(1, color="#555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Position along native source strip (%)", ylabel="Native source error (pixels)", title="Where errors occur", xlim=(0, 100))
    axes[1].legend(fontsize=8)

    for component, label, color in ((0, "Sample (across detector)", "#3a6ea5"), (1, "Line (along detector)", "#b05c00")):
        axes[2].scatter(observed[valid, 1] / height * 100, errors[valid, component], s=12, color=color, alpha=.6, label=label)
    axes[2].axhline(0, color="#555", linewidth=1)
    axes[2].set(xlabel="Position along native source strip (%)", ylabel="Signed native source error (pixels)", title="Directional residuals", xlim=(0, 100))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.17)
    fig.suptitle("IIRS 2024-01-19 → WAC: saved matcher evidence, not independent ground truth", fontsize=13)
    fig.savefig(OUT / "residual_diagnostics.png", dpi=170)
    fig.savefig(OUT / "residual_diagnostics.pdf")
    plt.close(fig)
    print(json.dumps({"all_held_out": all_stats, "screened_held_out": check_stats}, indent=2))


if __name__ == "__main__":
    main()
