"""Build report figures and summary data from a completed DRIVE pilot."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from dataset import DriveDataset, load_gray
from metrics import binary_metrics

METRICS = {
    "acc": "Accuracy", "se": "Sensitivity", "sp": "Specificity",
    "auc": "ROC-AUC", "dice": "Dice / F1", "iou_vessel": "Vessel IoU",
}
NAVY = "#172b45"
BLUE = "#1f608c"
TEAL = "#08756d"
ORANGE = "#ae4f20"
GRAY = "#4e5e6c"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "text.color": NAVY,
    "axes.labelcolor": NAVY, "axes.edgecolor": "#bcc8d3",
    "xtick.color": GRAY, "ytick.color": GRAY, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})


def save(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.pdf")
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.set(xlim=(0, 6.6), ylim=(0, 4.5))
    ax.set_axis_off()

    def box(x, y, width, text, tint="#edf4f9"):
        ax.add_patch(FancyBboxPatch((x, y), width, 0.64, boxstyle="round,pad=0.07",
                                   facecolor=tint, edgecolor="#8ca4b9", linewidth=0.8))
        ax.text(x + width / 2, y + 0.32, text, va="center", ha="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1,
                                "shrinkA": 0, "shrinkB": 0})

    box(1.5, 3.75, 3.6, "DRIVE training images\n+ manual masks")
    box(3.64, 2.66, 2.78, "Out-of-fold teachers")
    box(3.64, 1.52, 2.78, "SGL CON\nManual + soft BCE", "#e8f6f1")
    box(.18, 1.52, 2.78, "Baseline CON\nManual-label BCE")
    box(1.10, .20, 4.4, "Held-out DRIVE test images\nEnhanceNet → SegNet")
    arrow(2.16, 3.64, 1.66, 2.28)
    arrow(4.62, 3.64, 5.02, 3.40)
    arrow(5.02, 2.56, 5.02, 2.25)
    arrow(3.21, 3.64, 3.54, 1.86)
    arrow(1.60, 1.42, 2.31, .94)
    arrow(5.02, 1.42, 4.29, .94)
    save(fig, "workflow")


def read_labels(drive):
    records = {}
    masks = {}
    for split in ("training", "test"):
        for sample in DriveDataset(drive, split=split, mode="full").samples:
            image_id = sample["id"]
            fov = sample["ma"] > 0.5
            manual = sample["ve"] > 0.5
            row = {
                "split": split, "id": image_id, "height": fov.shape[0],
                "width": fov.shape[1], "fov_pixels": int(fov.sum()),
                "vessel_pixels": int((manual & fov).sum()),
            }
            row["vessel_fraction"] = row["vessel_pixels"] / row["fov_pixels"]
            if split == "test":
                path = drive / "test" / "2nd_manual" / f"{image_id}_manual2.gif"
                second = load_gray(path) > 200
                human = binary_metrics(second[fov], manual[fov])
                row.update(observer_dice=human["dice"], observer_se=human["se"],
                           observer_sp=human["sp"],
                           observer_disagreement=float((second[fov] != manual[fov]).mean()))
                masks[image_id] = fov, manual
            records[image_id] = row
    return records, masks


def plot_dataset(records):
    fig, axes = plt.subplots(1, 2, figsize=(7, 1.92), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.12]})
    ax = axes[0]
    for y, split, color in [(0, "test", TEAL), (1, "training", BLUE)]:
        rows = [row for row in records.values() if row["split"] == split]
        data = sorted(row["vessel_fraction"] for row in rows)
        shift = np.array([-.13, -.04, .05, .14, .02] * 4)
        ax.scatter(data, y + shift, s=19, c=color, alpha=.72, edgecolors="white", lw=.3)
        ax.vlines(np.mean(data), y - .23, y + .23, color=NAVY, lw=1.5)
    ax.set(yticks=[0, 1], yticklabels=["Test", "Train"], ylim=(-.49, 1.52),
           xlabel="Vessel share of field of view", title="(a) Class imbalance")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", alpha=.22)
    ax = axes[1]
    test = sorted((row for row in records.values() if row["split"] == "test"),
                  key=lambda row: row["observer_dice"])
    data = [row["observer_dice"] for row in test]
    ax.scatter(range(1, 21), data, s=20, c=TEAL, edgecolors="white", lw=.3, zorder=3)
    ax.axhline(np.mean(data), color=NAVY, lw=1.1, ls="--", label="Mean Dice")
    ax.set(xlabel="Test image rank by agreement", ylabel="Observer Dice",
           xticks=[1, 5, 10, 15, 20], title="(b) Agreement between graders")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontweight("bold")
    save(fig, "drive_data")


def load_result(path):
    return {row["id"]: row["fov"] for row in json.loads(path.read_text())["per_image"]}


def plot_effects(deltas, intervals):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    for y, metric in enumerate(METRICS):
        mean = deltas[metric].mean()
        lo, hi = intervals[metric]
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.3)
        ax.scatter(mean, y, color=BLUE, s=25, zorder=3)
    ax.axvline(0, color=GRAY, lw=.8, ls="--")
    ax.set(yticks=range(len(METRICS)), yticklabels=list(METRICS.values()),
           xlabel="SGL − baseline (percentage points)", title="(a) Mean paired changes")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda value, _: f"{100 * value:g}")
    ax = axes[1]
    for metric, color in [("se", TEAL), ("sp", ORANGE), ("auc", BLUE)]:
        ax.plot(range(1, 21), deltas[metric], "o-", ms=2.7, lw=.9, color=color,
                label=METRICS[metric])
    ax.axhline(0, color=GRAY, lw=.8, ls="--")
    ax.set(xlabel="DRIVE test image ID", ylabel="SGL − baseline (pp)",
           xticks=[1, 5, 10, 15, 20], title="(b) Per-image shifts")
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:g}")
    ax.legend(frameon=False, loc="best", ncol=1, fontsize=6.8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.title.set_fontweight("bold")
    save(fig, "pilot_effects")


def paired_errors(run, masks):
    # Pool in normalized image coordinates using coarse cells for visualization.
    size = 20
    grid = {key: np.zeros((size, size), dtype=np.int64)
            for key in ("shared_fp", "fov")}
    histograms = np.zeros((2, 2, 256), dtype=np.int64)
    rows = {}
    for image_id, (fov, manual) in sorted(masks.items()):
        predictions = []
        for arm, folder in enumerate(("baseline_test_predictions", "test_predictions")):
            with Image.open(run / folder / f"{image_id}_pred.png") as image:
                predictions.append(np.asarray(image.convert("L")) > 127)
            with Image.open(run / folder / f"{image_id}_prob.png") as image:
                probabilities = np.asarray(image.convert("L"))
            for label, mask in enumerate((~manual, manual)):
                histograms[arm, label] += np.bincount(probabilities[fov & mask], minlength=256)
        base, sgl = predictions
        slices = {
            "shared_fn": manual & ~base & ~sgl, "recovered_fn": manual & ~base & sgl,
            "new_fn": manual & base & ~sgl, "shared_fp": ~manual & base & sgl,
            "removed_fp": ~manual & base & ~sgl, "new_fp": ~manual & ~base & sgl,
        }
        rows[image_id] = {key: int((mask & fov).sum()) for key, mask in slices.items()}
        h, w = fov.shape
        ys = np.arange(h) * size // h
        xs = np.arange(w) * size // w
        bins = (ys[:, None] * size + xs).astype(int)
        for key, mask in {"shared_fp": slices["shared_fp"], "fov": fov}.items():
            grid[key] += np.bincount(bins[fov & mask], minlength=size * size).reshape(size, size)

    return rows, grid, histograms


def plot_operating_point(grid, histograms):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    nonempty = np.flatnonzero(histograms.sum(axis=(0, 1)))
    left = min(nonempty[0] / 255, 100 / 255) - .015
    right = max(nonempty[-1] / 255, 100 / 255) + .015
    for arm, ax in enumerate(axes[:2]):
        for label, color, caption in [(0, ORANGE, "Background"), (1, TEAL, "Vessel")]:
            counts = histograms[arm, label]
            ax.step(np.arange(256) / 255, 100 * counts / counts.sum(), where="post",
                    label=caption, color=color, lw=1.2)
        ax.axvline(100 / 255, color=GRAY, lw=.9, ls="--", label="Mask threshold")
        ax.set(xlim=(max(left, 0), min(right, 1)),
               xlabel="Predicted vessel probability",
               title=("(a) Baseline scores", "(b) SGL scores")[arm])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.20)
        ax.title.set_fontweight("bold")
    common_top = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(0, common_top)
    axes[0].set_ylabel("Pixels within class (%)")
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper right")
    rate = np.divide(grid["shared_fp"], grid["fov"],
                     out=np.full(grid["fov"].shape, np.nan),
                     where=grid["fov"] >= 500)
    im = axes[2].imshow(rate * 100, cmap="magma_r", norm=Normalize(0, 100),
                        interpolation="nearest")
    axes[2].set_title("(c) Shared false positives", fontsize=8, fontweight="bold")
    axes[2].set_axis_off()
    fig.colorbar(im, ax=axes[2], label="FOV pixels (%)", shrink=.7, pad=.06)
    save(fig, "operating_point")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    workflow()
    records, masks = read_labels(args.drive)
    plot_dataset(records)
    baseline = load_result(args.run / "baseline_test_results.json")
    sgl = load_result(args.run / "test_results.json")
    ids = sorted(masks)
    if set(ids) != set(baseline) or set(ids) != set(sgl):
        raise ValueError("The DRIVE test set and both result files must have the same image IDs.")
    deltas = {metric: np.array([sgl[i][metric] - baseline[i][metric] for i in ids])
              for metric in METRICS}
    indices = np.random.default_rng(2026).integers(0, len(ids), (10_000, len(ids)))
    intervals = {metric: np.quantile(values[indices].mean(axis=1), [.025, .975])
                 for metric, values in deltas.items()}
    plot_effects(deltas, intervals)
    errors, grid, histograms = paired_errors(args.run, masks)
    plot_operating_point(grid, histograms)
    for i in ids:
        for metric in METRICS:
            records[i][f"baseline_{metric}"] = baseline[i][metric]
            records[i][f"sgl_{metric}"] = sgl[i][metric]
        records[i].update(errors[i])

    summary = {
        "dataset": {
            split: {key: float(np.mean([row[key] for row in records.values()
                                        if row["split"] == split])) for key in
                    (["vessel_fraction", "fov_pixels"] if split == "training" else
                     ["vessel_fraction", "fov_pixels", "observer_dice", "observer_disagreement",
                      "observer_se", "observer_sp"])} for split in ("training", "test")
        },
        "pilot": {
            metric: {
                "baseline": float(np.mean([baseline[i][metric] for i in ids])),
                "sgl": float(np.mean([sgl[i][metric] for i in ids])),
                "delta": float(deltas[metric].mean()),
                "paired_bootstrap_ci": intervals[metric].tolist(),
                "improved_images": int((deltas[metric] > 0).sum()),
            } for metric in METRICS
        },
        "paired_error_counts": {key: sum(row[key] for row in errors.values())
                                for key in next(iter(errors.values()))},
        "quantized_probability_range": {
            name: (np.flatnonzero(histograms[arm].sum(axis=0))[[0, -1]] / 255).tolist()
            for arm, name in enumerate(("baseline", "sgl"))
        },
    }
    with (HERE / "evidence" / "per_image.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in records.values() for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records.values())
    with (HERE / "evidence" / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = (args.run / "pseudo_labels" / "folds.json").read_text()
    (HERE / "evidence" / "folds.json").write_text(manifest.rstrip() + "\n")


if __name__ == "__main__":
    main()
