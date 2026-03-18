import argparse
import csv
import math
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


FONT_FAMILY = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
COLORS = {
    "clean_source": "#355C7D",
    "clean_target": "#6C8EAD",
    "asr_source": "#C06C84",
    "asr_target": "#F67280",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready plots from experiment results.")
    parser.add_argument("--csv", type=str, default="experiment_results.csv")
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--keep_all", action="store_true")
    return parser.parse_args()


def to_float(value):
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def latest_rows(rows):
    grouped = OrderedDict()
    for row in rows:
        exp_name = row.get("exp_name", "")
        stamp = row.get("time", "")
        if exp_name not in grouped or stamp >= grouped[exp_name].get("time", ""):
            grouped[exp_name] = row
    return list(grouped.values())


def load_rows(csv_path, dataset_name="", keep_all=False):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if dataset_name:
        rows = [row for row in rows if row.get("dataset_name", "") == dataset_name]

    if not keep_all:
        rows = latest_rows(rows)

    return rows


def collect_target_indices(rows):
    indices = set()
    for row in rows:
        for key in row.keys():
            if key.startswith("clean_target_acc_"):
                indices.add(int(key.rsplit("_", 1)[1]))
    return sorted(indices)


def shorten_name(name):
    parts = name.split("_")
    return "\n".join(parts)


def prepare_metrics(rows):
    target_indices = collect_target_indices(rows)
    metrics = []

    for row in rows:
        clean_target_series = [to_float(row.get(f"clean_target_acc_{idx}", "")) for idx in target_indices]
        asr_target_series = [to_float(row.get(f"target_asr_{idx}", "")) for idx in target_indices]

        clean_targets = [x for x in clean_target_series if not math.isnan(x)]
        asr_targets = [x for x in asr_target_series if not math.isnan(x)]

        metrics.append(
            {
                "exp_name": row.get("exp_name", "unknown"),
                "label": shorten_name(row.get("exp_name", "unknown")),
                "clean_source_acc": to_float(row.get("clean_source_acc", "")),
                "source_asr": to_float(row.get("source_asr", "")),
                "clean_target_mean": float(np.mean(clean_targets)) if clean_targets else math.nan,
                "target_asr_mean": float(np.mean(asr_targets)) if asr_targets else math.nan,
                "clean_targets": clean_target_series,
                "asr_targets": asr_target_series,
            }
        )

    return metrics, target_indices


def make_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_FAMILY,
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
        }
    )


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_overview(metrics, outdir):
    labels = [item["label"] for item in metrics]
    x = np.arange(len(metrics))
    width = 0.34

    clean_source = np.array([item["clean_source_acc"] for item in metrics], dtype=float)
    clean_target = np.array([item["clean_target_mean"] for item in metrics], dtype=float)
    source_asr = np.array([item["source_asr"] for item in metrics], dtype=float)
    target_asr = np.array([item["target_asr_mean"] for item in metrics], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    bars1 = axes[0].bar(x - width / 2, clean_source, width, color=COLORS["clean_source"], label="Source")
    bars2 = axes[0].bar(x + width / 2, clean_target, width, color=COLORS["clean_target"], label="Target mean")
    axes[0].set_title("Clean Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, loc="upper left")
    annotate_bars(axes[0], bars1)
    annotate_bars(axes[0], bars2)

    bars3 = axes[1].bar(x - width / 2, source_asr, width, color=COLORS["asr_source"], label="Source")
    bars4 = axes[1].bar(x + width / 2, target_asr, width, color=COLORS["asr_target"], label="Target mean")
    axes[1].set_title("Attack Success Rate")
    axes[1].set_ylabel("ASR")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, loc="upper left")
    annotate_bars(axes[1], bars3)
    annotate_bars(axes[1], bars4)

    axes[2].set_title("Transferability Trade-off")
    axes[2].set_xlabel("Target Clean Accuracy")
    axes[2].set_ylabel("Target Mean ASR")
    axes[2].set_xlim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    for item in metrics:
        x_val = item["clean_target_mean"]
        y_val = item["target_asr_mean"]
        if math.isnan(x_val) or math.isnan(y_val):
            continue
        axes[2].scatter(x_val, y_val, s=95, color="#2A9D8F", edgecolors="white", linewidths=0.8)
        axes[2].annotate(
            item["exp_name"],
            (x_val, y_val),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    save_figure(fig, outdir, "overview")


def plot_per_target(metrics, target_indices, outdir):
    if not target_indices:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    x = np.arange(len(target_indices))
    width = 0.14 if metrics else 0.2

    for idx, item in enumerate(metrics):
        clean_vals = item["clean_targets"]
        asr_vals = item["asr_targets"]
        offset = (idx - (len(metrics) - 1) / 2) * width
        color = plt.cm.cividis(0.15 + 0.7 * idx / max(len(metrics) - 1, 1))
        axes[0].bar(x + offset, clean_vals, width=width, label=item["exp_name"], color=color)
        axes[1].bar(x + offset, asr_vals, width=width, label=item["exp_name"], color=color)

    domain_labels = [f"Target {idx}" for idx in target_indices]
    axes[0].set_title("Per-Domain Clean Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xticks(x, domain_labels)

    axes[1].set_title("Per-Domain ASR")
    axes[1].set_ylabel("ASR")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xticks(x, domain_labels)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    save_figure(fig, outdir, "per_target")


def save_figure(fig, outdir, stem):
    png_path = os.path.join(outdir, f"{stem}.png")
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    configure_style()
    make_outdir(args.outdir)

    rows = load_rows(args.csv, dataset_name=args.dataset, keep_all=args.keep_all)
    if not rows:
        raise ValueError("No matching experiment rows found in the CSV file.")

    metrics, target_indices = prepare_metrics(rows)
    plot_overview(metrics, args.outdir)
    plot_per_target(metrics, target_indices, args.outdir)

    print(f"Saved figures to {args.outdir}")


if __name__ == "__main__":
    main()
