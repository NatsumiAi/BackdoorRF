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
    "gap": "#2A9D8F",
    "trajectory": "#264653",
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
        rows = []
        for row in csv.DictReader(f):
            clean_row = {}
            for key, value in row.items():
                if key is None:
                    continue
                clean_row[key] = value
            rows.append(clean_row)

    if dataset_name:
        rows = [row for row in rows if row.get("dataset_name", "") == dataset_name]

    if not keep_all:
        rows = latest_rows(rows)

    return rows


def collect_target_indices(rows):
    indices = set()
    for row in rows:
        for key in row.keys():
            if key is None:
                continue
            if key.startswith("clean_target_acc_"):
                indices.add(int(key.rsplit("_", 1)[1]))
    return sorted(indices)


def shorten_name(name):
    parts = name.split("_")
    return "\n".join(parts)


def display_name(name):
    return name.replace("_", " ").title()


def infer_stage_rank(name):
    ordered = ["single_patch_post", "sparse_post", "sparse_eot", "sparse_eot_channel"]
    if name in ordered:
        return ordered.index(name)
    return len(ordered)


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
                "asr_gap": to_float(row.get("source_asr", "")) - (float(np.mean(asr_targets)) if asr_targets else math.nan),
                "clean_targets": clean_target_series,
                "asr_targets": asr_target_series,
            }
        )

    metrics.sort(key=lambda item: infer_stage_rank(item["exp_name"]))
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
            "savefig.bbox": "standard",
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


def plot_asr_gap(metrics, outdir):
    valid = [item for item in metrics if not math.isnan(item["asr_gap"]) and not math.isnan(item["target_asr_mean"])]
    if not valid:
        return

    labels = [item["label"] for item in valid]
    x = np.arange(len(valid))
    gaps = np.array([item["asr_gap"] for item in valid], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    bars = ax.bar(x, gaps, color=COLORS["gap"], width=0.58)
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_title("Cross-Domain Transfer Loss")
    ax.set_ylabel(r"ASR Gap: Source - Target Mean")
    ax.set_ylim(min(-0.05, np.nanmin(gaps) - 0.05), max(1.02, np.nanmax(gaps) + 0.08))
    ax.set_xticks(x, labels)
    annotate_bars(ax, bars)
    save_figure(fig, outdir, "asr_gap")


def plot_transfer_trajectory(metrics, outdir):
    valid = [item for item in metrics if not math.isnan(item["asr_gap"]) and not math.isnan(item["target_asr_mean"])]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=True)
    xs = [item["asr_gap"] for item in valid]
    ys = [item["target_asr_mean"] for item in valid]
    names = [item["exp_name"] for item in valid]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.88, len(valid)))

    for idx, (x_val, y_val, name) in enumerate(zip(xs, ys, names)):
        ax.scatter(x_val, y_val, s=135, color=cmap[idx], edgecolors="white", linewidths=1.0, zorder=3)
        ax.annotate(display_name(name), (x_val, y_val), textcoords="offset points", xytext=(6, 7), fontsize=9)
        if idx > 0:
            prev_x, prev_y = xs[idx - 1], ys[idx - 1]
            ax.annotate(
                "",
                xy=(x_val, y_val),
                xytext=(prev_x, prev_y),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": COLORS["trajectory"], "alpha": 0.65},
            )

    ax.set_title("Backdoor Transfer Trajectory")
    ax.set_xlabel(r"ASR Gap: Source - Target Mean (lower is better)")
    ax.set_ylabel("Target Mean ASR (higher is better)")
    ax.set_xlim(min(-0.02, min(xs) - 0.05), max(xs) + 0.08)
    ax.set_ylim(min(-0.02, min(ys) - 0.05), 1.02)
    ax.text(
        0.98,
        0.04,
        "Preferred region",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=COLORS["trajectory"],
    )
    ax.annotate(
        "",
        xy=(0.08, 0.92),
        xytext=(0.3, 0.72),
        textcoords="axes fraction",
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.3, "linestyle": "--", "color": COLORS["trajectory"]},
    )
    save_figure(fig, outdir, "transfer_trajectory")


def _make_base_signal(length=256):
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    i_comp = 0.85 * np.sin(2 * np.pi * 7 * t) + 0.18 * np.sin(2 * np.pi * 17 * t)
    q_comp = 0.8 * np.cos(2 * np.pi * 7 * t + 0.25) + 0.16 * np.cos(2 * np.pi * 15 * t)
    return np.stack([i_comp, q_comp], axis=0).astype(np.float32)


def _inject_segment(x, start, seg_len, amp=0.16, freq=6.0):
    t = np.arange(seg_len, dtype=np.float32)
    phase = 2 * np.pi * freq * t / max(seg_len, 1)
    x[0, start:start + seg_len] += amp * np.sin(phase)
    x[1, start:start + seg_len] += amp * np.cos(phase)
    return x


def build_schematic_signals(length=256):
    clean = _make_base_signal(length)

    single_patch = clean.copy()
    single_patch = _inject_segment(single_patch, int(length * 0.72), int(length * 0.18), amp=0.18, freq=7.0)

    sparse = clean.copy()
    sparse = _inject_segment(sparse, int(length * 0.08), int(length * 0.1), amp=0.16, freq=7.0)
    sparse = _inject_segment(sparse, int(length * 0.44), int(length * 0.1), amp=0.16, freq=7.0)
    sparse = _inject_segment(sparse, int(length * 0.78), int(length * 0.1), amp=0.16, freq=7.0)

    channel = sparse.copy()
    theta = np.deg2rad(12.0)
    i_comp = channel[0].copy()
    q_comp = channel[1].copy()
    channel[0] = 1.05 * (np.cos(theta) * i_comp - np.sin(theta) * q_comp)
    channel[1] = 1.05 * (np.sin(theta) * i_comp + np.cos(theta) * q_comp)
    channel = np.roll(channel, shift=4, axis=1)

    return [
        ("Clean Signal", clean),
        ("Single-Patch Trigger", single_patch),
        ("Sparse Trigger", sparse),
        ("Sparse + Channel", channel),
    ]


def plot_trigger_schematic(outdir):
    schematics = build_schematic_signals()
    fig, axes = plt.subplots(len(schematics), 1, figsize=(12.0, 8.2), sharex=True, constrained_layout=True)
    time_axis = np.arange(schematics[0][1].shape[1])

    for ax, (title, signal) in zip(axes, schematics):
        ax.plot(time_axis, signal[0], color="#1D4E89", linewidth=1.8, label="I channel")
        ax.plot(time_axis, signal[1], color="#D1495B", linewidth=1.5, label="Q channel")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, len(time_axis) - 1)
        ax.axhline(0.0, color="#999999", linewidth=0.8, alpha=0.7)

    axes[0].legend(frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Sample Index")
    save_figure(fig, outdir, "trigger_schematic")


def save_figure(fig, outdir, stem):
    png_path = os.path.join(outdir, f"{stem}.png")
    pdf_path = os.path.join(outdir, f"{stem}.pdf")

    if hasattr(fig, "get_layout_engine") and fig.get_layout_engine() is not None:
        fig.set_layout_engine(None)

    try:
        fig.tight_layout(pad=0.8)
    except Exception:
        pass

    fig.savefig(png_path, dpi=320, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
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
    plot_asr_gap(metrics, args.outdir)
    plot_transfer_trajectory(metrics, args.outdir)
    plot_trigger_schematic(args.outdir)

    print(f"Saved figures to {args.outdir}")


if __name__ == "__main__":
    main()
