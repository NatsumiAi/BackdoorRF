import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


FONT_FAMILY = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ASA-style semantic augmentation mechanisms.")
    parser.add_argument("--outdir", type=str, default="figures")
    return parser.parse_args()


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


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_figure(fig, outdir, stem):
    png_path = os.path.join(outdir, f"{stem}.png")
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    try:
        fig.tight_layout(pad=0.8)
    except Exception:
        pass
    fig.savefig(png_path, dpi=320, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


def generate_domains(num=80, seed=2026):
    rng = np.random.default_rng(seed)
    source_a = rng.normal(loc=[-1.5, 0.6], scale=[0.38, 0.26], size=(num, 2))
    source_b = rng.normal(loc=[1.3, 0.7], scale=[0.35, 0.24], size=(num, 2))
    target_a = rng.normal(loc=[-0.8, -0.8], scale=[0.45, 0.32], size=(num, 2))
    target_b = rng.normal(loc=[1.9, -0.6], scale=[0.42, 0.30], size=(num, 2))
    return source_a, source_b, target_a, target_b


def semantic_mix(points_a, points_b, alpha=0.55):
    count = min(len(points_a), len(points_b))
    lam = np.linspace(alpha, 1.0 - alpha, count).reshape(-1, 1)
    return lam * points_a[:count] + (1.0 - lam) * points_b[:count]


def plot_semantic_space(outdir):
    source_a, source_b, target_a, target_b = generate_domains()
    asa_a = semantic_mix(source_a, target_a, alpha=0.2)
    asa_b = semantic_mix(source_b, target_b, alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)

    axes[0].scatter(source_a[:, 0], source_a[:, 1], s=24, color="#355C7D", alpha=0.75, label="Source class A")
    axes[0].scatter(source_b[:, 0], source_b[:, 1], s=24, color="#C06C84", alpha=0.75, label="Source class B")
    axes[0].scatter(target_a[:, 0], target_a[:, 1], s=24, color="#6C8EAD", alpha=0.65, marker="^", label="Target class A")
    axes[0].scatter(target_b[:, 0], target_b[:, 1], s=24, color="#F67280", alpha=0.65, marker="^", label="Target class B")
    axes[0].set_title("Before ASA: Domain Shift in Semantic Space")
    axes[0].set_xlabel("Semantic axis 1")
    axes[0].set_ylabel("Semantic axis 2")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")

    axes[1].scatter(source_a[:, 0], source_a[:, 1], s=18, color="#355C7D", alpha=0.28)
    axes[1].scatter(source_b[:, 0], source_b[:, 1], s=18, color="#C06C84", alpha=0.28)
    axes[1].scatter(target_a[:, 0], target_a[:, 1], s=18, color="#6C8EAD", alpha=0.25, marker="^")
    axes[1].scatter(target_b[:, 0], target_b[:, 1], s=18, color="#F67280", alpha=0.25, marker="^")
    axes[1].scatter(asa_a[:, 0], asa_a[:, 1], s=34, color="#2A9D8F", label="ASA class A")
    axes[1].scatter(asa_b[:, 0], asa_b[:, 1], s=34, color="#E9C46A", label="ASA class B")
    axes[1].set_title("After ASA: Interpolated Cross-Domain Semantics")
    axes[1].set_xlabel("Semantic axis 1")
    axes[1].set_ylabel("Semantic axis 2")
    axes[1].legend(frameon=False, loc="upper center")

    save_figure(fig, outdir, "asa_semantic_space")


def build_feature_maps():
    x = np.linspace(-1, 1, 36)
    y = np.linspace(-1, 1, 72)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    source = np.exp(-((xx + 0.25) ** 2 + (yy - 0.15) ** 2) / 0.18) + 0.55 * np.exp(-((xx - 0.4) ** 2 + (yy + 0.35) ** 2) / 0.12)
    target = 0.8 * np.exp(-((xx + 0.05) ** 2 + (yy + 0.05) ** 2) / 0.22) + 0.65 * np.exp(-((xx - 0.48) ** 2 + (yy - 0.28) ** 2) / 0.16)
    asa = 0.55 * source + 0.45 * target
    return source, target, asa


def plot_feature_maps(outdir):
    source, target, asa = build_feature_maps()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)

    titles = ["Source semantic map", "Target semantic map", "ASA interpolated map"]
    data_list = [source, target, asa]
    for ax, title, data in zip(axes, titles, data_list):
        image = ax.imshow(data, cmap="magma", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    save_figure(fig, outdir, "asa_feature_maps")


def plot_pipeline(outdir):
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.axis("off")

    boxes = [
        (0.05, 0.35, 0.18, 0.3, "Source / Target\nIQ Samples", "#355C7D"),
        (0.29, 0.35, 0.18, 0.3, "Semantic Encoder\n(backbone)", "#6C8EAD"),
        (0.53, 0.35, 0.18, 0.3, "ASA Interpolation\nfeature mixing", "#2A9D8F"),
        (0.77, 0.35, 0.18, 0.3, "Robust Classifier\n+ DG objective", "#C06C84"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.9, edgecolor="none", transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=12)

    for start, end in [(0.23, 0.29), (0.47, 0.53), (0.71, 0.77)]:
        ax.annotate(
            "",
            xy=(end, 0.5),
            xytext=(start, 0.5),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#264653"},
        )

    ax.text(0.5, 0.12, "ASA aligns cross-domain semantics by interpolating domain-sensitive features into smoother, more transferable class manifolds.",
            transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#264653")
    save_figure(fig, outdir, "asa_pipeline")


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    configure_style()
    plot_semantic_space(args.outdir)
    plot_feature_maps(args.outdir)
    plot_pipeline(args.outdir)
    print(f"Saved ASA figures to {args.outdir}")


if __name__ == "__main__":
    main()
