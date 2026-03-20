import argparse
import csv
import glob
import inspect
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from util.CNNmodel import MACNN
from util.backdoor import make_poisoned_eval_set
from util.get_dataset import get_dataset


FONT_FAMILY = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export t-SNE plots from real model embeddings.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--results_csv", type=str, default="experiment_results.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="weight")
    parser.add_argument("--list_experiments", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="WiSig", choices=["ORACLE", "WiSig"])
    parser.add_argument("--model_size", type=str, default="S", choices=["S", "M", "L"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_per_split", type=int, default=300)
    parser.add_argument("--target_domain_idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--outdir", type=str, default="figures")

    parser.add_argument("--use_mixstyle", action="store_true")
    parser.add_argument("--mixstyle_p", type=float, default=0.5)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)
    parser.add_argument("--mixstyle_layers", type=str, default="1,2")
    parser.add_argument("--mixstyle_mode", type=str, default="random", choices=["random", "crossdomain"])

    parser.add_argument("--include_poison", action="store_true")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--trigger_type", type=str, default="sparse_sine")
    parser.add_argument("--trigger_amp", type=float, default=0.05)
    parser.add_argument("--trigger_len", type=int, default=96)
    parser.add_argument("--trigger_pos", type=str, default="tail")
    parser.add_argument("--trigger_freq", type=int, default=8)
    parser.add_argument("--trigger_segments", type=int, default=3)
    parser.add_argument("--trigger_anchor_positions", type=str, default="head,middle,tail")
    parser.add_argument("--trigger_jitter", type=int, default=16)
    parser.add_argument("--trigger_iq_mode", type=str, default="quadrature", choices=["quadrature", "mirror", "same"])
    parser.add_argument("--trigger_adaptive_amp", action="store_true")
    return parser.parse_args()


def get_param_value(model_size):
    mapping = {"S": 8, "M": 16, "L": 32}
    if model_size not in mapping:
        raise ValueError(f"Invalid model_size: {model_size}")
    return mapping[model_size]


def to_bool(value):
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def to_int(value, default=0):
    text = str(value).strip()
    if not text:
        return default
    return int(float(text))


def to_float(value, default=0.0):
    text = str(value).strip()
    if not text:
        return default
    return float(text)


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
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "--",
        }
    )


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_layer_spec(layer_spec):
    layers = []
    for item in str(layer_spec).split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    return layers


def latest_experiment_row(results_csv, exp_name):
    rows = []
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clean_row = {k: v for k, v in row.items() if k is not None}
            if clean_row.get("exp_name", "") == exp_name:
                rows.append(clean_row)

    if not rows:
        raise ValueError(f"Experiment '{exp_name}' not found in {results_csv}")

    rows.sort(key=lambda item: item.get("time", ""))
    return rows[-1]


def list_experiments(results_csv):
    rows = []
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clean_row = {k: v for k, v in row.items() if k is not None}
            if clean_row.get("exp_name", ""):
                rows.append(clean_row)

    if not rows:
        print(f"No experiments found in {results_csv}")
        return []

    latest = {}
    for row in rows:
        name = row.get("exp_name", "")
        if name not in latest or row.get("time", "") >= latest[name].get("time", ""):
            latest[name] = row

    print(f"Available experiments in {results_csv}:")
    names = sorted(latest.keys())
    for idx, exp_name in enumerate(names, start=1):
        row = latest[exp_name]
        dataset = row.get("dataset_name", "")
        stamp = row.get("time", "")
        backdoor = row.get("backdoor", "")
        trigger_stage = row.get("trigger_stage", "")
        print(f"{idx:>2}. {exp_name} | dataset={dataset} | time={stamp} | backdoor={backdoor} | trigger_stage={trigger_stage}")

    return names


def choose_experiment_interactively(results_csv):
    names = list_experiments(results_csv)
    if not names:
        raise ValueError(f"No experiments available in {results_csv}")

    while True:
        choice = input("Select experiment number to plot: ").strip()
        if not choice:
            continue
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        index = int(choice)
        if 1 <= index <= len(names):
            return names[index - 1]

        print(f"Please choose a number between 1 and {len(names)}.")


def sync_args_from_row(args, row):
    args.dataset_name = row.get("dataset_name", args.dataset_name) or args.dataset_name
    args.model_size = row.get("model_size", args.model_size) or args.model_size

    if row.get("use_mixstyle", ""):
        args.use_mixstyle = to_bool(row.get("use_mixstyle", 0))
    if row.get("mixstyle_p", ""):
        args.mixstyle_p = to_float(row.get("mixstyle_p"), args.mixstyle_p)
    if row.get("mixstyle_alpha", ""):
        args.mixstyle_alpha = to_float(row.get("mixstyle_alpha"), args.mixstyle_alpha)
    if row.get("mixstyle_layers", ""):
        args.mixstyle_layers = row.get("mixstyle_layers")
    if row.get("mixstyle_mode", ""):
        args.mixstyle_mode = row.get("mixstyle_mode")

    if row.get("backdoor", ""):
        has_backdoor = to_bool(row.get("backdoor", 0))
        if has_backdoor:
            args.include_poison = True
    if row.get("target_label", ""):
        args.target_label = to_int(row.get("target_label"), args.target_label)
    if row.get("trigger_type", ""):
        args.trigger_type = row.get("trigger_type")
    if row.get("trigger_amp", ""):
        args.trigger_amp = to_float(row.get("trigger_amp"), args.trigger_amp)
    if row.get("trigger_len", ""):
        args.trigger_len = to_int(row.get("trigger_len"), args.trigger_len)
    if row.get("trigger_pos", ""):
        args.trigger_pos = row.get("trigger_pos")
    if row.get("trigger_freq", ""):
        args.trigger_freq = to_int(row.get("trigger_freq"), args.trigger_freq)
    if row.get("trigger_segments", ""):
        args.trigger_segments = to_int(row.get("trigger_segments"), args.trigger_segments)
    if row.get("trigger_anchor_positions", ""):
        args.trigger_anchor_positions = row.get("trigger_anchor_positions")
    if row.get("trigger_jitter", ""):
        args.trigger_jitter = to_int(row.get("trigger_jitter"), args.trigger_jitter)
    if row.get("trigger_iq_mode", ""):
        args.trigger_iq_mode = row.get("trigger_iq_mode")
    if row.get("trigger_adaptive_amp", ""):
        args.trigger_adaptive_amp = to_bool(row.get("trigger_adaptive_amp", 0))


def candidate_patterns(args, row):
    patterns = []
    base = os.path.join(
        args.checkpoint_dir,
        f"Dataset={args.dataset_name}_Model={args.model_size}_*.pth",
    )
    patterns.append(base)
    if row.get("time", "") and row.get("exp_name", ""):
        patterns.append(os.path.join(args.checkpoint_dir, f"*Dataset={args.dataset_name}_Model={args.model_size}_*.pth"))
    return patterns


def score_checkpoint(path, args, row):
    name = os.path.basename(path)
    checks = [
        f"Dataset={args.dataset_name}_",
        f"Model={args.model_size}_",
        f"backdoor={to_int(row.get('backdoor', 0))}_",
        f"tt={args.trigger_type}_" if row.get("trigger_type", "") else "",
        f"tl={args.trigger_len}_" if row.get("trigger_len", "") else "",
        f"ts={row.get('trigger_stage', '')}.pth" if row.get("trigger_stage", "") else "",
        f"ms={int(args.use_mixstyle)}_",
        f"msp={args.mixstyle_p}_",
        f"msa={args.mixstyle_alpha}_",
        f"msl={args.mixstyle_layers.replace(',', '-')}_",
        f"msm={args.mixstyle_mode}_",
    ]
    score = 0
    for token in checks:
        if token and token in name:
            score += 1
    return score


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint

    if not args.exp_name:
        raise ValueError("Provide either --checkpoint or --exp_name.")

    row = latest_experiment_row(args.results_csv, args.exp_name)
    sync_args_from_row(args, row)

    saved_path = row.get("checkpoint_path", "").strip()
    if saved_path and os.path.exists(saved_path):
        return saved_path

    candidates = []
    for pattern in candidate_patterns(args, row):
        candidates.extend(glob.glob(pattern))

    if not candidates:
        raise ValueError(f"No checkpoint files found in {args.checkpoint_dir} for experiment '{args.exp_name}'")

    candidates = sorted(set(candidates), key=lambda item: (score_checkpoint(item, args, row), os.path.getmtime(item)), reverse=True)
    return candidates[0]


def build_model(args, num_classes):
    model = MACNN(
        in_channels=2,
        channels=get_param_value(args.model_size),
        num_classes=num_classes,
        use_mixstyle=args.use_mixstyle,
        mixstyle_p=args.mixstyle_p,
        mixstyle_alpha=args.mixstyle_alpha,
        mixstyle_layers=parse_layer_spec(args.mixstyle_layers),
        mixstyle_mode=args.mixstyle_mode,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def sample_subset(x, y, max_count, seed):
    if len(x) <= max_count:
        return x, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=max_count, replace=False)
    indices.sort()
    return x[indices], y[indices]


def get_embeddings(model, x, y, batch_size, split_name):
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
            embedding, _ = model(batch_x)
            outputs.append(embedding.cpu().numpy())
            labels.append(batch_y.numpy())
    emb = np.concatenate(outputs, axis=0)
    lab = np.concatenate(labels, axis=0)
    tags = np.array([split_name] * len(lab))
    return emb, lab, tags


def trigger_cfg(args):
    return {
        "trigger_type": args.trigger_type,
        "amp": args.trigger_amp,
        "length": args.trigger_len,
        "pos": args.trigger_pos,
        "freq": args.trigger_freq,
        "num_segments": args.trigger_segments,
        "anchors": args.trigger_anchor_positions,
        "jitter": args.trigger_jitter,
        "adaptive_amp": args.trigger_adaptive_amp,
        "iq_mode": args.trigger_iq_mode,
    }


def collect_sets(args, dataset):
    sets = []
    x_source, y_source = sample_subset(dataset["test_s"][0], dataset["test_s"][1], args.max_per_split, args.seed + 1)
    sets.append((x_source, y_source, "Clean Source"))

    target_idx = max(1, args.target_domain_idx) - 1
    x_target, y_target = dataset["test_t"][target_idx]
    x_target, y_target = sample_subset(x_target, y_target, args.max_per_split, args.seed + 2)
    sets.append((x_target, y_target, f"Clean Target {target_idx + 1}"))

    if args.include_poison:
        x_bd_s, y_bd_s = make_poisoned_eval_set(x_source, y_source, args.target_label, trigger_cfg(args))
        x_bd_t, y_bd_t = make_poisoned_eval_set(x_target, y_target, args.target_label, trigger_cfg(args))
        sets.append((x_bd_s, y_bd_s, "Poisoned Source"))
        sets.append((x_bd_t, y_bd_t, f"Poisoned Target {target_idx + 1}"))

    return sets


def run_tsne(embeddings, perplexity, seed, n_iter):
    max_perplexity = max(1.0, float(len(embeddings) - 1))
    tsne_kwargs = dict(
        n_components=2,
        perplexity=min(perplexity, max_perplexity),
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    tsne_signature = inspect.signature(TSNE.__init__)
    if "n_iter" in tsne_signature.parameters:
        tsne_kwargs["n_iter"] = n_iter
    elif "max_iter" in tsne_signature.parameters:
        tsne_kwargs["max_iter"] = n_iter

    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(embeddings)


def plot_tsne(points, labels, split_tags, outdir, dataset_name, include_poison):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

    unique_labels = np.unique(labels)
    palette = plt.cm.tab20(np.linspace(0.05, 0.95, max(len(unique_labels), 2)))
    label_to_color = {label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)}

    split_order = []
    for tag in split_tags:
        if tag not in split_order:
            split_order.append(tag)

    markers = {
        "Clean Source": "o",
        "Clean Target 1": "^",
        "Clean Target 2": "^",
        "Clean Target 3": "^",
        "Poisoned Source": "s",
        "Poisoned Target 1": "D",
        "Poisoned Target 2": "D",
        "Poisoned Target 3": "D",
    }

    for cls in unique_labels:
        mask = labels == cls
        axes[0].scatter(points[mask, 0], points[mask, 1], s=18, color=label_to_color[cls], alpha=0.75, label=f"Class {int(cls)}")
    axes[0].set_title(f"Real Embeddings by Class ({dataset_name})")
    axes[0].set_xlabel("t-SNE axis 1")
    axes[0].set_ylabel("t-SNE axis 2")
    axes[0].legend(frameon=False, ncol=2, loc="best")

    for tag in split_order:
        mask = split_tags == tag
        axes[1].scatter(
            points[mask, 0],
            points[mask, 1],
            s=30 if "Poisoned" in tag else 18,
            marker=markers.get(tag, "o"),
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            label=tag,
        )
    title_suffix = "with Backdoor Views" if include_poison else "Source/Target Shift"
    axes[1].set_title(f"Real Embeddings by Split ({title_suffix})")
    axes[1].set_xlabel("t-SNE axis 1")
    axes[1].set_ylabel("t-SNE axis 2")
    axes[1].legend(frameon=False, loc="best")

    stem = "embedding_tsne_poison" if include_poison else "embedding_tsne_clean"
    save_figure(fig, outdir, stem)


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


def main():
    args = parse_args()
    if args.list_experiments:
        list_experiments(args.results_csv)
        return

    if not args.checkpoint and not args.exp_name:
        args.exp_name = choose_experiment_interactively(args.results_csv)
        print(f"Selected experiment: {args.exp_name}")

    set_seed(args.seed)
    ensure_dir(args.outdir)
    configure_style()

    args.checkpoint = resolve_checkpoint(args)

    dataset = get_dataset(args.dataset_name)
    num_classes = 16 if args.dataset_name == "ORACLE" else 6
    model = build_model(args, num_classes)

    all_embeddings = []
    all_labels = []
    all_tags = []
    for idx, (x, y, name) in enumerate(collect_sets(args, dataset)):
        emb, lab, tags = get_embeddings(model, x, y, args.batch_size, name)
        all_embeddings.append(emb)
        all_labels.append(lab)
        all_tags.append(tags)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    split_tags = np.concatenate(all_tags, axis=0)

    tsne_points = run_tsne(embeddings, args.perplexity, args.seed, args.n_iter)
    plot_tsne(tsne_points, labels, split_tags, args.outdir, args.dataset_name, args.include_poison)

    print(f"Resolved checkpoint: {args.checkpoint}")
    print(f"Saved t-SNE figures to {args.outdir}")


if __name__ == "__main__":
    main()
