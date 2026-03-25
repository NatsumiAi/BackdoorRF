import argparse
import csv
import os
import subprocess
import sys


PYTHON_EXE = sys.executable
MAIN_FILE = "main.py"

# ========= 可改参数区域 =========
# 这里修改通过 test_from_results.py 触发 target clean adaptation 的默认参数。
ADAPT_DEFAULTS = {
    "adapt_target_clean": True,
    "adapt_epochs": 100,
    "adapt_lr": 1e-4,
    "adapt_batch_size": 32,
    "adapt_wd": 0.0,
    "adapt_val_ratio": 0.2,
    "adapt_subset_ratio": 1.0,
    "adapt_seed": 2023,
    "adapt_save_suffix": "_adapt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Select an experiment from results CSV and run test mode.")
    parser.add_argument("--csv", type=str, default="experiment_results.csv")
    parser.add_argument("--list_only", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--adapt_target_clean", action="store_true", default=ADAPT_DEFAULTS["adapt_target_clean"])
    parser.add_argument("--adapt_epochs", type=int, default=ADAPT_DEFAULTS["adapt_epochs"])
    parser.add_argument("--adapt_lr", type=float, default=ADAPT_DEFAULTS["adapt_lr"])
    parser.add_argument("--adapt_batch_size", type=int, default=ADAPT_DEFAULTS["adapt_batch_size"])
    parser.add_argument("--adapt_wd", type=float, default=ADAPT_DEFAULTS["adapt_wd"])
    parser.add_argument("--adapt_val_ratio", type=float, default=ADAPT_DEFAULTS["adapt_val_ratio"])
    parser.add_argument("--adapt_subset_ratio", type=float, default=ADAPT_DEFAULTS["adapt_subset_ratio"])
    parser.add_argument("--adapt_seed", type=int, default=ADAPT_DEFAULTS["adapt_seed"])
    parser.add_argument("--adapt_save_suffix", type=str, default=ADAPT_DEFAULTS["adapt_save_suffix"])
    return parser.parse_args()


def read_latest_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clean_row = {k: v for k, v in row.items() if k is not None}
            if clean_row.get("exp_name", ""):
                rows.append(clean_row)

    latest = {}
    for row in rows:
        key = (row.get("dataset_name", ""), row.get("exp_name", ""))
        if key not in latest or row.get("time", "") >= latest[key].get("time", ""):
            latest[key] = row

    return sorted(latest.values(), key=lambda item: (item.get("dataset_name", ""), item.get("exp_name", "")))


def to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def append_arg(cmd, key, value):
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    cmd.extend([f"--{key}", text])


def append_bool(cmd, key, row_key, row):
    if to_bool(row.get(row_key, "")):
        cmd.append(f"--{key}")


def build_test_command(row, args):
    cmd = [PYTHON_EXE, "-u", MAIN_FILE, "--mode", "test"]

    append_arg(cmd, "checkpoint_path", row.get("checkpoint_path", ""))
    append_arg(cmd, "dataset_name", row.get("dataset_name", ""))
    append_arg(cmd, "model_size", row.get("model_size", ""))

    append_arg(cmd, "num_workers", row.get("num_workers", "0"))
    append_bool(cmd, "pin_memory", "pin_memory", row)
    append_bool(cmd, "persistent_workers", "persistent_workers", row)
    append_arg(cmd, "prefetch_factor", row.get("prefetch_factor", ""))
    append_bool(cmd, "benchmark", "benchmark", row)

    append_bool(cmd, "use_mixstyle", "use_mixstyle", row)
    append_arg(cmd, "mixstyle_p", row.get("mixstyle_p", ""))
    append_arg(cmd, "mixstyle_alpha", row.get("mixstyle_alpha", ""))
    append_arg(cmd, "mixstyle_layers", row.get("mixstyle_layers", ""))
    append_arg(cmd, "mixstyle_mode", row.get("mixstyle_mode", ""))

    append_bool(cmd, "semantic_mix", "semantic_mix", row)
    append_arg(cmd, "semantic_mix_p", row.get("semantic_mix_p", ""))
    append_arg(cmd, "semantic_mix_alpha", row.get("semantic_mix_alpha", ""))
    append_arg(cmd, "lambda_sem", row.get("lambda_sem", ""))

    if to_bool(row.get("backdoor", "")):
        cmd.append("--backdoor")
        append_arg(cmd, "target_label", row.get("target_label", ""))
        append_arg(cmd, "poison_rate", row.get("poison_rate", ""))
        append_arg(cmd, "trigger_type", row.get("trigger_type", ""))
        append_arg(cmd, "trigger_amp", row.get("trigger_amp", ""))
        append_arg(cmd, "trigger_len", row.get("trigger_len", ""))
        append_arg(cmd, "trigger_pos", row.get("trigger_pos", ""))
        append_arg(cmd, "trigger_freq", row.get("trigger_freq", ""))
        append_arg(cmd, "trigger_segments", row.get("trigger_segments", ""))
        append_arg(cmd, "trigger_anchor_positions", row.get("trigger_anchor_positions", ""))
        append_arg(cmd, "trigger_jitter", row.get("trigger_jitter", ""))
        append_arg(cmd, "trigger_iq_mode", row.get("trigger_iq_mode", ""))
        append_bool(cmd, "trigger_adaptive_amp", "trigger_adaptive_amp", row)
        append_arg(cmd, "trigger_stage", row.get("trigger_stage", ""))
        append_arg(cmd, "trigger_position_mode", row.get("trigger_position_mode", ""))
        append_arg(cmd, "trigger_global_shift", row.get("trigger_global_shift", ""))
        append_arg(cmd, "trigger_hybrid_ratio", row.get("trigger_hybrid_ratio", ""))
        append_bool(cmd, "poison_channel_aug", "poison_channel_aug", row)
        append_arg(cmd, "channel_phase_max_deg", row.get("channel_phase_max_deg", ""))
        append_arg(cmd, "channel_scale_min", row.get("channel_scale_min", ""))
        append_arg(cmd, "channel_scale_max", row.get("channel_scale_max", ""))
        append_arg(cmd, "channel_shift_max", row.get("channel_shift_max", ""))
        append_arg(cmd, "channel_snr_db", row.get("channel_snr_db", ""))

    if args.adapt_target_clean:
        cmd.append("--adapt_target_clean")
        append_arg(cmd, "adapt_epochs", args.adapt_epochs)
        append_arg(cmd, "adapt_lr", args.adapt_lr)
        append_arg(cmd, "adapt_batch_size", args.adapt_batch_size)
        append_arg(cmd, "adapt_wd", args.adapt_wd)
        append_arg(cmd, "adapt_val_ratio", args.adapt_val_ratio)
        append_arg(cmd, "adapt_subset_ratio", args.adapt_subset_ratio)
        append_arg(cmd, "adapt_seed", args.adapt_seed)
        append_arg(cmd, "adapt_save_suffix", args.adapt_save_suffix)

    return cmd


def choose_row(rows):
    print(f"Available experiments in results CSV:")
    for idx, row in enumerate(rows, start=1):
        dataset = row.get("dataset_name", "")
        exp_name = row.get("exp_name", "")
        stamp = row.get("time", "")
        ckpt = os.path.basename(row.get("checkpoint_path", ""))
        print(f"{idx:>2}. {dataset} | {exp_name} | {stamp} | {ckpt}")

    while True:
        text = input("Select experiment number to test: ").strip()
        if not text:
            continue
        if text.isdigit():
            index = int(text)
            if 1 <= index <= len(rows):
                return rows[index - 1]
        print(f"Please choose a number between 1 and {len(rows)}.")


def ask_adaptation(args):
    prompt = (
        "Enable target clean adaptation? [y/N] "
        f"(epochs={args.adapt_epochs}, lr={args.adapt_lr}, batch={args.adapt_batch_size}, "
        f"subset={args.adapt_subset_ratio}, val_ratio={args.adapt_val_ratio}): "
    )
    choice = input(prompt).strip().lower()
    if choice in {"y", "yes"}:
        args.adapt_target_clean = True
    elif choice in {"n", "no", ""}:
        args.adapt_target_clean = False
    else:
        print("Unrecognized choice, using default: no adaptation.")
        args.adapt_target_clean = False


def main():
    args = parse_args()
    rows = read_latest_rows(args.csv)
    if not rows:
        raise ValueError(f"No experiments found in {args.csv}")

    if args.list_only:
        for idx, row in enumerate(rows, start=1):
            dataset = row.get("dataset_name", "")
            exp_name = row.get("exp_name", "")
            stamp = row.get("time", "")
            ckpt = os.path.basename(row.get("checkpoint_path", ""))
            print(f"{idx:>2}. {dataset} | {exp_name} | {stamp} | {ckpt}")
        return

    if args.index > 0:
        if args.index > len(rows):
            raise ValueError(f"Index {args.index} out of range. Found {len(rows)} experiments.")
        row = rows[args.index - 1]
    else:
        row = choose_row(rows)
        ask_adaptation(args)

    checkpoint_path = row.get("checkpoint_path", "")
    if checkpoint_path and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    cmd = build_test_command(row, args)
    print("Running test command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
