import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime


PYTHON_EXE = sys.executable
MAIN_FILE = "main.py"
LOG_DIR = "log"
RESULT_CSV = "experiment_results.csv"


def get_result_fields():
    return [
        "time",
        "exp_name",
        "checkpoint_path",
        "dataset_name",
        "model_size",
        "seed",
        "epochs",
        "batch_size",
        "test_batch_size",
        "num_workers",
        "benchmark",
        "amp",
        "main_aug_depth",
        "aux_aug_depth",
        "lr",
        "wd",
        "lambda_con",
        "clean_pretrain_epochs",
        "trigger_only_epochs",
        "poison_loss_weight",
        "aux_clean_weight",
        "joint_train_scope",
        "lambda_pos",
        "trigger_lr",
        "trigger_amp",
        "trigger_len",
        "trigger_iq_mode",
        "trigger_adaptive_amp",
        "trigger_position_mode",
        "trigger_smooth_kernel",
        "lambda_trigger_energy",
        "lambda_trigger_smooth",
        "environment_template_matching",
        "lambda_trigger_env",
        "env_template_mode",
        "env_template_seed",
        "env_low_freq_ratio",
        "env_high_freq_ratio",
        "env_template_smooth_kernel",
        "env_match_mode",
        "env_n_fft",
        "env_hop_length",
        "env_win_length",
        "target_label",
        "poison_rate",
        "clean_source_acc",
        "source_asr",
        "clean_target_acc_1",
        "target_asr_1",
        "clean_target_acc_2",
        "target_asr_2",
        "clean_target_acc_3",
        "target_asr_3",
    ]


def resolve_result_csv_path(csv_file):
    expected_fields = get_result_fields()
    if not os.path.exists(csv_file):
        return csv_file
    with open(csv_file, "r", encoding="utf-8", newline="") as file_obj:
        header = next(csv.reader(file_obj), [])
    if header == expected_fields:
        return csv_file
    base, ext = os.path.splitext(csv_file)
    version = 2
    while True:
        candidate = f"{base}_v{version}{ext}"
        if not os.path.exists(candidate):
            print(f"[INFO] Existing CSV header is outdated: {csv_file}")
            print(f"[INFO] Writing new results to: {candidate}")
            return candidate
        with open(candidate, "r", encoding="utf-8", newline="") as file_obj:
            header = next(csv.reader(file_obj), [])
        if header == expected_fields:
            print(f"[INFO] Using compatible results CSV: {candidate}")
            return candidate
        version += 1


COMMON_ARGS = {
    "dataset_name": "ORACLE",
    "mode": "train_test",
    "model_size": "S",
    "seed": 2023,
    "epochs": 2000,
    "batch_size": 32,
    "test_batch_size": 16,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "benchmark": True,
    "tensorboard": True,
    "monitor_backdoor": True,
    "monitor_interval": 5,
    "monitor_subset": 256,
    "lr": 1e-3,
    "wd": 0.0,
    "main_aug_depth": [2],
    "aux_aug_depth": [1],
    "lambda_con": [1.0, 100.0],
    "cuda": "0",
    "amp": True,
    "backdoor": True,
    "target_label": 0,
    "poison_rate": 0.01,
    "clean_pretrain_epochs": -1,
    "trigger_only_epochs": -1,
    "poison_loss_weight": 0.5,
    "aux_clean_weight": 0.0,
    "joint_train_scope": "layer3_fc",
    "trigger_amp": 0.03,
    "trigger_len": 512,
    "trigger_iq_mode": "quadrature",
    "trigger_adaptive_amp": True,
    "trigger_position_mode": "random",
    "trigger_smooth_kernel": 9,
    "trigger_lr": 5e-3,
    "lambda_pos": 0.2,
    "lambda_trigger_energy": 1e-3,
    "lambda_trigger_smooth": 1e-3,
    "environment_template_matching": True,
    "lambda_trigger_env": 0.08,
    "env_template_mode": "band_limited_noise",
    "env_template_seed": 2023,
    "env_low_freq_ratio": 0.05,
    "env_high_freq_ratio": 0.35,
    "env_template_smooth_kernel": 9,
    "env_match_mode": "spectrogram",
    "env_n_fft": 32,
    "env_hop_length": 8,
    "env_win_length": 32,
}


DATASET_DEFAULTS = {
    "ORACLE": {
        "dataset_name": "ORACLE",
        "batch_size": 32,
        "test_batch_size": 16,
        "main_aug_depth": [2],
        "aux_aug_depth": [1],
        "trigger_len": 512,
        "trigger_amp": 0.08,
        "monitor_subset": 0,
    },
    "WiSig": {
        "dataset_name": "WiSig",
        "batch_size": 64,
        "test_batch_size": 32,
        "main_aug_depth": [3],
        "aux_aug_depth": [1],
        "trigger_len": 64,
        "trigger_amp": 0.03,
        "monitor_subset": 256,
    },
}


MUST_RUN_EXPERIMENTS = [
    {
        "title": "Main results with repeated seeds",
        "items": [
            "Run the full method on ORACLE and WiSig with at least 3 random seeds.",
            "Report Clean Source Acc, Clean Target Acc, Source ASR, and Target ASR as mean +- std.",
        ],
    },
    {
        "title": "Position-independence validation",
        "items": [
            "Evaluate ASR across multiple trigger positions, not only one random placement.",
            "At minimum compare average random-position ASR with fixed-position or grouped-position ASR.",
        ],
    },
    {
        "title": "Core ablations",
        "items": [
            "Remove staged training and compare against the final method.",
            "Remove position consistency loss (lambda_pos=0).",
            "Remove environment-template matching (environment_template_matching=False, lambda_trigger_env=0).",
            "Compare multi-view trigger warmup against single-view trigger warmup if time permits.",
        ],
    },
    {
        "title": "Trigger trade-off study",
        "items": [
            "Sweep trigger length and report the CA/ASR trade-off.",
            "Sweep trigger amplitude and report the CA/ASR trade-off.",
        ],
    },
    {
        "title": "Stealth evidence",
        "items": [
            "Visualize learned trigger waveform and spectral representation.",
            "Report at least one quantitative trigger-energy or signal-to-trigger-ratio metric.",
        ],
    },
]


EXPERIMENT_VARIANTS = [
    # {
    #     "name": "clean_baseline",
    #     "description": "Clean-only SDG baseline without backdoor for paper comparison.",
    #     "overrides": {
    #         "backdoor": False,
    #         "poison_rate": 0.0,
    #         "seed": 2023,
    #     },
    # },
    # {
    #     "name": "paper_full",
    #     "description": "Full staged RF backdoor method, seed 2023.",
    #     "overrides": {"seed": 2023},
    # },
    # {
    #     "name": "no_stage",
    #     "description": "Ablation without staged training; train through a single joint phase.",
    #     "overrides": {
    #         "clean_pretrain_epochs": 0,
    #         "trigger_only_epochs": 0,
    #         "poison_loss_weight": 1.0,
    #     },
    # },
    {
        "name": "no_pos",
        "description": "Ablation without position consistency loss.",
        "overrides": {"lambda_pos": 0.0},
    },
    # {
    #     "name": "no_env",
    #     "description": "Ablation without environment-template matching.",
    #     "overrides": {
    #         "environment_template_matching": False,
    #         "lambda_trigger_env": 0.0,
    #     },
    # }
]


def build_experiments():
    experiments = []
    for dataset_name, dataset_overrides in DATASET_DEFAULTS.items():
        dataset_prefix = dataset_name.lower()
        for variant in EXPERIMENT_VARIANTS:
            exp_cfg = {
                "exp_name": f"{dataset_prefix}_{variant['name']}",
                "description": variant["description"],
                "overrides": {},
            }
            exp_cfg["overrides"].update(dataset_overrides)
            exp_cfg["overrides"].update(variant.get("overrides", {}))
            exp_cfg["overrides"].update(variant.get("dataset_overrides", {}).get(dataset_name, {}))
            experiments.append(exp_cfg)
    return experiments


EXPERIMENTS = build_experiments()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper-style backdoor experiments.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[],
        help="Run only experiments from the selected dataset(s), e.g. ORACLE WiSig.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[],
        help="Run only selected experiment names, e.g. oracle_paper_full wisig_psd_match.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments after filtering and exit.",
    )
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="Run experiments for all configured datasets instead of the default dataset in COMMON_ARGS.",
    )
    return parser.parse_args()


def build_command(common_args):
    cmd = [PYTHON_EXE, "-u", MAIN_FILE]
    for key, value in common_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            cmd.append(f"--{key}")
            cmd.extend([str(item) for item in value])
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def merge_args(base_args, overrides=None):
    merged = dict(base_args)
    if overrides:
        merged.update(overrides)
    return merged


def filter_experiments(experiments, dataset_filters=None, experiment_filters=None):
    dataset_filters = {item.strip().lower() for item in (dataset_filters or []) if item.strip()}
    experiment_filters = {item.strip() for item in (experiment_filters or []) if item.strip()}

    filtered = []
    for exp_cfg in experiments:
        dataset_name = str(exp_cfg.get("overrides", {}).get("dataset_name", COMMON_ARGS["dataset_name"]))
        exp_name = exp_cfg["exp_name"]
        if dataset_filters and dataset_name.lower() not in dataset_filters:
            continue
        if experiment_filters and exp_name not in experiment_filters:
            continue
        filtered.append(exp_cfg)
    return filtered


def print_experiments(experiments):
    for idx, exp_cfg in enumerate(experiments, start=1):
        dataset_name = exp_cfg.get("overrides", {}).get("dataset_name", COMMON_ARGS["dataset_name"])
        desc = exp_cfg.get("description", "")
        print(f"[{idx}] {exp_cfg['exp_name']} | dataset={dataset_name} | {desc}")


def print_must_run_checklist():
    print("[CHECKLIST] Must-run experiments for the final paper")
    for idx, section in enumerate(MUST_RUN_EXPERIMENTS, start=1):
        print(f"  {idx}. {section['title']}")
        for item in section["items"]:
            print(f"     - {item}")


def parse_metrics(log_text):
    metrics = {}
    patterns = {
        "checkpoint_path": r"Checkpoint path:\s*(.+)",
        "clean_source_acc": r"Clean Source Acc:\s*([0-9]*\.?[0-9]+)",
        "source_asr": r"Source ASR:\s*([0-9]*\.?[0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match:
            metrics[key] = match.group(1) if key == "checkpoint_path" else float(match.group(1))
    for idx, val in re.findall(r"Clean Target Acc #(\d+):\s*([0-9]*\.?[0-9]+)", log_text):
        metrics[f"clean_target_acc_{idx}"] = float(val)
    for idx, val in re.findall(r"Target ASR #(\d+):\s*([0-9]*\.?[0-9]+)", log_text):
        metrics[f"target_asr_{idx}"] = float(val)
    return metrics


def save_csv_row(csv_file, row_dict):
    file_exists = os.path.exists(csv_file)
    fields = get_result_fields()
    with open(csv_file, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row_dict.get(key, "") for key in fields})


def run_one_experiment(exp_cfg):
    exp_name = exp_cfg["exp_name"]
    run_args = merge_args(COMMON_ARGS, exp_cfg.get("overrides"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{timestamp}_{exp_name}.log")
    cmd = build_command(run_args)

    print("=" * 80)
    print(f"[RUN] {exp_name}")
    if exp_cfg.get("description"):
        print(f"[DESC] {exp_cfg['description']}")
    print("[CMD]", " ".join(cmd))
    print(f"[LOG] {log_path}")
    print("=" * 80)

    all_output = []
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            all_output.append(line)
        process.wait()
        if process.returncode != 0:
            print(f"\n[ERROR] Experiment failed with return code {process.returncode}")
            print(f"[ERROR] Check log file: {log_path}")
            return

    metrics = parse_metrics("".join(all_output))
    row = {
        "time": timestamp,
        "exp_name": exp_name,
        "dataset_name": run_args["dataset_name"],
        "model_size": run_args["model_size"],
        "epochs": run_args["epochs"],
        "batch_size": run_args["batch_size"],
        "test_batch_size": run_args["test_batch_size"],
        "num_workers": run_args["num_workers"],
        "benchmark": int(run_args["benchmark"]),
        "amp": int(run_args["amp"]),
        "main_aug_depth": ",".join(map(str, run_args["main_aug_depth"])),
        "aux_aug_depth": ",".join(map(str, run_args["aux_aug_depth"])),
        "lr": run_args["lr"],
        "wd": run_args["wd"],
        "lambda_con": ",".join(map(str, run_args["lambda_con"])),
        "clean_pretrain_epochs": run_args["clean_pretrain_epochs"],
        "trigger_only_epochs": run_args["trigger_only_epochs"],
        "poison_loss_weight": run_args["poison_loss_weight"],
        "aux_clean_weight": run_args["aux_clean_weight"],
        "joint_train_scope": run_args["joint_train_scope"],
        "lambda_pos": run_args["lambda_pos"],
        "trigger_lr": run_args["trigger_lr"],
        "trigger_amp": run_args["trigger_amp"],
        "trigger_len": run_args["trigger_len"],
        "trigger_iq_mode": run_args["trigger_iq_mode"],
        "trigger_adaptive_amp": int(run_args["trigger_adaptive_amp"]),
        "trigger_position_mode": run_args["trigger_position_mode"],
        "trigger_smooth_kernel": run_args["trigger_smooth_kernel"],
        "lambda_trigger_energy": run_args["lambda_trigger_energy"],
        "lambda_trigger_smooth": run_args["lambda_trigger_smooth"],
        "environment_template_matching": int(run_args["environment_template_matching"]),
        "lambda_trigger_env": run_args["lambda_trigger_env"],
        "env_template_mode": run_args["env_template_mode"],
        "env_template_seed": run_args["env_template_seed"],
        "env_low_freq_ratio": run_args["env_low_freq_ratio"],
        "env_high_freq_ratio": run_args["env_high_freq_ratio"],
        "env_template_smooth_kernel": run_args["env_template_smooth_kernel"],
        "env_match_mode": run_args["env_match_mode"],
        "env_n_fft": run_args["env_n_fft"],
        "env_hop_length": run_args["env_hop_length"],
        "env_win_length": run_args["env_win_length"],
        "target_label": run_args["target_label"],
        "poison_rate": run_args["poison_rate"],
    }
    row.update(metrics)
    save_csv_row(RESULT_CSV, row)


def main():
    global RESULT_CSV
    args = parse_args()
    ensure_dir(LOG_DIR)
    RESULT_CSV = resolve_result_csv_path(RESULT_CSV)
    dataset_filters = args.dataset
    if not dataset_filters and not args.all_datasets:
        dataset_filters = [COMMON_ARGS["dataset_name"]]
    selected_experiments = filter_experiments(EXPERIMENTS, dataset_filters, args.experiments)
    if not selected_experiments:
        raise ValueError("No experiments matched the provided filters.")
    print_must_run_checklist()
    print(f"[INFO] Total experiments: {len(selected_experiments)}")
    print_experiments(selected_experiments)
    if args.list:
        return
    for exp_cfg in selected_experiments:
        run_one_experiment(exp_cfg)


if __name__ == "__main__":
    main()
