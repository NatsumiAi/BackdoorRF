import os
import re
import csv
import sys
import subprocess
from datetime import datetime

# ========= 可改参数区域 =========
PYTHON_EXE = sys.executable        # 当前 python
MAIN_FILE = "main.py"             # 你的主训练脚本
LOG_DIR = "log"
RESULT_CSV = "experiment_results.csv"


def get_result_fields():
    return [
        "time", "exp_name",
        "checkpoint_path",
        "dataset_name", "model_size", "epochs",
        "main_aug_depth", "aux_aug_depth",
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "benchmark",
        "use_mixstyle", "mixstyle_p", "mixstyle_alpha", "mixstyle_layers", "mixstyle_mode",
        "semantic_mix", "semantic_mix_p", "semantic_mix_alpha", "lambda_sem",
        "backdoor", "target_label", "poison_rate",
        "trigger_type", "trigger_amp", "trigger_len", "trigger_pos", "trigger_freq",
        "trigger_segments", "trigger_anchor_positions", "trigger_jitter",
        "trigger_iq_mode", "trigger_adaptive_amp",
        "poison_channel_aug", "channel_phase_max_deg", "channel_scale_min",
        "channel_scale_max", "channel_shift_max", "channel_snr_db", "trigger_stage",
        "clean_source_acc", "source_asr",
        "clean_target_acc_1", "target_asr_1",
        "clean_target_acc_2", "target_asr_2",
        "clean_target_acc_3", "target_asr_3",
    ]


def resolve_result_csv_path(csv_file):
    expected_fields = get_result_fields()
    if not os.path.exists(csv_file):
        return csv_file

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])

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
        with open(candidate, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if header == expected_fields:
            print(f"[INFO] Using compatible results CSV: {candidate}")
            return candidate
        version += 1

COMMON_ARGS = {
    "dataset_name": "ORACLE",     # ORACLE / WiSig
    "mode": "train_test",
    "model_size": "S",
    "epochs": 200,
    "batch_size": 32,
    "test_batch_size": 16,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "benchmark": True,
    "tensorboard": True,
    "adapt_target_clean": False,
    "adapt_epochs": 10,
    "adapt_lr": 1e-5,
    "adapt_batch_size": 32,
    "adapt_wd": 0.0,
    "adapt_val_ratio": 0.2,
    "adapt_subset_ratio": 1.0,
    "adapt_seed": 2023,
    "adapt_save_suffix": "_adapt",
    "monitor_backdoor": True,
    "monitor_interval": 5,
    "monitor_subset": 256,
    "lr": 0.001,
    "main_aug_depth": [4],
    "aux_aug_depth": [1],
    "lambda_con": [1.0, 100.0],
    "wd": 0,
    "cuda": "0",
    "amp": True,
    "use_mixstyle": False,
    "mixstyle_p": 0.5,
    "mixstyle_alpha": 0.1,
    "mixstyle_layers": "1,2",
    "mixstyle_mode": "random",
    "semantic_mix": True,
    "semantic_mix_p": 0.5,
    "semantic_mix_alpha": 0.4,
    "lambda_sem": 0.5,
    "lambda_pos": 0.05,
    "trigger_lr": 1e-3,
}

# 默认后门参数
BACKDOOR_ARGS = {
    "target_label": 0,
    "poison_rate": 0.1,
    "trigger_type": "sparse_sine",   # sine / const / impulse / square / sparse_*
    "trigger_amp": 0.08,
    "trigger_len": 128,
    "trigger_pos": "tail",    # 稀疏 trigger 的兜底位置
    "trigger_freq": 8,
    "trigger_segments": 2,
    "trigger_anchor_positions": "head,middle,tail",
    "trigger_jitter": 16,
    "trigger_iq_mode": "quadrature",
    "trigger_adaptive_amp": True,
    "trigger_position_mode": "hybrid", # fixed / random_shift / energy_adaptive / hybrid
    "trigger_global_shift": 16,
    "trigger_hybrid_ratio": 0.2,
    "poison_channel_aug": True,
    "channel_phase_max_deg": 15.0,
    "channel_scale_min": 0.9,
    "channel_scale_max": 1.1,
    "channel_shift_max": 4,
    "channel_snr_db": 25.0,
}

# 要跑的实验列表
EXPERIMENTS = [
    # {
    #     "exp_name": "oracle_clean",
    #     "backdoor": False,
    # },
    {
        "exp_name": "oracle_sine_post",
        "backdoor": True,
        "trigger_stage": "post",
        "backdoor_overrides": {
            "trigger_type": "learnable_sparse",
            "trigger_amp": 0.08,
            "trigger_len": 256,
            "trigger_pos": "tail",
            "trigger_segments": 1,
            "trigger_anchor_positions": "tail",
            "trigger_jitter": 0,
            "trigger_adaptive_amp": False,
            "trigger_position_mode": "fixed",
            "trigger_global_shift": 0,
            "trigger_hybrid_ratio": 0.0,
            "poison_channel_aug": False,
        }
    },
    # {
    #     "exp_name": "oracle_old_sparse",
    #     "backdoor": True,
    #     "trigger_stage": "post",
    #     "backdoor_overrides": {
    #         "trigger_type": "sparse_sine",
    #         "trigger_amp": 0.12,
    #         "trigger_len": 2048,
    #         "trigger_segments": 2,
    #         "trigger_anchor_positions": "head,middle,tail",
    #         "trigger_jitter": 16,
    #         "trigger_adaptive_amp": True,
    #         "trigger_position_mode": "fixed",
    #         "trigger_global_shift": 0,
    #         "trigger_hybrid_ratio": 0.0,
    #         "poison_channel_aug": False,
    #     }
    # },
    # {
    #     "exp_name": "oracle_paper_sparse",
    #     "backdoor": True,
    #     "trigger_stage": "post",
    #     "backdoor_overrides": {
    #         "trigger_type": "sparse_sine",
    #         "trigger_amp": 0.12,
    #         "trigger_len": 2048,
    #         "trigger_segments": 2,
    #         "trigger_anchor_positions": "head,middle,tail",
    #         "trigger_jitter": 4,
    #         "trigger_adaptive_amp": True,
    #         "trigger_position_mode": "energy_adaptive",
    #         "trigger_global_shift": 8,
    #         "trigger_hybrid_ratio": 0.0,
    #         "poison_channel_aug": False,
    #     }
    # },
    # {
    #     "exp_name": "oracle_hybrid_sparse",
    #     "backdoor": True,
    #     "trigger_stage": "post",
    #     "backdoor_overrides": {
    #         "trigger_type": "sparse_sine",
    #         "trigger_amp": 0.12,
    #         "trigger_len": 2048,
    #         "trigger_segments": 2,
    #         "trigger_anchor_positions": "head,middle,tail",
    #         "trigger_jitter": 10,
    #         "trigger_adaptive_amp": True,
    #         "trigger_position_mode": "hybrid",
    #         "trigger_global_shift": 16,
    #         "trigger_hybrid_ratio": 0.2,
    #         "poison_channel_aug": False,
    #     }
    # },
]
# ===============================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_command(common_args, exp_cfg, backdoor_args):
    cmd = [PYTHON_EXE, "-u", MAIN_FILE]
    exp_backdoor_args = dict(backdoor_args)
    exp_backdoor_args.update(exp_cfg.get("backdoor_overrides", {}))

    # 通用参数
    for k, v in common_args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        elif isinstance(v, list):
            cmd.append(f"--{k}")
            cmd.extend([str(x) for x in v])
        else:
            cmd.extend([f"--{k}", str(v)])

    # 后门相关
    if exp_cfg.get("backdoor", False):
        cmd.append("--backdoor")
        cmd.extend(["--target_label", str(exp_backdoor_args["target_label"])])
        cmd.extend(["--poison_rate", str(exp_backdoor_args["poison_rate"])])
        cmd.extend(["--trigger_type", str(exp_backdoor_args["trigger_type"])])
        cmd.extend(["--trigger_amp", str(exp_backdoor_args["trigger_amp"])])
        cmd.extend(["--trigger_len", str(exp_backdoor_args["trigger_len"])])
        cmd.extend(["--trigger_pos", str(exp_backdoor_args["trigger_pos"])])
        cmd.extend(["--trigger_freq", str(exp_backdoor_args["trigger_freq"])])
        cmd.extend(["--trigger_segments", str(exp_backdoor_args["trigger_segments"])])
        cmd.extend(["--trigger_anchor_positions", str(exp_backdoor_args["trigger_anchor_positions"])])
        cmd.extend(["--trigger_jitter", str(exp_backdoor_args["trigger_jitter"])])
        cmd.extend(["--trigger_iq_mode", str(exp_backdoor_args["trigger_iq_mode"])])
        cmd.extend(["--trigger_position_mode", str(exp_backdoor_args["trigger_position_mode"])])
        cmd.extend(["--trigger_global_shift", str(exp_backdoor_args["trigger_global_shift"])])
        cmd.extend(["--trigger_hybrid_ratio", str(exp_backdoor_args["trigger_hybrid_ratio"])])
        if exp_backdoor_args.get("trigger_adaptive_amp", False):
            cmd.append("--trigger_adaptive_amp")
        if exp_backdoor_args.get("poison_channel_aug", False):
            cmd.append("--poison_channel_aug")
        cmd.extend(["--channel_phase_max_deg", str(exp_backdoor_args["channel_phase_max_deg"])])
        cmd.extend(["--channel_scale_min", str(exp_backdoor_args["channel_scale_min"])])
        cmd.extend(["--channel_scale_max", str(exp_backdoor_args["channel_scale_max"])])
        cmd.extend(["--channel_shift_max", str(exp_backdoor_args["channel_shift_max"])])
        cmd.extend(["--channel_snr_db", str(exp_backdoor_args["channel_snr_db"])])
        cmd.extend(["--trigger_stage", str(exp_cfg["trigger_stage"])])

    return cmd


def parse_metrics(log_text):
    metrics = {}

    patterns = {
        "checkpoint_path": r"Checkpoint path:\s*(.+)",
        "clean_source_acc": r"Clean Source Acc:\s*([0-9]*\.?[0-9]+)",
        "source_asr": r"Source ASR:\s*([0-9]*\.?[0-9]+)",
    }

    for k, p in patterns.items():
        m = re.search(p, log_text)
        if m:
            metrics[k] = m.group(1) if k == "checkpoint_path" else float(m.group(1))

    # 匹配多个 target 域结果
    clean_target_matches = re.findall(r"Clean Target Acc #(\d+):\s*([0-9]*\.?[0-9]+)", log_text)
    target_asr_matches = re.findall(r"Target ASR #(\d+):\s*([0-9]*\.?[0-9]+)", log_text)

    for idx, val in clean_target_matches:
        metrics[f"clean_target_acc_{idx}"] = float(val)

    for idx, val in target_asr_matches:
        metrics[f"target_asr_{idx}"] = float(val)

    return metrics


def save_csv_row(csv_file, row_dict):
    file_exists = os.path.exists(csv_file)

    base_fields = get_result_fields()

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)

        if not file_exists:
            writer.writeheader()

        # 不存在的字段补空
        final_row = {k: row_dict.get(k, "") for k in base_fields}
        writer.writerow(final_row)


def run_one_experiment(exp_cfg):
    exp_name = exp_cfg["exp_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{timestamp}_{exp_name}.log")
    exp_backdoor_args = dict(BACKDOOR_ARGS)
    exp_backdoor_args.update(exp_cfg.get("backdoor_overrides", {}))

    cmd = build_command(COMMON_ARGS, exp_cfg, BACKDOOR_ARGS)

    print("=" * 80)
    print(f"[RUN] {exp_name}")
    print("[CMD]", " ".join(cmd))
    print(f"[LOG] {log_path}")
    print("=" * 80)

    all_output = []

    with open(log_path, "w", encoding="utf-8") as log_f:
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
            log_f.write(line)
            all_output.append(line)

        process.wait()
        return_code = process.returncode
        if return_code != 0:
            print(f"\n[ERROR] Experiment failed with return code {return_code}")
            print(f"[ERROR] Check log file: {log_path}")
            return

    full_log = "".join(all_output)
    metrics = parse_metrics(full_log)

    result_row = {
        "time": timestamp,
        "exp_name": exp_name,
        "dataset_name": COMMON_ARGS["dataset_name"],
        "model_size": COMMON_ARGS["model_size"],
        "epochs": COMMON_ARGS["epochs"],
        "main_aug_depth": ",".join(map(str, COMMON_ARGS["main_aug_depth"])),
        "aux_aug_depth": ",".join(map(str, COMMON_ARGS["aux_aug_depth"])),
        "num_workers": COMMON_ARGS.get("num_workers", ""),
        "pin_memory": int(COMMON_ARGS.get("pin_memory", False)),
        "persistent_workers": int(COMMON_ARGS.get("persistent_workers", False)),
        "prefetch_factor": COMMON_ARGS.get("prefetch_factor", ""),
        "benchmark": int(COMMON_ARGS.get("benchmark", False)),
        "use_mixstyle": int(COMMON_ARGS.get("use_mixstyle", False)),
        "mixstyle_p": COMMON_ARGS.get("mixstyle_p", ""),
        "mixstyle_alpha": COMMON_ARGS.get("mixstyle_alpha", ""),
        "mixstyle_layers": COMMON_ARGS.get("mixstyle_layers", ""),
        "mixstyle_mode": COMMON_ARGS.get("mixstyle_mode", ""),
        "semantic_mix": int(COMMON_ARGS.get("semantic_mix", False)),
        "semantic_mix_p": COMMON_ARGS.get("semantic_mix_p", ""),
        "semantic_mix_alpha": COMMON_ARGS.get("semantic_mix_alpha", ""),
        "lambda_sem": COMMON_ARGS.get("lambda_sem", ""),
        "backdoor": int(exp_cfg.get("backdoor", False)),
        "target_label": exp_backdoor_args["target_label"] if exp_cfg.get("backdoor", False) else "",
        "poison_rate": exp_backdoor_args["poison_rate"] if exp_cfg.get("backdoor", False) else "",
        "trigger_type": exp_backdoor_args["trigger_type"] if exp_cfg.get("backdoor", False) else "",
        "trigger_amp": exp_backdoor_args["trigger_amp"] if exp_cfg.get("backdoor", False) else "",
        "trigger_len": exp_backdoor_args["trigger_len"] if exp_cfg.get("backdoor", False) else "",
        "trigger_pos": exp_backdoor_args["trigger_pos"] if exp_cfg.get("backdoor", False) else "",
        "trigger_freq": exp_backdoor_args["trigger_freq"] if exp_cfg.get("backdoor", False) else "",
        "trigger_segments": exp_backdoor_args["trigger_segments"] if exp_cfg.get("backdoor", False) else "",
        "trigger_anchor_positions": exp_backdoor_args["trigger_anchor_positions"] if exp_cfg.get("backdoor", False) else "",
        "trigger_jitter": exp_backdoor_args["trigger_jitter"] if exp_cfg.get("backdoor", False) else "",
        "trigger_iq_mode": exp_backdoor_args["trigger_iq_mode"] if exp_cfg.get("backdoor", False) else "",
        "trigger_adaptive_amp": int(exp_backdoor_args["trigger_adaptive_amp"]) if exp_cfg.get("backdoor", False) else "",
        "poison_channel_aug": int(exp_backdoor_args["poison_channel_aug"]) if exp_cfg.get("backdoor", False) else "",
        "channel_phase_max_deg": exp_backdoor_args["channel_phase_max_deg"] if exp_cfg.get("backdoor", False) else "",
        "channel_scale_min": exp_backdoor_args["channel_scale_min"] if exp_cfg.get("backdoor", False) else "",
        "channel_scale_max": exp_backdoor_args["channel_scale_max"] if exp_cfg.get("backdoor", False) else "",
        "channel_shift_max": exp_backdoor_args["channel_shift_max"] if exp_cfg.get("backdoor", False) else "",
        "channel_snr_db": exp_backdoor_args["channel_snr_db"] if exp_cfg.get("backdoor", False) else "",
        "trigger_stage": exp_cfg.get("trigger_stage", "") if exp_cfg.get("backdoor", False) else "",
    }

    result_row.update(metrics)
    save_csv_row(RESULT_CSV, result_row)

    print("\n[SUMMARY]")
    for k, v in result_row.items():
        if v != "":
            print(f"{k}: {v}")

    print(f"\n[OK] Result saved to: {RESULT_CSV}")
    print(f"[OK] Log saved to: {log_path}\n")


def main():
    ensure_dir(LOG_DIR)
    global RESULT_CSV
    RESULT_CSV = resolve_result_csv_path(RESULT_CSV)

    print("即将运行以下实验：")
    for exp in EXPERIMENTS:
        print(" -", exp["exp_name"])

    for exp in EXPERIMENTS:
        run_one_experiment(exp)


if __name__ == "__main__":
    main()
