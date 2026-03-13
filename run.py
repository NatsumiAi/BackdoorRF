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

COMMON_ARGS = {
    "dataset_name": "WiSig",     # ORACLE / WiSig
    "mode": "train_test",
    "model_size": "S",
    "epochs": 100,
    "batch_size": 32,
    "test_batch_size": 16,
    "lr": 0.001,
    "main_aug_depth": [4],
    "aux_aug_depth": [1],
    "lambda_con": [1.0, 100.0],
    "wd": 0,
    "cuda": "0",
    "amp": True
}

# 默认后门参数
BACKDOOR_ARGS = {
    "target_label": 0,
    "poison_rate": 0.1,
    "trigger_type": "sine",   # sine / const / impulse / square
    "trigger_amp": 0.05,
    "trigger_len": 96,
    "trigger_pos": "tail",    # head / middle / tail / random
    "trigger_freq": 8
}

# 要跑的实验列表
EXPERIMENTS = [
    {
        "exp_name": "baseline_clean",
        "backdoor": False
    },
    # {
    #     "exp_name": "backdoor_post",
    #     "backdoor": True,
    #     "trigger_stage": "post"
    # },
    # {
    #     "exp_name": "backdoor_pre",
    #     "backdoor": True,
    #     "trigger_stage": "pre"
    # }
]
# ===============================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_command(common_args, exp_cfg, backdoor_args):
    cmd = [PYTHON_EXE, MAIN_FILE]

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
        cmd.extend(["--target_label", str(backdoor_args["target_label"])])
        cmd.extend(["--poison_rate", str(backdoor_args["poison_rate"])])
        cmd.extend(["--trigger_type", str(backdoor_args["trigger_type"])])
        cmd.extend(["--trigger_amp", str(backdoor_args["trigger_amp"])])
        cmd.extend(["--trigger_len", str(backdoor_args["trigger_len"])])
        cmd.extend(["--trigger_pos", str(backdoor_args["trigger_pos"])])
        cmd.extend(["--trigger_freq", str(backdoor_args["trigger_freq"])])
        cmd.extend(["--trigger_stage", str(exp_cfg["trigger_stage"])])

    return cmd


def parse_metrics(log_text):
    metrics = {}

    patterns = {
        "clean_source_acc": r"Clean Source Acc:\s*([0-9]*\.?[0-9]+)",
        "source_asr": r"Source ASR:\s*([0-9]*\.?[0-9]+)",
    }

    for k, p in patterns.items():
        m = re.search(p, log_text)
        if m:
            metrics[k] = float(m.group(1))

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

    # 收集所有可能字段
    base_fields = [
        "time", "exp_name",
        "dataset_name", "model_size", "epochs",
        "main_aug_depth", "aux_aug_depth",
        "backdoor", "target_label", "poison_rate",
        "trigger_type", "trigger_amp", "trigger_len", "trigger_pos", "trigger_freq", "trigger_stage",
        "clean_source_acc", "source_asr",
        "clean_target_acc_1", "target_asr_1",
        "clean_target_acc_2", "target_asr_2",
        "clean_target_acc_3", "target_asr_3",
    ]

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
            universal_newlines=True
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
        "backdoor": int(exp_cfg.get("backdoor", False)),
        "target_label": BACKDOOR_ARGS["target_label"] if exp_cfg.get("backdoor", False) else "",
        "poison_rate": BACKDOOR_ARGS["poison_rate"] if exp_cfg.get("backdoor", False) else "",
        "trigger_type": BACKDOOR_ARGS["trigger_type"] if exp_cfg.get("backdoor", False) else "",
        "trigger_amp": BACKDOOR_ARGS["trigger_amp"] if exp_cfg.get("backdoor", False) else "",
        "trigger_len": BACKDOOR_ARGS["trigger_len"] if exp_cfg.get("backdoor", False) else "",
        "trigger_pos": BACKDOOR_ARGS["trigger_pos"] if exp_cfg.get("backdoor", False) else "",
        "trigger_freq": BACKDOOR_ARGS["trigger_freq"] if exp_cfg.get("backdoor", False) else "",
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

    print("即将运行以下实验：")
    for exp in EXPERIMENTS:
        print(" -", exp["exp_name"])

    for exp in EXPERIMENTS:
        run_one_experiment(exp)


if __name__ == "__main__":
    main()