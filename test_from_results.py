import argparse
import csv
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from main import (
    build_loader_kwargs,
    build_model,
    build_trigger,
    load_training_checkpoint,
    make_targeted_eval_subset,
    parse_args,
    save_training_checkpoint,
)
from util.get_dataset import get_dataset


RESULT_FILES = [
    "experiment_results_paper.csv",
    "experiment_results_v2.csv",
    "experiment_results.csv",
]

ADAPT_DEFAULTS = {
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "batch_size": 32,
    "val_ratio": 0.2,
    "subset_ratio": 1.0,
    "seed": 2023,
    "target_index": 1,
    "save_adapted": False,
}


def str_to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def pick_results_csv(path=""):
    if path:
        return path
    for candidate in RESULT_FILES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("No results CSV found.")


def load_experiments(csv_path):
    latest = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if not row or not row.get("exp_name"):
                continue
            key = (row.get("dataset_name", ""), row["exp_name"])
            latest[key] = row
    experiments = [latest[key] for key in sorted(latest.keys())]
    if not experiments:
        raise ValueError(f"No experiments found in {csv_path}")
    return experiments


def print_experiments(experiments):
    for idx, row in enumerate(experiments, start=1):
        print(
            f"[{idx}] dataset={row.get('dataset_name', '')} | exp={row.get('exp_name', '')} | "
            f"checkpoint={row.get('checkpoint_path', '')}"
        )


def select_experiment(experiments, index=0):
    if index:
        if index < 1 or index > len(experiments):
            raise ValueError(f"Index out of range: {index}")
        return experiments[index - 1]
    print_experiments(experiments)
    picked = int(input("Select experiment index: ").strip())
    if picked < 1 or picked > len(experiments):
        raise ValueError(f"Index out of range: {picked}")
    return experiments[picked - 1]


def fill_conf_from_row(row):
    conf = parse_args([])
    conf.dataset_name = row.get("dataset_name", conf.dataset_name)
    conf.model_size = row.get("model_size", conf.model_size)
    conf.epochs = int(row.get("epochs") or conf.epochs)
    conf.batch_size = int(row.get("batch_size") or conf.batch_size)
    conf.test_batch_size = int(row.get("test_batch_size") or conf.test_batch_size)
    conf.num_workers = int(row.get("num_workers") or conf.num_workers)
    conf.benchmark = str_to_bool(row.get("benchmark", conf.benchmark))
    conf.amp = str_to_bool(row.get("amp", conf.amp))
    conf.lr = float(row.get("lr") or conf.lr)
    conf.wd = float(row.get("wd") or conf.wd)
    conf.lambda_pos = float(row.get("lambda_pos") or conf.lambda_pos)
    conf.trigger_lr = float(row.get("trigger_lr") or conf.trigger_lr)
    conf.trigger_amp = float(row.get("trigger_amp") or conf.trigger_amp)
    conf.trigger_len = int(row.get("trigger_len") or conf.trigger_len)
    conf.trigger_iq_mode = row.get("trigger_iq_mode", conf.trigger_iq_mode)
    conf.trigger_adaptive_amp = str_to_bool(row.get("trigger_adaptive_amp", conf.trigger_adaptive_amp))
    conf.trigger_position_mode = row.get("trigger_position_mode", conf.trigger_position_mode)
    conf.trigger_smooth_kernel = int(row.get("trigger_smooth_kernel") or conf.trigger_smooth_kernel)
    conf.lambda_trigger_energy = float(row.get("lambda_trigger_energy") or conf.lambda_trigger_energy)
    conf.lambda_trigger_smooth = float(row.get("lambda_trigger_smooth") or conf.lambda_trigger_smooth)
    conf.environment_template_matching = str_to_bool(
        row.get("environment_template_matching", conf.environment_template_matching)
    )
    conf.lambda_trigger_env = float(row.get("lambda_trigger_env") or conf.lambda_trigger_env)
    conf.env_template_mode = row.get("env_template_mode", conf.env_template_mode)
    conf.env_template_seed = int(row.get("env_template_seed") or conf.env_template_seed)
    conf.env_low_freq_ratio = float(row.get("env_low_freq_ratio") or conf.env_low_freq_ratio)
    conf.env_high_freq_ratio = float(row.get("env_high_freq_ratio") or conf.env_high_freq_ratio)
    conf.env_template_smooth_kernel = int(
        row.get("env_template_smooth_kernel") or conf.env_template_smooth_kernel
    )
    conf.env_match_mode = row.get("env_match_mode", conf.env_match_mode)
    conf.env_n_fft = int(row.get("env_n_fft") or conf.env_n_fft)
    conf.env_hop_length = int(row.get("env_hop_length") or conf.env_hop_length)
    conf.env_win_length = int(row.get("env_win_length") or conf.env_win_length)
    conf.target_label = int(row.get("target_label") or conf.target_label)
    conf.poison_rate = float(row.get("poison_rate") or conf.poison_rate)
    main_aug = row.get("main_aug_depth")
    if main_aug:
        conf.main_aug_depth = [int(item) for item in str(main_aug).split(",") if item != ""]
    aux_aug = row.get("aux_aug_depth")
    if aux_aug:
        conf.aux_aug_depth = [int(item) for item in str(aux_aug).split(",") if item != ""]
    conf.backdoor = True
    conf.mode = "test"
    conf.checkpoint_path = row.get("checkpoint_path", "")
    return conf


def build_eval_loader(x, y, batch_size, loader_kwargs):
    return DataLoader(
        TensorDataset(torch.Tensor(x), torch.Tensor(y)),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )


def evaluate_once(model, loader, desc, target_label=None, trigger_module=None, apply_trigger=False):
    model.eval()
    if trigger_module is not None:
        trigger_module.eval()
    correct = 0
    target_pred_count = 0
    with torch.no_grad():
        for data, target in loader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            if apply_trigger and trigger_module is not None:
                data = trigger_module(data, mode="random")
            _, output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if target_label is not None:
                target_pred_count += (pred.view(-1) == int(target_label)).sum().item()

    num_samples = len(loader.dataset)
    acc = correct / max(num_samples, 1)
    target_pred_rate = None if target_label is None else target_pred_count / max(num_samples, 1)
    if target_pred_rate is None:
        print(f"{desc}: {acc:.4f} | samples={num_samples}")
    else:
        print(f"{desc}: {acc:.4f} | samples={num_samples} | pred_target_{target_label}={target_pred_rate:.4f}")
    return acc, target_pred_rate, num_samples


def evaluate_repeated_random_asr(model, loader, desc, target_label, trigger_module, repeats):
    metrics = []
    pred_rates = []
    sample_count = len(loader.dataset)
    for repeat_idx in range(1, repeats + 1):
        acc, pred_rate, _ = evaluate_once(
            model,
            loader,
            desc=f"{desc} [run {repeat_idx}/{repeats}]",
            target_label=target_label,
            trigger_module=trigger_module,
            apply_trigger=True,
        )
        metrics.append(acc)
        pred_rates.append(0.0 if pred_rate is None else pred_rate)

    metric_array = np.array(metrics, dtype=np.float64)
    pred_array = np.array(pred_rates, dtype=np.float64)
    print(
        f"{desc} Summary: mean={metric_array.mean():.4f} | std={metric_array.std(ddof=0):.4f} | "
        f"min={metric_array.min():.4f} | max={metric_array.max():.4f} | samples={sample_count} | "
        f"pred_target_{target_label}_mean={pred_array.mean():.4f}"
    )
    return metric_array


def evaluate_backdoor_suite(model, trigger_module, dataset, conf, loader_kwargs, prefix="", random_repeats=1):
    if prefix:
        print(prefix)
    clean_source_loader = build_eval_loader(dataset["test_s"][0], dataset["test_s"][1], conf.test_batch_size, loader_kwargs)
    evaluate_once(model, clean_source_loader, desc="Clean Source Acc", target_label=conf.target_label)

    x_bd_s, y_bd_s = make_targeted_eval_subset(dataset["test_s"][0], dataset["test_s"][1], conf.target_label)
    bd_source_loader = build_eval_loader(x_bd_s, y_bd_s, conf.test_batch_size, loader_kwargs)
    evaluate_repeated_random_asr(model, bd_source_loader, desc="Source ASR", target_label=conf.target_label, trigger_module=trigger_module, repeats=random_repeats)

    for idx, (x_test, y_test) in enumerate(dataset["test_t"]):
        clean_target_loader = build_eval_loader(x_test, y_test, conf.test_batch_size, loader_kwargs)
        evaluate_once(model, clean_target_loader, desc=f"Clean Target Acc #{idx + 1}", target_label=conf.target_label)
        x_bd_t, y_bd_t = make_targeted_eval_subset(x_test, y_test, conf.target_label)
        bd_target_loader = build_eval_loader(x_bd_t, y_bd_t, conf.test_batch_size, loader_kwargs)
        evaluate_repeated_random_asr(
            model,
            bd_target_loader,
            desc=f"Target ASR #{idx + 1}",
            target_label=conf.target_label,
            trigger_module=trigger_module,
            repeats=random_repeats,
        )


def adapt_on_target_clean(model, dataset, conf, loader_kwargs, adapt_cfg):
    target_idx = max(1, int(adapt_cfg["target_index"])) - 1
    if "adapt_t" not in dataset or not dataset["adapt_t"]:
        raise ValueError(f"Dataset {conf.dataset_name} does not provide target adaptation data.")
    if target_idx >= len(dataset["adapt_t"]):
        raise ValueError(f"Target index out of range: {adapt_cfg['target_index']}")

    x_adapt, y_adapt = dataset["adapt_t"][target_idx]
    total = len(x_adapt)
    subset_ratio = float(adapt_cfg["subset_ratio"])
    if subset_ratio < 1.0:
        keep = max(1, int(total * subset_ratio))
        generator = torch.Generator().manual_seed(int(adapt_cfg["seed"]))
        perm = torch.randperm(total, generator=generator)[:keep].numpy()
        x_adapt = x_adapt[perm]
        y_adapt = y_adapt[perm]

    adapt_dataset = TensorDataset(torch.Tensor(x_adapt), torch.Tensor(y_adapt))
    val_ratio = float(adapt_cfg["val_ratio"])
    val_size = max(1, int(len(adapt_dataset) * val_ratio)) if len(adapt_dataset) > 1 else 0
    train_size = len(adapt_dataset) - val_size
    if train_size <= 0:
        train_size = len(adapt_dataset)
        val_size = 0
    generator = torch.Generator().manual_seed(int(adapt_cfg["seed"]))
    if val_size > 0:
        train_dataset, val_dataset = random_split(adapt_dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = adapt_dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(adapt_cfg["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=conf.test_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(adapt_cfg["lr"]), weight_decay=float(adapt_cfg["weight_decay"]))

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    print(
        f"[Adapt] target_index={adapt_cfg['target_index']} | train_samples={train_size} | "
        f"val_samples={val_size} | epochs={adapt_cfg['epochs']} | lr={adapt_cfg['lr']}"
    )
    for epoch in range(1, int(adapt_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for data, target in train_loader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad(set_to_none=True)
            _, output = model(data)
            log_prob = F.log_softmax(output, dim=1)
            loss = F.nll_loss(log_prob, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = log_prob.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_count += data.size(0)

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)

        val_loss = float("nan")
        val_acc = float("nan")
        if val_loader is not None:
            model.eval()
            correct = 0
            count = 0
            total = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    target = target.long()
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()
                    _, output = model(data)
                    log_prob = F.log_softmax(output, dim=1)
                    loss = F.nll_loss(log_prob, target)
                    total += loss.item() * data.size(0)
                    pred = log_prob.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    count += data.size(0)
            val_loss = total / max(count, 1)
            val_acc = correct / max(count, 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
        else:
            best_state = deepcopy(model.state_dict())

        print(
            f"[Adapt] epoch {epoch:>3}/{adapt_cfg['epochs']} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}"
            + (f" | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}" if val_loader is not None else "")
        )

    model.load_state_dict(best_state)
    return model


def parse_script_args():
    parser = argparse.ArgumentParser(description="Select a past experiment, optionally adapt on target clean data, and evaluate backdoor metrics.")
    parser.add_argument("--results_csv", type=str, default="")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--list_experiments", action="store_true")
    parser.add_argument("--enable_adapt", action="store_true")
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--adapt_epochs", type=int, default=ADAPT_DEFAULTS["epochs"])
    parser.add_argument("--adapt_lr", type=float, default=ADAPT_DEFAULTS["lr"])
    parser.add_argument("--adapt_wd", type=float, default=ADAPT_DEFAULTS["weight_decay"])
    parser.add_argument("--adapt_batch_size", type=int, default=ADAPT_DEFAULTS["batch_size"])
    parser.add_argument("--adapt_val_ratio", type=float, default=ADAPT_DEFAULTS["val_ratio"])
    parser.add_argument("--adapt_subset_ratio", type=float, default=ADAPT_DEFAULTS["subset_ratio"])
    parser.add_argument("--adapt_seed", type=int, default=ADAPT_DEFAULTS["seed"])
    parser.add_argument("--adapt_target_index", type=int, default=ADAPT_DEFAULTS["target_index"])
    parser.add_argument("--save_adapted", action="store_true")
    return parser.parse_args()


def main():
    args = parse_script_args()
    csv_path = pick_results_csv(args.results_csv)
    experiments = load_experiments(csv_path)
    if args.list_experiments:
        print_experiments(experiments)
        return

    row = select_experiment(experiments, args.index)
    conf = fill_conf_from_row(row)
    dataset = get_dataset(conf.dataset_name)
    _, eval_loader_kwargs = build_loader_kwargs(conf)

    model = build_model(conf, 16 if conf.dataset_name == "ORACLE" else 6)
    trigger_module = build_trigger(conf)
    load_training_checkpoint(conf.checkpoint_path, model, trigger_module)
    if torch.cuda.is_available():
        model = model.cuda()
        trigger_module = trigger_module.cuda()

    print(f"Selected experiment: {row.get('exp_name', '')} | dataset={conf.dataset_name}")
    print(f"Checkpoint: {conf.checkpoint_path}")
    print(f"Random ASR repeats: {args.random_repeats}")
    print("\n[Before adaptation]")
    evaluate_backdoor_suite(model, trigger_module, dataset, conf, eval_loader_kwargs, random_repeats=max(1, args.random_repeats))

    if args.enable_adapt:
        adapt_cfg = {
            "epochs": args.adapt_epochs,
            "lr": args.adapt_lr,
            "weight_decay": args.adapt_wd,
            "batch_size": args.adapt_batch_size,
            "val_ratio": args.adapt_val_ratio,
            "subset_ratio": args.adapt_subset_ratio,
            "seed": args.adapt_seed,
            "target_index": args.adapt_target_index,
        }
        model = adapt_on_target_clean(model, dataset, conf, eval_loader_kwargs, adapt_cfg)
        print("\n[After target clean adaptation]")
        evaluate_backdoor_suite(model, trigger_module, dataset, conf, eval_loader_kwargs, random_repeats=max(1, args.random_repeats))
        if args.save_adapted:
            base, ext = os.path.splitext(conf.checkpoint_path)
            save_path = f"{base}_adapt_t{args.adapt_target_index}{ext or '.pth'}"
            save_training_checkpoint(save_path, model, trigger_module)
            print(f"Saved adapted checkpoint: {save_path}")


if __name__ == "__main__":
    main()
