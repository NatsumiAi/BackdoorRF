import argparse
import hashlib
import inspect
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from torchvision import transforms

from util.CNNmodel import MACNN
from util.augmentation import augmentations
from util.get_dataset import get_dataset
from util.learnable_trigger import LearnableSparseTrigger
from util.residual_prior import load_or_create_environment_template, environment_template_matching_loss
from util.training_monitor import TrainingMonitor, format_eta


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(2023)


def get_param_value(model_size):
    mapping = {"S": 8, "M": 16, "L": 32}
    if model_size not in mapping:
        raise ValueError(f"Invalid model_size: {model_size}. Use 'S', 'M', or 'L'.")
    return mapping[model_size]


def build_save_path(conf):
    os.makedirs("weight", exist_ok=True)
    tags = [
        f"Dataset={conf.dataset_name}",
        f"Model={conf.model_size}",
        f"mad={','.join(map(str, conf.main_aug_depth))}",
        f"aad={','.join(map(str, conf.aux_aug_depth))}",
        f"nw={conf.num_workers}",
        f"pm={int(conf.pin_memory)}",
        f"pw={int(conf.persistent_workers)}",
        f"pf={conf.prefetch_factor}",
        f"bm={int(conf.benchmark)}",
        f"bd={int(conf.backdoor)}",
        f"target={conf.target_label}",
        f"pr={conf.poison_rate}",
        f"ta={conf.trigger_amp}",
        f"tl={conf.trigger_len}",
        f"tiq={conf.trigger_iq_mode}",
        f"tad={int(conf.trigger_adaptive_amp)}",
        f"tpm={conf.trigger_position_mode}",
        f"tsk={conf.trigger_smooth_kernel}",
        f"lp={conf.lambda_pos}",
        f"lte={conf.lambda_trigger_energy}",
        f"lts={conf.lambda_trigger_smooth}",
        f"etm={int(conf.environment_template_matching)}",
        f"ltev={conf.lambda_trigger_env}",
        f"etmode={conf.env_template_mode}",
        f"ets={conf.env_template_seed}",
        f"etlow={conf.env_low_freq_ratio}",
        f"ethigh={conf.env_high_freq_ratio}",
        f"etsk={conf.env_template_smooth_kernel}",
        f"emm={conf.env_match_mode}",
        f"efft={conf.env_n_fft}",
        f"ehop={conf.env_hop_length}",
        f"ewin={conf.env_win_length}",
    ]
    digest = hashlib.sha1("_".join(tags).encode("utf-8")).hexdigest()[:10]
    return os.path.join("weight", f"Dataset={conf.dataset_name}_Model={conf.model_size}_{digest}.pth")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Paper-style learnable RF backdoor training.")
    parser.add_argument("--dataset_name", type=str, default="ORACLE", choices=["ORACLE", "WiSig"])
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "train_test"])
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--model_size", type=str, default="S", choices=["S", "M", "L"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--tb_dir", type=str, default="runs")
    parser.add_argument("--monitor_backdoor", action="store_true")
    parser.add_argument("--monitor_interval", type=int, default=5)
    parser.add_argument("--monitor_subset", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--main_aug_depth", type=int, nargs="+", default=[4])
    parser.add_argument("--aux_aug_depth", type=int, nargs="+", default=[1])
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--backdoor", action="store_true")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poison_rate", type=float, default=0.1)

    parser.add_argument("--trigger_amp", type=float, default=0.08)
    parser.add_argument("--trigger_len", type=int, default=256)
    parser.add_argument("--trigger_iq_mode", type=str, default="quadrature", choices=["quadrature", "mirror", "same"])
    parser.add_argument("--trigger_adaptive_amp", action="store_true")
    parser.add_argument("--trigger_position_mode", type=str, default="random", choices=["fixed", "random", "high_energy", "low_energy"])
    parser.add_argument("--trigger_smooth_kernel", type=int, default=9)
    parser.add_argument("--trigger_lr", type=float, default=5e-4)
    parser.add_argument("--lambda_pos", type=float, default=1.0)
    parser.add_argument("--lambda_trigger_energy", type=float, default=1e-3)
    parser.add_argument("--lambda_trigger_smooth", type=float, default=1e-3)

    parser.add_argument("--environment_template_matching", action="store_true")
    parser.add_argument("--lambda_trigger_env", type=float, default=0.1)
    parser.add_argument("--env_template_mode", type=str, default="band_limited_noise", choices=["band_limited_noise"])
    parser.add_argument("--env_template_path", type=str, default="")
    parser.add_argument("--env_template_dir", type=str, default="env_templates")
    parser.add_argument("--env_template_seed", type=int, default=2023)
    parser.add_argument("--env_low_freq_ratio", type=float, default=0.05)
    parser.add_argument("--env_high_freq_ratio", type=float, default=0.35)
    parser.add_argument("--env_template_smooth_kernel", type=int, default=9)
    parser.add_argument("--env_match_mode", type=str, default="spectrogram", choices=["spectrogram", "psd"])
    parser.add_argument("--env_n_fft", type=int, default=32)
    parser.add_argument("--env_hop_length", type=int, default=8)
    parser.add_argument("--env_win_length", type=int, default=32)
    return parser.parse_args(args)


def position_consistency_loss(embedding_a, embedding_b):
    if embedding_a.numel() == 0 or embedding_b.numel() == 0:
        return embedding_a.new_tensor(0.0)
    ref = F.normalize(embedding_a, dim=1)
    cur = F.normalize(embedding_b, dim=1)
    return F.mse_loss(cur, ref)


def set_module_trainable(module, enabled):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(enabled)


def apply_trigger_to_views(data_views, poison_flag, trigger_module, mode="random", detach_trigger=False):
    if trigger_module is None or not torch.any(poison_flag):
        return data_views
    patched = []
    for view in data_views:
        view = view.clone()
        poisoned_view = trigger_module(view[poison_flag], mode=mode)
        if detach_trigger:
            poisoned_view = poisoned_view.detach()
        view[poison_flag] = poisoned_view
        patched.append(view)
    return patched


def sample_eval_subset(x, y, max_count, seed):
    if max_count <= 0 or len(x) <= max_count:
        return x, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=max_count, replace=False)
    indices.sort()
    return x[indices], y[indices]


def make_targeted_eval_subset(x, y, target_label):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = y != int(target_label)
    return x[mask], np.full(int(mask.sum()), int(target_label), dtype=y.dtype)


def save_training_checkpoint(save_path, model, trigger_module):
    payload = {
        "model": model.state_dict(),
        "learnable_trigger": trigger_module.state_dict(),
    }
    torch.save(payload, save_path)


def load_training_checkpoint(load_path, model, trigger_module):
    payload = torch.load(load_path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Checkpoint format is incompatible with the current paper-style workflow.")
    model.load_state_dict(payload["model"])
    missing, unexpected = trigger_module.load_state_dict(payload.get("learnable_trigger", {}), strict=False)
    if missing or unexpected:
        print(f"[Warning] Learnable trigger checkpoint mismatch | missing={missing} | unexpected={unexpected}")


def sample_augmented_views(iq_data, preprocess, main_aug_depth, aux_aug_depth):
    base = np.array(iq_data, dtype=np.float32, copy=True)
    views = []
    for aug_depth in aux_aug_depth:
        view = np.array(base, dtype=np.float32, copy=True)
        if aug_depth > 0:
            for op in np.random.choice(augmentations, aug_depth):
                view = op(view)
        views.append(np.squeeze(preprocess(view.astype(np.float32))))

    main_view = np.array(base, dtype=np.float32, copy=True)
    if main_aug_depth[0] > 0:
        for op in np.random.choice(augmentations, main_aug_depth[0]):
            main_view = op(main_view)
    views.append(np.squeeze(preprocess(main_view.astype(np.float32))))
    return views


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, preprocess, main_aug_depth, aux_aug_depth, backdoor=False, target_label=0, poison_rate=0.0):
        self.dataset = [(np.squeeze(x_train[i, :, :]), int(y_train[i])) for i in range(np.shape(x_train)[0])]
        self.preprocess = preprocess
        self.main_aug_depth = main_aug_depth
        self.aux_aug_depth = aux_aug_depth
        self.backdoor = backdoor
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.poison_mask = np.zeros(len(self.dataset), dtype=bool)

        if self.backdoor and self.poison_rate > 0:
            labels = np.array([y for _, y in self.dataset])
            candidate_idx = np.where(labels != self.target_label)[0]
            num_poison = int(len(candidate_idx) * self.poison_rate)
            poison_idx = np.random.choice(candidate_idx, num_poison, replace=False)
            self.poison_mask[poison_idx] = True
            print(f"[Backdoor] poison samples: {num_poison}/{len(candidate_idx)}")
            print(f"[Backdoor] target label: {self.target_label}")

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x_aug = sample_augmented_views(x, self.preprocess, self.main_aug_depth, self.aux_aug_depth)
        return x_aug, y, int(self.poison_mask[idx])

    def __len__(self):
        return len(self.dataset)


def masked_nll_loss(log_prob, labels, mask):
    if mask is None or not torch.any(mask):
        return log_prob.new_tensor(0.0)
    return F.nll_loss(log_prob[mask], labels[mask])


def mean_poison_loss(log_prob, target_label, poison_mask):
    if poison_mask is None or not torch.any(poison_mask):
        return log_prob.new_tensor(0.0)
    poison_targets = torch.full((int(poison_mask.sum().item()),), int(target_label), dtype=torch.long, device=log_prob.device)
    return F.nll_loss(log_prob[poison_mask], poison_targets)


def mean_poison_view_loss(prob_list, target_label, poison_mask):
    if poison_mask is None or not torch.any(poison_mask):
        return prob_list[-1].new_tensor(0.0)
    losses = [mean_poison_loss(log_prob, target_label, poison_mask) for log_prob in prob_list]
    return torch.stack(losses).mean()


def train(model, train_dataloader, optimizer, scaler, epoch, conf, trigger_module, trigger_optimizer, env_template=None):
    model.train()
    trigger_module.train()
    correct = 0
    correct_total = 0
    all_loss = 0.0

    for data, target, poison_flag in train_dataloader:
        target = target.long()
        poison_flag = poison_flag.bool()

        if torch.cuda.is_available():
            data = [view.cuda() for view in data]
            target = target.cuda()
            poison_flag = poison_flag.cuda()

        clean_mask = ~poison_flag
        num_views = len(data)

        if torch.any(poison_flag):
            set_module_trainable(model, False)
            set_module_trainable(trigger_module, True)
            model.eval()
            trigger_module.train()
            trigger_optimizer.zero_grad(set_to_none=True)

            trigger_view = data[-1]
            with autocast("cuda", enabled=conf.amp and torch.cuda.is_available()):
                random_view = trigger_view.clone()
                high_view = trigger_view.clone()
                low_view = trigger_view.clone()
                random_view[poison_flag] = trigger_module(trigger_view[poison_flag], mode="random")
                high_view[poison_flag] = trigger_module(trigger_view[poison_flag], mode="high_energy")
                low_view[poison_flag] = trigger_module(trigger_view[poison_flag], mode="low_energy")

                random_embedding, random_output = model(random_view)
                high_embedding, _ = model(high_view)
                low_embedding, _ = model(low_view)

                random_log_prob = F.log_softmax(random_output, dim=1)
                trigger_loss = mean_poison_loss(random_log_prob, conf.target_label, poison_flag)
                trigger_loss = trigger_loss + conf.lambda_pos * position_consistency_loss(high_embedding[poison_flag], low_embedding[poison_flag])
                trigger_loss = trigger_loss + trigger_module.regularization_loss(
                    lambda_energy=conf.lambda_trigger_energy,
                    lambda_smooth=conf.lambda_trigger_smooth,
                )
                if conf.environment_template_matching and conf.lambda_trigger_env > 0.0:
                    env_loss = environment_template_matching_loss(
                        trigger_module.effective_pattern(),
                        env_template,
                        match_mode=conf.env_match_mode,
                        n_fft=conf.env_n_fft,
                        hop_length=conf.env_hop_length,
                        win_length=conf.env_win_length,
                    )
                    trigger_loss = trigger_loss + conf.lambda_trigger_env * env_loss

            scaler.scale(trigger_loss).backward()
            scaler.step(trigger_optimizer)
            scaler.update()

        set_module_trainable(model, True)
        set_module_trainable(trigger_module, False)
        model.train()
        trigger_module.eval()
        optimizer.zero_grad(set_to_none=True)

        poisoned_views = apply_trigger_to_views(data, poison_flag, trigger_module, mode="random", detach_trigger=True)
        data_all = torch.cat(poisoned_views, dim=0)
        with autocast("cuda", enabled=conf.amp and torch.cuda.is_available()):
            _, output = model(data_all)
            prob = F.log_softmax(output, dim=1)
            prob_list = torch.split(prob, data[0].size(0))
            clean_cls_loss = masked_nll_loss(prob_list[-1], target, clean_mask)
            poison_cls_loss = mean_poison_view_loss(prob_list, conf.target_label, poison_flag)
            result_loss = clean_cls_loss + poison_cls_loss

        scaler.scale(result_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        set_module_trainable(trigger_module, True)
        trigger_module.train()

        batch_size = data[0].size(0)
        all_loss += result_loss.item() * batch_size
        pred = prob_list[-1].argmax(dim=1, keepdim=True)
        if torch.any(clean_mask):
            correct += pred[clean_mask].eq(target[clean_mask].view_as(pred[clean_mask])).sum().item()
            correct_total += int(clean_mask.sum().item())

    print(
        "Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n".format(
            epoch,
            all_loss / len(train_dataloader.dataset),
            correct,
            max(correct_total, 1),
            100.0 * correct / max(correct_total, 1),
        )
    )
    return all_loss / len(train_dataloader.dataset), correct / max(correct_total, 1)


def evaluate(model, loss_fn, test_dataloader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            _, output = model(data)
            log_prob = F.log_softmax(output, dim=1)
            test_loss += loss_fn(log_prob, target).item() * data.size(0)
            pred = log_prob.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    print(
        "\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n".format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return test_loss, correct / len(test_dataloader.dataset)


def test(model, test_dataloader, desc="Test", target_label=None, trigger_module=None, apply_trigger=False):
    model.eval()
    if trigger_module is not None:
        trigger_module.eval()
    correct = 0
    target_pred_count = 0
    with torch.no_grad():
        for data, target in test_dataloader:
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

    num_samples = len(test_dataloader.dataset)
    acc = correct / num_samples
    if target_label is not None:
        target_pred_rate = target_pred_count / max(num_samples, 1)
        print(f"{desc}: {acc:.4f} | samples={num_samples} | pred_target_{target_label}={target_pred_rate:.4f}")
    else:
        print(f"{desc}: {acc:.4f} | samples={num_samples}")
    return acc


def monitor_backdoor_metrics(model, dataset, conf, eval_loader_kwargs, epoch, trigger_module):
    if not conf.backdoor:
        return {}

    source_x, source_y = sample_eval_subset(dataset["test_s"][0], dataset["test_s"][1], conf.monitor_subset, 2023 + epoch)
    x_bd_s, y_bd_s = make_targeted_eval_subset(source_x, source_y, conf.target_label)
    source_loader = DataLoader(TensorDataset(torch.Tensor(x_bd_s), torch.Tensor(y_bd_s)), batch_size=conf.test_batch_size, shuffle=False, **eval_loader_kwargs)
    monitor_source_asr = test(model, source_loader, desc="Monitor Source ASR", trigger_module=trigger_module, apply_trigger=True)

    target_metrics = []
    for idx, (x_t, y_t) in enumerate(dataset["test_t"]):
        x_t, y_t = sample_eval_subset(x_t, y_t, conf.monitor_subset, 4023 + epoch + idx)
        x_bd_t, y_bd_t = make_targeted_eval_subset(x_t, y_t, conf.target_label)
        target_loader = DataLoader(TensorDataset(torch.Tensor(x_bd_t), torch.Tensor(y_bd_t)), batch_size=conf.test_batch_size, shuffle=False, **eval_loader_kwargs)
        target_metrics.append(test(model, target_loader, desc=f"Monitor Target ASR #{idx + 1}", trigger_module=trigger_module, apply_trigger=True))

    result = {
        "monitor_source_asr": monitor_source_asr,
        "monitor_target_asr_mean": float(np.mean(target_metrics)) if target_metrics else math.nan,
    }
    if target_metrics:
        result["monitor_asr_gap"] = monitor_source_asr - result["monitor_target_asr_mean"]
    return result


def train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, scaler, epochs, save_path, conf, dataset, eval_loader_kwargs, trigger_module, trigger_optimizer, env_template):
    best_loss = float("inf")
    monitor = TrainingMonitor(save_path, use_tensorboard=conf.tensorboard, log_dir=conf.log_dir, tb_root=conf.tb_dir)
    train_start = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_acc = train(
                model,
                train_loader,
                optimizer,
                scaler,
                epoch,
                conf,
                trigger_module,
                trigger_optimizer,
                env_template=env_template,
            )
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            if val_loss < best_loss:
                print(f"Saving model at epoch {epoch} with loss {val_loss:.4f}")
                best_loss = val_loss
                save_training_checkpoint(save_path, model, trigger_module)

            epoch_time = time.perf_counter() - epoch_start
            elapsed = time.perf_counter() - train_start
            eta_seconds = (elapsed / max(epoch, 1)) * max(epochs - epoch, 0)
            row = {
                "epoch": epoch,
                "stage": "joint_paper",
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
                "eta_minutes": eta_seconds / 60.0,
            }
            if conf.monitor_backdoor and epoch % max(conf.monitor_interval, 1) == 0:
                row.update(monitor_backdoor_metrics(model, dataset, conf, eval_loader_kwargs, epoch, trigger_module))
            monitor.update(row)

            print(
                f"Epoch {epoch:>3}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | best_val={best_loss:.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e} | epoch_time={epoch_time:.1f}s | eta={format_eta(eta_seconds)}"
            )
            if "monitor_source_asr" in row:
                print(
                    f"Backdoor monitor | source_asr={row['monitor_source_asr']:.4f} "
                    f"| target_asr_mean={row['monitor_target_asr_mean']:.4f} "
                    f"| asr_gap={row.get('monitor_asr_gap', math.nan):.4f}"
                )
            monitor_msg = f"Training monitor updated: {monitor.history_csv_path}"
            if monitor.tensorboard_dir is not None:
                monitor_msg += f" | TensorBoard: {monitor.tensorboard_dir}"
            print(monitor_msg)
    finally:
        monitor.close()


def build_loader_kwargs(conf):
    train_loader_kwargs = {
        "num_workers": conf.num_workers,
        "pin_memory": conf.pin_memory,
    }
    if conf.num_workers > 0 and conf.persistent_workers:
        train_loader_kwargs["persistent_workers"] = True
    if conf.num_workers > 0 and "prefetch_factor" in inspect.signature(DataLoader.__init__).parameters:
        train_loader_kwargs["prefetch_factor"] = conf.prefetch_factor

    eval_loader_kwargs = {
        "num_workers": 0 if os.name == "nt" else conf.num_workers,
        "pin_memory": conf.pin_memory,
    }
    if eval_loader_kwargs["num_workers"] > 0 and "prefetch_factor" in inspect.signature(DataLoader.__init__).parameters:
        eval_loader_kwargs["prefetch_factor"] = conf.prefetch_factor
    return train_loader_kwargs, eval_loader_kwargs


def build_model(conf, num_classes):
    return MACNN(in_channels=2, channels=get_param_value(conf.model_size), num_classes=num_classes)


def build_trigger(conf):
    return LearnableSparseTrigger(
        total_length=conf.trigger_len,
        amp=conf.trigger_amp,
        iq_mode=conf.trigger_iq_mode,
        adaptive_amp=conf.trigger_adaptive_amp,
        position_mode=conf.trigger_position_mode,
        smooth_kernel=conf.trigger_smooth_kernel,
    )


def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda
    torch.backends.cudnn.benchmark = bool(conf.benchmark)
    torch.backends.cudnn.deterministic = not conf.benchmark

    num_classes = 16 if conf.dataset_name == "ORACLE" else 6
    input_shape = (2, 6000) if conf.dataset_name == "ORACLE" else (2, 256)
    save_path = build_save_path(conf)
    load_path = conf.checkpoint_path or save_path
    print(f"Checkpoint path: {save_path}")
    if conf.checkpoint_path:
        print(f"Checkpoint override for loading: {conf.checkpoint_path}")
    if conf.tensorboard:
        print(f"TensorBoard root: {conf.tb_dir}")

    dataset = get_dataset(conf.dataset_name)
    train_loader_kwargs, eval_loader_kwargs = build_loader_kwargs(conf)

    train_loader = DataLoader(
        AugDataset(
            dataset["train"][0],
            dataset["train"][1],
            transforms.ToTensor(),
            conf.main_aug_depth,
            conf.aux_aug_depth,
            backdoor=conf.backdoor,
            target_label=conf.target_label,
            poison_rate=conf.poison_rate,
        ),
        batch_size=conf.batch_size,
        shuffle=True,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        TensorDataset(torch.Tensor(dataset["val"][0]), torch.Tensor(dataset["val"][1])),
        batch_size=conf.test_batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )

    model = build_model(conf, num_classes)
    trigger_module = build_trigger(conf)
    env_template = None
    if conf.environment_template_matching and conf.lambda_trigger_env > 0.0:
        env_template, env_template_path, created = load_or_create_environment_template(
            length=conf.trigger_len,
            template_mode=conf.env_template_mode,
            template_path=conf.env_template_path,
            save_dir=conf.env_template_dir,
            seed=conf.env_template_seed,
            low_freq_ratio=conf.env_low_freq_ratio,
            high_freq_ratio=conf.env_high_freq_ratio,
            smooth_kernel=conf.env_template_smooth_kernel,
        )
        action = "Created" if created else "Loaded"
        print(
            f"{action} environment template: {tuple(env_template.shape)} | "
            f"path={env_template_path} | mode={conf.env_template_mode}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    trigger_optimizer = torch.optim.Adam(trigger_module.parameters(), lr=conf.trigger_lr, weight_decay=0.0)
    scaler = GradScaler("cuda", enabled=conf.amp and torch.cuda.is_available())
    loss_fn = nn.NLLLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        trigger_module = trigger_module.cuda()
        if env_template is not None:
            env_template = env_template.cuda()
        loss_fn = loss_fn.cuda()
        summary(model, input_shape)

    if conf.mode in ["train", "train_test"]:
        print("Starting training...")
        train_and_evaluate(
            model,
            loss_fn,
            train_loader,
            val_loader,
            optimizer,
            scaler,
            conf.epochs,
            save_path,
            conf,
            dataset,
            eval_loader_kwargs,
            trigger_module,
            trigger_optimizer,
            env_template,
        )

    if conf.mode in ["test", "train_test"]:
        model = build_model(conf, num_classes)
        trigger_module = build_trigger(conf)
        load_training_checkpoint(load_path, model, trigger_module)
        if torch.cuda.is_available():
            model = model.cuda()
            trigger_module = trigger_module.cuda()

        print("Starting clean testing on source domain...")
        clean_source_loader = DataLoader(
            TensorDataset(torch.Tensor(dataset["test_s"][0]), torch.Tensor(dataset["test_s"][1])),
            batch_size=conf.test_batch_size,
            shuffle=False,
            **eval_loader_kwargs,
        )
        test(model, clean_source_loader, desc="Clean Source Acc", target_label=conf.target_label)

        if conf.backdoor:
            print("Starting backdoor testing on source domain...")
            x_bd_s, y_bd_s = make_targeted_eval_subset(dataset["test_s"][0], dataset["test_s"][1], conf.target_label)
            bd_source_loader = DataLoader(
                TensorDataset(torch.Tensor(x_bd_s), torch.Tensor(y_bd_s)),
                batch_size=conf.test_batch_size,
                shuffle=False,
                **eval_loader_kwargs,
            )
            test(model, bd_source_loader, desc="Source ASR", target_label=conf.target_label, trigger_module=trigger_module, apply_trigger=True)

        print("Starting clean testing on target domain...")
        for idx, (x_test, y_test) in enumerate(dataset["test_t"]):
            clean_target_loader = DataLoader(
                TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)),
                batch_size=conf.test_batch_size,
                shuffle=False,
                **eval_loader_kwargs,
            )
            test(model, clean_target_loader, desc=f"Clean Target Acc #{idx + 1}", target_label=conf.target_label)

            if conf.backdoor:
                print(f"Starting backdoor testing on target domain #{idx + 1}...")
                x_bd_t, y_bd_t = make_targeted_eval_subset(x_test, y_test, conf.target_label)
                bd_target_loader = DataLoader(
                    TensorDataset(torch.Tensor(x_bd_t), torch.Tensor(y_bd_t)),
                    batch_size=conf.test_batch_size,
                    shuffle=False,
                    **eval_loader_kwargs,
                )
                test(model, bd_target_loader, desc=f"Target ASR #{idx + 1}", target_label=conf.target_label, trigger_module=trigger_module, apply_trigger=True)


if __name__ == "__main__":
    main()
