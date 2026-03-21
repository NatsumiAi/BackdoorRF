import os
import argparse
import hashlib
import inspect
import time
import torch
import numpy as np
import random
import math
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from util.get_dataset import get_dataset
from util.CNNmodel import *
from util.augmentation import augmentations
from util.channel_aug import apply_channel_perturbation
from torchsummary import summary
from util.con_losses import SupConLoss
from util.backdoor import add_trigger, make_poisoned_eval_set
from util.training_monitor import TrainingMonitor, format_eta

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

setup_seed(2023)

def get_param_value(model_size: str) -> int:
    """Returns the parameter value based on the input size: S, M, or L."""
    model_size_mapping = {'S': 8, 'M': 16, 'L': 32}
    if model_size in model_size_mapping:
        return model_size_mapping[model_size]
    else:
        raise ValueError(f"Invalid model_size: {model_size}. Use 'S', 'M', or 'L'.")


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
        f"lam={','.join(map(str, conf.lambda_con))}",
        f"bd={int(conf.backdoor)}",
        f"target={conf.target_label}",
        f"pr={conf.poison_rate}",
        f"tt={conf.trigger_type}",
        f"ta={conf.trigger_amp}",
        f"tl={conf.trigger_len}",
        f"tp={conf.trigger_pos}",
        f"tf={conf.trigger_freq}",
        f"tseg={conf.trigger_segments}",
        f"tj={conf.trigger_jitter}",
        f"tiq={conf.trigger_iq_mode}",
        f"tad={int(conf.trigger_adaptive_amp)}",
        f"tpm={conf.trigger_position_mode}",
        f"tgs={conf.trigger_global_shift}",
        f"thr={conf.trigger_hybrid_ratio}",
        f"pca={int(conf.poison_channel_aug)}",
        f"cph={conf.channel_phase_max_deg}",
        f"cs={conf.channel_scale_min}-{conf.channel_scale_max}",
        f"csh={conf.channel_shift_max}",
        f"csnr={conf.channel_snr_db}",
        f"ms={int(conf.use_mixstyle)}",
        f"msp={conf.mixstyle_p}",
        f"msa={conf.mixstyle_alpha}",
        f"msl={conf.mixstyle_layers.replace(',', '-')}",
        f"msm={conf.mixstyle_mode}",
        f"sm={int(conf.semantic_mix)}",
        f"smp={conf.semantic_mix_p}",
        f"sma={conf.semantic_mix_alpha}",
        f"ls={conf.lambda_sem}",
        f"lp={conf.lambda_pos}",
        f"ts={conf.trigger_stage}",
    ]
    long_name = "_".join(tags)
    digest = hashlib.sha1(long_name.encode("utf-8")).hexdigest()[:10]
    short_name = f"Dataset={conf.dataset_name}_Model={conf.model_size}_{digest}.pth"
    return os.path.join("weight", short_name)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Single-source Domain Generalization")
    parser.add_argument("--dataset_name", type=str, default="ORACLE", choices=["ORACLE", "WiSig"])   
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "train_test"])
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--model_size", type=str, default="S")
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--monitor_backdoor", action="store_true")
    parser.add_argument("--monitor_interval", type=int, default=5)
    parser.add_argument("--monitor_subset", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--main_aug_depth", type=int, nargs='+', default=[2])
    parser.add_argument("--aux_aug_depth", type=int, nargs='+', default=[1])
    parser.add_argument("--lambda_con", type=float, nargs='+', default=[1.0, 100.0])
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--backdoor", action="store_true")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    # 后门攻击参数
    parser.add_argument("--trigger_type", type=str, default="sine",
                        choices=[
                            "sine", "const", "impulse", "square",
                            "sparse_sine", "sparse_const", "sparse_impulse", "sparse_square"
                        ])
    parser.add_argument("--trigger_amp", type=float, default=0.05)
    parser.add_argument("--trigger_len", type=int, default=64)
    parser.add_argument("--trigger_pos", type=str, default="tail",
                        choices=["head", "middle", "tail", "random"])
    parser.add_argument("--trigger_freq", type=int, default=8)
    parser.add_argument("--trigger_segments", type=int, default=3)
    parser.add_argument("--trigger_anchor_positions", type=str, default="head,middle,tail")
    parser.add_argument("--trigger_jitter", type=int, default=8)
    parser.add_argument("--trigger_iq_mode", type=str, default="quadrature",
                        choices=["quadrature", "mirror", "same"])
    parser.add_argument("--trigger_adaptive_amp", action="store_true")
    parser.add_argument("--trigger_position_mode", type=str, default="fixed",
                        choices=["fixed", "random_shift", "energy_adaptive", "hybrid"])
    parser.add_argument("--trigger_global_shift", type=int, default=0)
    parser.add_argument("--trigger_hybrid_ratio", type=float, default=0.25)
    parser.add_argument("--poison_channel_aug", action="store_true")
    parser.add_argument("--channel_phase_max_deg", type=float, default=15.0)
    parser.add_argument("--channel_scale_min", type=float, default=0.9)
    parser.add_argument("--channel_scale_max", type=float, default=1.1)
    parser.add_argument("--channel_shift_max", type=int, default=4)
    parser.add_argument("--channel_snr_db", type=float, default=25.0)
    parser.add_argument("--use_mixstyle", action="store_true")
    parser.add_argument("--mixstyle_p", type=float, default=0.5)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)
    parser.add_argument("--mixstyle_layers", type=str, default="1,2")
    parser.add_argument("--mixstyle_mode", type=str, default="random",
                        choices=["random", "crossdomain"])
    parser.add_argument("--semantic_mix", action="store_true")
    parser.add_argument("--semantic_mix_p", type=float, default=0.5)
    parser.add_argument("--semantic_mix_alpha", type=float, default=0.4)
    parser.add_argument("--lambda_sem", type=float, default=0.5)
    parser.add_argument("--lambda_pos", type=float, default=0.0)

    # trigger 什么时候加：pre/post augmentation or EOT sampling
    parser.add_argument("--trigger_stage", type=str, default="post",
                        choices=["pre", "post", "eot"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def parse_layer_spec(layer_spec):
    layers = []
    for item in str(layer_spec).split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(int(item))
    return layers


def soft_cross_entropy(logits, soft_targets):
    log_prob = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()


def position_consistency_loss(embedding_list, poison_mask):
    if poison_mask is None or not torch.any(poison_mask) or len(embedding_list) < 2:
        device = embedding_list[-1].device if embedding_list else poison_mask.device
        return torch.tensor(0.0, device=device)

    ref = F.normalize(embedding_list[-1][poison_mask], dim=1)
    losses = []
    for view_embedding in embedding_list[:-1]:
        cur = F.normalize(view_embedding[poison_mask], dim=1)
        losses.append(F.mse_loss(cur, ref))

    if not losses:
        return torch.tensor(0.0, device=ref.device)
    return torch.stack(losses).mean()


def semantic_mixup(embeddings, labels, num_classes, alpha=0.4, prob=0.5):
    if alpha <= 0.0 or prob <= 0.0 or embeddings.size(0) < 2:
        return None

    if random.random() > prob:
        return None

    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample((embeddings.size(0), 1)).to(embeddings.device)
    perm = torch.randperm(embeddings.size(0), device=embeddings.device)

    mixed_embeddings = F.normalize(lam * embeddings + (1.0 - lam) * embeddings[perm], dim=1)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    mixed_targets = lam * one_hot + (1.0 - lam) * one_hot[perm]
    return mixed_embeddings, mixed_targets


def sample_eval_subset(x, y, max_count, seed):
    if max_count <= 0 or len(x) <= max_count:
        return x, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=max_count, replace=False)
    indices.sort()
    return x[indices], y[indices]


def monitor_backdoor_metrics(model, dataset, conf, eval_loader_kwargs, trigger_cfg, epoch):
    if not conf.backdoor:
        return {}

    source_x, source_y = sample_eval_subset(
        dataset['test_s'][0],
        dataset['test_s'][1],
        conf.monitor_subset,
        2023 + epoch,
    )
    x_bd_s, y_bd_s = make_poisoned_eval_set(source_x, source_y, conf.target_label, trigger_cfg)
    source_loader = DataLoader(
        TensorDataset(torch.Tensor(x_bd_s), torch.Tensor(y_bd_s)),
        batch_size=conf.test_batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )
    monitor_source_asr = test(model, source_loader, desc="Monitor Source ASR")

    target_metrics = []
    for idx, (x_t, y_t) in enumerate(dataset['test_t']):
        x_t, y_t = sample_eval_subset(x_t, y_t, conf.monitor_subset, 4023 + epoch + idx)
        x_bd_t, y_bd_t = make_poisoned_eval_set(x_t, y_t, conf.target_label, trigger_cfg)
        target_loader = DataLoader(
            TensorDataset(torch.Tensor(x_bd_t), torch.Tensor(y_bd_t)),
            batch_size=conf.test_batch_size,
            shuffle=False,
            **eval_loader_kwargs,
        )
        target_metrics.append(test(model, target_loader, desc=f"Monitor Target ASR #{idx + 1}"))

    result = {
        "monitor_source_asr": monitor_source_asr,
        "monitor_target_asr_mean": float(np.mean(target_metrics)) if target_metrics else math.nan,
    }
    if target_metrics:
        result["monitor_asr_gap"] = monitor_source_asr - result["monitor_target_asr_mean"]
    return result


def _sample_augmented_view(base, preprocess, aug_depth, aug_list,
                           poison=False, trigger_cfg=None, channel_cfg=None,
                           trigger_stage="post"):
    iq_data_aug = np.array(base, dtype=np.float32, copy=True)

    if poison and trigger_cfg is not None and trigger_stage == "eot":
        iq_data_aug = add_trigger(iq_data_aug, **trigger_cfg)
        if channel_cfg is not None:
            iq_data_aug = apply_channel_perturbation(iq_data_aug, **channel_cfg)

    if aug_depth != 0:
        sampled_ops = np.random.choice(aug_list, aug_depth)
        for op in sampled_ops:
            iq_data_aug = op(iq_data_aug)

    if poison and trigger_cfg is not None and trigger_stage == "post":
        iq_data_aug = add_trigger(iq_data_aug, **trigger_cfg)
        if channel_cfg is not None:
            iq_data_aug = apply_channel_perturbation(iq_data_aug, **channel_cfg)

    iq_data_aug = np.squeeze(preprocess(iq_data_aug.astype(np.float32)))
    return iq_data_aug


def aug(iq_data, preprocess, main_aug_depth, aux_aug_depth,
        poison=False, trigger_cfg=None, channel_cfg=None, trigger_stage="post"):
    aug_list = augmentations
    iq_data_aug_list = []

    base = np.array(iq_data, dtype=np.float32, copy=True)

    # 先加 trigger，再增强
    if poison and trigger_cfg is not None and trigger_stage == "pre":
        base = add_trigger(base, **trigger_cfg)
        if channel_cfg is not None:
            base = apply_channel_perturbation(base, **channel_cfg)

    # aux views
    for i in range(len(aux_aug_depth)):
        view_base = iq_data if trigger_stage == "eot" else base
        iq_data_aug = _sample_augmented_view(
            view_base,
            preprocess,
            aux_aug_depth[i],
            aug_list,
            poison=poison,
            trigger_cfg=trigger_cfg,
            channel_cfg=channel_cfg,
            trigger_stage=trigger_stage,
        )
        iq_data_aug_list.append(iq_data_aug)

    # main view
    view_base = iq_data if trigger_stage == "eot" else base
    iq_data_aug = _sample_augmented_view(
        view_base,
        preprocess,
        main_aug_depth[0],
        aug_list,
        poison=poison,
        trigger_cfg=trigger_cfg,
        channel_cfg=channel_cfg,
        trigger_stage=trigger_stage,
    )
    iq_data_aug_list.append(iq_data_aug)

    return iq_data_aug_list

class AugDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, preprocess, main_aug_depth, aux_aug_depth,
                 backdoor=False, target_label=0, poison_rate=0.0,
                 trigger_cfg=None, channel_cfg=None, trigger_stage="post"):
        self.dataset = []
        for i in range(np.shape(x_train)[0]):
            self.dataset.append((np.squeeze(x_train[i, :, :]), int(y_train[i])))

        self.preprocess = preprocess
        self.main_aug_depth = main_aug_depth
        self.aux_aug_depth = aux_aug_depth

        self.backdoor = backdoor
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.trigger_cfg = trigger_cfg
        self.channel_cfg = channel_cfg
        self.trigger_stage = trigger_stage

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
        poison = self.backdoor and self.poison_mask[idx]

        if poison:
            y = self.target_label

        x_aug = aug(
            x, self.preprocess,
            self.main_aug_depth, self.aux_aug_depth,
            poison=poison,
            trigger_cfg=self.trigger_cfg,
            channel_cfg=self.channel_cfg,
            trigger_stage=self.trigger_stage
        )
        return x_aug, y, int(poison)

    def __len__(self):
        return len(self.dataset)

def train(model, loss, train_dataloader, optimizer, scaler, epoch, conf):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        data, target, poison_flag = data_nn
        target = target.long()
        poison_flag = poison_flag.bool()
        domain_target = []
        target_all = []
        num_data= len(conf.main_aug_depth) + len(conf.aux_aug_depth)
        for i in range(num_data):
            domain_target.append(i*torch.ones(data[0].size(0)).long())   
            target_all.append(target)   

        if torch.cuda.is_available():
            data_all = torch.cat(data, 0).cuda()
            target = target.cuda()
            poison_flag = poison_flag.cuda()
            target_all = torch.cat(target_all, 0).cuda()
            domain_target= torch.cat(domain_target, 0).cuda()
        # AMP加速
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=conf.amp and torch.cuda.is_available()):
            embedding, output = model(data_all)
            prob = F.log_softmax(output, dim=1)
            porb_list = torch.split(prob, data[0].size(0))
            embedding_list = torch.split(embedding, data[0].size(0))
            cls_loss = loss[0](porb_list[num_data - 1], target)
            con_loss = loss[1](embedding.unsqueeze(1), target_all, adv=False)
            adv_con_loss = loss[1](embedding.unsqueeze(1), domain_target, adv=True)
            sem_loss = torch.tensor(0.0, device=embedding.device)
            pos_loss = position_consistency_loss(embedding_list, poison_flag)

            if conf.semantic_mix:
                sem_pack = semantic_mixup(
                    embedding_list[num_data - 1],
                    target,
                    num_classes=conf.num_classes,
                    alpha=conf.semantic_mix_alpha,
                    prob=conf.semantic_mix_p,
                )
                if sem_pack is not None:
                    mixed_embeddings, mixed_targets = sem_pack
                    mixed_logits = model.classify_embedding(mixed_embeddings)
                    sem_loss = soft_cross_entropy(mixed_logits, mixed_targets)

            result_loss = (
                cls_loss
                + conf.lambda_con[0] * con_loss
                + conf.lambda_con[1] * adv_con_loss
                + conf.lambda_sem * sem_loss
                + conf.lambda_pos * pos_loss
            )

        scaler.scale(result_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # -------------
        batch_size = data[0].size(0)
        all_loss += result_loss.item()*batch_size
        pred = porb_list[num_data -  1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset),
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    return all_loss / len(train_dataloader.dataset), correct / len(train_dataloader.dataset)

def evaluate(model, loss, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding, output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += loss[0](output, target).item()*data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return test_loss, correct / len(test_dataloader.dataset)

def test(model, test_dataloader, desc="Test", target_label=None):
    model.eval()
    correct = 0
    target_pred_count = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding, output = model(data)
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


def train_and_evaluate(model, loss, train_loader, val_loader, optimizer, scaler, epochs, save_path, conf,
                       dataset=None, eval_loader_kwargs=None, trigger_cfg=None):
    """Train and evaluate the model, saving the best model."""
    best_loss = float('inf')
    conf.best_val_loss = best_loss
    model.best_val_loss = best_loss
    monitor = TrainingMonitor(
        save_path,
        use_tensorboard=conf.tensorboard,
        log_dir=conf.log_dir,
        tb_root=conf.tb_dir,
    )
    train_start = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_acc = train(model, loss, train_loader, optimizer, scaler, epoch, conf)
            val_loss, val_acc = evaluate(model, loss, val_loader, epoch)
            if val_loss < best_loss:
                print(f"Saving model at epoch {epoch} with loss {val_loss:.4f}")
                best_loss = val_loss
                conf.best_val_loss = best_loss
                model.best_val_loss = best_loss
                torch.save(model.state_dict(), save_path)
            else:
                conf.best_val_loss = best_loss
                model.best_val_loss = best_loss

            epoch_time = time.perf_counter() - epoch_start
            elapsed = time.perf_counter() - train_start
            avg_epoch_time = elapsed / epoch
            eta_seconds = avg_epoch_time * max(epochs - epoch, 0)
            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_loss,
                "lr": current_lr,
                "epoch_time": epoch_time,
                "eta_minutes": eta_seconds / 60.0,
            }

            if (
                conf.monitor_backdoor
                and dataset is not None
                and eval_loader_kwargs is not None
                and trigger_cfg is not None
                and epoch % max(conf.monitor_interval, 1) == 0
            ):
                row.update(monitor_backdoor_metrics(model, dataset, conf, eval_loader_kwargs, trigger_cfg, epoch))

            monitor.update(row)

            print(
                f"Epoch {epoch:>3}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | best_val={best_loss:.4f} "
                f"| lr={current_lr:.2e} | epoch_time={epoch_time:.1f}s | eta={format_eta(eta_seconds)}"
            )
            if "monitor_source_asr" in row:
                monitor_msg = (
                    f"Backdoor monitor | source_asr={row['monitor_source_asr']:.4f} "
                    f"| target_asr_mean={row['monitor_target_asr_mean']:.4f} "
                    f"| asr_gap={row.get('monitor_asr_gap', math.nan):.4f}"
                )
                print(monitor_msg)
            monitor_msg = f"Training monitor updated: {monitor.history_csv_path} | {monitor.history_plot_path}"
            if monitor.tensorboard_dir is not None:
                monitor_msg += f" | TensorBoard: {monitor.tensorboard_dir}"
            print(monitor_msg)
    finally:
        monitor.close()

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

    trigger_cfg = {
        "trigger_type": conf.trigger_type,
        "amp": conf.trigger_amp,
        "length": conf.trigger_len,
        "pos": conf.trigger_pos,
        "freq": conf.trigger_freq,
        "num_segments": conf.trigger_segments,
        "anchors": conf.trigger_anchor_positions,
        "jitter": conf.trigger_jitter,
        "adaptive_amp": conf.trigger_adaptive_amp,
        "iq_mode": conf.trigger_iq_mode,
        "position_mode": conf.trigger_position_mode,
        "global_shift": conf.trigger_global_shift,
        "hybrid_ratio": conf.trigger_hybrid_ratio,
    }
    channel_cfg = {
        "enable": conf.poison_channel_aug,
        "phase_max_deg": conf.channel_phase_max_deg,
        "scale_min": conf.channel_scale_min,
        "scale_max": conf.channel_scale_max,
        "shift_max": conf.channel_shift_max,
        "snr_db": conf.channel_snr_db,
    }
    
    dataset = get_dataset(conf.dataset_name)
    conf.num_classes = num_classes
    train_loader_kwargs = {
        "num_workers": conf.num_workers,
        "pin_memory": conf.pin_memory,
    }
    if conf.num_workers > 0 and conf.persistent_workers:
        train_loader_kwargs["persistent_workers"] = True
    if conf.num_workers > 0:
        data_loader_sig = inspect.signature(DataLoader.__init__)
        if "prefetch_factor" in data_loader_sig.parameters:
            train_loader_kwargs["prefetch_factor"] = conf.prefetch_factor

    eval_loader_kwargs = {
        "num_workers": 0 if os.name == "nt" else conf.num_workers,
        "pin_memory": conf.pin_memory,
    }
    if eval_loader_kwargs["num_workers"] > 0:
        data_loader_sig = inspect.signature(DataLoader.__init__)
        if "prefetch_factor" in data_loader_sig.parameters:
            eval_loader_kwargs["prefetch_factor"] = conf.prefetch_factor

    train_loader = DataLoader(
        AugDataset(
            dataset['train'][0], dataset['train'][1],
            transforms.ToTensor(),
            conf.main_aug_depth, conf.aux_aug_depth,
            backdoor=conf.backdoor,
            target_label=conf.target_label,
            poison_rate=conf.poison_rate,
            trigger_cfg=trigger_cfg,
            channel_cfg=channel_cfg,
            trigger_stage=conf.trigger_stage
        ),
        batch_size=conf.batch_size,
        shuffle=True,
        **train_loader_kwargs,
    )

    val_loader = DataLoader(TensorDataset(torch.Tensor(dataset['val'][0]), torch.Tensor(dataset['val'][1])), 
                            batch_size=conf.test_batch_size, shuffle=True, **eval_loader_kwargs)

    mixstyle_layers = parse_layer_spec(conf.mixstyle_layers)
    model = MACNN(
        in_channels=2,
        channels=get_param_value(conf.model_size),
        num_classes=num_classes,
        use_mixstyle=conf.use_mixstyle,
        mixstyle_p=conf.mixstyle_p,
        mixstyle_alpha=conf.mixstyle_alpha,
        mixstyle_layers=mixstyle_layers,
        mixstyle_mode=conf.mixstyle_mode,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    scaler = GradScaler("cuda", enabled=conf.amp and torch.cuda.is_available())
    cls_loss = nn.NLLLoss()
    con_loss = SupConLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        cls_loss = cls_loss.cuda()
        con_loss = con_loss.cuda()
        summary(model, input_shape)

    loss = cls_loss, con_loss

    if conf.mode in ["train", "train_test"]:
        print("Starting training...")
        train_and_evaluate(
            model,
            loss,
            train_loader,
            val_loader,
            optimizer,
            scaler,
            conf.epochs,
            save_path,
            conf,
            dataset=dataset,
            eval_loader_kwargs=eval_loader_kwargs,
            trigger_cfg=trigger_cfg,
        )

    if conf.mode in ["test", "train_test"]:
        # 读取模型
        model = MACNN(
            in_channels=2,
            channels=get_param_value(conf.model_size),
            num_classes=num_classes,
            use_mixstyle=conf.use_mixstyle,
            mixstyle_p=conf.mixstyle_p,
            mixstyle_alpha=conf.mixstyle_alpha,
            mixstyle_layers=mixstyle_layers,
            mixstyle_mode=conf.mixstyle_mode,
        )
        state_dict = torch.load(load_path, map_location="cpu")
        model.load_state_dict(state_dict)

        if torch.cuda.is_available():
            model = model.cuda()

        # -------------------------
        # 1) 干净 Source 测试
        # -------------------------
        print("Starting clean testing on source domain...")
        clean_source_loader = DataLoader(
            TensorDataset(
                torch.Tensor(dataset['test_s'][0]),
                torch.Tensor(dataset['test_s'][1])
            ),
            batch_size=conf.test_batch_size,
            shuffle=False,
            **eval_loader_kwargs,
        )
        clean_acc_s = test(model, clean_source_loader, desc="Clean Source Acc", target_label=conf.target_label)

        # -------------------------
        # 2) Source ASR 测试
        # -------------------------
        if conf.backdoor:
            print("Starting backdoor testing on source domain...")
            x_bd_s, y_bd_s = make_poisoned_eval_set(
                dataset['test_s'][0],
                dataset['test_s'][1],
                conf.target_label,
                trigger_cfg
            )
            bd_source_loader = DataLoader(
                TensorDataset(
                    torch.Tensor(x_bd_s),
                    torch.Tensor(y_bd_s)
                ),
                batch_size=conf.test_batch_size,
                shuffle=False,
                **eval_loader_kwargs,
            )
            asr_s = test(model, bd_source_loader, desc="Source ASR", target_label=conf.target_label)

        # -------------------------
        # 3) 干净 Target 测试
        # -------------------------
        print("Starting clean testing on target domain...")
        for i, (x_test, y_test) in enumerate(dataset['test_t']):
            clean_target_loader = DataLoader(
                TensorDataset(
                    torch.Tensor(x_test),
                    torch.Tensor(y_test)
                ),
                batch_size=conf.test_batch_size,
                shuffle=False,
                **eval_loader_kwargs,
            )
            clean_acc_t = test(model, clean_target_loader, desc=f"Clean Target Acc #{i + 1}", target_label=conf.target_label)

            # -------------------------
            # 4) Target ASR 测试
            # -------------------------
            if conf.backdoor:
                print(f"Starting backdoor testing on target domain #{i + 1}...")
                x_bd_t, y_bd_t = make_poisoned_eval_set(
                    x_test,
                    y_test,
                    conf.target_label,
                    trigger_cfg
                )
                bd_target_loader = DataLoader(
                    TensorDataset(
                        torch.Tensor(x_bd_t),
                        torch.Tensor(y_bd_t)
                    ),
                    batch_size=conf.test_batch_size,
                    shuffle=False,
                    **eval_loader_kwargs,
                )
                asr_t = test(model, bd_target_loader, desc=f"Target ASR #{i + 1}", target_label=conf.target_label)

if __name__ == "__main__":
    main()
