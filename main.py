import os
import argparse
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from torchsummary import summary

from util.get_dataset import get_dataset
from util.CNNmodel import MACNN
from util.con_losses import SupConLoss

# augmentation 导入：如果没有 fast_augmentations，就退化为 augmentations
try:
    from util.augmentation import augmentations, fast_augmentations
except Exception:
    from util.augmentation import augmentations
    fast_augmentations = augmentations

# backdoor 导入：如果没有 util/backdoor.py，则使用本文件内 fallback
try:
    from util.backdoor import add_trigger, make_poisoned_eval_set
except Exception:
    def _get_start(L, length, pos="tail"):
        length = min(length, L)
        if pos == "head":
            return 0
        elif pos == "middle":
            return (L - length) // 2
        elif pos == "random":
            return np.random.randint(0, L - length + 1)
        else:
            return L - length

    def add_trigger(x, trigger_type="sine", amp=0.05, length=64, pos="tail", freq=8):
        x = np.array(x, dtype=np.float32, copy=True)
        assert x.ndim == 2 and x.shape[0] == 2, f"expect [2, L], got {x.shape}"

        L = x.shape[1]
        length = max(4, min(int(length), L))
        start = _get_start(L, length, pos)
        t = np.arange(length, dtype=np.float32)

        if trigger_type == "sine":
            phase = 2 * np.pi * freq * t / max(length, 1)
            trig_i = amp * np.sin(phase)
            trig_q = amp * np.cos(phase)
        elif trigger_type == "const":
            trig_i = np.full(length, amp, dtype=np.float32)
            trig_q = np.full(length, -amp, dtype=np.float32)
        elif trigger_type == "impulse":
            trig_i = np.zeros(length, dtype=np.float32)
            trig_q = np.zeros(length, dtype=np.float32)
            trig_i[::4] = amp
            trig_q[2::4] = -amp
        elif trigger_type == "square":
            base = amp * np.sign(np.sin(2 * np.pi * freq * t / max(length, 1)))
            trig_i = base
            trig_q = -base
        else:
            raise ValueError(f"Unsupported trigger_type: {trigger_type}")

        x[0, start:start + length] += trig_i
        x[1, start:start + length] += trig_q
        return x

    def make_poisoned_eval_set(x, y, target_label, trigger_cfg):
        idx = y != target_label
        x_sel = x[idx].copy()
        for i in range(len(x_sel)):
            x_sel[i] = add_trigger(x_sel[i], **trigger_cfg)
        y_sel = np.full(len(x_sel), target_label, dtype=y.dtype)
        return x_sel, y_sel


def setup_seed(seed=2023, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


setup_seed(2023, deterministic=False)


def get_param_value(model_size: str) -> int:
    model_size_mapping = {'S': 8, 'M': 16, 'L': 32}
    if model_size in model_size_mapping:
        return model_size_mapping[model_size]
    raise ValueError(f"Invalid model_size: {model_size}. Use 'S', 'M', or 'L'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-domain RF Fingerprinting + Backdoor + TensorBoard")

    # base
    parser.add_argument("--dataset_name", type=str, default="WiSig", choices=["ORACLE", "WiSig"])
    parser.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"])
    parser.add_argument("--model_size", type=str, default="S", choices=["S", "M", "L"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--main_aug_depth", type=int, nargs='+', default=[4])
    parser.add_argument("--aux_aug_depth", type=int, nargs='+', default=[1])
    parser.add_argument("--lambda_con", type=float, nargs='+', default=[1.0, 100.0])
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2023)

    # speed
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--fast_aug", action="store_true")

    # backdoor
    parser.add_argument("--backdoor", action="store_true")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--trigger_type", type=str, default="sine",
                        choices=["sine", "const", "impulse", "square"])
    parser.add_argument("--trigger_amp", type=float, default=0.05)
    parser.add_argument("--trigger_len", type=int, default=32)
    parser.add_argument("--trigger_pos", type=str, default="tail",
                        choices=["head", "middle", "tail", "random"])
    parser.add_argument("--trigger_freq", type=int, default=8)
    parser.add_argument("--trigger_stage", type=str, default="post", choices=["pre", "post"])

    # TensorBoard
    parser.add_argument("--use_tb", action="store_true")
    parser.add_argument("--tb_root", type=str, default="runs")
    parser.add_argument("--tb_eval_interval", type=int, default=5)

    return parser.parse_args()


def build_model(conf, num_classes):
    return MACNN(
        in_channels=2,
        channels=get_param_value(conf.model_size),
        num_classes=num_classes
    )


def build_save_path(conf):
    os.makedirs("weight", exist_ok=True)
    save_path = (
        f"weight/"
        f"Dataset={conf.dataset_name}_"
        f"Model={conf.model_size}_"
        f"main_aug_depth={','.join(map(str, conf.main_aug_depth))}_"
        f"aux_aug_depth={','.join(map(str, conf.aux_aug_depth))}_"
        f"lambda={','.join(map(str, conf.lambda_con))}_"
        f"backdoor={int(conf.backdoor)}_"
        f"target={conf.target_label}_"
        f"pr={conf.poison_rate}_"
        f"tt={conf.trigger_type}_"
        f"amp={conf.trigger_amp}_"
        f"len={conf.trigger_len}_"
        f"pos={conf.trigger_pos}_"
        f"freq={conf.trigger_freq}_"
        f"stage={conf.trigger_stage}.pth"
    )
    return save_path


def load_model_state(save_path, conf, num_classes):
    model = build_model(conf, num_classes)
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)

    # 兼容 state_dict / 整模型两种格式
    if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    return model


def plot_iq_signal(x, title="IQ Signal"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].plot(x[0], color='b')
    axes[0].set_title(title + " - I")
    axes[1].plot(x[1], color='r')
    axes[1].set_title(title + " - Q")
    plt.tight_layout()
    return fig


def plot_clean_vs_trigger(clean_x, poison_x, title="Clean vs Triggered"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(clean_x[0], label="Clean I", alpha=0.9)
    axes[0].plot(poison_x[0], label="Triggered I", alpha=0.7)
    axes[0].legend()
    axes[0].set_title(title + " - I")

    axes[1].plot(clean_x[1], label="Clean Q", alpha=0.9)
    axes[1].plot(poison_x[1], label="Triggered Q", alpha=0.7)
    axes[1].legend()
    axes[1].set_title(title + " - Q")

    plt.tight_layout()
    return fig


def build_tb_writer(conf):
    if not conf.use_tb:
        return None

    os.makedirs(conf.tb_root, exist_ok=True)
    tb_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{conf.dataset_name}_"
        f"{conf.model_size}_"
        f"bd{int(conf.backdoor)}_"
        f"pr{conf.poison_rate}_"
        f"tg{conf.target_label}_"
        f"{conf.trigger_type}_"
        f"{conf.trigger_stage}"
    )
    tb_logdir = os.path.join(conf.tb_root, tb_name)
    writer = SummaryWriter(log_dir=tb_logdir)
    print(f"[TensorBoard] logdir = {tb_logdir}")
    return writer


def aug(iq_data, preprocess, main_aug_depth, aux_aug_depth,
        poison=False, trigger_cfg=None, trigger_stage="post", fast_aug=False):
    aug_list = fast_augmentations if fast_aug else augmentations
    iq_data_aug_list = []

    base = np.array(iq_data, dtype=np.float32, copy=True)

    if poison and trigger_cfg is not None and trigger_stage == "pre":
        base = add_trigger(base, **trigger_cfg)

    # aux views
    for i in range(len(aux_aug_depth)):
        iq_data_aug = base.copy()
        if aux_aug_depth[i] != 0:
            sampled_ops = np.random.choice(aug_list, aux_aug_depth[i])
            for op in sampled_ops:
                iq_data_aug = op(iq_data_aug)

        if poison and trigger_cfg is not None and trigger_stage == "post":
            iq_data_aug = add_trigger(iq_data_aug, **trigger_cfg)

        iq_data_aug = np.squeeze(preprocess(iq_data_aug.astype(np.float32)))
        iq_data_aug_list.append(iq_data_aug)

    # main view
    iq_data_aug = base.copy()
    if main_aug_depth[0] != 0:
        sampled_ops = np.random.choice(aug_list, main_aug_depth[0])
        for op in sampled_ops:
            iq_data_aug = op(iq_data_aug)

    if poison and trigger_cfg is not None and trigger_stage == "post":
        iq_data_aug = add_trigger(iq_data_aug, **trigger_cfg)

    iq_data_aug = np.squeeze(preprocess(iq_data_aug.astype(np.float32)))
    iq_data_aug_list.append(iq_data_aug)

    return iq_data_aug_list


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, preprocess, main_aug_depth, aux_aug_depth,
                 backdoor=False, target_label=0, poison_rate=0.0,
                 trigger_cfg=None, trigger_stage="post", fast_aug=False):
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
        self.trigger_stage = trigger_stage
        self.fast_aug = fast_aug

        self.poison_mask = np.zeros(len(self.dataset), dtype=bool)

        if self.backdoor and self.poison_rate > 0:
            labels = np.array([y for _, y in self.dataset])
            candidate_idx = np.where(labels != self.target_label)[0]
            num_poison = int(len(candidate_idx) * self.poison_rate)
            if num_poison > 0:
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
            trigger_stage=self.trigger_stage,
            fast_aug=self.fast_aug
        )
        return x_aug, y

    def __len__(self):
        return len(self.dataset)


def build_loader(dataset_obj, batch_size, shuffle, conf):
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=conf.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(conf.num_workers > 0)
    )


def train(model, loss, train_dataloader, optimizer, scaler, epoch, conf):
    model.train()
    correct = 0
    all_loss = 0

    for data_nn in train_dataloader:
        data, target = data_nn
        target = target.long()

        domain_target = []
        target_all = []
        num_data = len(conf.main_aug_depth) + len(conf.aux_aug_depth)

        for i in range(num_data):
            domain_target.append(i * torch.ones(data[0].size(0)).long())
            target_all.append(target)

        if torch.cuda.is_available():
            data_all = torch.cat(data, 0).cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_all = torch.cat(target_all, 0).cuda(non_blocking=True)
            domain_target = torch.cat(domain_target, 0).cuda(non_blocking=True)
        else:
            data_all = torch.cat(data, 0)
            target_all = torch.cat(target_all, 0)
            domain_target = torch.cat(domain_target, 0)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=conf.amp and torch.cuda.is_available()):
            embedding, output = model(data_all)
            prob = F.log_softmax(output, dim=1)
            prob_list = torch.split(prob, data[0].size(0))

            cls_loss = loss[0](prob_list[num_data - 1], target)
            con_loss = loss[1](embedding.unsqueeze(1), target_all, adv=False)
            adv_con_loss = loss[1](embedding.unsqueeze(1), domain_target, adv=True)
            result_loss = cls_loss + conf.lambda_con[0] * con_loss + conf.lambda_con[1] * adv_con_loss

        scaler.scale(result_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        all_loss += result_loss.item() * data[0].size(0)
        pred = prob_list[num_data - 1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    epoch_loss = all_loss / len(train_dataloader.dataset)
    epoch_acc = correct / len(train_dataloader.dataset)

    print(
        f"Train Epoch: {epoch}\tLoss: {epoch_loss:.6f}, "
        f"Accuracy: {correct}/{len(train_dataloader.dataset)} ({100.0 * epoch_acc:.2f}%)"
    )
    return epoch_loss, epoch_acc


def evaluate(model, loss, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            embedding, output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += loss[0](output, target).item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    acc = correct / len(test_dataloader.dataset)

    print(
        f"\nValidation set: Loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_dataloader.dataset)} ({100.0 * acc:.2f}%)\n"
    )
    return test_loss, acc


def test(model, test_dataloader, desc="Test"):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            _, output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_dataloader.dataset)
    print(f"{desc}: {acc:.4f}")
    return acc


def periodic_tb_eval(model, dataset, conf, writer, epoch, trigger_cfg):
    """每隔若干个 epoch 记录 source/target clean acc 和 ASR 到 TensorBoard。"""
    if writer is None or conf.tb_eval_interval <= 0:
        return

    if epoch % conf.tb_eval_interval != 0:
        return

    # clean source
    clean_source_loader = build_loader(
        TensorDataset(torch.Tensor(dataset['test_s'][0]), torch.Tensor(dataset['test_s'][1])),
        conf.test_batch_size, False, conf
    )
    clean_source_acc = test(model, clean_source_loader, desc=f"[Epoch {epoch}] Clean Source Acc")
    writer.add_scalar("Test/Clean_Source_Acc", clean_source_acc, epoch)

    # clean target
    for i, (x_test, y_test) in enumerate(dataset['test_t']):
        clean_target_loader = build_loader(
            TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)),
            conf.test_batch_size, False, conf
        )
        clean_target_acc = test(model, clean_target_loader, desc=f"[Epoch {epoch}] Clean Target Acc #{i+1}")
        writer.add_scalar(f"Test/Clean_Target_Acc_{i+1}", clean_target_acc, epoch)

    # ASR
    if conf.backdoor:
        x_bd_s, y_bd_s = make_poisoned_eval_set(
            dataset['test_s'][0], dataset['test_s'][1],
            conf.target_label, trigger_cfg
        )
        bd_source_loader = build_loader(
            TensorDataset(torch.Tensor(x_bd_s), torch.Tensor(y_bd_s)),
            conf.test_batch_size, False, conf
        )
        source_asr = test(model, bd_source_loader, desc=f"[Epoch {epoch}] Source ASR")
        writer.add_scalar("Backdoor/Source_ASR", source_asr, epoch)

        for i, (x_test, y_test) in enumerate(dataset['test_t']):
            x_bd_t, y_bd_t = make_poisoned_eval_set(
                x_test, y_test, conf.target_label, trigger_cfg
            )
            bd_target_loader = build_loader(
                TensorDataset(torch.Tensor(x_bd_t), torch.Tensor(y_bd_t)),
                conf.test_batch_size, False, conf
            )
            target_asr = test(model, bd_target_loader, desc=f"[Epoch {epoch}] Target ASR #{i+1}")
            writer.add_scalar(f"Backdoor/Target_ASR_{i+1}", target_asr, epoch)

    writer.flush()


def train_and_evaluate(model, loss, train_loader, val_loader, optimizer, scaler,
                       epochs, save_path, conf, writer=None, dataset=None, trigger_cfg=None):
    best_loss = float('inf')
    best_epoch = -1
    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, loss, train_loader, optimizer, scaler, epoch, conf)
        val_loss, val_acc = evaluate(model, loss, val_loader, epoch)

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        periodic_tb_eval(model, dataset, conf, writer, epoch, trigger_cfg)

        if val_loss < best_loss:
            print(f"Saving model at epoch {epoch} with loss {val_loss:.4f}")
            best_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            print(f"No improvement for {wait} epoch(s)")

        if wait >= conf.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_epoch, best_loss


def final_test_and_log(model, dataset, conf, trigger_cfg, writer=None, global_step=0):
    results = {}

    print("Starting clean testing on source domain...")
    test_loader = build_loader(
        TensorDataset(torch.Tensor(dataset['test_s'][0]), torch.Tensor(dataset['test_s'][1])),
        conf.test_batch_size, False, conf
    )
    results["clean_source_acc"] = test(model, test_loader, desc="Clean Source Acc")
    if writer is not None:
        writer.add_scalar("Final/Clean_Source_Acc", results["clean_source_acc"], global_step)

    if conf.backdoor:
        print("Starting backdoor testing on source domain...")
        x_bd, y_bd = make_poisoned_eval_set(
            dataset['test_s'][0],
            dataset['test_s'][1],
            conf.target_label,
            trigger_cfg
        )
        bd_loader = build_loader(
            TensorDataset(torch.Tensor(x_bd), torch.Tensor(y_bd)),
            conf.test_batch_size, False, conf
        )
        results["source_asr"] = test(model, bd_loader, desc="Source ASR")
        if writer is not None:
            writer.add_scalar("Final/Source_ASR", results["source_asr"], global_step)

    print("Starting testing on target domain...")
    for i, (x_test, y_test) in enumerate(dataset['test_t']):
        clean_loader = build_loader(
            TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)),
            conf.test_batch_size, False, conf
        )
        results[f"clean_target_acc_{i+1}"] = test(model, clean_loader, desc=f"Clean Target Acc #{i+1}")
        if writer is not None:
            writer.add_scalar(f"Final/Clean_Target_Acc_{i+1}", results[f'clean_target_acc_{i+1}'], global_step)

        if conf.backdoor:
            x_bd, y_bd = make_poisoned_eval_set(
                x_test, y_test,
                conf.target_label,
                trigger_cfg
            )
            bd_loader = build_loader(
                TensorDataset(torch.Tensor(x_bd), torch.Tensor(y_bd)),
                conf.test_batch_size, False, conf
            )
            results[f"target_asr_{i+1}"] = test(model, bd_loader, desc=f"Target ASR #{i+1}")
            if writer is not None:
                writer.add_scalar(f"Final/Target_ASR_{i+1}", results[f'target_asr_{i+1}'], global_step)

    print("\n===== Summary =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    if writer is not None:
        writer.flush()

    return results


def main():
    conf = parse_args()
    setup_seed(conf.seed, deterministic=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda

    num_classes = 16 if conf.dataset_name == "ORACLE" else 6
    input_shape = (2, 6000) if conf.dataset_name == "ORACLE" else (2, 256)

    trigger_cfg = {
        "trigger_type": conf.trigger_type,
        "amp": conf.trigger_amp,
        "length": conf.trigger_len,
        "pos": conf.trigger_pos,
        "freq": conf.trigger_freq
    }

    save_path = build_save_path(conf)
    dataset = get_dataset(conf.dataset_name)
    writer = build_tb_writer(conf)

    # TensorBoard: 记录一个 clean / triggered 信号图
    if writer is not None:
        sample_x = dataset['train'][0][0]
        fig = plot_iq_signal(sample_x, title="Train Sample Clean")
        writer.add_figure("Signals/Clean_Sample", fig, global_step=0)
        plt.close(fig)

        if conf.backdoor:
            poison_x = add_trigger(sample_x, **trigger_cfg)
            fig = plot_clean_vs_trigger(sample_x, poison_x, title="Trigger Injection Example")
            writer.add_figure("Signals/Clean_vs_Triggered", fig, global_step=0)
            plt.close(fig)

    train_dataset = AugDataset(
        dataset['train'][0], dataset['train'][1],
        transforms.ToTensor(),
        conf.main_aug_depth, conf.aux_aug_depth,
        backdoor=conf.backdoor,
        target_label=conf.target_label,
        poison_rate=conf.poison_rate,
        trigger_cfg=trigger_cfg,
        trigger_stage=conf.trigger_stage,
        fast_aug=conf.fast_aug
    )

    train_loader = build_loader(train_dataset, conf.batch_size, True, conf)
    val_loader = build_loader(
        TensorDataset(torch.Tensor(dataset['val'][0]), torch.Tensor(dataset['val'][1])),
        conf.test_batch_size, False, conf
    )

    model = build_model(conf, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)

    cls_loss = nn.NLLLoss()
    con_loss = SupConLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        cls_loss = cls_loss.cuda()
        con_loss = con_loss.cuda()
        summary(model, input_shape)

    scaler = GradScaler("cuda", enabled=conf.amp and torch.cuda.is_available())
    loss = (cls_loss, con_loss)

    best_epoch = 0

    if conf.mode in ["train", "train_test"]:
        print("Starting training...")
        best_epoch, best_loss = train_and_evaluate(
            model, loss, train_loader, val_loader, optimizer, scaler,
            conf.epochs, save_path, conf,
            writer=writer,
            dataset=dataset,
            trigger_cfg=trigger_cfg
        )
        print(f"Best epoch: {best_epoch}, best val loss: {best_loss:.4f}")

    if conf.mode in ["test", "train_test"]:
        print("Loading best model for testing...")
        model = load_model_state(save_path, conf, num_classes)
        final_test_and_log(
            model, dataset, conf, trigger_cfg,
            writer=writer,
            global_step=best_epoch if best_epoch > 0 else conf.epochs
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()