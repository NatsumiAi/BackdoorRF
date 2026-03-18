import os
import argparse
import torch
import numpy as np
import random
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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2023)

def get_param_value(model_size: str) -> int:
    """Returns the parameter value based on the input size: S, M, or L."""
    model_size_mapping = {'S': 8, 'M': 16, 'L': 32}
    if model_size in model_size_mapping:
        return model_size_mapping[model_size]
    else:
        raise ValueError(f"Invalid model_size: {model_size}. Use 'S', 'M', or 'L'.")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Single-source Domain Generalization")
    parser.add_argument("--dataset_name", type=str, default="ORACLE", choices=["ORACLE", "WiSig"])   
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "train_test"])
    parser.add_argument("--model_size", type=str, default="S")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
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
    parser.add_argument("--poison_channel_aug", action="store_true")
    parser.add_argument("--channel_phase_max_deg", type=float, default=15.0)
    parser.add_argument("--channel_scale_min", type=float, default=0.9)
    parser.add_argument("--channel_scale_max", type=float, default=1.1)
    parser.add_argument("--channel_shift_max", type=int, default=4)
    parser.add_argument("--channel_snr_db", type=float, default=25.0)

    # trigger 什么时候加：pre/post augmentation or EOT sampling
    parser.add_argument("--trigger_stage", type=str, default="post",
                        choices=["pre", "post", "eot"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


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
        return x_aug, y

    def __len__(self):
        return len(self.dataset)

def train(model, loss, train_dataloader, optimizer, scaler, epoch, conf):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        data, target = data_nn
        target = target.long()
        domain_target = []
        target_all = []
        num_data= len(conf.main_aug_depth) + len(conf.aux_aug_depth)
        for i in range(num_data):
            domain_target.append(i*torch.ones(data[0].size(0)).long())   
            target_all.append(target)   

        if torch.cuda.is_available():
            data_all = torch.cat(data, 0).cuda()
            target = target.cuda()
            target_all = torch.cat(target_all, 0).cuda()
            domain_target= torch.cat(domain_target, 0).cuda()
        # AMP加速
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=conf.amp and torch.cuda.is_available()):
            embedding, output = model(data_all)
            prob = F.log_softmax(output, dim=1)
            porb_list = torch.split(prob, data[0].size(0))
            cls_loss = loss[0](porb_list[num_data - 1], target)
            con_loss = loss[1](embedding.unsqueeze(1), target_all, adv=False)
            adv_con_loss = loss[1](embedding.unsqueeze(1), domain_target, adv=True)
            result_loss = cls_loss + conf.lambda_con[0] * con_loss + conf.lambda_con[1] * adv_con_loss

        scaler.scale(result_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # -------------
        all_loss += result_loss.item()*data[0].size(0)
        pred = porb_list[num_data -  1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset),
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )

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

    return test_loss

def test(model, test_dataloader, desc="Test"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding, output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_dataloader.dataset)
    print(f"{desc}: {acc:.4f}")
    return acc


def train_and_evaluate(model, loss, train_loader, val_loader, optimizer, scaler, epochs, save_path, conf):
    """Train and evaluate the model, saving the best model."""
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train(model, loss, train_loader, optimizer, scaler, epoch, conf)
        val_loss = evaluate(model, loss, val_loader, epoch)
        if val_loss < best_loss:
            print(f"Saving model at epoch {epoch} with loss {val_loss:.4f}")
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)

def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda

    num_classes = 16 if conf.dataset_name == "ORACLE" else 6
    input_shape = (2, 6000) if conf.dataset_name == "ORACLE" else (2, 256)

    save_path = (
        f"weight/Dataset={conf.dataset_name}_"
        f"Model={conf.model_size}_"
        f"main_aug_depth={','.join(map(str, conf.main_aug_depth))}_"
        f"aux_aug_depth={','.join(map(str, conf.aux_aug_depth))}_"
        f"lambda={','.join(map(str, conf.lambda_con))}_"
        f"backdoor={int(conf.backdoor)}_"
        f"target={conf.target_label}_"
        f"pr={conf.poison_rate}_"
        f"tt={conf.trigger_type}_"
        f"ta={conf.trigger_amp}_"
        f"tl={conf.trigger_len}_"
        f"tp={conf.trigger_pos}_"
        f"tf={conf.trigger_freq}_"
        f"tseg={conf.trigger_segments}_"
        f"tanch={conf.trigger_anchor_positions.replace(',', '-')}_"
        f"tj={conf.trigger_jitter}_"
        f"tiq={conf.trigger_iq_mode}_"
        f"tad={int(conf.trigger_adaptive_amp)}_"
        f"pca={int(conf.poison_channel_aug)}_"
        f"cph={conf.channel_phase_max_deg}_"
        f"cs={conf.channel_scale_min}-{conf.channel_scale_max}_"
        f"csh={conf.channel_shift_max}_"
        f"csnr={conf.channel_snr_db}_"
        f"ts={conf.trigger_stage}.pth"
    )

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
        shuffle=True
    )

    val_loader = DataLoader(TensorDataset(torch.Tensor(dataset['val'][0]), torch.Tensor(dataset['val'][1])), 
                            batch_size=conf.test_batch_size, shuffle=True)

    model = MACNN(in_channels=2, channels=get_param_value(conf.model_size), num_classes=num_classes)

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
        train_and_evaluate(model, loss, train_loader, val_loader, optimizer, scaler, conf.epochs, save_path, conf)

    if conf.mode in ["test", "train_test"]:
        # 读取模型
        model = MACNN(
            in_channels=2,
            channels=get_param_value(conf.model_size),
            num_classes=num_classes
        )
        state_dict = torch.load(save_path, map_location="cpu")
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
            shuffle=False
        )
        clean_acc_s = test(model, clean_source_loader, desc="Clean Source Acc")

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
                shuffle=False
            )
            asr_s = test(model, bd_source_loader, desc="Source ASR")

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
                shuffle=False
            )
            clean_acc_t = test(model, clean_target_loader, desc=f"Clean Target Acc #{i + 1}")

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
                    shuffle=False
                )
                asr_t = test(model, bd_target_loader, desc=f"Target ASR #{i + 1}")

if __name__ == "__main__":
    main()
