import os

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset


def build_adaptation_save_path(save_path, suffix="_adapt"):
    root, ext = os.path.splitext(save_path)
    return f"{root}{suffix}{ext}"


def build_target_adaptation_loaders(target_data, conf, loader_kwargs=None):
    x_target, y_target = target_data
    if conf.adapt_subset_ratio < 1.0:
        x_target, _, y_target, _ = train_test_split(
            x_target,
            y_target,
            train_size=conf.adapt_subset_ratio,
            random_state=conf.adapt_seed,
            stratify=y_target,
        )

    x_adapt_train, x_adapt_val, y_adapt_train, y_adapt_val = train_test_split(
        x_target,
        y_target,
        test_size=conf.adapt_val_ratio,
        random_state=conf.adapt_seed,
        stratify=y_target,
    )

    if loader_kwargs is None:
        loader_kwargs = {}

    adapt_batch_size = conf.adapt_batch_size or conf.batch_size
    train_loader = DataLoader(
        TensorDataset(torch.Tensor(x_adapt_train), torch.Tensor(y_adapt_train)),
        batch_size=adapt_batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TensorDataset(torch.Tensor(x_adapt_val), torch.Tensor(y_adapt_val)),
        batch_size=conf.test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def evaluate_adaptation(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            _, output = model(data)
            output = F.log_softmax(output, dim=1)
            total_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = correct / len(dataloader.dataset)
    return avg_loss, avg_acc


def adapt_on_target_clean(model, target_data, conf, eval_loader_kwargs=None, save_path=None):
    if eval_loader_kwargs is None:
        eval_loader_kwargs = {}

    adapt_loader_kwargs = dict(eval_loader_kwargs)
    if adapt_loader_kwargs.get("num_workers", 0) > 0:
        adapt_loader_kwargs["persistent_workers"] = False

    train_loader, val_loader = build_target_adaptation_loaders(target_data, conf, loader_kwargs=adapt_loader_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.adapt_lr, weight_decay=conf.adapt_wd)
    scaler = GradScaler("cuda", enabled=conf.amp and torch.cuda.is_available())
    loss_fn = torch.nn.NLLLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    best_val_loss = float("inf")
    best_state_dict = None

    print("Starting target clean adaptation...")
    for epoch in range(1, conf.adapt_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        for data, target in train_loader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=conf.amp and torch.cuda.is_available()):
                _, output = model(data)
                output = F.log_softmax(output, dim=1)
                train_loss = loss_fn(output, target)

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += train_loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_train_acc = correct / len(train_loader.dataset)
        val_loss, val_acc = evaluate_adaptation(model, loss_fn, val_loader)
        print(
            f"Adapt Epoch {epoch:>3}/{conf.adapt_epochs} | train_loss={avg_train_loss:.4f} "
            f"| train_acc={avg_train_acc:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Saved adapted checkpoint: {save_path}")

    return model, save_path
