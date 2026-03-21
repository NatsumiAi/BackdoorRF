import csv
import os

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def build_history_paths(save_path, log_dir="log"):
    stem = os.path.splitext(os.path.basename(save_path))[0]
    os.makedirs(log_dir, exist_ok=True)
    return (
        os.path.join(log_dir, f"{stem}_history.csv"),
        os.path.join(log_dir, f"{stem}_history.png"),
    )


def build_tensorboard_dir(save_path, tb_root="runs"):
    stem = os.path.splitext(os.path.basename(save_path))[0]
    os.makedirs(tb_root, exist_ok=True)
    return os.path.join(tb_root, stem)


def format_eta(seconds):
    minutes = max(seconds, 0.0) / 60.0
    return f"{minutes:.1f}m"


class TrainingMonitor:
    def __init__(self, save_path, use_tensorboard=False, log_dir="log", tb_root="runs"):
        self.history_csv_path, self.history_plot_path = build_history_paths(save_path, log_dir=log_dir)
        self.history = []
        self.writer = None
        self.tensorboard_dir = None

        if use_tensorboard and SummaryWriter is not None:
            self.tensorboard_dir = build_tensorboard_dir(save_path, tb_root=tb_root)
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def update(self, row):
        self.history.append(row)
        self._save_history_csv()
        self._save_history_plot()
        self._write_tensorboard(row)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def _save_history_csv(self):
        base_fields = [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "best_val_loss",
            "lr",
            "epoch_time",
            "eta_minutes",
        ]
        dynamic_fields = []
        for row in self.history:
            for key in row.keys():
                if key not in base_fields and key not in dynamic_fields:
                    dynamic_fields.append(key)
        fieldnames = base_fields + dynamic_fields
        with open(self.history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def _save_history_plot(self):
        if plt is None or not self.history:
            return

        epochs = [row["epoch"] for row in self.history]
        train_loss = [row["train_loss"] for row in self.history]
        val_loss = [row["val_loss"] for row in self.history]
        train_acc = [row["train_acc"] for row in self.history]
        val_acc = [row["val_acc"] for row in self.history]
        best_val = [row["best_val_loss"] for row in self.history]

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
        axes[0].plot(epochs, train_loss, color="#355C7D", linewidth=2.0, label="Train loss")
        axes[0].plot(epochs, val_loss, color="#C06C84", linewidth=2.0, label="Val loss")
        axes[0].plot(epochs, best_val, color="#2A9D8F", linewidth=1.6, linestyle="--", label="Best val loss")
        axes[0].set_title("Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(frameon=False)

        axes[1].plot(epochs, train_acc, color="#355C7D", linewidth=2.0, label="Train acc")
        axes[1].plot(epochs, val_acc, color="#6C8EAD", linewidth=2.0, label="Val acc")
        axes[1].set_title("Accuracy Curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].legend(frameon=False)

        fig.savefig(self.history_plot_path, dpi=220, facecolor="white")
        plt.close(fig)

    def _write_tensorboard(self, row):
        if self.writer is None:
            return

        epoch = row["epoch"]
        self.writer.add_scalar("train/loss", row["train_loss"], epoch)
        self.writer.add_scalar("train/acc", row["train_acc"], epoch)
        self.writer.add_scalar("val/loss", row["val_loss"], epoch)
        self.writer.add_scalar("val/acc", row["val_acc"], epoch)
        self.writer.add_scalar("train/best_val_loss", row["best_val_loss"], epoch)
        self.writer.add_scalar("train/lr", row["lr"], epoch)
        self.writer.add_scalar("time/epoch_time", row["epoch_time"], epoch)
        self.writer.add_scalar("time/eta_minutes", row["eta_minutes"], epoch)
        for key, value in row.items():
            if key in {"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_loss", "lr", "epoch_time", "eta_minutes"}:
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"monitor/{key}", value, epoch)
