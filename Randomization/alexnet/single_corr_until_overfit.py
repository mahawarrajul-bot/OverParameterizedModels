import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from alexnet import AlexNetModel, randomize_labels


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


class SingleCorruptionOverfitTrainer:
    def __init__(self, num_classes=10, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            AlexNetModel(num_classes=num_classes),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def _build_loader(self, x, y, batch_size, shuffle):
        x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        y_tensor = torch.from_numpy(y).long()
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * y_batch.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)

        return total_loss / total, correct / total

    def train_until_overfit(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size,
        max_epochs,
        min_epochs,
        overfit_patience,
        train_acc_margin,
        log_path,
    ):
        train_loader = self._build_loader(x_train, y_train, batch_size, shuffle=True)
        test_loader = self._build_loader(x_test, y_test, batch_size, shuffle=False)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        overfit_counter = 0
        history = []

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "train_error",
                    "test_loss",
                    "test_accuracy",
                    "test_error",
                ]
            )

            for epoch in range(1, max_epochs + 1):
                self.model.train()
                total_loss = 0.0
                total = 0
                correct = 0

                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    self.optimizer.zero_grad()
                    logits = self.model(x_batch)
                    loss = self.criterion(logits, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() * y_batch.size(0)
                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == y_batch).sum().item()
                    total += y_batch.size(0)

                train_loss = total_loss / total
                train_acc = correct / total
                train_error = 1.0 - train_acc

                test_loss, test_acc = self._evaluate(test_loader)
                test_error = 1.0 - test_acc

                writer.writerow(
                    [
                        epoch,
                        f"{train_loss:.6f}",
                        f"{train_acc:.6f}",
                        f"{train_error:.6f}",
                        f"{test_loss:.6f}",
                        f"{test_acc:.6f}",
                        f"{test_error:.6f}",
                    ]
                )

                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "train_error": train_error,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_error": test_error,
                    }
                )

                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
                )

                if epoch >= min_epochs and train_acc >= (1.0 - train_acc_margin):
                    overfit_counter += 1
                else:
                    overfit_counter = 0

                if overfit_counter >= overfit_patience:
                    print(
                        "Overfitting detected: train accuracy stayed near 1.0 "
                        f"(margin={train_acc_margin}) for "
                        f"{overfit_patience} consecutive epochs."
                    )
                    break

        return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AlexNet for one corruption level until overfitting is detected."
    )
    parser.add_argument("--corruption-level", type=float, required=True)
    parser.add_argument("--seed", type=int, default=25292)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--overfit-patience", type=int, default=5)
    parser.add_argument(
        "--train-acc-margin",
        type=float,
        default=0.01,
        help="Overfit when train_accuracy >= 1 - margin for patience epochs.",
    )
    parser.add_argument("--data-fraction", type=float, default=0.4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--log-dir", default=str(SCRIPT_DIR / "logs"))
    parser.add_argument("--checkpoint-dir", default=str(SCRIPT_DIR / "checkpoints"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.max_epochs = min(args.max_epochs, 300)

    if not 0.0 <= args.corruption_level <= 1.0:
        raise ValueError("--corruption-level must be between 0.0 and 1.0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test_dataset = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)

    x_train = train_dataset.data.astype("float32") / 255.0
    y_train = np.array(train_dataset.targets, dtype=np.int64)
    x_test = test_dataset.data.astype("float32") / 255.0
    y_test = np.array(test_dataset.targets, dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    train_subset_size = int(len(x_train) * args.data_fraction)
    test_subset_size = int(len(x_test) * args.data_fraction)

    train_idx = rng.choice(len(x_train), size=train_subset_size, replace=False)
    test_idx = rng.choice(len(x_test), size=test_subset_size, replace=False)

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    train_mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    train_var = x_train.var(axis=(0, 1, 2), keepdims=True)
    x_train = (x_train - train_mean) / np.sqrt(train_var + 1e-7)
    x_test = (x_test - train_mean) / np.sqrt(train_var + 1e-7)

    y_train_corrupted = randomize_labels(
        y_train.copy(), args.corruption_level, seed=args.seed
    )

    trainer = SingleCorruptionOverfitTrainer(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    log_path = os.path.join(
        args.log_dir,
        f"alexnet_single_corr_{args.corruption_level:.2f}.csv",
    )

    history = trainer.train_until_overfit(
        x_train=x_train,
        y_train=y_train_corrupted,
        x_test=x_test,
        y_test=y_test,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        overfit_patience=args.overfit_patience,
        train_acc_margin=args.train_acc_margin,
        log_path=log_path,
    )

    last_epoch = history[-1]["epoch"] if history else 0
    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"alexnet_single_corr_{args.corruption_level:.2f}_epoch_{last_epoch}.pth",
    )
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "history": history,
            "args": vars(args),
        },
        ckpt_path,
    )

    print(f"Saved log: {log_path}")
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
