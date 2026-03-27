import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from alexnet import AlexNetModel, randomize_labels


class AlexNetTrainer:
    def __init__(self, num_classes=10, learning_rate=0.0001):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            AlexNetModel(num_classes=self.num_classes),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

    def _build_dataloader(self, x, y, batch_size, shuffle):
        x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        y_tensor = torch.from_numpy(y).long()
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _evaluate_loader(self, loader):
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
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc

    def train_and_log(
        self, x_train, y_train, x_test, y_test, epochs, batch_size, log_file_path
    ):
        train_loader = self._build_dataloader(
            x_train, y_train, batch_size, shuffle=True
        )
        test_loader = self._build_dataloader(x_test, y_test, batch_size, shuffle=False)

        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,train_loss,train_accuracy,train_error,test_loss,test_accuracy,test_error\n"
            )

            for epoch in range(1, epochs + 1):
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
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

                train_loss = total_loss / total
                train_acc = correct / total
                train_error = 1.0 - train_acc

                test_loss, test_acc = self._evaluate_loader(test_loader)
                test_error = 1.0 - test_acc

                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},{train_error:.6f},"
                    f"{test_loss:.6f},{test_acc:.6f},{test_error:.6f}\n"
                )

                print(
                    f"Epoch {epoch:02d} | train_error={train_error:.4f} | "
                    f"test_error={test_error:.4f}"
                )


if __name__ == "__main__":
    torch.manual_seed(25292)
    np.random.seed(25292)

    num_classes = 10
    epochs = 10
    batch_size = 256
    seed = 25292
    data_fraction = 0.4

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)

    x_train = train_dataset.data.astype("float32") / 255.0
    x_test = test_dataset.data.astype("float32") / 255.0
    y_train = np.array(train_dataset.targets, dtype=np.int64)
    y_test = np.array(test_dataset.targets, dtype=np.int64)

    # Use a deterministic subset to reduce runtime/memory.
    rng = np.random.default_rng(seed)
    train_subset_size = int(len(x_train) * data_fraction)
    test_subset_size = int(len(x_test) * data_fraction)

    train_idx = rng.choice(len(x_train), size=train_subset_size, replace=False)
    test_idx = rng.choice(len(x_test), size=test_subset_size, replace=False)

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    print(
        f"Using {data_fraction:.1f} fraction | "
        f"train: {len(x_train)} samples | test: {len(x_test)} samples"
    )

    train_mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    train_var = x_train.var(axis=(0, 1, 2), keepdims=True)
    x_train = (x_train - train_mean) / np.sqrt(train_var + 1e-7)
    x_test = (x_test - train_mean) / np.sqrt(train_var + 1e-7)

    os.makedirs("logs", exist_ok=True)

    corruption_levels = np.round(np.arange(0.0, 1.01, 0.1), 1)

    for corruption_level in corruption_levels:
        print("=" * 80)
        print(f"Running corruption level: {corruption_level:.1f}")

        y_train_randomized = randomize_labels(y_train.copy(), corruption_level, seed)

        trainer = AlexNetTrainer(num_classes=num_classes)

        log_file = f"logs/alexnet_corr_{corruption_level:.1f}.csv"
        trainer.train_and_log(
            x_train=x_train,
            y_train=y_train_randomized,
            x_test=x_test,
            y_test=y_test,
            epochs=epochs,
            batch_size=batch_size,
            log_file_path=log_file,
        )

        print(f"Saved log: {log_file}")
