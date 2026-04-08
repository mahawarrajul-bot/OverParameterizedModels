import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models
from cifar10_utils import randomize_labels


class AlexNetTrainer:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, learning_rate=0.0001):
        """
        Initialize AlexNet-based classifier for CIFAR-10.

        Args:
            input_shape: Shape of input images (default: (32, 32, 3) for CIFAR-10)
            num_classes: Number of output classes (default: 10)
            learning_rate: Learning rate for Adam optimizer (default: 0.0001)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.history = None
        self.epochs_to_overfit = None

        self._build_model()

    def _build_model(self):
        """Build AlexNet and resize CIFAR inputs to 224x224 inside the model."""
        self.model = nn.Sequential(
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            models.alexnet(num_classes=self.num_classes),
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def summary(self):
        """Display model architecture summary."""
        print(self.model)
        return self.model

    def _build_dataloader(self, x, y, batch_size, shuffle):
        x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        y_tensor = torch.from_numpy(y).long()
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _evaluate_loader(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=100,
        batch_size=128,
        verbose=1,
        overfit_threshold=0.0001,
        patience=30,
        label_corruption_level=0.0,
    ):
        """
        Train the model with overfit detection based on training loss.

        Args:
            x_train: Training data
            y_train: Training labels
            x_test: Test/validation data for per-epoch loss tracking (default: None)
            y_test: Test/validation labels for per-epoch loss tracking (default: None)
            epochs: Maximum number of training epochs (default: 200)
            batch_size: Batch size (default: 128)
            verbose: Verbosity mode (default: 1)
            overfit_threshold: Training loss threshold to detect overfit (default: 0.01)
            patience: Number of consecutive epochs below threshold to confirm overfit (default: 3)

        Returns:
            Training history, final test error, and epochs to overfit
        """
        train_loader = self._build_dataloader(
            x_train, y_train, batch_size, shuffle=True
        )
        test_loader = None
        if x_test is not None and y_test is not None:
            test_loader = self._build_dataloader(
                x_test, y_test, batch_size, shuffle=False
            )

        # Track first overfit landmark while continuing full training
        patience_counter = 0
        epochs_to_overfit = None

        # Initialize history tracking for epoch-wise plotting
        all_history = {
            "loss": [],
            "accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_error": [],
        }

        for epoch in range(epochs):
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

            current_train_loss = total_loss / total
            current_train_acc = correct / total

            # Get current training loss
            # Evaluate test loss at each epoch if test data is provided
            current_test_loss = np.nan
            current_test_acc = np.nan
            if test_loader is not None:
                current_test_loss, current_test_acc = self._evaluate_loader(test_loader)

            # Store history
            all_history["loss"].append(current_train_loss)
            all_history["accuracy"].append(current_train_acc)
            all_history["test_loss"].append(current_test_loss)
            all_history["test_accuracy"].append(current_test_acc)
            all_history["test_error"].append(
                np.nan if np.isnan(current_test_acc) else 1 - current_test_acc
            )

            if verbose:
                if test_loader is not None:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"train_loss: {current_train_loss:.4f} - "
                        f"train_acc: {current_train_acc:.4f} - "
                        f"test_loss: {current_test_loss:.4f} - "
                        f"test_acc: {current_test_acc:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"train_loss: {current_train_loss:.4f} - "
                        f"train_acc: {current_train_acc:.4f}"
                    )

            # Check for overfit: training loss below threshold
            if current_train_loss < overfit_threshold:
                patience_counter += 1
            else:
                patience_counter = 0

            # Record the first epoch when overfit condition is met, but keep training.
            if patience_counter >= patience and epochs_to_overfit is None:
                epochs_to_overfit = (
                    epoch + 1 - patience + 1
                )  # First epoch when it went below threshold
                if verbose:
                    print(
                        f"\nOverfit detected at epoch {epochs_to_overfit} (training loss < {overfit_threshold})"
                    )

        # Plot train/test loss against epochs
        epoch_indices = range(1, len(all_history["loss"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_indices, all_history["loss"], linestyle=":", label="Train Loss")
        plt.plot(
            epoch_indices, all_history["test_loss"], color="orange", label="Test Loss"
        )
        plt.title("Train vs Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"alexnet_train_test_loss_{label_corruption_level:.1f}.png")
        plt.close()

        # Plot test error against epochs.
        plt.figure(figsize=(8, 5))
        plt.plot(
            epoch_indices, all_history["test_error"], color="red", label="Test Error"
        )
        plt.title("Test Error vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"alexnet_test_error_epochs_{label_corruption_level:.1f}.png")
        plt.close()

        if epochs_to_overfit is None:
            epochs_to_overfit = epochs

        self.epochs_to_overfit = epochs_to_overfit

        final_test_error = np.nan
        if test_loader is not None:
            _, final_test_acc = self._evaluate_loader(test_loader)
            final_test_error = 1 - final_test_acc

        if verbose and epochs_to_overfit == epochs:
            print(
                f"\nDid not overfit within {epochs} epochs (training loss did not stay below {overfit_threshold})"
            )

        self.history = all_history
        return self.history, final_test_error, epochs_to_overfit

    def evaluate(self, x_test, y_test):
        test_loader = self._build_dataloader(
            x_test, y_test, batch_size=256, shuffle=False
        )
        test_loss, test_acc = self._evaluate_loader(test_loader)
        test_error = 1 - test_acc
        print(
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Error: {test_error:.4f}"
        )
        return test_loss, test_acc, test_error


if __name__ == "__main__":
    # Load and preprocess CIFAR-10 data
    print("starting here")
    torch.manual_seed(25292)
    np.random.seed(25292)

    NUM_CLASSES = 10
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)

    x_train = train_dataset.data.astype("float32") / 255.0
    x_test = test_dataset.data.astype("float32") / 255.0
    y_train = np.array(train_dataset.targets, dtype=np.int64)
    y_test = np.array(test_dataset.targets, dtype=np.int64)

    # Normalize with training-set statistics only, then reuse for test data.
    train_mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    train_var = x_train.var(axis=(0, 1, 2), keepdims=True)
    x_train = (x_train - train_mean) / np.sqrt(train_var + 1e-7)
    x_test = (x_test - train_mean) / np.sqrt(train_var + 1e-7)

    # keep_fraction is the fraction of labels left unchanged.
    keep_fractions = [1.0]
    seed = 25292
    test_errors = []
    time_to_overfit = []
    for keep_fraction in keep_fractions:
        corruption_level = 1 - keep_fraction
        print(
            f"Keep fraction: {keep_fraction:.1f} | Corruption level: {corruption_level:.1f}"
        )
        y_train_randomized = randomize_labels(y_train.copy(), corruption_level, seed)
        model = AlexNetTrainer(num_classes=NUM_CLASSES)

        _, final_test_error, epochs_to_overfit_val = model.train(
            x_train,
            y_train_randomized,
            x_test,
            y_test,
            epochs=10,
            batch_size=256,
            label_corruption_level=corruption_level,
        )
        test_errors.append(final_test_error)
        time_to_overfit.append(epochs_to_overfit_val)
        print(f"  Final test error: {final_test_error:.4f}")
        print(f"  Epochs to overfit: {epochs_to_overfit_val}")

    # Plot test error vs randomization
