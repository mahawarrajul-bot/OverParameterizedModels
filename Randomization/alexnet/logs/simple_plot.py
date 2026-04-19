#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


def plot_single_level_error(corruption_level):
    """Plot train/test error vs epochs for a single corruption level."""
    log_file = Path(__file__).parent / f"alexnet_single_corr_{corruption_level:.2f}.csv"

    if not log_file.exists():
        print(f"Error: File not found: {log_file}")
        return False

    rows = []
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_error": float(row["train_error"]),
                    "test_error": float(row["test_error"]),
                }
            )

    if not rows:
        print(f"No data found in {log_file}")
        return False

    epochs = [r["epoch"] for r in rows]
    train_error = [r["train_error"] for r in rows]
    test_error = [r["test_error"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_error, "b--o", label="Train Error", linewidth=2)
    plt.plot(epochs, test_error, "r-o", label="Test Error", linewidth=2)
    plt.title(
        f"Train/Test Error vs Epochs (Corruption {corruption_level:.2f})", fontsize=14
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = Path(__file__).parent / f"error_plot_{corruption_level:.2f}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")
    plt.close()
    return True


def plot_all_levels_error():
    """Plot training errors for all available corruption levels."""
    log_dir = Path(__file__).parent
    csv_files = sorted(log_dir.glob("alexnet_single_corr_*.csv"))

    if not csv_files:
        print("No CSV files found!")
        return False

    plt.figure(figsize=(12, 7))

    for csv_file in csv_files:
        corruption = float(csv_file.stem.split("_")[-1])

        rows = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "train_error": float(row["train_error"]),
                    }
                )

        if rows:
            epochs = [r["epoch"] for r in rows]
            train_error = [r["train_error"] for r in rows]
            plt.plot(
                epochs, train_error, "o-", label=f"Corr {corruption:.2f}", linewidth=2
            )

    plt.title("Training Error vs Epochs (All Corruption Levels)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Error", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = Path(__file__).parent / "all_train_errors.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")
    plt.close()
    return True


def plot_all_levels_test_error():
    """Plot test errors for all available corruption levels."""
    log_dir = Path(__file__).parent
    csv_files = sorted(log_dir.glob("alexnet_single_corr_*.csv"))

    if not csv_files:
        print("No CSV files found!")
        return False

    plt.figure(figsize=(12, 7))

    for csv_file in csv_files:
        corruption = float(csv_file.stem.split("_")[-1])

        rows = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "test_error": float(row["test_error"]),
                    }
                )

        if rows:
            epochs = [r["epoch"] for r in rows]
            test_error = [r["test_error"] for r in rows]
            plt.plot(
                epochs, test_error, "o-", label=f"Corr {corruption:.2f}", linewidth=2
            )

    plt.title("Test Error vs Epochs (All Corruption Levels)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Test Error", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = Path(__file__).parent / "all_test_errors.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")
    plt.close()
    return True


if __name__ == "__main__":
    print("Generating plots...\n")

    # Plot individual levels
    for corr in [0.0, 0.1, 0.2, 0.5, 0.8]:
        print(f"Plotting corruption level {corr:.2f}...")
        plot_single_level_error(corr)

    print("\nPlotting all training errors...")
    plot_all_levels_error()

    print("Plotting all test errors...")
    plot_all_levels_test_error()

    print("\nDone!")
