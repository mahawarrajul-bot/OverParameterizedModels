import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _validate_corruption(corruption, log_dir="."):
    if not (0.0 <= corruption <= 1.0):
        raise ValueError("corruption must be between 0.0 and 1.0")

    log_file = _log_file_for_corruption(log_dir, corruption)
    if not log_file.exists():
        raise FileNotFoundError(
            f"No single-corr log found for {corruption:.2f}: {log_file}"
        )


def _log_file_for_corruption(log_dir, corruption):
    return Path(log_dir) / f"alexnet_single_corr_{corruption:.2f}.csv"


def _read_log_rows(log_file):
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "train_error": float(row["train_error"]),
                    "test_accuracy": float(row["test_accuracy"]),
                    "test_error": float(row["test_error"]),
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {log_file}")

    return rows


def plot_error_vs_epochs(corruption, log_dir=".", save_path=None):
    _validate_corruption(corruption, log_dir)
    log_file = _log_file_for_corruption(log_dir, corruption)
    rows = _read_log_rows(log_file)

    epochs = [r["epoch"] for r in rows]
    train_error = [r["train_error"] for r in rows]
    test_error = [r["test_error"] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_error, linestyle="--", marker="o", label="Train Error")
    plt.plot(epochs, test_error, linestyle="-", marker="o", label="Test Error")
    plt.title(f"Train/Test Error vs Epochs (corruption={corruption:.1f})")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if not save_path:
        save_path = Path(log_dir) / f"error_{corruption:.2f}.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def plot_accuracy_vs_epochs(corruption, log_dir=".", save_path=None):
    _validate_corruption(corruption, log_dir)
    log_file = _log_file_for_corruption(log_dir, corruption)
    rows = _read_log_rows(log_file)

    epochs = [r["epoch"] for r in rows]
    train_accuracy = [r["train_accuracy"] for r in rows]
    test_accuracy = [r["test_accuracy"] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs,
        train_accuracy,
        linestyle="--",
        marker="o",
        label="Train Accuracy",
    )
    plt.plot(epochs, test_accuracy, linestyle="-", marker="o", label="Test Accuracy")
    plt.title(f"Train/Test Accuracy vs Epochs (corruption={corruption:.1f})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if not save_path:
        save_path = Path(log_dir) / f"accuracy_{corruption:.2f}.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def _find_all_single_corrs(log_dir):
    """Find all available single corruption CSV files."""
    log_dir = Path(log_dir)
    files = sorted(log_dir.glob("alexnet_single_corr_*.csv"))
    corruptions = []
    for f in files:
        try:
            corr = float(f.stem.split("_")[-1])
            corruptions.append(corr)
        except ValueError:
            pass
    return corruptions


def plot_all_training_errors_vs_epochs(log_dir=".", save_path=None):
    """Plot training error vs epochs for all available single corruption levels."""
    corruptions = _find_all_single_corrs(log_dir)

    if not corruptions:
        print(f"No single corruption CSV files found in {log_dir}")
        return

    plt.figure(figsize=(11, 6))

    for corruption in sorted(corruptions):
        log_file = _log_file_for_corruption(log_dir, corruption)
        try:
            rows = _read_log_rows(log_file)
            epochs = [r["epoch"] for r in rows]
            train_error = [r["train_error"] for r in rows]
            plt.plot(
                epochs,
                train_error,
                marker="o",
                label=f"Corruption {corruption:.2f}",
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {log_file}: {e}")

    plt.title("Training Error vs Epochs (All Single Corruption Levels)")
    plt.xlabel("Epoch")
    plt.ylabel("Training Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if not save_path:
        save_path = Path(log_dir) / "all_train_errors.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def plot_all_test_errors_vs_epochs(log_dir=".", save_path=None):
    """Plot test error vs epochs for all available single corruption levels."""
    corruptions = _find_all_single_corrs(log_dir)

    if not corruptions:
        print(f"No single corruption CSV files found in {log_dir}")
        return

    plt.figure(figsize=(11, 6))

    for corruption in sorted(corruptions):
        log_file = _log_file_for_corruption(log_dir, corruption)
        try:
            rows = _read_log_rows(log_file)
            epochs = [r["epoch"] for r in rows]
            test_error = [r["test_error"] for r in rows]
            plt.plot(
                epochs,
                test_error,
                marker="o",
                label=f"Corruption {corruption:.2f}",
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {log_file}: {e}")

    plt.title("Test Error vs Epochs (All Single Corruption Levels)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if not save_path:
        save_path = Path(log_dir) / "all_test_errors.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot AlexNet train/test error or accuracy vs epochs"
    )
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="Plot all single corruption levels together",
    )
    parser.add_argument(
        "--corruption",
        type=float,
        help="Specific corruption level, e.g. 0.5 (use with --metric)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=("error", "accuracy", "train-error", "test-error", "both"),
        default="both",
        help="Which graph to draw",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing alexnet_single_corr_*.csv files",
    )
    args = parser.parse_args()

    if args.all_levels:
        if args.metric in ("train-error", "error", "both"):
            out = Path(args.log_dir) / "all_single_corr_train_errors.png"
            plot_all_training_errors_vs_epochs(log_dir=args.log_dir, save_path=out)

        if args.metric in ("test-error", "error", "both"):
            out = Path(args.log_dir) / "all_single_corr_test_errors.png"
            plot_all_test_errors_vs_epochs(log_dir=args.log_dir, save_path=out)
    else:
        if not args.corruption:
            parser.error("--corruption is required when not using --all-levels")

        if args.metric in ("error", "both"):
            out = Path(args.log_dir) / f"single_corr_error_{args.corruption:.2f}.png"
            plot_error_vs_epochs(args.corruption, log_dir=args.log_dir, save_path=out)

        if args.metric in ("accuracy", "both"):
            out = Path(args.log_dir) / f"single_corr_accuracy_{args.corruption:.2f}.png"
            plot_accuracy_vs_epochs(
                args.corruption, log_dir=args.log_dir, save_path=out
            )


if __name__ == "__main__":
    main()
