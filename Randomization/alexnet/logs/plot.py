import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PREFIX = "alexnet_corr_"


def _level_to_corruption(level):
    return level / 10.0


def _corruption_to_filename(corruption_level):
    return f"{LOG_PREFIX}{corruption_level:.1f}.csv"


def _parse_levels(level_selector):
    if level_selector == 11:
        return list(range(0, 11))
    if 0 <= level_selector <= 10:
        return [level_selector]
    raise ValueError("level_selector must be an integer from 0 to 11.")


def _load_csv_rows(log_dir, level):
    corruption_level = _level_to_corruption(level)
    file_path = Path(log_dir) / _corruption_to_filename(corruption_level)
    if not file_path.exists():
        return None, None

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "train_error": float(row["train_error"]),
                    "test_loss": float(row["test_loss"]),
                    "test_accuracy": float(row["test_accuracy"]),
                    "test_error": float(row["test_error"]),
                }
            )
    return file_path, rows


def _collect_data(log_dir, level_selector):
    data_by_level = {}
    missing_levels = []

    for level in _parse_levels(level_selector):
        file_path, rows = _load_csv_rows(log_dir, level)
        if rows is None:
            missing_levels.append(level)
            continue
        data_by_level[level] = rows

    if not data_by_level:
        raise FileNotFoundError(
            f"No log files found in {log_dir} for selector {level_selector}."
        )

    if missing_levels:
        print(f"Skipping missing levels: {missing_levels}")

    return data_by_level


def plot_train_test_error_vs_epochs(level_selector, log_dir=".", save_path=None):
    data_by_level = _collect_data(log_dir, level_selector)
    plt.figure(figsize=(10, 6))

    for level, rows in sorted(data_by_level.items()):
        epochs = [r["epoch"] for r in rows]
        train_error = [r["train_error"] for r in rows]
        test_error = [r["test_error"] for r in rows]
        corr = _level_to_corruption(level)

        plt.plot(
            epochs, train_error, linestyle="--", label=f"Train Err (corr={corr:.1f})"
        )
        plt.plot(epochs, test_error, linestyle="-", label=f"Test Err (corr={corr:.1f})")

    title_suffix = "all levels" if level_selector == 11 else f"level={level_selector}"
    plt.title(f"Train/Test Error vs Epochs ({title_suffix})")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_train_test_accuracy_vs_epochs(level_selector, log_dir=".", save_path=None):
    data_by_level = _collect_data(log_dir, level_selector)
    plt.figure(figsize=(10, 6))

    for level, rows in sorted(data_by_level.items()):
        epochs = [r["epoch"] for r in rows]
        train_acc = [r["train_accuracy"] for r in rows]
        test_acc = [r["test_accuracy"] for r in rows]
        corr = _level_to_corruption(level)

        plt.plot(
            epochs, train_acc, linestyle="--", label=f"Train Acc (corr={corr:.1f})"
        )
        plt.plot(epochs, test_acc, linestyle="-", label=f"Test Acc (corr={corr:.1f})")

    title_suffix = "all levels" if level_selector == 11 else f"level={level_selector}"
    plt.title(f"Train/Test Accuracy vs Epochs ({title_suffix})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_train_test_loss_vs_epochs(level_selector, log_dir=".", save_path=None):
    data_by_level = _collect_data(log_dir, level_selector)
    plt.figure(figsize=(10, 6))

    for level, rows in sorted(data_by_level.items()):
        epochs = [r["epoch"] for r in rows]
        train_loss = [r["train_loss"] for r in rows]
        test_loss = [r["test_loss"] for r in rows]
        corr = _level_to_corruption(level)

        plt.plot(
            epochs, train_loss, linestyle="--", label=f"Train Loss (corr={corr:.1f})"
        )
        plt.plot(epochs, test_loss, linestyle="-", label=f"Test Loss (corr={corr:.1f})")

    title_suffix = "all levels" if level_selector == 11 else f"level={level_selector}"
    plt.title(f"Train/Test Loss vs Epochs ({title_suffix})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        type=int,
        default=11,
        help="0..10 for one level; 11 to plot all levels.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["error", "accuracy", "loss", "all"],
        default="all",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing alexnet_corr_*.csv logs.",
    )
    args = parser.parse_args()

    if args.plot in ("error", "all"):
        plot_train_test_error_vs_epochs(args.level, args.log_dir)
    if args.plot in ("accuracy", "all"):
        plot_train_test_accuracy_vs_epochs(args.level, args.log_dir)
    if args.plot in ("loss", "all"):
        plot_train_test_loss_vs_epochs(args.level, args.log_dir)


if __name__ == "__main__":
    main()
