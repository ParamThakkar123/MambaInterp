from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_history_csv(history: list[dict[str, float | int]], output_path: str | Path) -> str:
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        output_path.write_text("", encoding="utf-8")
        return str(output_path)

    preferred_order = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_macro_f1",
        "val_macro_f1",
        "lr",
    ]
    keys = list(history[0].keys())
    fieldnames = [k for k in preferred_order if k in keys] + [
        k for k in keys if k not in preferred_order
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return str(output_path)


def save_training_curves(
    history: list[dict[str, float | int]],
    output_dir: str | Path,
    model_name: str,
    val_fold: int,
) -> str | None:
    if not history:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_accuracy"]) for row in history]
    val_acc = [float(row["val_accuracy"]) for row in history]
    train_f1 = [float(row["train_macro_f1"]) for row in history]
    val_f1 = [float(row["val_macro_f1"]) for row in history]
    lrs = [float(row["lr"]) for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{model_name} (fold {val_fold})", fontsize=13)

    ax = axes[0][0]
    ax.plot(epochs, train_loss, label="train", marker="o", linewidth=2)
    ax.plot(epochs, val_loss, label="val", marker="o", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0][1]
    ax.plot(epochs, train_acc, label="train", marker="o", linewidth=2)
    ax.plot(epochs, val_acc, label="val", marker="o", linewidth=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1][0]
    ax.plot(epochs, train_f1, label="train", marker="o", linewidth=2)
    ax.plot(epochs, val_f1, label="val", marker="o", linewidth=2)
    ax.set_title("Macro F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1][1]
    ax.plot(epochs, lrs, marker="o", linewidth=2, color="#c44e52")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return str(plot_path)


def _load_history(history_path: str | Path) -> list[dict[str, Any]]:
    path = Path(history_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def save_small_experiment_plots(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories: list[tuple[str, list[dict[str, Any]]]] = []
    for summary in summaries:
        model_name = str(summary.get("model", "unknown"))
        history_path = summary.get("history_path")
        if not history_path:
            continue
        history = _load_history(history_path)
        if history:
            histories.append((model_name, history))

    if not histories:
        return {}

    generated: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for model_name, history in histories:
        epochs = [int(row["epoch"]) for row in history]
        val_acc = [float(row["val_accuracy"]) for row in history]
        val_f1 = [float(row["val_macro_f1"]) for row in history]
        axes[0].plot(epochs, val_acc, marker="o", linewidth=2, label=model_name)
        axes[1].plot(epochs, val_f1, marker="o", linewidth=2, label=model_name)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    curve_plot = output_dir / "small_experiments_curves.png"
    fig.savefig(curve_plot, dpi=180)
    plt.close(fig)
    generated["curves"] = str(curve_plot)

    ranked = sorted(
        (
            (str(summary.get("model", "unknown")), float(summary.get("best_val_accuracy", 0.0)))
            for summary in summaries
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    if ranked:
        labels = [item[0] for item in ranked]
        values = [item[1] for item in ranked]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(labels, values, color="#4c72b0")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title("Best Validation Accuracy by Model")
        ax.set_ylabel("Best Val Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        rank_plot = output_dir / "small_experiments_best_accuracy.png"
        fig.savefig(rank_plot, dpi=180)
        plt.close(fig)
        generated["ranking"] = str(rank_plot)

    return generated
