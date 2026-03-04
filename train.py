from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ESC50AudioConfig, ESC50Dataset
from .engine import evaluate, set_seed, train_one_epoch
from .models import MODEL_NAMES, ModelConfig, build_model
from .plotting import save_history_csv, save_training_curves


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str = "runs"
    model: str = "mamba_spectrogram"
    val_fold: int = 1
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    max_train_steps: int = 0
    max_val_steps: int = 0
    # Audio config
    sample_rate: int = 16000
    clip_duration_s: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    # Model config
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
    spectrogram_patch_freq: int = 16
    spectrogram_patch_time: int = 16
    spectrogram_stride_freq: int = 16
    spectrogram_stride_time: int = 16


def _normalize_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _as_jsonable_config(cfg: TrainConfig) -> dict[str, object]:
    data = asdict(cfg)
    return data


def _build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    audio_cfg = ESC50AudioConfig(
        sample_rate=cfg.sample_rate,
        clip_duration_s=cfg.clip_duration_s,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    train_folds = [f for f in [1, 2, 3, 4, 5] if f != cfg.val_fold]
    val_folds = [cfg.val_fold]

    train_ds = ESC50Dataset(
        data_root=cfg.data_root,
        folds=train_folds,
        audio_config=audio_cfg,
        random_crop=True,
    )
    val_ds = ESC50Dataset(
        data_root=cfg.data_root,
        folds=val_folds,
        audio_config=audio_cfg,
        random_crop=False,
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def run_training(cfg: TrainConfig) -> dict[str, object]:
    if cfg.model not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{cfg.model}'. Choices: {MODEL_NAMES}")

    set_seed(cfg.seed)
    device = _normalize_device(cfg.device)

    output_dir = Path(cfg.output_dir) / f"{cfg.model}_fold{cfg.val_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _build_dataloaders(cfg)
    model_cfg = ModelConfig(
        model_name=cfg.model,
        num_classes=50,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        dropout=cfg.dropout,
        spectrogram_patch_freq=cfg.spectrogram_patch_freq,
        spectrogram_patch_time=cfg.spectrogram_patch_time,
        spectrogram_stride_freq=cfg.spectrogram_stride_freq,
        spectrogram_stride_time=cfg.spectrogram_stride_time,
    )
    model = build_model(model_cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    max_train_steps = cfg.max_train_steps if cfg.max_train_steps > 0 else None
    max_val_steps = cfg.max_val_steps if cfg.max_val_steps > 0 else None

    best_val_acc = -1.0
    best_ckpt_path = output_dir / "best.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_steps=max_train_steps,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_steps=max_val_steps,
            desc="val",
        )
        scheduler.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "train_macro_f1": train_metrics.macro_f1,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "val_macro_f1": val_metrics.macro_f1,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"[{cfg.model}] epoch={epoch:03d} "
            f"train_acc={train_metrics.accuracy:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} "
            f"val_f1={val_metrics.macro_f1:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": cfg.model,
            "model_config": asdict(model_cfg),
            "train_config": _as_jsonable_config(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics.to_dict(),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            torch.save(checkpoint, best_ckpt_path)

    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    history_csv_path = output_dir / "history.csv"
    save_history_csv(history, history_csv_path)

    training_plot_path: str | None = None
    plot_warning: str | None = None
    try:
        training_plot_path = save_training_curves(
            history=history,
            output_dir=output_dir,
            model_name=cfg.model,
            val_fold=cfg.val_fold,
        )
    except ImportError:
        plot_warning = (
            "matplotlib is not installed; skipped saving training curve plots. "
            "Install with: pip install matplotlib"
        )
        print(f"Warning: {plot_warning}")

    summary = {
        "model": cfg.model,
        "val_fold": cfg.val_fold,
        "best_val_accuracy": best_val_acc,
        "best_checkpoint": str(best_ckpt_path),
        "history_path": str(history_path),
        "history_csv_path": str(history_csv_path),
        "training_plot_path": training_plot_path,
        "plot_warning": plot_warning,
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train spectrogram-based Mamba audio classifier on ESC-50"
    )
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--output-dir", default="runs", type=str)
    parser.add_argument("--model", default="mamba_spectrogram", choices=MODEL_NAMES)
    parser.add_argument("--val-fold", default=1, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--max-train-steps", default=0, type=int)
    parser.add_argument("--max-val-steps", default=0, type=int)
    parser.add_argument("--sample-rate", default=16000, type=int)
    parser.add_argument("--clip-duration-s", default=5.0, type=float)
    parser.add_argument("--n-fft", default=1024, type=int)
    parser.add_argument("--hop-length", default=512, type=int)
    parser.add_argument("--n-mels", default=128, type=int)
    parser.add_argument("--d-model", default=256, type=int)
    parser.add_argument("--n-layers", default=6, type=int)
    parser.add_argument("--d-state", default=16, type=int)
    parser.add_argument("--d-conv", default=4, type=int)
    parser.add_argument("--expand", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--spectrogram-patch-freq", default=16, type=int)
    parser.add_argument("--spectrogram-patch-time", default=16, type=int)
    parser.add_argument("--spectrogram-stride-freq", default=16, type=int)
    parser.add_argument("--spectrogram-stride-time", default=16, type=int)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model=args.model,
        val_fold=args.val_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        sample_rate=args.sample_rate,
        clip_duration_s=args.clip_duration_s,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        spectrogram_patch_freq=args.spectrogram_patch_freq,
        spectrogram_patch_time=args.spectrogram_patch_time,
        spectrogram_stride_freq=args.spectrogram_stride_freq,
        spectrogram_stride_time=args.spectrogram_stride_time,
    )
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
