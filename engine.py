from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: move_to_device(v, device) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(move_to_device(v, device) for v in inputs)
    return inputs


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


def _compute_macro_f1(targets: list[int], preds: list[int]) -> float:
    if not targets:
        return 0.0
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        return 0.0
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_steps: int | None = None,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, (inputs, targets) in enumerate(progress, start=1):
        if max_steps is not None and step > max_steps:
            break

        inputs = move_to_device(inputs, device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == targets).sum().item())
            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            progress.set_postfix(
                loss=f"{(total_loss / max(total_samples, 1)):.4f}",
                acc=f"{(total_correct / max(total_samples, 1)):.4f}",
            )

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
        macro_f1=_compute_macro_f1(all_targets, all_preds),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_steps: int | None = None,
    desc: str = "eval",
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    progress = tqdm(dataloader, desc=desc, leave=False)
    for step, (inputs, targets) in enumerate(progress, start=1):
        if max_steps is not None and step > max_steps:
            break

        inputs = move_to_device(inputs, device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=-1)
        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == targets).sum().item())
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        progress.set_postfix(
            loss=f"{(total_loss / max(total_samples, 1)):.4f}",
            acc=f"{(total_correct / max(total_samples, 1)):.4f}",
        )

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
        macro_f1=_compute_macro_f1(all_targets, all_preds),
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
