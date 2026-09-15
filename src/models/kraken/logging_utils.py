"""Local metric logging for Kraken fine-tuning."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


def scalar(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LocalMetricsLogger(Callback):
    """Write train/validation losses to JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), **payload}) + "\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = scalar(outputs.get("loss") if isinstance(outputs, dict) else outputs)
        loss = loss if loss is not None else scalar(trainer.callback_metrics.get("train_loss"))
        if loss is not None:
            self.write({
                "event": "train_batch",
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "batch_idx": int(batch_idx),
                "train_loss": loss,
            })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = scalar(outputs.get("val_loss") if isinstance(outputs, dict) else None)
        loss = loss if loss is not None else scalar(trainer.callback_metrics.get("val_loss"))
        if loss is not None:
            self.write({
                "event": "validation_batch",
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "batch_idx": int(batch_idx),
                "val_loss": loss,
            })

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            key: scalar(value)
            for key, value in trainer.callback_metrics.items()
            if key.startswith("train_")
        }
        metrics = {key: value for key, value in metrics.items() if value is not None}
        if metrics:
            self.write({
                "event": "train_epoch",
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                **metrics,
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {
            key: scalar(value)
            for key, value in trainer.callback_metrics.items()
            if key.startswith("val_")
        }
        metrics = {key: value for key, value in metrics.items() if value is not None}
        if metrics:
            self.write({
                "event": "validation_epoch",
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                **metrics,
            })
