"""Checkpoint evaluation helpers for the canonical PyTorch Calamari workflow."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_calamari_checkpoint
from .codec import CharacterCodec
from .data import CalamariLineDataset, collate_ctc
from .trainer import evaluate_model


def evaluate_checkpoint(
    checkpoint: Path,
    data_root: Path,
    *,
    split: str,
    batch_size: int,
    workers: int,
    device: str,
) -> dict[str, float]:
    """Evaluate a saved checkpoint against one dataset split."""
    model, metadata = load_calamari_checkpoint(checkpoint)
    codec = CharacterCodec(metadata.charset)
    dataset = CalamariLineDataset(data_root, split, codec, metadata.line_height)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_ctc,
    )
    torch_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    return evaluate_model(model.to(torch_device), loader, codec, torch_device, nn.CTCLoss(blank=0))
