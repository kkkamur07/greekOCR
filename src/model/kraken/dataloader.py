"""Kraken datamodule wrapper for teacher pseudo-label channels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kraken.configs import BLLASegmentationTrainingDataConfig
from kraken.train import BLLASegmentationDataModule


class TeacherPseudoLabelDataset:
    """Dataset wrapper that fills missing Kraken channels from a frozen teacher."""

    def __init__(
        self,
        wrapped,
        *,
        teacher_net: torch.nn.Module,
        pseudo_channels: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        self.wrapped = wrapped
        # Kraken's trainer expects train_set.dataset to expose BaselineSet fields.
        self.dataset = wrapped.dataset if hasattr(wrapped, "dataset") else wrapped
        self.teacher_net = teacher_net
        self.pseudo_channels = pseudo_channels

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, idx):
        sample = self.wrapped[idx]
        image = sample["image"]
        target = sample["target"].clone()

        teacher_device = next(self.teacher_net.parameters()).device
        with torch.no_grad():
            logits, _ = self.teacher_net(image.unsqueeze(0).to(teacher_device))
            logits = F.interpolate(logits, size=target.shape[-2:])
            teacher_probs = torch.sigmoid(logits).squeeze(0).cpu()

        for channel in self.pseudo_channels:
            if channel < target.shape[0] and channel < teacher_probs.shape[0]:
                target[channel] = teacher_probs[channel]

        sample["target"] = target
        return sample


class TeacherPseudoLabelDataModule(BLLASegmentationDataModule):
    """Kraken data module that injects teacher heatmaps before training."""

    def __init__(
        self,
        data_config: BLLASegmentationTrainingDataConfig,
        *,
        teacher_net: torch.nn.Module,
        pseudo_channels: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        super().__init__(data_config)
        self.teacher_net = teacher_net
        self.pseudo_channels = pseudo_channels

    def setup(self, stage: str = None):
        super().setup(stage)
        if stage not in (None, "fit"):
            return
        if not isinstance(self.train_set, TeacherPseudoLabelDataset):
            self.train_set = TeacherPseudoLabelDataset(
                self.train_set,
                teacher_net=self.teacher_net,
                pseudo_channels=self.pseudo_channels,
            )
        if not isinstance(self.val_set, TeacherPseudoLabelDataset):
            self.val_set = TeacherPseudoLabelDataset(
                self.val_set,
                teacher_net=self.teacher_net,
                pseudo_channels=self.pseudo_channels,
            )
