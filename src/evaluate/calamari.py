"""Evaluate a PyTorch Calamari checkpoint with the composed Hydra config."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from ..models.calamari.evaluate import evaluate_checkpoint


@hydra.main(version_base=None, config_path="../../config/calamari", config_name="configs")
def main(cfg: DictConfig) -> None:
    metrics = evaluate_checkpoint(
        Path(to_absolute_path(cfg.evaluation.checkpoint)).expanduser().resolve(),
        Path(to_absolute_path(cfg.data.dir)).expanduser().resolve(),
        split=str(cfg.evaluation.split),
        batch_size=int(cfg.evaluation.batch_size),
        workers=int(cfg.training.num_workers),
        device=str(cfg.training.device),
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
