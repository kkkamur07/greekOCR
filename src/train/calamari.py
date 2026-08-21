"""Train or fine-tune the Calamari CTC recognizer with Hydra."""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..logging.wandb import WandbLogger
from ..models.calamari.trainer import CalamariTrainingSettings, train_calamari


def _set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../../config/calamari", config_name="configs")
def main(cfg: DictConfig) -> None:
    _set_seed(int(cfg.training.seed))
    data_root = Path(to_absolute_path(cfg.data.dir)).expanduser().resolve()
    output_dir = Path(to_absolute_path(cfg.output.root)).expanduser().resolve()
    checkpoint = (
        Path(to_absolute_path(cfg.training.checkpoint)).expanduser().resolve()
        if cfg.training.checkpoint is not None
        else None
    )
    settings = CalamariTrainingSettings(
        mode=str(cfg.training.mode),
        checkpoint=checkpoint,
        epochs=int(cfg.training.epochs),
        batch_size=int(cfg.training.batch_size),
        workers=int(cfg.training.num_workers),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        line_height=int(cfg.model.line_height),
        device=str(cfg.training.device),
        temperature=float(cfg.model.temperature),
        train_split=str(cfg.data.train_split),
        validation_split=str(cfg.data.validation_split),
        n_augmentations=int(cfg.augmentation.n_augmentations),
        augmentation_probability=float(cfg.augmentation.probability),
        ema_decay=float(cfg.training.ema_decay),
        logging_steps=int(cfg.logging.steps),
        warmup_ratio=float(cfg.training.warmup_ratio),
        checkpoint_top_k=int(cfg.training.checkpoint_top_k),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    log_dir = (
        Path(to_absolute_path(cfg.logging.root)).expanduser().resolve()
        / "calamari"
        / output_dir.name
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("Resolved Calamari configuration must be a mapping.")
    wandb_logger = WandbLogger(
        enabled=bool(cfg.wandb.enabled),
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity) if cfg.wandb.entity is not None else None,
        name=str(cfg.wandb.name) if cfg.wandb.name is not None else None,
        mode=str(cfg.wandb.mode),
        save_dir=log_dir,
        config=resolved_config,
    )
    metrics_file = log_dir / "metrics.jsonl"

    def report(metrics: dict[str, float]) -> None:
        print(json.dumps(metrics, sort_keys=True))
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        wandb_logger.log_metrics(metrics, step=int(metrics["step"]))

    try:
        _, _, best = train_calamari(data_root, output_dir, settings, report=report)
    finally:
        wandb_logger.finish()
    print(json.dumps({"best": best, "checkpoint": str(output_dir / "best.pt")}, sort_keys=True))


if __name__ == "__main__":
    main()
