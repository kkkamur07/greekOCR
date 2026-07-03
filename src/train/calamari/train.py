#!/usr/bin/env python3
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.logging.wandb_logger import WandbLogger  # noqa: E402
from train_utils import build_calamari_train_command, stream_process, validate_pack  # noqa: E402


@hydra.main(version_base=None, config_path="../../../configs", config_name="calamari_train")
def main(cfg: DictConfig) -> None:
    pack_dir = Path(to_absolute_path(cfg.data.pack_dir)).expanduser().resolve()
    output_dir = Path(to_absolute_path(cfg.output.root)).expanduser().resolve()
    train_images, val_images = validate_pack(pack_dir)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else f"calamari-train-{timestamp}"

    base_cmd, cmd_env = build_calamari_train_command()
    cmd = base_cmd + [
        "--network",
        str(cfg.model.network),
        "--n_augmentations",
        str(cfg.training.n_augmentations),
        "--trainer.output_dir",
        str(output_dir),
        "--trainer.epochs",
        str(cfg.training.epochs),
        "--early_stopping.n_to_go",
        str(cfg.training.early_stopping_patience),
        "--early_stopping.frequency",
        str(cfg.training.early_stopping_frequency),
        "--train.gt_extension",
        ".gt.txt",
        "--val.gt_extension",
        ".gt.txt",
        "--train.images",
        *train_images,
        "--val.images",
        *val_images,
    ]
    if cfg.training.gpu is not None and str(cfg.training.gpu) != "":
        cmd.extend(["--device.gpus", str(cfg.training.gpu)])

    header = [
        "=" * 40,
        f"Calamari training started: {datetime.now()}",
        f"  Pack:              {pack_dir}",
        f"  Output:            {output_dir}",
        f"  Train images:      {len(train_images)}",
        f"  Val images:        {len(val_images)}",
        f"  Network:           {cfg.model.network}",
        f"  Epochs:            {cfg.training.epochs}",
        f"  Augmentations:     {cfg.training.n_augmentations}",
        f"  Early stopping:    {cfg.training.early_stopping_patience}",
        f"  GPU:               {cfg.training.gpu if cfg.training.gpu is not None else 'CPU'}",
        "=" * 40,
    ]
    with log_file.open("w", encoding="utf-8") as handle:
        for line in header:
            print(line)
            handle.write(line + "\n")

    wandb_logger = WandbLogger(
        enabled=bool(cfg.wandb.enabled),
        project=str(cfg.wandb.project),
        entity=cfg.wandb.entity,
        name=run_name,
        mode=str(cfg.wandb.mode),
        save_dir=log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    try:
        stream_process(cmd, log_file, env=cmd_env)
    finally:
        wandb_logger.finish()
    print(f"Log saved to: {log_file}")
    print(f"Checkpoints under: {output_dir}")


if __name__ == "__main__":
    main()
