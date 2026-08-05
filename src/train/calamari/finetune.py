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
from train_utils import (  # noqa: E402
    build_calamari_command,
    stream_process,
    uses_gpu,
    validate_pack,
    write_header,
)


@hydra.main(version_base=None, config_path="../../../configs", config_name="calamari_finetune")
def main(cfg: DictConfig) -> None:
    pack_dir = Path(to_absolute_path(cfg.data.pack_dir)).expanduser().resolve()
    train_images, val_images = validate_pack(pack_dir)
    warmstart = Path(to_absolute_path(cfg.model.warmstart_checkpoint)).expanduser().resolve()
    # Calamari stores the trainer params (including the preprocessing pipeline) in a
    # sidecar .json, so a checkpoint without it cannot be warmstarted.
    if not warmstart.exists() or not Path(str(warmstart) + ".json").exists():
        raise SystemExit(f"Missing warmstart checkpoint or metadata: {warmstart}")

    output_dir = Path(to_absolute_path(cfg.output.root)).expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"finetune_{timestamp}.log"
    cer_log_file = log_dir / f"finetune_{timestamp}_cer.log"
    run_name = (
        str(cfg.wandb.name) if cfg.wandb.name is not None else f"calamari-finetune-{timestamp}"
    )

    cmd, cmd_env = build_calamari_command(
        cfg,
        output_dir=output_dir,
        train_images=train_images,
        val_images=val_images,
        extra_args=[
            "--warmstart.model",
            str(warmstart),
            "--codec.keep_loaded",
            "true",
            "--learning_rate.lr",
            str(cfg.training.learning_rate),
        ],
    )

    write_header(
        log_file,
        "Calamari finetune started",
        {
            "Pack": pack_dir,
            "Warmstart model": warmstart,
            "Output": output_dir,
            "Train images": len(train_images),
            "Val images": len(val_images),
            "Epochs": cfg.training.epochs,
            "Learning rate": cfg.training.learning_rate,
            "Augmentations": cfg.training.n_augmentations,
            "Early stopping": cfg.training.early_stopping_patience,
            "GPU": cfg.training.gpu if uses_gpu(cfg) else "CPU",
        },
    )

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
        stream_process(cmd, log_file, cer_log_file, env=cmd_env)
    finally:
        wandb_logger.finish()
    print(f"Log saved to: {log_file}")
    print(f"CER log saved to: {cer_log_file}")
    print(f"Fine-tuned checkpoints under: {output_dir}")


if __name__ == "__main__":
    main()
