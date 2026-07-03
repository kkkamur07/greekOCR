#!/usr/bin/env python3
"""Fine-tune Kraken polygon segmentation with refined annotations."""

from __future__ import annotations

import copy
import logging
import os
import sys
import threading
from importlib import resources
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


LOGGER = logging.getLogger(__name__)


def tee_terminal_to_file(path: Path) -> None:
    """Mirror stdout/stderr to both the terminal and log.err."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    read_fd, write_fd = os.pipe()

    def _tee() -> None:
        with os.fdopen(read_fd, "rb", buffering=0) as reader:
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                os.write(saved_stdout, chunk)
                os.write(log_fd, chunk)

    thread = threading.Thread(target=_tee, name="kraken-log-tee", daemon=True)
    thread.start()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    # Keep descriptors alive for the process lifetime.
    tee_terminal_to_file._fds = (log_fd, saved_stdout, saved_stderr, thread)  # type: ignore[attr-defined]


def ensure_local_kraken_importable() -> None:
    local_kraken_src = Path(to_absolute_path("kraken"))
    if local_kraken_src.is_dir() and str(local_kraken_src) not in sys.path:
        sys.path.insert(0, str(local_kraken_src))


def optional_path(value: str | None) -> Path | None:
    return Path(to_absolute_path(value)) if value else None


def default_kraken_model_path() -> Path:
    model_path = Path(str(resources.files("kraken").joinpath("blla.mlmodel")))
    if not model_path.exists():
        raise FileNotFoundError("Could not find Kraken's bundled blla.mlmodel. Set model.load in kraken_seg.yaml.")
    return model_path


def build_class_mapping(cfg: DictConfig) -> dict:
    return {
        "aux": {
            "_start_separator": int(cfg.classes.start_separator_channel),
            "_end_separator": int(cfg.classes.end_separator_channel),
        },
        "baselines": {str(cfg.classes.baseline_name): int(cfg.classes.baseline_channel)},
        "regions": {str(cfg.classes.region_name): int(cfg.classes.region_channel)},
    }


def fine_tune(cfg: DictConfig, *, run_dir: Path, log_dir: Path) -> Path | None:
    ensure_local_kraken_importable()

    import torch
    import torch.nn.functional as F
    from lightning.pytorch.callbacks import ModelCheckpoint

    from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig
    from kraken.models.convert import convert_models
    from kraken.models.writers import write_safetensors
    from kraken.train import BLLASegmentationModel, KrakenTrainer

    from model.transcription.kraken.dataloader import TeacherPseudoLabelDataModule
    from model.transcription.kraken.dataset import build_segmentation_documents, split_documents
    from model.transcription.kraken.logging_utils import LocalMetricsLogger

    class FineTuneSegmentationModel(BLLASegmentationModel):
        """Kraken model with explicit validation-loss logging."""

        def validation_step(self, batch, batch_idx):
            x, y = batch["image"], batch["target"]
            output, _ = self.net(x)
            output = F.interpolate(output, size=(y.size(2), y.size(3)))
            loss = self.criterion(output, y)
            if self.dice is not None:
                loss = loss + self.dice_weight * self.dice(torch.sigmoid(output), y)
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            super().validation_step(batch, batch_idx)
            return {"val_loss": loss.detach()}

    def convert_best_checkpoint(checkpoint_path: Path, output_path: Path, weights_format: str) -> Path:
        try:
            return Path(convert_models([checkpoint_path], output_path, weights_format=weights_format))
        except Exception:
            if weights_format != "safetensors":
                raise
            LOGGER.warning("Default converter failed; retrying with trusted local checkpoint load.", exc_info=True)
            module = FineTuneSegmentationModel.load_from_checkpoint(
                checkpoint_path,
                weights_only=False,
                map_location="cpu",
            )
            return Path(write_safetensors([module.net], output_path))

    checkpoint_dir = run_dir / str(cfg.output.checkpoint_dirname)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Run directory: %s", run_dir)
    class_mapping = build_class_mapping(cfg)
    model_config = BLLASegmentationTrainingConfig(
        resize=str(cfg.model.resize),
        quit="fixed",
        epochs=int(cfg.training.epochs),
        min_epochs=int(cfg.training.min_epochs),
        lrate=float(cfg.training.lrate),
        weight_decay=float(cfg.training.weight_decay),
        dice_weight=float(cfg.training.dice_weight),
        weights_format=str(cfg.model.weights_format),
        schedule="cosine",
        cos_t_max=int(cfg.training.epochs),
        cos_min_lr=float(cfg.training.lrate) / 10,
    )

    load_path = optional_path(cfg.model.load) or default_kraken_model_path()
    LOGGER.info("Loading base Kraken segmentation model: %s", load_path)
    if str(load_path).endswith(".ckpt"):
        model = FineTuneSegmentationModel.load_from_checkpoint(load_path, config=model_config, weights_only=False)
    else:
        model = FineTuneSegmentationModel.load_from_weights(load_path, config=model_config)

    # This is fine-tuning: start from Kraken weights and preserve non-polygon channels with a frozen teacher.
    model.net.user_metadata["class_mapping"] = copy.deepcopy(class_mapping)
    model.net.user_metadata["_full_class_mapping"] = copy.deepcopy(class_mapping)
    teacher_net = copy.deepcopy(model.net)
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

    docs = build_segmentation_documents(
        data_root=Path(to_absolute_path(cfg.data.root)),
        annotations_dir=Path(to_absolute_path(cfg.data.annotations_dir)),
        region_name=str(cfg.classes.region_name),
        use_kraken_ceiling=bool(cfg.data.use_kraken_ceiling),
        prefer_processed=bool(cfg.data.prefer_processed),
    )
    train_docs, val_docs = split_documents(
        docs,
        validation_ratio=float(cfg.data.validation_ratio),
        seed=int(cfg.data.seed),
    )
    LOGGER.info("Train pages: %d; validation pages: %d", len(train_docs), len(val_docs))

    data_config = BLLASegmentationTrainingDataConfig(
        format_type=None,
        training_data=train_docs,    #!training data
        evaluation_data=val_docs,    #!validation data
        num_workers=int(cfg.training.num_workers),
        augment=bool(cfg.training.augment),
        line_width=int(cfg.training.line_width),
        line_class_mapping=class_mapping["baselines"],
        region_class_mapping=class_mapping["regions"],
    )
    data_module = TeacherPseudoLabelDataModule(
        data_config,
        teacher_net=teacher_net,
        pseudo_channels=tuple(int(channel) for channel in cfg.classes.pseudo_channels),
    )

    if bool(cfg.get("dry_run", False)):
        LOGGER.info("Dry run complete. Data and model configuration were built successfully.")
        return None

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="region_polygon_finetune_{epoch:02d}-{val_metric:.4f}",
        monitor="val_metric",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback, LocalMetricsLogger(log_dir / str(cfg.logging.metrics_file))]
    LOGGER.info("Writing local metrics to %s", log_dir / str(cfg.logging.metrics_file))

    trainer_logger, pl_logger = None, None
    if bool(cfg.wandb.enabled):
        from lightning.pytorch.loggers import WandbLogger

        trainer_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            save_dir=str(log_dir),
            log_model=bool(cfg.wandb.log_model),
            mode=cfg.wandb.mode,
        )
        pl_logger = "custom"
        LOGGER.info("Enabled W&B logging for project %s", cfg.wandb.project)

    trainer = KrakenTrainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        max_epochs=int(cfg.training.epochs),
        min_epochs=int(cfg.training.min_epochs),
        freeze_backbone=int(cfg.training.freeze_backbone),
        enable_progress_bar=True,
        enable_summary=True,
        callbacks=callbacks,
        logger=trainer_logger,
        pl_logger=pl_logger,
        log_dir=log_dir,
        gradient_clip_val=model_config.gradient_clip_val,
        accumulate_grad_batches=model_config.accumulate_grad_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        use_distributed_sampler=False,
    )
    trainer.fit(model, data_module)

    if not checkpoint_callback.best_model_path:
        LOGGER.warning("Fine-tuning finished without a best checkpoint.")
        return None

    score = checkpoint_callback.best_model_score.item()
    weights_format = str(cfg.model.weights_format)
    weights_path = run_dir / f"best_region_polygon_{score:.4f}.{weights_format}"
    converted = convert_best_checkpoint(Path(checkpoint_callback.best_model_path), weights_path, weights_format)
    LOGGER.info("Best checkpoint: %s", checkpoint_callback.best_model_path)
    LOGGER.info("Converted weights: %s", converted)
    return converted


@hydra.main(version_base=None, config_path="../../../configs", config_name="kraken_seg")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log_dir = run_dir / "logs"
    tee_terminal_to_file(log_dir / str(cfg.logging.terminal_file))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    fine_tune(cfg, run_dir=run_dir, log_dir=log_dir)


if __name__ == "__main__":
    main()
