"""Fine-tune a Syriac TrOCR model with Hydra configuration."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import Seq2SeqTrainingArguments

from ..logging.local import configure_file_logging
from ..models.trocr.augmentation import LineAugmentation
from ..models.trocr.dataloader import LineDataset, TrOCRCollator
from ..models.trocr.metrics import compute_token_metrics
from ..models.trocr.model_builder import build_model
from ..models.trocr.tokenizer import build_processor, load_tokenizer, resolve_tokenizer_path
from ..models.trocr.trainer import MetricsCsvCallback, TrOCRTrainer


LOGGER = logging.getLogger(__name__)

_SWEEP_PARAMETER_PATHS = frozenset(
    {
        "experiment",
    }
)


def log_training_summary(
    cfg: DictConfig,
    model,
    tokenizer,
    processor,
    train_dataset: LineDataset,
    eval_dataset: LineDataset,
    training_args: Seq2SeqTrainingArguments,
) -> None:
    """Log the resolved training setup and actual model sizes as a table."""
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    decoder_parameters = sum(parameter.numel() for parameter in model.decoder.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    accumulation_steps = training_args.gradient_accumulation_steps
    image_size = processor.image_processor.size
    rows = [
        ("Model", "Source", model.config.name_or_path),
        ("Model", "Encoder frozen", str(bool(cfg.model.freeze_encoder))),
        ("Parameters", "Total", f"{total_parameters:,}"),
        ("Parameters", "Trainable", f"{trainable_parameters:,}"),
        ("Parameters", "Encoder", f"{encoder_parameters:,}"),
        ("Parameters", "Decoder", f"{decoder_parameters:,}"),
        ("Tokenizer", "Type", tokenizer.__class__.__name__),
        ("Tokenizer", "Path", str(cfg.tokenizer.path)),
        ("Tokenizer", "Vocabulary size", f"{len(tokenizer):,}"),
        ("Tokenizer", "Fast tokenizer", str(bool(cfg.tokenizer.use_fast))),
        ("Decoder", "Reinitialize mode", str(cfg.decoder.reinitialize)),
        ("Decoder", "Tied input/output embeddings", str(bool(cfg.decoder.tied))),
        ("Decoder", "Dropout", str(cfg.decoder.dropout)),
        ("Tokenization", "Special tokens", "Not added; EOS is appended manually"),
        ("Tokenization", "Max input tokens", str(cfg.training.max_target_length - 1)),
        ("Tokenization", "Max label tokens", str(cfg.training.max_target_length)),
        ("Image preprocessing", "Resize", str(image_size)),
        ("Data", "Training examples", f"{len(train_dataset):,}"),
        ("Data", "Validation examples", f"{len(eval_dataset):,}"),
        ("Training", "Epochs", str(cfg.training.epochs)),
        ("Training", "Per-device train batch size", str(training_args.per_device_train_batch_size)),
        ("Training", "Per-device eval batch size", str(training_args.per_device_eval_batch_size)),
        ("Training", "Gradient accumulation steps", str(accumulation_steps)),
        (
            "Training",
            "Effective batch size per process",
            str(training_args.per_device_train_batch_size * accumulation_steps),
        ),
        ("Training", "Mixed precision (FP16)", str(training_args.fp16)),
        ("Training", "Data loader workers", str(training_args.dataloader_num_workers)),
        ("Checkpoints", "Ranking metric", "Evaluation CER"),
        ("Checkpoints", "Retained", str(cfg.training.checkpoint_top_k)),
        ("Optimizer", "Learning rate", str(training_args.learning_rate)),
        ("Optimizer", "Weight decay", str(training_args.weight_decay)),
        ("Optimizer", "Max gradient norm", str(training_args.max_grad_norm)),
        ("Optimizer", "Warmup ratio", str(training_args.warmup_ratio)),
        ("Optimizer", "LR scheduler", str(training_args.lr_scheduler_type)),
        ("LoRA", "Enabled", str(bool(cfg.lora_adaptors.enabled))),
        ("LoRA", "Rank", str(cfg.lora_adaptors.rank)),
        (
            "LoRA",
            "Alpha/rank ratio",
            str(cfg.lora_adaptors.alpha_rank_ratio),
        ),
        ("LoRA", "Dropout", str(cfg.lora_adaptors.dropout)),
        ("LoRA", "Encoder layers", f"Last {cfg.lora_adaptors.num_layers}"),
        ("LoRA", "Attention targets", ", ".join(cfg.lora_adaptors.target_modules)),
        ("Augmentation", "Probability", str(cfg.augmentation.probability)),
        ("Augmentation", "Mode", str(cfg.augmentation.mode)),
        ("Augmentation", "Operations per image", str(cfg.augmentation.num_operations)),
        ("Augmentation", "Magnitude", str(cfg.augmentation.magnitude)),
    ]
    widths = [max(len(str(row[index])) for row in rows + [("Section", "Setting", "Value")]) for index in range(3)]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header = "| " + " | ".join(
        value.ljust(width) for value, width in zip(("Section", "Setting", "Value"), widths)
    ) + " |"
    body = [
        "| " + " | ".join(str(value).ljust(width) for value, width in zip(row, widths)) + " |"
        for row in rows
    ]
    LOGGER.info("Training configuration:\n%s\n%s\n%s\n%s", separator, header, separator, "\n".join(body + [separator]))


def configure_wandb(cfg: DictConfig, log_dir: Path) -> list[str]:
    """Configure Hugging Face's optional W&B integration from Hydra settings."""
    if not cfg.wandb.enabled:
        return []

    os.environ["WANDB_PROJECT"] = str(cfg.wandb.project)
    os.environ["WANDB_MODE"] = str(cfg.wandb.mode)
    os.environ["WANDB_DIR"] = str(log_dir)
    if cfg.wandb.entity is not None:
        os.environ["WANDB_ENTITY"] = str(cfg.wandb.entity)
    else:
        os.environ.pop("WANDB_ENTITY", None)
    if cfg.wandb.name is not None:
        os.environ["WANDB_NAME"] = str(cfg.wandb.name)
    else:
        os.environ.pop("WANDB_NAME", None)
    return ["wandb"]


def apply_sweep_experiment(cfg: DictConfig, experiment_name: str) -> None:
    """Apply the fixed configuration assigned to a named sweep experiment."""
    experiment = cfg.sweep.experiments.get(experiment_name)
    if experiment is None:
        available = ", ".join(sorted(cfg.sweep.experiments.keys()))
        raise ValueError(
            f"Unknown W&B sweep experiment {experiment_name!r}. "
            f"Available experiments: {available}."
        )

    overrides = OmegaConf.to_container(experiment, resolve=True)
    if not isinstance(overrides, dict):
        raise TypeError(
            f"Sweep experiment {experiment_name!r} must contain configuration overrides."
        )
    for path, value in overrides.items():
        OmegaConf.update(cfg, str(path), value, merge=False)
    LOGGER.info("Applied W&B sweep experiment %s: %s", experiment_name, overrides)


def initialize_run(cfg: DictConfig, log_dir: Path) -> str:
    """Initialize W&B when enabled and return a unique run directory name."""
    if not cfg.wandb.enabled:
        run_name = str(cfg.wandb.name or "tr_ocr")
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-")
        return f"{safe_name or 'tr_ocr'}-{run_id}"

    configure_wandb(cfg, log_dir)
    import wandb

    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity) if cfg.wandb.entity is not None else None,
        mode=str(cfg.wandb.mode),
        dir=str(log_dir),
    )
    if "WANDB_SWEEP_ID" in os.environ:
        unknown_parameters = set(run.config.keys()) - _SWEEP_PARAMETER_PATHS
        if unknown_parameters:
            raise ValueError(
                "Unsupported W&B sweep parameters: "
                f"{sorted(unknown_parameters)}"
            )
        apply_sweep_experiment(cfg, str(run.config["experiment"]))

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run.name).strip("-")
    return f"{safe_name or 'tr_ocr'}-{run.id}"


@hydra.main(version_base=None, config_path="../../config/trocr", config_name="configs")
def main(cfg: DictConfig) -> None:
    initial_output_dir = Path(to_absolute_path(cfg.output.root)).expanduser().resolve()
    initial_log_dir = (
        Path(to_absolute_path(cfg.logging.root)).expanduser().resolve()
        / "trocr"
        / initial_output_dir.name
    )
    run_directory = initialize_run(cfg, initial_log_dir)
    OmegaConf.update(
        cfg,
        "output.root",
        str(Path(str(cfg.output.root)) / run_directory),
        merge=False,
    )

    output_dir = Path(to_absolute_path(cfg.output.root)).expanduser().resolve()
    log_dir = (
        Path(to_absolute_path(cfg.logging.root)).expanduser().resolve()
        / "trocr"
        / output_dir.name
    )
    configure_file_logging(
        log_file=log_dir / "train.log",
        level=str(cfg.logging.level),
    )

    checkpoint_dir = cfg.model.get("checkpoint_dir")
    model_root = (
        Path(to_absolute_path(checkpoint_dir)).expanduser().resolve()
        if checkpoint_dir
        else None
    )
    is_resume_checkpoint = (
        model_root is not None and (model_root / "trainer_state.json").exists()
    )
    model_source = str(model_root) if model_root and model_root.exists() else str(cfg.model.name)
    tokenizer_path = resolve_tokenizer_path(cfg.tokenizer.path)
    tokenizer = load_tokenizer(
        str(tokenizer_path),
        use_fast=bool(cfg.tokenizer.use_fast),
        pad_token=str(cfg.tokenizer.pad_token),
    )
    processor = build_processor(model_source, tokenizer)
    lora_config = OmegaConf.to_container(cfg.lora_adaptors, resolve=True)
    if not isinstance(lora_config, dict):
        raise TypeError("LoRA configuration must resolve to a mapping.")
    model = build_model(
        model_source,
        tokenizer,
        max_target_length=cfg.training.max_target_length,
        freeze_visual_encoder=bool(cfg.model.freeze_encoder),
        reinitialize_decoder=(
            "none" if is_resume_checkpoint else str(cfg.decoder.reinitialize)
        ),
        tie_decoder_embeddings=bool(cfg.decoder.tied),
        decoder_dropout=float(cfg.decoder.dropout),
        lora_config=lora_config,
    )

    augmentation = LineAugmentation(
        probability=cfg.augmentation.probability,
        mode=str(cfg.augmentation.mode),
        num_operations=int(cfg.augmentation.num_operations),
        magnitude=cfg.augmentation.magnitude,
        exclude_groups=tuple(cfg.augmentation.exclude_groups),
        exclude_operations=tuple(cfg.augmentation.exclude_operations),
    )
    data_dir = Path(to_absolute_path(cfg.data.dir)).expanduser().resolve()
    train_dataset = LineDataset(data_dir, "train", augmentation=augmentation)
    eval_dataset = LineDataset(data_dir, "val")

    def compute_metrics(prediction) -> dict[str, float]:
        predictions, labels = prediction
        return compute_token_metrics(predictions, labels, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.epochs,
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        logging_steps=cfg.logging.steps,
        eval_strategy="epoch",
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=cfg.training.max_target_length,
        fp16=torch.cuda.is_available() and not cfg.training.no_fp16,
        dataloader_num_workers=cfg.training.num_workers,
        remove_unused_columns=False,
        report_to=configure_wandb(cfg, log_dir),
        run_name=cfg.wandb.name,
        seed=cfg.training.seed,
    )
    log_training_summary(
        cfg,
        model,
        tokenizer,
        processor,
        train_dataset,
        eval_dataset,
        training_args,
    )
    trainer = TrOCRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrOCRCollator(processor, cfg.training.max_target_length),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCsvCallback(log_dir / "metrics.csv")],
        checkpoint_top_k=int(cfg.training.checkpoint_top_k),
    )
    resume_checkpoint = str(model_root) if is_resume_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.image_processor.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
