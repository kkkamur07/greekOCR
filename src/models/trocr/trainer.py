"""Training metrics for the Hugging Face TrOCR trainer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from transformers import Seq2SeqTrainer, TrainerCallback

from .metrics import compute_token_metrics


OCR_METRIC_NAMES = (
    "cer",
    "wer",
    "wpa",
    "normalized_edit_distance",
    "exact_match_accuracy",
    "sroie_precision",
    "sroie_recall",
    "sroie_f1",
)


class MetricsCsvCallback(TrainerCallback):
    """Write one row per validation epoch to a local CSV file."""

    fields = (
        "epoch",
        "step",
        "train_loss",
        "encoder_grad_norm",
        "decoder_grad_norm",
        "decoder_input_embedding_grad_norm",
        "decoder_last_4_layer_1_grad_norm",
        "decoder_last_4_layer_2_grad_norm",
        "decoder_last_4_layer_3_grad_norm",
        "decoder_last_4_layer_4_grad_norm",
        "eval_loss",
        *(f"train_{name}" for name in OCR_METRIC_NAMES),
        *(f"eval_{name}" for name in OCR_METRIC_NAMES),
        "learning_rate",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.writer = None
        self.latest_train_loss = None
        self.latest_encoder_grad_norm = None
        self.latest_decoder_grad_norm = None
        self.latest_decoder_input_embedding_grad_norm = None
        self.latest_decoder_layer_grad_norms = {
            position: None for position in range(1, 5)
        }
        self.latest_train_ocr_metrics = {
            name: None for name in OCR_METRIC_NAMES
        }
        self.latest_learning_rate = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return control
        self.latest_train_loss = logs.get("loss", self.latest_train_loss)
        self.latest_encoder_grad_norm = logs.get(
            "encoder_grad_norm", self.latest_encoder_grad_norm
        )
        self.latest_decoder_grad_norm = logs.get(
            "decoder_grad_norm", self.latest_decoder_grad_norm
        )
        self.latest_decoder_input_embedding_grad_norm = logs.get(
            "decoder_input_embedding_grad_norm",
            self.latest_decoder_input_embedding_grad_norm,
        )
        for position in self.latest_decoder_layer_grad_norms:
            metric_name = f"decoder_last_4_layer_{position}_grad_norm"
            self.latest_decoder_layer_grad_norms[position] = logs.get(
                metric_name, self.latest_decoder_layer_grad_norms[position]
            )
        for name in OCR_METRIC_NAMES:
            self.latest_train_ocr_metrics[name] = logs.get(
                f"train_{name}", self.latest_train_ocr_metrics[name]
            )
        self.latest_learning_rate = logs.get("learning_rate", self.latest_learning_rate)
        if "eval_loss" in logs and self.writer is not None:
            self.writer.writerow(
                {
                    "epoch": logs.get("epoch", state.epoch),
                    "step": state.global_step,
                    "train_loss": self.latest_train_loss,
                    "encoder_grad_norm": self.latest_encoder_grad_norm,
                    "decoder_grad_norm": self.latest_decoder_grad_norm,
                    "decoder_input_embedding_grad_norm": (
                        self.latest_decoder_input_embedding_grad_norm
                    ),
                    **{
                        f"decoder_last_4_layer_{position}_grad_norm": value
                        for position, value in self.latest_decoder_layer_grad_norms.items()
                    },
                    "eval_loss": logs["eval_loss"],
                    **{
                        f"train_{name}": value
                        for name, value in self.latest_train_ocr_metrics.items()
                    },
                    **{
                        f"eval_{name}": logs.get(f"eval_{name}")
                        for name in OCR_METRIC_NAMES
                    },
                    "learning_rate": self.latest_learning_rate,
                }
            )
            self.file.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.file is not None:
            self.file.close()
        return control


class TrOCRTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that logs generated OCR and gradient metrics."""

    def _gradient_norm(self, parameters) -> float:
        squared_norm = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in parameters
            if parameter.grad is not None
        )
        scale = self.scaler.get_scale() if getattr(self, "do_grad_scaling", False) else 1.0
        return squared_norm**0.5 / scale

    def _gradient_metrics(self) -> dict[str, float]:
        decoder_layers = self.model.decoder.model.decoder.layers
        if len(decoder_layers) < 4:
            raise ValueError("TrOCR decoder must have at least four transformer layers.")

        metrics = {
            "encoder_grad_norm": self._gradient_norm(self.model.encoder.parameters()),
            "decoder_grad_norm": self._gradient_norm(self.model.decoder.parameters()),
            "decoder_input_embedding_grad_norm": self._gradient_norm(
                self.model.decoder.get_input_embeddings().parameters()
            ),
        }
        for position, layer in enumerate(decoder_layers[-4:], start=1):
            metrics[f"decoder_last_4_layer_{position}_grad_norm"] = self._gradient_norm(
                layer.parameters()
            )
        return metrics

    def training_step(self, model, inputs):
        should_log = (self.state.global_step + 1) % self.args.logging_steps == 0
        metrics = {}
        if should_log:
            batch = self._prepare_inputs(dict(inputs))
            was_training = model.training
            model.eval()
            with torch.no_grad(), self.autocast_smart_context_manager():
                generated_ids = model.generate(
                    pixel_values=batch["pixel_values"],
                    max_length=self.args.generation_max_length or model.config.max_length,
                )
            metrics.update(
                compute_token_metrics(
                    generated_ids,
                    batch["labels"],
                    self.tokenizer,
                    prefix="train_",
                )
            )
            if was_training:
                model.train()

        loss = super().training_step(model, inputs)
        if should_log:
            metrics.update(self._gradient_metrics())
            self.log(metrics)
        return loss
