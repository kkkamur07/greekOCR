"""Hugging Face Trainer integration for the PyTorch CTC Calamari recognizer."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ...metrics.languages import language_indices
from ...metrics.metrics import compute_sequence_length_metrics, compute_text_metrics
from .checkpoint import load_calamari_checkpoint, save_calamari_checkpoint
from .codec import CharacterCodec
from .config import default_model_config
from .data import CalamariAugmentedDataset, CalamariLineDataset, collect_samples, collate_ctc
from .model import CalamariTorchModel


TRAIN_OCR_METRIC_NAMES = ("cer", "wer", "exact_match")
EVAL_OCR_METRIC_NAMES = (
    *TRAIN_OCR_METRIC_NAMES,
    "sroie_precision",
    "sroie_recall",
    "sroie_f1",
)
_EMA_FILENAME = "calamari_ema.pt"
_METADATA_FILENAME = "calamari_metadata.json"


@dataclass(frozen=True)
class CalamariTrainingSettings:
    epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    line_height: int
    device: str
    temperature: float = -1.0
    checkpoint: Path | None = None
    mode: str = "train"
    train_split: str = "train"
    validation_split: str = "val"
    n_augmentations: int = 0
    augmentation_probability: float = 1.0
    ema_decay: float = 0.99
    logging_steps: int = 10
    warmup_ratio: float = 0.09
    checkpoint_top_k: int = 1


class ExponentialMovingAverage:
    """Maintain an evaluation-ready exponential moving average of model weights."""

    def __init__(self, model: CalamariTorchModel, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                "Calamari ema_decay must be greater than or equal to zero and less than one."
            )
        self.decay = decay
        self.model = copy.deepcopy(model).requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, model: CalamariTorchModel) -> None:
        source = model.state_dict()
        target = self.model.state_dict()
        for name, target_value in target.items():
            source_value = source[name].detach()
            if torch.is_floating_point(target_value):
                target_value.lerp_(source_value, 1.0 - self.decay)
            else:
                target_value.copy_(source_value)


class _EMACallback(TrainerCallback):
    """Update Calamari's EMA after each completed optimizer step."""

    def __init__(self, ema: ExponentialMovingAverage) -> None:
        self.ema = ema

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.ema.model.to(next(model.parameters()).device)
            self.ema.update(model)
        return control


class _ReportCallback(TrainerCallback):
    """Forward normalized Trainer logs to the CLI's JSONL and W&B reporter."""

    def __init__(self, report: Callable[[dict[str, float]], None] | None) -> None:
        self.report = report

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.report is None or not state.is_world_process_zero or not logs:
            return control
        metrics = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        metrics["step"] = float(state.global_step)
        if state.epoch is not None:
            metrics.setdefault("epoch", float(state.epoch))
        self.report(metrics)
        return control


class _TrainerCompatibleConfig:
    """Expose Calamari's immutable config through Trainer's mutable interface."""

    def __init__(self, calamari_config: object) -> None:
        self._calamari_config = calamari_config
        self.use_cache = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._calamari_config, name)


class CalamariTrainer(Trainer):
    """Adapt Hugging Face Trainer's lifecycle to Calamari's CTC contract."""

    def __init__(
        self,
        *args: Any,
        codec: CharacterCodec,
        ema: ExponentialMovingAverage,
        line_height: int,
        temperature: float,
        language_eval_datasets: dict[str, Dataset[dict[str, object]]],
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if not isinstance(model, CalamariTorchModel):
            raise TypeError("CalamariTrainer requires a CalamariTorchModel.")
        self.lstm_layers = sum(layer.kind == "bilstm" for layer in model.config.layers)
        model.config = _TrainerCompatibleConfig(model.config)
        super().__init__(*args, **kwargs)
        self.codec = codec
        self.ema = ema
        self.line_height = line_height
        self.temperature = temperature
        self.language_eval_datasets = language_eval_datasets
        self.loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
        self.latest_train_text_metrics: dict[str, float] | None = None
        self.latest_train_loss: float | None = None
        self.latest_learning_rate: float | None = None

    def compute_loss(
        self,
        model: CalamariTorchModel,
        inputs: dict[str, object],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        del num_items_in_batch
        outputs = model(
            _tensor(inputs["image"], "image"),
            image_lengths=_tensor(inputs["image_lengths"], "image_lengths"),
        )
        logits = outputs["logits"]
        loss = self.loss_function(
            logits.log_softmax(-1).transpose(0, 1),
            _tensor(inputs["targets"], "targets"),
            outputs["out_len"],
            _tensor(inputs["target_lengths"], "target_lengths"),
        )
        if model.training:
            texts = [str(text) for text in inputs["texts"]]
            decoded = self.codec.decode_logits(logits, outputs["out_len"])
            self.latest_train_text_metrics = compute_text_metrics(texts, decoded)
            self.latest_train_loss = float(loss.detach().cpu())
            if self.optimizer is not None:
                self.latest_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: CalamariTorchModel,
        inputs: dict[str, object],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor | None, Tensor | tuple[Tensor, Tensor] | None, Tensor | None]:
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        return (
            loss.detach(),
            (outputs["logits"].detach(), outputs["out_len"].detach()),
            _tensor(inputs["labels"], "labels").detach(),
        )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        normalized = dict(logs)
        if "loss" in normalized:
            normalized["train_loss"] = normalized.pop("loss")
        if isinstance(normalized.get("train_loss"), int | float):
            self.latest_train_loss = float(normalized["train_loss"])
        if isinstance(normalized.get("learning_rate"), int | float):
            self.latest_learning_rate = float(normalized["learning_rate"])
        if "eval_loss" in normalized:
            if self.latest_train_loss is not None:
                normalized.setdefault("train_loss", self.latest_train_loss)
            if self.latest_learning_rate is not None:
                normalized.setdefault("learning_rate", self.latest_learning_rate)
        if self.latest_train_text_metrics is not None:
            normalized.update(
                {
                    f"train_{name}": self.latest_train_text_metrics[name]
                    for name in TRAIN_OCR_METRIC_NAMES
                }
            )
        super().log(normalized, start_time)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluate EMA weights while retaining raw weights for checkpoint resumes."""
        model = self.accelerator.unwrap_model(self.model)
        raw_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.ema.model.state_dict())
        try:
            metrics = super().evaluate(*args, **kwargs)
            language_metrics: dict[str, float] = {}
            for language, dataset in self.language_eval_datasets.items():
                language_metrics.update(
                    self.predict(dataset, metric_key_prefix=f"eval_{language}").metrics
                )
            if language_metrics:
                self.log(language_metrics)
                metrics.update(language_metrics)
            return metrics
        finally:
            model.load_state_dict(raw_state)

    def _save_checkpoint(self, model: nn.Module, trial: Any) -> None:
        super()._save_checkpoint(model, trial)
        if not self.is_world_process_zero():
            return
        checkpoint_dir = Path(self.args.output_dir) / (
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )
        torch.save(self.ema.model.state_dict(), checkpoint_dir / _EMA_FILENAME)
        _write_metadata(
            checkpoint_dir / _METADATA_FILENAME,
            self.codec,
            self.line_height,
            self.temperature,
            self.lstm_layers,
        )

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module | None = None
    ) -> None:
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        ema_path = Path(resume_from_checkpoint) / _EMA_FILENAME
        if not ema_path.is_file():
            raise ValueError(
                f"Calamari Trainer checkpoint is missing EMA state: {ema_path}."
            )
        self.ema.model.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=True))

    def load_ema_checkpoint(self, checkpoint: Path) -> None:
        """Load the EMA weights selected by Hugging Face best-checkpoint tracking."""
        self.ema.model.load_state_dict(
            torch.load(checkpoint / _EMA_FILENAME, map_location="cpu", weights_only=True)
        )


def train_calamari(
    data_root: Path,
    output_dir: Path,
    settings: CalamariTrainingSettings,
    *,
    report: Callable[[dict[str, float]], None] | None = None,
) -> tuple[CalamariTorchModel, CharacterCodec, dict[str, float]]:
    """Train or fine-tune Calamari through Hugging Face Trainer and save ``best.pt``."""
    if settings.mode not in {"train", "finetune"}:
        raise ValueError("Calamari training.mode must be 'train' or 'finetune'.")
    if settings.logging_steps <= 0:
        raise ValueError("Calamari logging_steps must be greater than zero.")
    if not 0.0 <= settings.warmup_ratio < 1.0:
        raise ValueError(
            "Calamari warmup_ratio must be greater than or equal to zero and less than one."
        )
    if settings.checkpoint_top_k <= 0:
        raise ValueError("Calamari checkpoint_top_k must be greater than zero.")

    train_samples = collect_samples(data_root, settings.train_split)
    if not train_samples:
        raise ValueError(f"No Calamari training samples found in {data_root}.")
    validation_samples = collect_samples(data_root, settings.validation_split)
    model, codec = _initial_model([*train_samples, *validation_samples], settings)
    train_dataset = CalamariAugmentedDataset(
        CalamariLineDataset(data_root, settings.train_split, codec, settings.line_height),
        settings.n_augmentations,
        probability=settings.augmentation_probability,
    )
    validation_dataset = CalamariLineDataset(
        data_root, settings.validation_split, codec, settings.line_height
    )
    language_eval_datasets = {
        language: Subset(validation_dataset, indices)
        for language, indices in language_indices(
            [sample.language for sample in validation_dataset.samples]
        ).items()
    }

    # The architecture has lazy LSTM and classifier layers, which must exist
    # before Trainer creates its optimizer or restores a checkpoint.
    _materialize_model(model, collate_ctc([train_dataset[0]]), torch.device("cpu"))
    ema = ExponentialMovingAverage(model, settings.ema_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        max_grad_norm=5.0,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=settings.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=settings.checkpoint_top_k,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        dataloader_num_workers=settings.workers,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=1111,
        use_cpu=_resolve_device(settings.device).type == "cpu",
        fp16=_resolve_device(settings.device).type == "cuda",
    )
    trainer = CalamariTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_ctc,
        compute_metrics=lambda prediction: _compute_ctc_metrics(prediction, codec),
        codec=codec,
        ema=ema,
        line_height=settings.line_height,
        temperature=settings.temperature,
        language_eval_datasets=language_eval_datasets,
        callbacks=[_EMACallback(ema), _ReportCallback(report)],
    )
    resume_checkpoint = _trainer_checkpoint(settings)
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)

    best_checkpoint = (
        Path(trainer.state.best_model_checkpoint)
        if trainer.state.best_model_checkpoint is not None
        else None
    )
    if best_checkpoint is not None:
        trainer.load_ema_checkpoint(best_checkpoint)
    ema.model.eval()
    ema_device = next(ema.model.parameters()).device
    save_calamari_checkpoint(
        output_dir / "best.pt",
        ema.model.cpu(),
        charset=codec.charset,
        line_height=settings.line_height,
        temperature=ema.model.config.temperature,
    )
    ema.model.to(ema_device)
    best_metrics = _best_metrics(trainer.state.log_history)
    return ema.model, codec, best_metrics


@torch.no_grad()
def evaluate_model(
    model: CalamariTorchModel,
    loader: DataLoader[dict[str, object]],
    codec: CharacterCodec,
    device: torch.device,
    loss_function: nn.CTCLoss | None = None,
) -> dict[str, float]:
    """Evaluate a model using CTC-decoded line transcription metrics."""
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    losses: list[float] = []
    for batch in loader:
        loss, decoded, texts = _batch_loss(model, batch, codec, device, loss_function)
        predictions.extend(decoded)
        references.extend(texts)
        if loss_function is not None:
            losses.append(float(loss.cpu()))
    metrics = compute_text_metrics(references, predictions)
    metrics.update(compute_sequence_length_metrics(references, predictions))
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    return metrics


def _initial_model(
    samples: list[object], settings: CalamariTrainingSettings
) -> tuple[CalamariTorchModel, CharacterCodec]:
    resume_checkpoint = _trainer_checkpoint(settings)
    if resume_checkpoint is not None:
        metadata = _read_metadata(resume_checkpoint / _METADATA_FILENAME)
        codec = CharacterCodec(tuple(metadata["charset"]))
        return (
            CalamariTorchModel(
                default_model_config(
                    classes=codec.classes,
                    temperature=float(metadata["temperature"]),
                    lstm_layers=_metadata_lstm_layers(metadata),
                )
            ),
            codec,
        )
    if settings.mode == "finetune":
        if settings.checkpoint is None:
            raise ValueError("Fine-tuning requires training.checkpoint.")
        model, metadata = load_calamari_checkpoint(settings.checkpoint)
        if metadata.line_height != settings.line_height:
            raise ValueError(
                "Fine-tuning line height must match the checkpoint: "
                f"{metadata.line_height}, received {settings.line_height}."
            )
        codec = CharacterCodec(metadata.charset)
        unsupported = sorted(
            {character for sample in samples for character in sample.text} - set(codec.charset)
        )
        if unsupported:
            raise ValueError(
                f"Fine-tuning data has characters absent from the checkpoint codec: {unsupported}"
            )
        return model, codec
    codec = CharacterCodec.from_texts(sample.text for sample in samples)
    return (
        CalamariTorchModel(
            default_model_config(classes=codec.classes, temperature=settings.temperature)
        ),
        codec,
    )


def _trainer_checkpoint(settings: CalamariTrainingSettings) -> Path | None:
    checkpoint = settings.checkpoint
    if (
        settings.mode == "train"
        and checkpoint is not None
        and checkpoint.is_dir()
        and (checkpoint / "trainer_state.json").is_file()
    ):
        return checkpoint
    return None


def _compute_ctc_metrics(prediction: Any, codec: CharacterCodec) -> dict[str, float]:
    logits, output_lengths = prediction.predictions
    hypotheses = codec.decode_logits(
        torch.as_tensor(logits),
        torch.as_tensor(output_lengths),
    )
    labels = torch.as_tensor(prediction.label_ids)
    references = [
        "".join(codec.charset[int(token)] for token in row.tolist() if token > 0)
        for row in labels
    ]
    metrics = compute_text_metrics(references, hypotheses)
    metrics.update(compute_sequence_length_metrics(references, hypotheses))
    return metrics


def _best_metrics(history: list[dict[str, Any]]) -> dict[str, float]:
    evaluations = [
        metrics
        for metrics in history
        if isinstance(metrics.get("eval_cer"), int | float)
    ]
    if not evaluations:
        raise RuntimeError("Calamari training did not produce evaluation metrics.")
    best = min(evaluations, key=lambda metrics: float(metrics["eval_cer"]))
    return {
        key: float(value)
        for key, value in best.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _write_metadata(
    path: Path,
    codec: CharacterCodec,
    line_height: int,
    temperature: float,
    lstm_layers: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "charset": codec.charset,
                "line_height": line_height,
                "temperature": temperature,
                "lstm_layers": lstm_layers,
            }
        ),
        encoding="utf-8",
    )


def _read_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Calamari Trainer checkpoint is missing metadata: {path}.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid Calamari Trainer checkpoint metadata: {path}.")
    return metadata


def _metadata_lstm_layers(metadata: dict[str, object]) -> int:
    lstm_layers = metadata.get("lstm_layers", 1)
    if (
        not isinstance(lstm_layers, int)
        or isinstance(lstm_layers, bool)
        or lstm_layers not in {1, 2}
    ):
        raise ValueError("Calamari Trainer checkpoint has an invalid LSTM layer count.")
    return lstm_layers


def _loader(
    dataset: Dataset[dict[str, object]], settings: CalamariTrainingSettings, *, shuffle: bool
) -> DataLoader[dict[str, object]]:
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.workers,
        collate_fn=collate_ctc,
    )


def _materialize_model(
    model: CalamariTorchModel, batch: dict[str, object], device: torch.device
) -> None:
    image = _tensor(batch["image"], "image").to(device)
    lengths = _tensor(batch["image_lengths"], "image_lengths").to(device)
    with torch.no_grad():
        model(image, image_lengths=lengths)


def _batch_loss(
    model: CalamariTorchModel,
    batch: dict[str, object],
    codec: CharacterCodec,
    device: torch.device,
    loss_function: nn.CTCLoss | None,
) -> tuple[Tensor, list[str], list[str]]:
    image = _tensor(batch["image"], "image").to(device)
    image_lengths = _tensor(batch["image_lengths"], "image_lengths").to(device)
    targets = _tensor(batch["targets"], "targets").to(device)
    target_lengths = _tensor(batch["target_lengths"], "target_lengths").to(device)
    outputs = model(image, image_lengths=image_lengths)
    output_lengths = outputs["out_len"]
    logits = outputs["logits"]
    if loss_function is None:
        loss = torch.zeros((), device=device)
    else:
        loss = loss_function(
            logits.log_softmax(-1).transpose(0, 1), targets, output_lengths, target_lengths
        )
    texts = [str(value) for value in batch["texts"]]
    return loss, codec.decode_logits(logits, output_lengths), texts


def _tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"Calamari batch {name} must be a Tensor.")
    return value


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)
