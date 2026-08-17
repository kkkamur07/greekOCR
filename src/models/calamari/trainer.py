"""CTC training and warm-start fine-tuning for PyTorch Calamari."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .checkpoint import load_calamari_checkpoint, save_calamari_checkpoint
from .codec import CharacterCodec
from .config import default_model_config
from .data import CalamariAugmentedDataset, CalamariLineDataset, collect_samples, collate_ctc
from .metrics import compute_text_metrics
from .model import CalamariTorchModel


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


class ExponentialMovingAverage:
    """Maintain an evaluation-ready exponential moving average of model weights."""

    def __init__(self, model: CalamariTorchModel, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("Calamari ema_decay must be greater than or equal to zero and less than one.")
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


def train_calamari(
    data_root: Path,
    output_dir: Path,
    settings: CalamariTrainingSettings,
    *,
    report: Callable[[dict[str, float]], None] | None = None,
) -> tuple[CalamariTorchModel, CharacterCodec, dict[str, float]]:
    """Train or warm-start fine-tune a Calamari model and persist its best checkpoint."""
    if settings.mode not in {"train", "finetune"}:
        raise ValueError("Calamari training.mode must be 'train' or 'finetune'.")
    train_samples = collect_samples(data_root, settings.train_split)
    if not train_samples:
        raise ValueError(f"No Calamari training samples found in {data_root}.")

    validation_samples = collect_samples(data_root, settings.validation_split)
    model, codec = _initial_model([*train_samples, *validation_samples], settings)
    base_train_dataset = CalamariLineDataset(
        data_root, settings.train_split, codec, settings.line_height
    )
    train_dataset = CalamariAugmentedDataset(
        base_train_dataset,
        settings.n_augmentations,
        probability=settings.augmentation_probability,
    )
    validation_dataset = CalamariLineDataset(
        data_root, settings.validation_split, codec, settings.line_height
    )
    train_loader = _loader(train_dataset, settings, shuffle=True)
    validation_loader = _loader(validation_dataset, settings, shuffle=False)
    device = _resolve_device(settings.device)
    model.to(device)

    # Lazy recurrent/classifier layers must exist before the optimizer sees them.
    _materialize_model(model, next(iter(train_loader)), device)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    ema = ExponentialMovingAverage(model, settings.ema_decay)
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Tensor] | None = None
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            loss, _, _ = _batch_loss(model, batch, codec, device, loss_function)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(model)
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_model(ema.model, validation_loader, codec, device, loss_function)
        metrics.update({"epoch": float(epoch), "loss": sum(losses) / max(len(losses), 1)})
        if report is not None:
            report(metrics)
        if best_metrics is None or metrics["cer"] < best_metrics["cer"]:
            best_metrics = dict(metrics)
            best_state = {
                name: value.detach().cpu().clone() for name, value in ema.model.state_dict().items()
            }
            save_calamari_checkpoint(
                output_dir / "best.pt",
                ema.model.cpu(),
                charset=codec.charset,
                line_height=settings.line_height,
                temperature=ema.model.config.temperature,
            )
            ema.model.to(device)

    if best_metrics is None or best_state is None:
        raise RuntimeError("Calamari training did not produce validation metrics.")
    ema.model.load_state_dict(best_state)
    ema.model.eval()
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
    metrics = compute_text_metrics(predictions, references)
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    return metrics


def _initial_model(samples, settings: CalamariTrainingSettings) -> tuple[CalamariTorchModel, CharacterCodec]:
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
        unsupported = sorted({character for sample in samples for character in sample.text} - set(codec.charset))
        if unsupported:
            raise ValueError(f"Fine-tuning data has characters absent from the checkpoint codec: {unsupported}")
        return model, codec
    codec = CharacterCodec.from_texts(sample.text for sample in samples)
    return (
        CalamariTorchModel(
            default_model_config(classes=codec.classes, temperature=settings.temperature)
        ),
        codec,
    )


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
        loss = loss_function(logits.log_softmax(-1).transpose(0, 1), targets, output_lengths, target_lengths)
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
