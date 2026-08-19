from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.models.calamari.checkpoint import load_calamari_checkpoint
from src.models.calamari.config import default_model_config
from src.models.calamari.codec import CharacterCodec
from src.models.calamari.data import CalamariAugmentedDataset, CalamariLineDataset, collate_ctc
from src.models.calamari.model import CalamariTorchModel
from src.models.calamari.trainer import ExponentialMovingAverage, CalamariTrainingSettings, train_calamari


def _dataset_root(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        directory = tmp_path / split
        directory.mkdir()
        texts = ("ab", "ba") if split != "val" else ("ac", "ca")
        for index, text in enumerate(texts):
            Image.new("L", (32 + index * 4, 16), color=255).save(directory / f"{index}.png")
            (directory / f"{index}.gt.txt").write_text(text, encoding="utf-8")
    return tmp_path


def test_codec_and_ctc_collation(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    codec = CharacterCodec.from_texts(("ab", "ba"))
    dataset = CalamariLineDataset(root, "train", codec, line_height=16)
    batch = collate_ctc([dataset[0], dataset[1]])
    assert codec.charset == ("", "a", "b")
    assert batch["image"].shape == (2, 36, 16, 1)
    assert batch["target_lengths"].tolist() == [2, 2]


def test_trocr_manifest_virtual_pack(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image_dir = source / "image"
    image_dir.mkdir(parents=True)
    Image.new("L", (32, 16), color=255).save(image_dir / "line.png")
    (source / "gt_train.txt").write_text("line.png\tab\n", encoding="utf-8")
    virtual = tmp_path / "calamari"
    virtual.mkdir()
    (virtual / "source.txt").write_text(str(source), encoding="utf-8")

    dataset = CalamariLineDataset(
        virtual, "train", CharacterCodec.from_texts(("ab",)), line_height=16
    )
    assert len(dataset) == 1


def test_legacy_augmentation_expands_training_samples_without_changing_labels(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    codec = CharacterCodec.from_texts(("ab", "ba"))
    base = CalamariLineDataset(root, "train", codec, line_height=16)
    augmented = CalamariAugmentedDataset(base, n_augmentations=2)

    original = augmented[0]
    variant = augmented[1]

    assert len(augmented) == len(base) * 3
    assert variant["text"] == original["text"]
    assert torch.equal(variant["targets"], original["targets"])
    assert variant["image"].shape[1:] == original["image"].shape[1:]

    disabled = CalamariAugmentedDataset(base, n_augmentations=2, probability=0.0)
    assert len(disabled) == len(base)
    assert torch.equal(disabled[0]["image"], base[0]["image"])


def test_ema_tracks_model_weights() -> None:
    model = CalamariTorchModel(default_model_config(classes=3))
    model(torch.zeros((1, 32, 16, 1)), image_lengths=torch.tensor([32]))
    ema = ExponentialMovingAverage(model, decay=0.99)
    parameter = next(model.parameters())
    expected = parameter.detach().clone()

    with torch.no_grad():
        parameter.add_(1)
    ema.update(model)

    ema_parameter = next(ema.model.parameters())
    assert torch.allclose(ema_parameter, expected + 0.01)

    raw_weight_ema = ExponentialMovingAverage(model, decay=0.0)
    raw_weight_ema.update(model)
    assert torch.equal(next(raw_weight_ema.model.parameters()), parameter)


def test_training_metrics_follow_logging_steps(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    settings = CalamariTrainingSettings(
        epochs=1,
        batch_size=1,
        workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        line_height=16,
        device="cpu",
        logging_steps=1,
        warmup_ratio=0.0,
    )
    reported_metrics: list[dict[str, float]] = []
    train_calamari(root, root / "train-run", settings, report=reported_metrics.append)

    assert len(reported_metrics) == 2
    assert set(reported_metrics[0]) == {
        "epoch",
        "step",
        "learning_rate",
        "train_loss",
        "train_cer",
        "train_wer",
        "train_exact_match",
    }
    assert "eval_loss" in reported_metrics[-1]
    assert reported_metrics[0]["step"] == 1.0
    assert reported_metrics[-1]["step"] == 2.0
    assert reported_metrics[0]["learning_rate"] > reported_metrics[-1]["learning_rate"]


def test_train_checkpoint_and_finetune(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    settings = CalamariTrainingSettings(
        epochs=1,
        batch_size=2,
        workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        line_height=16,
        device="cpu",
    )
    reported_metrics: list[dict[str, float]] = []
    _, codec, _ = train_calamari(
        root,
        root / "train-run",
        settings,
        report=reported_metrics.append,
    )
    assert set(reported_metrics[0]) == {
        "epoch",
        "step",
        "learning_rate",
        "train_loss",
        "eval_loss",
        "train_cer",
        "train_wer",
        "train_exact_match",
        "eval_cer",
        "eval_wer",
        "eval_exact_match",
        "eval_sroie_precision",
        "eval_sroie_recall",
        "eval_sroie_f1",
    }
    assert "c" in codec.charset
    checkpoint = root / "train-run" / "best.pt"
    _, metadata = load_calamari_checkpoint(checkpoint)
    assert metadata.charset == codec.charset

    fine_tune_settings = CalamariTrainingSettings(
        **{**settings.__dict__, "mode": "finetune", "checkpoint": checkpoint}
    )
    train_calamari(root, root / "fine-tune-run", fine_tune_settings)
    assert (root / "fine-tune-run" / "best.pt").is_file()
