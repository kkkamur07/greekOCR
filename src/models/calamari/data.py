"""Line-image dataset and CTC batching for PyTorch Calamari."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .augmentation import augment_legacy_line_image
from .codec import CharacterCodec


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


@dataclass(frozen=True)
class LineSample:
    image_path: Path
    text: str


class CalamariLineDataset(Dataset[dict[str, object]]):
    """Read paired line images and ``.gt.txt`` transcriptions for one split."""

    def __init__(self, root: Path, split: str, codec: CharacterCodec, line_height: int) -> None:
        self.codec = codec
        self.line_height = line_height
        self.samples = collect_samples(root, split)
        if not self.samples:
            raise ValueError(f"No labeled Calamari samples found for split {split!r} in {root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "image": _load_line_image(sample.image_path, self.line_height),
            "targets": self.codec.encode(sample.text),
            "text": sample.text,
        }


class CalamariAugmentedDataset(Dataset[dict[str, object]]):
    """Expose each training sample together with legacy Calamari augmentations."""

    def __init__(
        self,
        dataset: Dataset[dict[str, object]],
        n_augmentations: int,
        probability: float = 1.0,
    ) -> None:
        if n_augmentations < 0:
            raise ValueError("Calamari n_augmentations must be zero or greater.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Calamari augmentation probability must be between zero and one.")
        self.dataset = dataset
        self.n_augmentations = n_augmentations
        self.probability = probability

    def __len__(self) -> int:
        return len(self.dataset) * self._variants_per_sample

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_index, variant = divmod(index, self._variants_per_sample)
        sample = dict(self.dataset[sample_index])
        if variant and numpy.random.random() < self.probability:
            image = sample["image"]
            if not isinstance(image, Tensor):
                raise TypeError("Calamari augmentation requires tensor images.")
            sample["image"] = augment_legacy_line_image(image)
        return sample

    @property
    def _variants_per_sample(self) -> int:
        return self.n_augmentations + 1 if self.probability > 0.0 else 1


def collect_samples(root: Path, split: str) -> list[LineSample]:
    """Support canonical flat packs and the source ``images/labels`` layout."""
    root = _resolve_virtual_root(root)
    manifest = root / f"gt_{split}.txt"
    trocr_images = root / "image"
    if manifest.is_file() and trocr_images.is_dir():
        samples = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line:
                image_name, text = line.split("\t", 1)
                samples.append(LineSample(image_path=trocr_images / image_name, text=text))
        return samples

    flat_split = root / split
    images_dir = flat_split if flat_split.is_dir() else root / "images" / split
    labels_dir = flat_split if flat_split.is_dir() else root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []

    samples: list[LineSample] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.gt.txt"
        if label_path.is_file():
            samples.append(LineSample(image_path=image_path, text=label_path.read_text(encoding="utf-8")))
    return samples


def _resolve_virtual_root(root: Path) -> Path:
    marker = root / "source.txt"
    if not marker.is_file():
        return root
    source = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
    if not source.is_absolute():
        source = marker.parent / source
    resolved = source.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Calamari source marker points to a missing directory: {resolved}")
    return resolved


def collate_ctc(samples: list[dict[str, object]]) -> dict[str, object]:
    """Pad variable-width images and concatenate CTC label targets."""
    images = [sample["image"] for sample in samples]
    targets = [sample["targets"] for sample in samples]
    if not all(isinstance(image, Tensor) for image in images) or not all(
        isinstance(target, Tensor) for target in targets
    ):
        raise TypeError("Invalid Calamari batch.")
    typed_images = [image for image in images if isinstance(image, Tensor)]
    typed_targets = [target for target in targets if isinstance(target, Tensor)]
    widths = torch.tensor([image.shape[0] for image in typed_images], dtype=torch.long)
    height = typed_images[0].shape[1]
    batch = torch.zeros((len(typed_images), int(widths.max()), height, 1), dtype=torch.float32)
    for index, image in enumerate(typed_images):
        batch[index, : image.shape[0]] = image
    return {
        "image": batch,
        "image_lengths": widths,
        "targets": torch.cat(typed_targets),
        "target_lengths": torch.tensor([target.numel() for target in typed_targets], dtype=torch.long),
        "texts": [str(sample["text"]) for sample in samples],
    }


def _load_line_image(path: Path, line_height: int) -> Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        width = max(1, round(image.width * line_height / image.height))
        image = image.resize((width, line_height), Image.Resampling.BILINEAR)
        return torch.from_numpy(numpy.asarray(image).T.copy()).unsqueeze(-1)
