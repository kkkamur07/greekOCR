"""Apply augmentation to sample images and save originals + augmented side by side."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from src.models.trocr.augmentation import LineAugmentation


def main() -> None:
    data_dir = REPO_ROOT / "data" / "trocr_processed" / "combined" / "pretraining"
    gt_file = data_dir / "gt_train.txt"
    image_dir = data_dir / "image"
    output_dir = REPO_ROOT / "augmentation_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentation = LineAugmentation(
        probability=1.0,
        mode="random",
        num_operations=3,
        magnitude=None,
        exclude_groups=("geometry",),
        exclude_operations=("Invert",),
    )

    samples = []
    for line in gt_file.read_text(encoding="utf-8").splitlines()[:50]:
        if not line:
            continue
        image_name, text = line.split("\t", 1)
        image_path = image_dir / image_name
        if image_path.exists():
            samples.append((image_name, image_path, text))
        if len(samples) == 10:
            break

    for i, (name, path, text) in enumerate(samples):
        original = Image.open(path).convert("RGB")
        augmented = augmentation(original.copy())

        width = max(original.width, augmented.width)
        height = original.height + augmented.height + 10
        combined = Image.new("RGB", (width, height), (255, 255, 255))
        combined.paste(original, (0, 0))
        combined.paste(augmented, (0, original.height + 10))

        out_path = output_dir / f"{i:02d}_{name}"
        combined.save(out_path)
        print(f"[{i}] {name}: {text}")
        print(f"    Saved to {out_path}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
