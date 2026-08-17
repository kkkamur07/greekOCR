"""Normalize all TrOCR datasets to deterministic 80/10/10 splits.

Only manifest files are rewritten; images remain in place. Combined manifests
are rebuilt from the language-specific assignments, so a crop always has the
same split in its language dataset and the combined dataset.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPO_ROOT / "data" / "trocr_processed"
LANGUAGES = ("greek", "syriac", "armenian")
PARTITIONS = ("pretraining", "finetuning")
SPLITS = ("train", "val", "test")
SEED = 1111


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    """Read one TrOCR filename/transcription manifest."""
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            image_name, text = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid manifest row {path}:{line_number}") from exc
        rows.append((image_name, text))
    return rows


def pooled_rows(root: Path) -> list[tuple[str, str]]:
    """Pool all current splits and reject duplicate image names."""
    rows: list[tuple[str, str]] = []
    for split in SPLITS:
        rows.extend(parse_manifest(root / f"gt_{split}.txt"))
    names = [image_name for image_name, _ in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image names across splits under {root}")
    for image_name in names:
        if not (root / "image" / image_name).is_file():
            raise FileNotFoundError(f"Missing manifest image: {root / 'image' / image_name}")
    return rows


def source_group(language: str, partition: str, image_name: str) -> str:
    """Keep imported Greek corpora independently balanced."""
    if language == "greek" and partition == "pretraining":
        if image_name.startswith("esteban__"):
            return "esteban"
        if image_name.startswith("labelled__"):
            return "labelled"
        return "greek_base"
    return language


def split_rows(
    rows: Iterable[tuple[str, str]],
    seed: int,
) -> dict[str, list[tuple[str, str]]]:
    """Assign rows using rounded cumulative 80/10/10 boundaries."""
    shuffled = sorted(rows)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.8)
    val_end = round(len(shuffled) * 0.9)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_manifest_atomic(path: Path, rows: list[tuple[str, str]]) -> None:
    """Replace one manifest atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for image_name, text in rows:
            output.write(f"{image_name}\t{text}\n")
    os.replace(temporary, path)


def resplit_language_partition(language: str, partition: str) -> dict[str, int]:
    """Resplit one language partition, balancing Greek sources separately."""
    root = PROCESSED_ROOT / language / partition
    groups: dict[str, list[tuple[str, str]]] = {}
    for row in pooled_rows(root):
        group = source_group(language, partition, row[0])
        groups.setdefault(group, []).append(row)

    split_manifests = {split: [] for split in SPLITS}
    for group_index, group in enumerate(sorted(groups)):
        assignments = split_rows(groups[group], SEED + group_index)
        for split in SPLITS:
            split_manifests[split].extend(assignments[split])

    for split in SPLITS:
        split_manifests[split].sort()
        write_manifest_atomic(root / f"gt_{split}.txt", split_manifests[split])
    return {split: len(split_manifests[split]) for split in SPLITS}


def rebuild_combined_partition(partition: str) -> dict[str, int]:
    """Rebuild combined manifests from the normalized language manifests."""
    combined_root = PROCESSED_ROOT / "combined" / partition
    summary: dict[str, int] = {}
    for split in SPLITS:
        rows: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        for language in LANGUAGES:
            language_root = PROCESSED_ROOT / language / partition
            for image_name, text in parse_manifest(language_root / f"gt_{split}.txt"):
                if image_name in seen_names:
                    raise ValueError(
                        f"Duplicate image name in combined {partition}/{split}: {image_name}"
                    )
                seen_names.add(image_name)
                if not (combined_root / "image" / image_name).is_file():
                    raise FileNotFoundError(
                        f"Missing combined image: {combined_root / 'image' / image_name}"
                    )
                rows.append((image_name, text))
        rows.sort()
        write_manifest_atomic(combined_root / f"gt_{split}.txt", rows)
        summary[split] = len(rows)
    return summary


def validate_partition(root: Path) -> None:
    """Validate uniqueness, image presence, and split isolation."""
    split_names: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = parse_manifest(root / f"gt_{split}.txt")
        names = {image_name for image_name, _ in rows}
        if len(names) != len(rows):
            raise ValueError(f"Duplicate rows in {root}/gt_{split}.txt")
        for image_name in names:
            if not (root / "image" / image_name).is_file():
                raise FileNotFoundError(f"Missing image: {root / 'image' / image_name}")
        split_names[split] = names
    if split_names["train"] & split_names["val"]:
        raise ValueError(f"Train/validation overlap under {root}")
    if split_names["train"] & split_names["test"]:
        raise ValueError(f"Train/test overlap under {root}")
    if split_names["val"] & split_names["test"]:
        raise ValueError(f"Validation/test overlap under {root}")


def main() -> None:
    """Normalize language and combined TrOCR manifests."""
    summary: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        for language in LANGUAGES:
            key = f"{language}/{partition}"
            summary[key] = resplit_language_partition(language, partition)
        summary[f"combined/{partition}"] = rebuild_combined_partition(partition)

    for partition in PARTITIONS:
        for dataset in (*LANGUAGES, "combined"):
            root = PROCESSED_ROOT / dataset / partition
            validate_partition(root)
            if partition == "finetuning":
                names = {
                    image_name
                    for split in SPLITS
                    for image_name, _ in parse_manifest(root / f"gt_{split}.txt")
                }
                if any(name.startswith(("esteban__", "labelled__")) for name in names):
                    raise ValueError(f"External pretraining data leaked into {root}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
