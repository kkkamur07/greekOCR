"""Replace MS_P_172 and MS_UCLA_MS in processed Armenian with the new exports.

Removes existing ms_p_172* / ms_ucla* crops from pretraining and finetuning,
then adds each new folder as 30% pretraining / 70% finetuning, each with an
80/10/10 page split so lines from one page stay in one split.
"""

from __future__ import annotations

import os
import random
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing_data.armenian import (
    DROP_CROPS,
    normalize_latin_typos,
    normalize_print_punctuation,
)
from src.preprocessing_data.syriac import crop_polygon, parse_points, save_crop

PROCESSED = REPO_ROOT / "data" / "processed"
SOURCES = (
    ("ms_p_172", REPO_ROOT / "data" / "new_armenian" / "MS_P_172"),
    ("ms_ucla_ms", REPO_ROOT / "data" / "new_armenian" / "MS_UCLA_MS"),
)
REMOVE_PREFIXES = ("ms_p_172", "ms_ucla")
PARTITIONS = ("pretraining", "finetuning")
SPLITS = ("train", "val", "test")
SEED = 1111
PRETRAINING_RATIO = 0.3
SPLIT_RATIOS = (0.8, 0.1, 0.1)
PADDING = 12


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if local_tag(child.tag) == name]


def descendant(element: ET.Element, *names: str) -> ET.Element | None:
    current = element
    for name in names:
        found = next((child for child in current if local_tag(child.tag) == name), None)
        if found is None:
            return None
        current = found
    return current


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = " ".join(text.split())
    return unicodedata.normalize("NFC", text)


def page_lines(xml_path: Path) -> tuple[str, list[tuple[int, list[list[int]], str]]]:
    root = ET.parse(xml_path).getroot()
    page = next((child for child in root if local_tag(child.tag) == "Page"), None)
    if page is None or not page.get("imageFilename"):
        raise ValueError(f"PAGE XML missing Page/imageFilename: {xml_path}")
    lines: list[tuple[int, list[list[int]], str]] = []
    for region in children(page, "TextRegion"):
        for line in children(region, "TextLine"):
            text_el = descendant(line, "TextEquiv", "Unicode")
            text = (
                normalize_latin_typos(
                    normalize_print_punctuation(normalize_text(text_el.text or ""))
                )
                if text_el is not None
                else ""
            )
            if not text:
                continue
            coords = next((child for child in line if local_tag(child.tag) == "Coords"), None)
            polygon = parse_points(coords.get("points") if coords is not None else None)
            if len(polygon) < 3:
                continue
            lines.append((len(lines), polygon, text))
    return page.get("imageFilename"), lines


def split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train = round(total * ratios[0] / sum(ratios))
    val = round(total * ratios[1] / sum(ratios))
    test = total - train - val
    if total >= 3:
        counts = [max(1, train), max(1, val), max(1, test)]
        while sum(counts) > total:
            index = max(range(3), key=counts.__getitem__)
            if counts[index] == 1:
                break
            counts[index] -= 1
        while sum(counts) < total:
            counts[0] += 1
        return counts[0], counts[1], counts[2]
    return (max(1, total - 1), 1 if total > 1 else 0, 0)


def assign_pages(xml_files: list[Path], source_index: int) -> dict[str, dict[str, list[Path]]]:
    pages = list(xml_files)
    random.Random(SEED + source_index).shuffle(pages)
    pretraining_count = min(max(1, round(len(pages) * PRETRAINING_RATIO)), len(pages) - 1)
    partitions = {
        "pretraining": pages[:pretraining_count],
        "finetuning": pages[pretraining_count:],
    }
    assigned: dict[str, dict[str, list[Path]]] = {}
    for partition, partition_pages in partitions.items():
        train, val, test = split_counts(len(partition_pages), SPLIT_RATIOS)
        assigned[partition] = {
            "train": partition_pages[:train],
            "val": partition_pages[train : train + val],
            "test": partition_pages[train + val : train + val + test],
        }
    return assigned


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        image_name, text = line.split("\t", 1)
        rows.append((image_name, text))
    return rows


def write_manifest(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for image_name, text in rows:
            handle.write(f"{image_name}\t{text}\n")


def matches_removed(name: str) -> bool:
    return name.startswith(REMOVE_PREFIXES)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        destination.write_bytes(source.read_bytes())


def strip_old_crops(root: Path) -> int:
    removed = 0
    image_dir = root / "image"
    for image in list(image_dir.iterdir()):
        if image.is_file() and matches_removed(image.name):
            image.unlink()
            removed += 1
    for split in SPLITS:
        manifest = root / f"gt_{split}.txt"
        write_manifest(
            manifest, [row for row in parse_manifest(manifest) if not matches_removed(row[0])]
        )
    return removed


def main() -> None:
    for language in ("armenian", "combined"):
        for partition in PARTITIONS:
            removed = strip_old_crops(PROCESSED / language / partition)
            print(f"removed {removed} old crops from {language}/{partition}")

    added: dict[str, dict[str, int]] = {
        partition: {split: 0 for split in SPLITS} for partition in PARTITIONS
    }
    page_plan: dict[str, dict[str, dict[str, list[str]]]] = {}

    for source_index, (slug, source_root) in enumerate(SOURCES):
        xml_files = sorted(source_root.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No PAGE XML in {source_root}")
        assignments = assign_pages(xml_files, source_index)
        page_plan[slug] = {
            partition: {split: [path.stem for path in pages] for split, pages in splits.items()}
            for partition, splits in assignments.items()
        }
        for partition, splits in assignments.items():
            image_dir = PROCESSED / "armenian" / partition / "image"
            combined_dir = PROCESSED / "combined" / partition / "image"
            image_dir.mkdir(parents=True, exist_ok=True)
            for split, pages in splits.items():
                new_rows: list[tuple[str, str]] = []
                for xml_path in pages:
                    image_name, lines = page_lines(xml_path)
                    page_image = cv2.imread(str(source_root / image_name), cv2.IMREAD_COLOR)
                    if page_image is None:
                        raise ValueError(f"Unreadable page image: {source_root / image_name}")
                    for line_index, polygon, text in lines:
                        crop_name = f"{slug}__{xml_path.stem}__{line_index:03d}.jpg"
                        if crop_name in DROP_CROPS:
                            continue
                        crop, _ = crop_polygon(page_image, polygon, PADDING, keep_color=False)
                        crop_path = image_dir / crop_name
                        if crop_path.exists():
                            raise FileExistsError(crop_name)
                        save_crop(crop_path, crop, keep_color=False)
                        hardlink_or_copy(crop_path, combined_dir / crop_name)
                        new_rows.append((crop_name, text))
                armenian_manifest = PROCESSED / "armenian" / partition / f"gt_{split}.txt"
                combined_manifest = PROCESSED / "combined" / partition / f"gt_{split}.txt"
                write_manifest(armenian_manifest, parse_manifest(armenian_manifest) + new_rows)
                write_manifest(combined_manifest, parse_manifest(combined_manifest) + new_rows)
                added[partition][split] += len(new_rows)

    print("page assignments", page_plan)
    print("added lines", added)
    for language in ("armenian", "combined"):
        for partition in PARTITIONS:
            counts = {}
            for split in SPLITS:
                counts[split] = len(
                    parse_manifest(PROCESSED / language / partition / f"gt_{split}.txt")
                )
            print(f"{language}/{partition}", counts)


if __name__ == "__main__":
    main()
