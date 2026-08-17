#!/usr/bin/env python3
"""Prepare a flat Calamari pack from split line-crop data."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _read_gt_source(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _write_gt_utf8(dest: Path, text: str) -> None:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff")
    with dest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return path.resolve()


def _validate_root(root: Path) -> None:
    if not root.is_dir():
        raise SystemExit(
            f"-- data.root is not a directory:\n  {root}\n\n"
            "Point it at a dataset root with images/<split>/ and labels/<split>/."
        )
    images_root = root / "images"
    if not images_root.is_dir():
        raise SystemExit(
            f"Missing expected folder:\n  {images_root}\n\n"
            "Expected dataset layout:\n"
            "  DATA_ROOT/images/train  DATA_ROOT/labels/train/*.gt.txt\n"
            "  (same for val, test)"
        )


def _prepare_split(root: Path, out_root: Path, split: str, use_symlink: bool, force: bool) -> int:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not img_dir.is_dir():
        print(f"Skip missing {img_dir}")
        return 0

    dest_split = out_root / split
    if force and dest_split.exists():
        shutil.rmtree(dest_split)
    dest_split.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS or img_path.name.startswith("."):
            continue
        gt_src = lbl_dir / f"{img_path.stem}.gt.txt"
        if not gt_src.is_file():
            print(f"Missing GT, skipping {img_path.name}")
            continue

        img_dst = dest_split / img_path.name
        gt_dst = dest_split / f"{img_path.stem}.gt.txt"
        if img_dst.exists() or gt_dst.exists():
            raise FileExistsError(f"Destination exists: {img_dst} or {gt_dst}")

        if use_symlink:
            img_dst.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, img_dst)
        _write_gt_utf8(gt_dst, _read_gt_source(gt_src))

    exported = sum(1 for path in dest_split.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    print(f"{split}: exported {exported} line images to {dest_split}")
    return exported


def prepare_pack(
    data_root: Path,
    output_root: Path,
    *,
    copy_images: bool = False,
    force: bool = False,
) -> int:
    """Create a flat train/val/test Calamari pack from paired line crops."""
    root = _resolve_path(data_root)
    out_root = _resolve_path(output_root)
    _validate_root(root)
    total_exported = 0
    for split in ("train", "val", "test"):
        total_exported += _prepare_split(
            root=root,
            out_root=out_root,
            split=split,
            use_symlink=not copy_images,
            force=force,
        )
    if total_exported == 0:
        raise ValueError(f"No line images were exported from {root}.")
    return total_exported


def prepare_trocr_manifest_pack(
    data_root: Path,
    output_root: Path,
    *,
    copy_images: bool = False,
    force: bool = False,
) -> int:
    """Create a native Calamari pack from TrOCR manifests."""
    root = _resolve_path(data_root)
    images_root = root / "image"
    if not images_root.is_dir():
        raise ValueError(f"Missing TrOCR image directory: {images_root}")

    output_root = _resolve_path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"Destination exists: {output_root}; use force=true.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total_exported = 0
    for split in ("train", "val", "test"):
        manifest = root / f"gt_{split}.txt"
        if not manifest.is_file():
            continue
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            image_name, text = line.split("\t", 1)
            source = images_root / image_name
            if not source.is_file():
                raise FileNotFoundError(f"Missing TrOCR image declared by {manifest}: {source}")
            image_destination = split_root / image_name
            label_destination = split_root / f"{Path(image_name).stem}.gt.txt"
            if image_destination.exists() or label_destination.exists():
                raise FileExistsError(
                    f"Destination exists: {image_destination} or {label_destination}"
                )
            if copy_images:
                shutil.copy2(source, image_destination)
            else:
                try:
                    os.link(source, image_destination)
                except OSError:
                    shutil.copy2(source, image_destination)
            _write_gt_utf8(label_destination, text)
            total_exported += 1
    if total_exported == 0:
        raise ValueError(f"No TrOCR manifest rows were exported from {root}.")
    return total_exported


def prepare_combined_trocr_manifest_pack(
    data_root: Path,
    output_root: Path,
    source_packs: tuple[Path, ...],
    *,
    force: bool = False,
) -> int:
    """Create a native combined pack by hardlinking existing Calamari pairs.

    The individual language packs must already be materialized. Reusing their
    image and ``.gt.txt`` inodes keeps a combined view within file-count quotas.
    """
    root = _resolve_path(data_root)
    if not (root / "image").is_dir():
        raise ValueError(f"Missing TrOCR image directory: {root / 'image'}")

    output_root = _resolve_path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"Destination exists: {output_root}; use force=true.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_packs = tuple(_resolve_path(pack) for pack in source_packs)
    total_exported = 0
    for split in ("train", "val", "test"):
        manifest = root / f"gt_{split}.txt"
        if not manifest.is_file():
            continue
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)

        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            image_name, text = line.split("\t", 1)
            label_name = f"{Path(image_name).stem}.gt.txt"
            matches = [
                (pack / split / image_name, pack / split / label_name)
                for pack in resolved_packs
                if (pack / split / image_name).is_file()
                and (pack / split / label_name).is_file()
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one source pair for {split}/{image_name}; "
                    f"found {len(matches)}."
                )
            source_image, source_label = matches[0]
            if _read_gt_source(source_label) != text:
                raise ValueError(f"Source label differs from combined manifest: {source_label}")

            image_destination = split_root / image_name
            label_destination = split_root / label_name
            if image_destination.exists() or label_destination.exists():
                raise FileExistsError(
                    f"Destination exists: {image_destination} or {label_destination}"
                )
            try:
                os.link(source_image, image_destination)
                os.link(source_label, label_destination)
            except OSError as exc:
                raise OSError(
                    "Combined Calamari packs require hardlinks to reuse the "
                    "already-materialized language pairs."
                ) from exc
            total_exported += 1
    if total_exported == 0:
        raise ValueError(f"No TrOCR manifest rows were exported from {root}.")
    return total_exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Calamari train/val/test pack from labeled line images."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        prepare_pack(
            args.data_root,
            args.output_root,
            copy_images=args.copy_images,
            force=args.force,
        )
    except ValueError as error:
        raise SystemExit(f"\nerror: {error}") from error


if __name__ == "__main__":
    main()
