#!/usr/bin/env python3
"""
Copy BibleDataset-style line crops into a flat folder for Calamari training.

Input layout::
    DATA_ROOT/images/<split>/*.png
    DATA_ROOT/labels/<split>/*.gt.txt

Output::
    OUT_DIR/<split>/image.png + image.gt.txt  (same basename, side by side).
    ``*.gt.txt`` are always written as UTF-8 (sources may be UTF-8 / UTF-16 / etc.).

Usage::
    python ocr/calamari_ocr/prepare_calamari_pack.py

With explicit paths::

    python ocr/calamari_ocr/prepare_calamari_pack.py \\
        --data-root ./data/labelledData \\
        --out-dir ./data/calamari_pack
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _read_gt_source(path: Path) -> str:
    """Decode a ground-truth file regardless of source encoding (Calamari expects UTF-8)."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("utf-16-le")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("utf-16-be")
    except UnicodeDecodeError:
        pass
    return raw.decode("latin-1")


def _write_gt_utf8(dest: Path, text: str) -> None:
    """Write normalized UTF-8 text for Calamari (.gt.txt)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip("\ufeff")
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> None:
    default_data = Path(
        os.environ.get(
            "BIBLE_DATA_ROOT",
            str(_REPO_ROOT / "data" / "labelledData"),
        )
    )
    default_out = _REPO_ROOT / "data" / "calamari_pack"

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        default=default_data,
        help=f"BibleDataset root (images/, labels/). Default: {default_data} or $BIBLE_DATA_ROOT.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"Flat Calamari pack root. Default: {default_out}",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to export (must exist under images/<split>).",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images (default). Omit to symlink images. Ground truth is always rewritten as UTF-8.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Remove existing split folders under --out-dir before exporting.",
    )
    args = p.parse_args()
    use_symlink = not args.copy

    root = args.data_root.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()

    if not root.is_dir():
        print(
            f"error: --data-root is not a directory:\n  {root}\n\n"
            "Point it at your BibleDataset root (with images/<split>/ and labels/<split>/), e.g.:\n"
            "  python ocr/calamari_ocr/prepare_calamari_pack.py --data-root /path/to/labelledData\n"
            "or set environment variable BIBLE_DATA_ROOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    images_root = root / "images"
    if not images_root.is_dir():
        print(
            f"error: expected folder missing:\n  {images_root}\n\n"
            "Create BibleDataset layout:\n"
            "  DATA_ROOT/images/train  DATA_ROOT/labels/train/*.gt.txt\n"
            "  (same for val, test)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    total_exported = 0
    for split in args.splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            print(
                f"Skip missing {img_dir}\n"
                f"  (create it or remove '{split}' from --splits)",
                file=sys.stderr,
            )
            continue

        dest_split = out_root / split
        if args.force and dest_split.exists():
            shutil.rmtree(dest_split)
        dest_split.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS or img_path.name.startswith("."):
                continue
            stem = img_path.stem
            gt_src = lbl_dir / f"{stem}.gt.txt"
            if not gt_src.is_file():
                print(f"Missing GT, skipping {img_path.name}", file=sys.stderr)
                continue

            img_dst = dest_split / img_path.name
            gt_dst = dest_split / f"{stem}.gt.txt"

            if img_dst.exists() or gt_dst.exists():
                raise FileExistsError(f"Destination exists: {img_dst} or {gt_dst}")

            gt_text = _read_gt_source(gt_src)
            if use_symlink:
                img_dst.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, img_dst)
            _write_gt_utf8(gt_dst, gt_text)

        n = sum(
            1
            for p in dest_split.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
        total_exported += n
        print(f"{split}: exported {n} line images to {dest_split}")

    if total_exported == 0:
        print(
            "\nerror: no line images were exported.\n"
            f"  data-root used: {root}\n\n"
            "Fix one of:\n"
            "  • Put crops under images/train (and labels/train/*.gt.txt), etc.\n"
            "  • Pass the real dataset path: --data-root /path/to/labelledData\n"
            "  • export BIBLE_DATA_ROOT=/path/to/labelledData\n",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
