#!/usr/bin/env python3
"""Prepare a flat Calamari pack from split line-crop data."""

from __future__ import annotations

import shutil
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
REPO_ROOT = Path(__file__).resolve().parents[2]

# Edit these values directly before running the script.
DATA_ROOT = REPO_ROOT / "data" / "labelledData"   #! change the path run the script
OUTPUT_ROOT = REPO_ROOT / "data" / "calamari_pack"
SPLITS = ("train", "val", "test")
SYMLINK_IMAGES = True
FORCE_OVERWRITE = False


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
    return (REPO_ROOT / path).resolve()


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


def main() -> None:
    root = _resolve_path(DATA_ROOT)
    out_root = _resolve_path(OUTPUT_ROOT)
    _validate_root(root)

    total_exported = 0
    for split in SPLITS:
        total_exported += _prepare_split(
            root=root,
            out_root=out_root,
            split=split,
            use_symlink=bool(SYMLINK_IMAGES),
            force=bool(FORCE_OVERWRITE),
        )

    if total_exported == 0:
        raise SystemExit(
            "\nerror: no line images were exported.\n"
            f"  data.root used: {root}\n\n"
            "Fix one of:\n"
            "  • Put crops under images/train (and labels/train/*.gt.txt), etc.\n"
            "  • Override data.root=/path/to/dataset"
        )


if __name__ == "__main__":
    main()
