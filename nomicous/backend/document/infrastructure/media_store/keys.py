"""Logical object keys for document part page images."""

import re
from uuid import UUID

_SAFE_IMAGE_KEY = re.compile(
    r"^parts/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?:/[a-z0-9][a-z0-9_-]{0,127})?\.[a-z0-9]{1,16}$"
)

DEFAULT_PART_IMAGE_SUFFIX = "webp"


def part_image_key(
    part_id: UUID, *, suffix: str = DEFAULT_PART_IMAGE_SUFFIX, filename_stem: str | None = None
) -> str:
    safe = re.sub(r"[^a-z0-9]", "", suffix.lstrip(".").lower())[:16] or DEFAULT_PART_IMAGE_SUFFIX
    if filename_stem is None:
        return f"parts/{part_id}.{safe}"
    stem = re.sub(r"[^a-z0-9_-]", "-", filename_stem.lower()).strip("-_")[:128]
    if not stem:
        stem = str(part_id)
    return f"parts/{part_id}/{stem}.{safe}"


def validate_image_key(image_key: str) -> None:
    if not image_key or image_key.startswith("/"):
        raise ValueError("Invalid image key")
    if ".." in image_key.split("/"):
        raise ValueError("Invalid image key")
    if not _SAFE_IMAGE_KEY.match(image_key):
        raise ValueError("Invalid image key")
