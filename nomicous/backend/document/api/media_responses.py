"""Shared OpenAPI + response helpers for part image routes."""
from hashlib import sha256

from fastapi.responses import Response

from backend.core.schemas.errors import ApiErrorResponse
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.orm_models import DocumentPart

IMAGE_MEDIA_RESPONSES: dict = {
    200: {
        "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}},
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
            "image/gif": {"schema": {"type": "string", "format": "binary"}},
            "image/webp": {"schema": {"type": "string", "format": "binary"}},
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        },
        "description": "Document part image bytes",
    },
    304: {"description": "Media has not changed"},
    404: {"model": ApiErrorResponse, "description": "Part or image not found"},
}

MAX_THUMBNAIL_WIDTH = 2048
PRIVATE_MEDIA_CACHE_CONTROL = "private, max-age=86400"
PUBLIC_MEDIA_CACHE_CONTROL = "public, max-age=300, must-revalidate"
THUMBNAIL_ENCODER_VERSION = "webp-q85-v1"


def media_type_for_image_key(image_key: str) -> str:
    suffix = image_key.rsplit(".", 1)[-1].lower()
    if suffix in ("png", "jpg", "jpeg", "gif", "webp"):
        return f"image/{'jpeg' if suffix == 'jpg' else suffix}"
    return "application/octet-stream"


def part_image_etag(part: DocumentPart, width: int | None) -> str:
    variant = "full" if width is None else f"thumbnail:w={width}:{THUMBNAIL_ENCODER_VERSION}"
    digest = sha256(f"{part.image_key}:{variant}".encode()).hexdigest()
    return f'"{digest}"'


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    if if_none_match is None:
        return False
    return any(candidate.strip() in ("*", etag) for candidate in if_none_match.split(","))


async def part_image_response(
    service: DocumentService,
    part: DocumentPart,
    *,
    width: int | None,
    if_none_match: str | None,
    cache_control: str,
) -> Response:
    etag = part_image_etag(part, width)
    headers = {"Cache-Control": cache_control, "ETag": etag}
    if etag_matches(if_none_match, etag):
        return Response(status_code=304, headers=headers)

    data = await service.read_part_bytes(part, width=width)
    media_type = "image/webp" if width is not None else media_type_for_image_key(part.image_key)
    return Response(content=data, media_type=media_type, headers=headers)
