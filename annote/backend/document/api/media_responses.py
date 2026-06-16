"""Shared OpenAPI + FileResponse helpers for part image routes."""

from fastapi.responses import FileResponse

from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.orm_models import DocumentPart

IMAGE_MEDIA_RESPONSES: dict = {
    200: {
        "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}},
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
            "image/gif": {"schema": {"type": "string", "format": "binary"}},
            "image/webp": {"schema": {"type": "string", "format": "binary"}},
            "application/octet-stream": {
                "schema": {"type": "string", "format": "binary"}
            },
        },
        "description": "Document part image bytes",
    },
    404: {"description": "Part or image not found"},
}


def media_type_for_image_key(image_key: str) -> str:
    suffix = image_key.rsplit(".", 1)[-1].lower()
    if suffix in ("png", "jpg", "jpeg", "gif", "webp"):
        return f"image/{'jpeg' if suffix == 'jpg' else suffix}"
    return "application/octet-stream"


def part_image_file_response(service: DocumentService, part: DocumentPart) -> FileResponse:
    path = service.resolve_part_image_path(part)
    return FileResponse(path, media_type=media_type_for_image_key(part.image_key))
