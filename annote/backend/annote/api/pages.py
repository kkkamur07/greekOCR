"""Page routes — catalogue, images, transcriptions, annotations, export."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from annote.schemas.annotation import PageAnnotation
from annote.schemas.auto_segment import AutoSegmentRequest
from annote.schemas.export import ExportErrorEvent, ExportRequest
from annote.schemas.warnings import ExportResponse
from annote.schemas.pages import PageListResponse, PageSummary, TextLineOut, TranscriptionResponse
from annote.services.annotation_store import load_annotation, save_annotation
from annote.services.export_service import export_page, export_page_events
from annote.services.kraken_segment import auto_segment_page
from annote.services.page_catalogue import image_media_type, list_pages, resolve_page_image
from annote.services.page_import import import_page, import_summary
from annote.services.text_lines import split_text_lines
from annote.settings import get_settings

router = APIRouter(prefix="/pages", tags=["pages"])
logger = logging.getLogger(__name__)


def _data_root() -> Path:
    return get_settings().data_root


@router.get("", response_model=PageListResponse)
async def get_pages() -> PageListResponse:
    return PageListResponse(pages=list_pages(_data_root()))


@router.post("/import", response_model=PageSummary)
async def post_import_page(
    image: UploadFile = File(...),
    transcription: UploadFile | None = File(None),
) -> PageSummary:
    root = _data_root()
    stem = await import_page(root, image, transcription)
    return import_summary(root, stem)


@router.get("/{stem}/image")
async def get_page_image(stem: str) -> FileResponse:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    return FileResponse(image_path, media_type=image_media_type(image_path))


@router.get("/{stem}/transcription", response_model=TranscriptionResponse)
async def get_transcription(stem: str) -> TranscriptionResponse:
    path = _data_root() / "transcriptions" / "pages" / f"{stem}.txt"
    if not path.is_file():
        return TranscriptionResponse(raw_text=None, text_lines=[], status="missing")
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return TranscriptionResponse(raw_text=raw, text_lines=[], status="ok")
    lines = split_text_lines(raw)
    return TranscriptionResponse(
        raw_text=raw,
        text_lines=[TextLineOut(index=line.index, text=line.text) for line in lines],
        status="ok",
    )


@router.get("/{stem}/annotation", response_model=PageAnnotation)
async def get_annotation(stem: str) -> PageAnnotation:
    return load_annotation(_data_root(), stem)


@router.post("/{stem}/segment", response_model=PageAnnotation)
async def post_auto_segment(stem: str, body: AutoSegmentRequest | None = None) -> PageAnnotation:
    """Run Kraken BLLA line segmentation and save segments to the page annotation JSON."""
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    opts = body or AutoSegmentRequest()
    try:
        return auto_segment_page(
            _data_root(),
            stem,
            replace=opts.replace,
            pair_transcription=opts.pair_transcription,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.put("/{stem}/annotation", response_model=PageAnnotation)
async def put_annotation(stem: str, annotation: PageAnnotation) -> PageAnnotation:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    existing = load_annotation(_data_root(), stem)
    if existing.export_metadata and annotation.export_metadata is None:
        annotation.export_metadata = existing.export_metadata
    return save_annotation(_data_root(), stem, annotation)


@router.post("/{stem}/export", response_model=ExportResponse)
async def post_export(stem: str, body: ExportRequest | None = None) -> ExportResponse:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    opts = body or ExportRequest()
    try:
        return export_page(_data_root(), stem, binarize=opts.binarize)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{stem}/export/stream")
async def post_export_stream(stem: str, body: ExportRequest | None = None) -> StreamingResponse:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    opts = body or ExportRequest()
    data_root = _data_root()

    def generate():
        try:
            for event in export_page_events(data_root, stem, binarize=opts.binarize):
                yield json.dumps(event.model_dump()) + "\n"
        except (RuntimeError, ValueError) as e:
            yield json.dumps(ExportErrorEvent(detail=str(e)).model_dump()) + "\n"
        except Exception as e:
            logger.exception("Unexpected export stream failure")
            yield json.dumps(ExportErrorEvent(detail="Unexpected export failure").model_dump()) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
