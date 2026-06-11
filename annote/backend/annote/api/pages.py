"""Page routes — catalogue, images, transcriptions, annotations, export."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from annote.schemas.annotation import PageAnnotation
from annote.schemas.auto_segment import AutoSegmentRequest
from annote.schemas.export import ExportErrorEvent, ExportResponse
from annote.schemas.history import HistoryListResponse
from annote.schemas.ocr import OcrErrorEvent
from annote.schemas.pages import PageListResponse, PageSummary, TextLineOut, TranscriptionResponse
from annote.services.annotation_history import (
    capture_snapshot,
    list_history,
    maybe_capture_on_save,
    restore_snapshot,
)
from annote.services.annotation_merge import clear_stale_model_transcriptions
from annote.services.annotation_store import load_annotation, save_annotation
from annote.services.calamari_ocr import ocr_page_events, run_segment_ocr
from annote.services.export_service import export_page, export_page_events
from annote.services.kraken_binarize import binarize_page, clear_binarized_page
from annote.services.kraken_segment import auto_segment_page
from annote.services.image_export import encode_page_display_jpeg, needs_display_jpeg
from annote.services.page_catalogue import build_page_summary, image_media_type, list_pages, resolve_page_image
from annote.services.page_image import resolve_source_page_image, resolve_working_page_image
from annote.services.page_lock import assert_page_unlocked, lock_page, unlock_page
from annote.services.page_import import import_page
from annote.services.preview_service import preview_segment_jpeg
from annote.services.text_lines import split_text_lines
from annote.services.transcription_pdf import generate_transcription_pdf
from annote.services.transcription_pdf_share import (
    read_share_pdf,
    remove_share_pdf,
    write_share_pdf_bytes,
)
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
    return build_page_summary(root, stem)


@router.get("/{stem}/image", response_model=None)
async def get_page_image(stem: str):
    root = _data_root()
    if resolve_source_page_image(root, stem) is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    image_path = resolve_working_page_image(root, stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    if needs_display_jpeg(image_path):
        return Response(content=encode_page_display_jpeg(image_path), media_type="image/jpeg")
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


@router.post("/{stem}/binarize", response_model=PageAnnotation)
async def post_binarize_page(stem: str) -> PageAnnotation:
    """Binarize the whole page with Kraken nlbin and use it for display and processing."""
    root = _data_root()
    if resolve_source_page_image(root, stem) is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    assert_page_unlocked(load_annotation(root, stem))
    try:
        saved = binarize_page(root, stem)
        maybe_capture_on_save(root, stem, saved)
        return saved
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/{stem}/binarize", response_model=PageAnnotation)
async def delete_binarized_page(stem: str) -> PageAnnotation:
    """Revert to the original manuscript image for display and processing."""
    root = _data_root()
    if resolve_source_page_image(root, stem) is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    assert_page_unlocked(load_annotation(root, stem))
    saved = clear_binarized_page(root, stem)
    maybe_capture_on_save(root, stem, saved)
    return saved


@router.post("/{stem}/segment", response_model=PageAnnotation)
async def post_auto_segment(stem: str, body: AutoSegmentRequest | None = None) -> PageAnnotation:
    """Run Kraken BLLA line segmentation and save segments to the page annotation JSON."""
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    assert_page_unlocked(load_annotation(_data_root(), stem))
    opts = body or AutoSegmentRequest()
    try:
        root = _data_root()
        saved = auto_segment_page(
            root,
            stem,
            replace=opts.replace,
            pair_transcription=opts.pair_transcription,
        )
        maybe_capture_on_save(root, stem, saved)
        return saved
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
    assert_page_unlocked(existing)
    if existing.export_metadata and annotation.export_metadata is None:
        annotation.export_metadata = existing.export_metadata
    annotation.locked = existing.locked
    annotation.binarized_at = existing.binarized_at
    annotation = clear_stale_model_transcriptions(existing, annotation)
    saved = save_annotation(_data_root(), stem, annotation)
    maybe_capture_on_save(_data_root(), stem, saved)
    return saved


@router.post("/{stem}/lock", response_model=PageAnnotation)
async def post_lock_page(stem: str) -> PageAnnotation:
    root = _data_root()
    image_path = resolve_page_image(root / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    try:
        pdf_bytes = generate_transcription_pdf(root, stem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    saved = lock_page(root, stem)
    try:
        write_share_pdf_bytes(root, stem, pdf_bytes)
        capture_snapshot(root, stem, saved, reason="lock", protected=True)
    except Exception as e:
        unlock_page(root, stem)
        remove_share_pdf(root, stem)
        raise HTTPException(status_code=500, detail="Failed to write share PDF") from e
    return saved


@router.post("/{stem}/unlock", response_model=PageAnnotation)
async def post_unlock_page(stem: str) -> PageAnnotation:
    root = _data_root()
    saved = unlock_page(root, stem)
    capture_snapshot(root, stem, saved, reason="unlock", protected=True)
    remove_share_pdf(root, stem)
    return saved


@router.get("/{stem}/history", response_model=HistoryListResponse)
async def get_page_history(stem: str) -> HistoryListResponse:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    return list_history(_data_root(), stem)


@router.post("/{stem}/history/{snapshot_id}/restore", response_model=PageAnnotation)
async def post_restore_history(stem: str, snapshot_id: str) -> PageAnnotation:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    return restore_snapshot(_data_root(), stem, snapshot_id)


@router.get("/{stem}/transcription.share.pdf")
async def get_transcription_share_pdf(stem: str) -> Response:
    root = _data_root()
    image_path = resolve_page_image(root / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    annotation = load_annotation(root, stem)
    if not annotation.locked:
        raise HTTPException(status_code=404, detail="Share PDF is only available for locked pages")
    pdf_bytes = read_share_pdf(root, stem)
    if pdf_bytes is None:
        raise HTTPException(status_code=404, detail="Share PDF not found")
    filename = f"{stem}_transcription.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{stem}/transcription.pdf")
async def get_transcription_pdf(stem: str) -> Response:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    try:
        pdf_bytes = generate_transcription_pdf(_data_root(), stem)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    filename = f"{stem}_transcription.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.get("/{stem}/segments/{segment_id}/preview")
async def get_segment_preview(stem: str, segment_id: str) -> Response:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    try:
        jpeg = preview_segment_jpeg(_data_root(), stem, segment_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Response(content=jpeg, media_type="image/jpeg")


@router.post("/{stem}/segments/{segment_id}/ocr", response_model=PageAnnotation)
async def post_segment_ocr(stem: str, segment_id: str) -> PageAnnotation:
    """Run Calamari OCR on one segment and persist model transcription fields."""
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    try:
        root = _data_root()
        saved = run_segment_ocr(root, stem, segment_id)
        maybe_capture_on_save(root, stem, saved)
        return saved
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{stem}/ocr/stream")
async def post_page_ocr_stream(stem: str) -> StreamingResponse:
    """Stream NDJSON progress while running OCR on every segment on the page."""
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    data_root = _data_root()

    def generate():
        try:
            for event in ocr_page_events(data_root, stem):
                yield json.dumps(event.model_dump()) + "\n"
            maybe_capture_on_save(data_root, stem, load_annotation(data_root, stem))
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            yield json.dumps(OcrErrorEvent(detail=str(e)).model_dump()) + "\n"
        except Exception as e:
            logger.exception("Unexpected page OCR stream failure")
            yield json.dumps(OcrErrorEvent(detail="Unexpected OCR failure").model_dump()) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/{stem}/export", response_model=ExportResponse)
async def post_export(stem: str) -> ExportResponse:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    try:
        return export_page(_data_root(), stem)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{stem}/export/stream")
async def post_export_stream(stem: str) -> StreamingResponse:
    image_path = resolve_page_image(_data_root() / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")
    data_root = _data_root()

    def generate():
        try:
            for event in export_page_events(data_root, stem):
                yield json.dumps(event.model_dump()) + "\n"
        except (RuntimeError, ValueError) as e:
            yield json.dumps(ExportErrorEvent(detail=str(e)).model_dump()) + "\n"
        except Exception as e:
            logger.exception("Unexpected export stream failure")
            yield json.dumps(ExportErrorEvent(detail="Unexpected export failure").model_dump()) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
