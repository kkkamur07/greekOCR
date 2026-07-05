"""Document and DocumentPart routes under projects."""

from io import BytesIO
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.export_service import AnnotationExportService
from backend.annotation.application.page_xml_export_service import PageXmlExportService
from backend.annotation.application.transcription_pdf_service import TranscriptionPdfService
from backend.core.exceptions import ValidationError
from backend.document.api.line_responses import line_response
from backend.document.api.responses import (
    document_response,
    document_with_parts_response,
    part_response,
)
from backend.document.api.schemas import (
    BlockCreateRequest,
    BlockPatchRequest,
    BlockResponse,
    CopyToGroundTruthRequest,
    CopyToGroundTruthResponse,
    DocumentCreateRequest,
    DocumentPartResponse,
    DocumentPartUpdateRequest,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentWithPartsResponse,
    ExportArtifactResponse,
    ExportResponse,
    ExportWarningsResponse,
    LayoutResetRequest,
    LayoutResponse,
    LineCreateRequest,
    LinePatchRequest,
    LineResponse,
    LinesReplaceRequest,
    LineTranscriptionPatchRequest,
    LineTranscriptionResponse,
    PagePairingResponse,
    PageTranscriptionImportRequest,
    PageTranscriptionTextLineResponse,
    PairingProgressResponse,
    PairTextLineRequest,
    ReorderPartsRequest,
    SegmentPartRequest,
    TranscribePartRequest,
    TranscriptionLayerResponse,
)
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Block
from backend.jobs.api.schemas import EnqueueJobResponse
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_service = DocumentService()
_document_repo = DocumentRepository()
_export_service = AnnotationExportService()
_page_xml_export_service = PageXmlExportService()
_transcription_pdf_service = TranscriptionPdfService()
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 200_000_000
PDF_RESPONSE = {
    200: {
        "content": {
            "application/pdf": {"schema": {"type": "string", "format": "binary"}}
        },
        "description": "Transcription PDF bytes",
    }
}
XML_RESPONSE = {
    200: {
        "content": {
            "application/xml": {"schema": {"type": "string", "format": "binary"}}
        },
        "description": "PAGE XML bytes",
    }
}


def _block_response(block: Block) -> BlockResponse:
    return BlockResponse(
        id=block.id,
        part_id=block.part_id,
        order=block.order,
        box=block.box,
        manual_geometry=block.manual_geometry,
        created_at=block.created_at,
    )


def _pairing_response(text_lines: list, progress: dict[str, int]) -> PagePairingResponse:
    return PagePairingResponse(
        text_lines=[
            PageTranscriptionTextLineResponse(
                order=line.order,
                text=line.text,
                paired_line_id=line.paired_line_id,
            )
            for line in text_lines
        ],
        pairing_progress=PairingProgressResponse(**progress),
    )


def _export_response(result) -> ExportResponse:
    return ExportResponse(
        exported_count=result.exported_count,
        artifacts=[
            ExportArtifactResponse(
                line_id=artifact.line_id,
                segment_number=artifact.segment_number,
                image_filename=artifact.image_filename,
                transcription_filename=artifact.transcription_filename,
                transcription_text=artifact.transcription_text,
                image_base64=artifact.image_base64,
            )
            for artifact in result.artifacts
        ],
        warnings=ExportWarningsResponse(
            unpaired_segments=result.warnings.unpaired_segments,
            unused_text_lines=result.warnings.unused_text_lines,
        ),
        steps=result.steps,
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_archived: bool = Query(default=False),
) -> list[DocumentResponse]:
    documents = await _service.list_documents(
        db, current_user, project_id, include_archived=include_archived
    )
    part_counts = await _document_repo.count_parts_by_document_ids(
        db, [document.id for document in documents]
    )
    return [
        document_response(document, part_count=part_counts.get(document.id, 0))
        for document in documents
    ]


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    project_id: UUID,
    body: DocumentCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    document = await _service.create_document(
        db, current_user, project_id, name=body.name
    )
    return document_response(document, part_count=0)


@router.get("/{document_id}", response_model=DocumentWithPartsResponse)
async def get_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentWithPartsResponse:
    document = await _service.get_document(db, current_user, project_id, document_id)
    return document_with_parts_response(document)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    project_id: UUID,
    document_id: UUID,
    body: DocumentUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    updates = body.model_dump(exclude_unset=True)
    document = await _service.update_document(
        db, current_user, project_id, document_id, **updates
    )
    part_counts = await _document_repo.count_parts_by_document_ids(db, [document.id])
    return document_response(document, part_count=part_counts.get(document.id, 0))


@router.get("/{document_id}/transcriptions", response_model=list[TranscriptionLayerResponse])
async def list_transcriptions(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TranscriptionLayerResponse]:
    transcriptions = await _service.list_transcriptions(
        db, current_user, project_id, document_id
    )
    return [TranscriptionLayerResponse.model_validate(t) for t in transcriptions]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_document(db, current_user, project_id, document_id)


@router.post(
    "/{document_id}/parts",
    response_model=DocumentPartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_part(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
) -> DocumentPartResponse:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ValidationError("Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError("File exceeds the 100 MB upload limit")
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValidationError("Uploaded file is not a valid image") from exc
    part = await _service.upload_part(
        db,
        current_user,
        project_id,
        document_id,
        data=data,
        filename=file.filename,
    )
    return part_response(part)


@router.patch("/{document_id}/parts/reorder", response_model=list[DocumentPartResponse])
async def reorder_parts(
    project_id: UUID,
    document_id: UUID,
    body: ReorderPartsRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DocumentPartResponse]:
    parts = await _service.reorder_parts(
        db, current_user, project_id, document_id, body.part_ids
    )
    return [part_response(p) for p in parts]


@router.patch("/{document_id}/parts/{part_id}", response_model=DocumentPartResponse)
async def update_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: DocumentPartUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentPartResponse:
    part = await _service.update_part_review_status(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        reviewed=body.reviewed,
    )
    return part_response(part)


@router.get("/{document_id}/parts/{part_id}/layout", response_model=LayoutResponse)
async def list_part_layout(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LayoutResponse:
    blocks, lines = await _service.list_part_layout(
        db, current_user, project_id, document_id, part_id
    )
    return LayoutResponse(
        blocks=[_block_response(block) for block in blocks],
        lines=[line_response(line) for line in lines],
    )


@router.post(
    "/{document_id}/parts/{part_id}/blocks",
    response_model=BlockResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_part_block(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: BlockCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BlockResponse:
    block = await _service.create_part_block(
        db, current_user, project_id, document_id, part_id, order=body.order, box=body.box
    )
    return _block_response(block)


@router.patch("/{document_id}/parts/{part_id}/blocks/{block_id}", response_model=BlockResponse)
async def patch_part_block(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    block_id: UUID,
    body: BlockPatchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BlockResponse:
    block = await _service.patch_part_block(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        block_id,
        **body.model_dump(exclude_unset=True),
    )
    return _block_response(block)


@router.delete(
    "/{document_id}/parts/{part_id}/blocks/{block_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_part_block(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    block_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_part_block(db, current_user, project_id, document_id, part_id, block_id)


@router.get("/{document_id}/parts/{part_id}/lines", response_model=list[LineResponse])
async def list_part_lines(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _service.list_part_lines(db, current_user, project_id, document_id, part_id)
    return [line_response(line) for line in lines]


@router.post(
    "/{document_id}/parts/{part_id}/lines",
    response_model=LineResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_part_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: LineCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LineResponse:
    line = await _service.create_part_line(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        order=body.order,
        kind=body.kind,
        points=body.points,
        block_id=body.block_id,
        baseline=body.baseline,
        mask=body.mask,
    )
    return line_response(line)


@router.patch("/{document_id}/parts/{part_id}/lines/{line_id}", response_model=LineResponse)
async def patch_part_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    line_id: UUID,
    body: LinePatchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LineResponse:
    line = await _service.patch_part_line(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        line_id,
        **body.model_dump(exclude_unset=True),
    )
    return line_response(line)


@router.delete(
    "/{document_id}/parts/{part_id}/lines/{line_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_part_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    line_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_part_line(db, current_user, project_id, document_id, part_id, line_id)


@router.put("/{document_id}/parts/{part_id}/lines", response_model=list[LineResponse])
async def replace_part_lines(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: LinesReplaceRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _service.replace_part_lines(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        lines=[line.model_dump() for line in body.lines],
    )
    return [line_response(line) for line in lines]


@router.post("/{document_id}/parts/{part_id}/layout/reset", response_model=LayoutResponse)
async def reset_part_layout(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: LayoutResetRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LayoutResponse:
    blocks, lines = await _service.reset_part_layout(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        line_ids=body.line_ids,
    )
    return LayoutResponse(
        blocks=[_block_response(block) for block in blocks],
        lines=[line_response(line) for line in lines],
    )


@router.put("/{document_id}/parts/{part_id}/page-transcription", response_model=PagePairingResponse)
async def import_page_transcription(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: PageTranscriptionImportRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PagePairingResponse:
    text_lines, progress = await _service.import_page_transcription(
        db, current_user, project_id, document_id, part_id, text=body.text
    )
    return _pairing_response(text_lines, progress)


@router.get("/{document_id}/parts/{part_id}/pairing", response_model=PagePairingResponse)
async def get_page_pairing(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PagePairingResponse:
    text_lines, progress = await _service.get_page_pairing(
        db, current_user, project_id, document_id, part_id
    )
    return _pairing_response(text_lines, progress)


@router.post("/{document_id}/parts/{part_id}/pairings", response_model=PagePairingResponse)
async def pair_page_text_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: PairTextLineRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PagePairingResponse:
    text_lines, progress = await _service.pair_page_text_line(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        line_id=body.line_id,
        text_line_order=body.text_line_order,
    )
    return _pairing_response(text_lines, progress)


@router.post("/{document_id}/parts/{part_id}/export", response_model=ExportResponse)
async def export_approved_line_artifacts(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ExportResponse:
    result = await _export_service.export_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    return _export_response(result)


@router.post(
    "/{document_id}/parts/{part_id}/transcription-pdf",
    response_class=Response,
    responses=PDF_RESPONSE,
)
async def generate_transcription_pdf(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    pdf_bytes = await _transcription_pdf_service.generate_part_pdf(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="transcription.pdf"'},
    )


@router.get(
    "/{document_id}/parts/{part_id}/page-xml",
    response_class=Response,
    responses=XML_RESPONSE,
)
async def export_part_page_xml(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    xml_bytes = await _page_xml_export_service.export_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    return Response(
        content=xml_bytes,
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="page.xml"'},
    )


@router.post(
    "/{document_id}/parts/{part_id}/segment",
    response_model=EnqueueJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def segment_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: SegmentPartRequest | None = None,
) -> EnqueueJobResponse:
    body = body or SegmentPartRequest()
    job = await _service.enqueue_segment_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        ml_params=body.model_dump(),
    )
    return EnqueueJobResponse(job_id=job.id)


@router.post(
    "/{document_id}/parts/{part_id}/transcribe",
    response_model=EnqueueJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def transcribe_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: TranscribePartRequest | None = None,
) -> EnqueueJobResponse:
    body = body or TranscribePartRequest()
    job = await _service.enqueue_transcribe_part(
        db, current_user, project_id, document_id, part_id, model_id=body.model_id, line_ids=body.line_ids
    )
    return EnqueueJobResponse(job_id=job.id)


@router.post(
    "/{document_id}/transcriptions/{transcription_id}/copy-to-ground-truth",
    response_model=CopyToGroundTruthResponse,
)
async def copy_to_ground_truth(
    project_id: UUID,
    document_id: UUID,
    transcription_id: UUID,
    body: CopyToGroundTruthRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CopyToGroundTruthResponse:
    copied_line_ids = await _service.copy_to_ground_truth(
        db,
        current_user,
        project_id,
        document_id,
        transcription_id,
        line_ids=body.line_ids,
    )
    return CopyToGroundTruthResponse(copied_line_ids=copied_line_ids)


@router.patch(
    "/{document_id}/transcriptions/{transcription_id}/lines/{line_id}",
    response_model=LineTranscriptionResponse,
)
async def patch_ground_truth_line_text(
    project_id: UUID,
    document_id: UUID,
    transcription_id: UUID,
    line_id: UUID,
    body: LineTranscriptionPatchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LineTranscriptionResponse:
    line_transcription = await _service.patch_ground_truth_line_text(
        db,
        current_user,
        project_id,
        document_id,
        transcription_id,
        line_id,
        text=body.text,
    )
    return LineTranscriptionResponse(
        id=line_transcription.id,
        transcription_id=line_transcription.transcription_id,
        transcription_kind="ground_truth",
        text=line_transcription.text,
        confidence=line_transcription.confidence,
    )


@router.delete("/{document_id}/parts/{part_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_part(db, current_user, project_id, document_id, part_id)
