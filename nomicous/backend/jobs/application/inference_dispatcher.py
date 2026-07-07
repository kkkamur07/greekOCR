"""Translate Product jobs into inference job submissions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from inference.contracts.common import InferenceTask as WireInferenceTask
from inference.contracts.jobs import JobSubmitRequest

from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.document.infrastructure.media_store import MediaStore
from backend.document.infrastructure.orm_models import DocumentPart
from backend.jobs.infrastructure.orm_models import Job, JobType
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from infrastructure.db import SyncSessionLocal

_DEFAULT_SEGMENT_REGISTRY_MODEL = "kraken-blla"
_DEFAULT_SEGMENT_REGISTRY_TAG = "stable"
_DEFAULT_TRANSCRIBE_REGISTRY_MODEL = "syriac-calamariv1"
_DEFAULT_TRANSCRIBE_REGISTRY_TAG = "stable"


@dataclass(frozen=True)
class RegistrySelection:
    model_id: str
    tag: str


def _registry_selection_from_artifact_ref(artifact_ref: str) -> RegistrySelection:
    parsed = urlparse(artifact_ref)

    if parsed.scheme != "registry" or not parsed.netloc or parsed.path not in ("", "/"):
        raise ValueError(
            "Inference model artifact_ref must be registry://<registry_model_id>?tag=<tag>"
        )

    tag = parse_qs(parsed.query).get("tag", ["stable"])[0] or "stable"
    return RegistrySelection(model_id=parsed.netloc, tag=tag)


def _job_registry_selection(
    session,
    job: Job,
    *,
    task: InferenceTask,
    fallback_model_id: str,
    fallback_tag: str,
) -> RegistrySelection:
    if job.model_id is None:
        return RegistrySelection(model_id=fallback_model_id, tag=fallback_tag)

    model = session.get(InferenceModel, job.model_id)
    if model is None:
        raise ValueError("Selected inference model not found")
    if model.task != task:
        raise ValueError(f"Selected inference model does not support {task.value}")

    return _registry_selection_from_artifact_ref(model.artifact_ref)


def _build_segment_request(job: Job) -> JobSubmitRequest:
    with SyncSessionLocal() as session:
        if job.document_part_id is None:
            raise ValueError("Segment job is missing its target document part")
        part = session.get(DocumentPart, job.document_part_id)
        if part is None:
            raise ValueError("Document part not found")
        selection = _job_registry_selection(
            session,
            job,
            task=InferenceTask.segment,
            fallback_model_id=_DEFAULT_SEGMENT_REGISTRY_MODEL,
            fallback_tag=_DEFAULT_SEGMENT_REGISTRY_TAG,
        )
        image_path = MediaStore().absolute_path(part.image_key)
        params = dict((job.payload or {}).get("ml_params") or {})
        return JobSubmitRequest(
            task=WireInferenceTask.segment,
            registry_model_id=selection.model_id,
            registry_tag=selection.tag,
            product_job_id=job.id,
            image_bytes=image_path.read_bytes(),
            params=params,
        )


def _build_transcribe_request(job: Job) -> JobSubmitRequest:
    if job.document_id is None or job.document_part_id is None:
        raise TranscribeJobHandlerError("Transcribe job is missing its target document part")

    with SyncSessionLocal() as session:
        part = session.get(DocumentPart, job.document_part_id)
        if part is None or part.document_id != job.document_id:
            raise TranscribeJobHandlerError("Document part not found")
        selection = _job_registry_selection(
            session,
            job,
            task=InferenceTask.transcribe,
            fallback_model_id=_DEFAULT_TRANSCRIBE_REGISTRY_MODEL,
            fallback_tag=_DEFAULT_TRANSCRIBE_REGISTRY_TAG,
        )
        lines = TranscribeMergeService.load_lines(session, part.id)
        payload = job.payload or {}
        selected_line_ids = payload.get("line_ids")
        if selected_line_ids:
            allowed = {uuid.UUID(str(line_id)) for line_id in selected_line_ids}
            lines = [line for line in lines if line.id in allowed]
            if not lines:
                raise TranscribeJobHandlerError("No matching lines to transcribe")

        image_bytes = MediaStore().absolute_path(part.image_key).read_bytes()
        base_params = dict((job.payload or {}).get("ml_params") or {})
        line_regions = [
            {
                "line_id": str(line.id),
                "line_index": index,
                "points": line.points,
            }
            for index, line in enumerate(lines)
        ]
        return JobSubmitRequest(
            task=WireInferenceTask.transcribe,
            registry_model_id=selection.model_id,
            registry_tag=selection.tag,
            product_job_id=job.id,
            image_bytes=image_bytes,
            params={**base_params, "lines": line_regions},
        )


def build_inference_submit_request(job: Job) -> JobSubmitRequest:
    if job.type == JobType.segment:
        return _build_segment_request(job)
    if job.type == JobType.transcribe:
        return _build_transcribe_request(job)
    raise ValueError(f"No inference submission for job type {job.type.value}")
