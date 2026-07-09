"""Document and part use cases with project membership checks."""

from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import MediaStore, get_media_store
from backend.ml.application.model_service import InferenceModelService
from backend.project.infrastructure.project_repository import ProjectRepository

from backend.document.application.document_crud import DocumentCrudMixin
from backend.document.application.document_job_enqueue import DocumentJobEnqueueMixin
from backend.document.application.document_service_shared import DocumentServiceSharedMixin
from backend.document.application.layout_service import LayoutServiceMixin
from backend.document.application.pairing_service import PairingServiceMixin
from backend.document.application.part_service import PartServiceMixin
from backend.document.application.transcription_service import TranscriptionServiceMixin


class DocumentService(
    DocumentCrudMixin,
    PartServiceMixin,
    LayoutServiceMixin,
    PairingServiceMixin,
    DocumentJobEnqueueMixin,
    TranscriptionServiceMixin,
    DocumentServiceSharedMixin,
):
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        media: MediaStore | None = None,
        inference_models: InferenceModelService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._media = media or get_media_store()
        self._inference_models = inference_models or InferenceModelService()
