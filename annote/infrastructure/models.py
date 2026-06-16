"""Import all ORM models so Alembic metadata is complete."""

from backend.annotation.infrastructure.orm_models import AnnotationHistorySnapshot  # noqa: F401
from backend.document.infrastructure.orm_models import (  # noqa: F401
    Block,
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
)
from backend.inference.infrastructure.orm_models import (  # noqa: F401
    InferenceModel,
    Job,
    ModelBinding,
)
from backend.project.infrastructure.orm_models import Project, project_shared_users  # noqa: F401
from backend.users.infrastructure.orm_models import User  # noqa: F401
