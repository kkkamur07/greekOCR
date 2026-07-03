"""Import all ORM models so Alembic metadata is complete.

Import order follows foreign-key / relationship dependencies so string-based
relationship targets resolve when mappers configure.
"""

from backend.users.infrastructure.orm_models import AuthRateLimitAttempt, User  # noqa: F401
from backend.jobs.infrastructure.orm_models import Job  # noqa: F401
from backend.project.infrastructure.orm_models import Project, project_shared_users  # noqa: F401
from backend.ml.infrastructure.orm_models import (  # noqa: F401
    InferenceModel,
    ModelBinding,
)
from backend.document.infrastructure.orm_models import (  # noqa: F401
    Block,
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
)
from backend.annotation.infrastructure.orm_models import AnnotationHistorySnapshot  # noqa: F401
