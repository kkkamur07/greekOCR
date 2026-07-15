"""Document access policy - members vs public published read."""

from backend.core.exceptions import AccessDeniedError, NotFoundError
from backend.document.infrastructure.orm_models import Document, DocumentWorkflow
from backend.project.domain.access import is_member
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


def can_read_document(
    document: Document,
    project: Project,
    user: User | None,
) -> bool:
    """Members read any workflow; others only published."""
    if user is not None and is_member(project, user.id):
        return True
    return document.workflow == DocumentWorkflow.published


def require_can_read(
    document: Document,
    project: Project,
    user: User | None,
) -> None:
    if can_read_document(document, project, user):
        return
    if document.workflow != DocumentWorkflow.published:
        raise NotFoundError("Document not found")
    raise AccessDeniedError("You do not have access to this document")
