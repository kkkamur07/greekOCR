"""Document access policy - members vs public published read.

A published document used to be readable by anyone who had the two path UUIDs: there
was no secret in the URL, so a link could never be revoked short of unpublishing, which
broke every other link that had been shared too. ``public_share_token`` is that secret.
It is compared with :func:`secrets.compare_digest` rather than ``==`` so a wrong guess
takes the same time to reject regardless of how many leading characters it gets right -
an equality check would leak that timing to a patient attacker.
"""

import secrets

from backend.core.exceptions import NotFoundError
from backend.document.infrastructure.orm_models import Document, DocumentWorkflow
from backend.project.domain.access import is_member
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


def can_read_document(
    document: Document,
    project: Project,
    user: User | None,
    token: str | None = None,
) -> bool:
    """Members read any workflow. Everyone else needs a published document *and* the
    token that matches its current share secret - a missing, wrong, or stale (rotated
    away) token reads exactly like the document was never published at all.
    """
    if user is not None and is_member(project, user.id):
        return True
    if document.workflow != DocumentWorkflow.published:
        return False
    if document.public_share_token is None or token is None:
        return False
    return secrets.compare_digest(document.public_share_token, token)


def require_can_read(
    document: Document,
    project: Project,
    user: User | None,
    token: str | None = None,
) -> None:
    """Raise if this caller may not read ``document``.

    Always ``NotFoundError``, never ``AccessDeniedError``: the only caller that ever
    reaches this predicate is the anonymous one (see ``DocumentAccess`` for why), and a
    403 here would admit that a document exists at this address even though the caller
    holds no token for it - exactly the confirmation the public surface must not give.
    """
    if can_read_document(document, project, user, token):
        return
    raise NotFoundError("Document not found")
