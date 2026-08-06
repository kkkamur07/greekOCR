"""One question per endpoint: hand me the document (or part) this caller may touch.

Before this module every read and write in the document context opened with the same
three or four steps — load the project, decide membership or fall back to the published
exception, load the document and confirm it really belongs to that project, then load the
part and confirm it belongs to the document. Thirty-odd application methods repeated some
prefix of that, so the rule was written thirty-odd times, a new endpoint could silently
omit a step, and none of it could be exercised without going through HTTP.

``user is None`` selects the anonymous audience, and that is not a convenience overload:
the public routers carry no authentication dependency at all, so ``None`` is the only
caller identity they can produce, and :func:`require_can_read` has only ever been reached
with that value. Members read any workflow; anonymous callers read ``published`` and
nothing else.

The status codes are behaviour, not detail, and are preserved exactly:

* missing project, missing document, document filed under another project, missing part,
  part filed under another document → ``NotFoundError`` (404);
* authenticated non-member → ``AccessDeniedError`` (403). The authenticated surface
  already admits that the project exists, so masking it here would buy nothing;
* anonymous caller against a *draft* → ``NotFoundError`` (404), never 403, so the public
  surface never confirms that an unpublished document exists.

Two loaders, deliberately. The project-scoped entry points use ``get_by_id``/``get_part``,
which eager-load parts, lines and transcriptions because their callers go straight on to
use them. :meth:`require_part_by_id` — the media hot path, one request per page image —
uses the row-only ``get_by_id_for_authz``/``get_part_row``, because serving bytes needs
``image_key`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AccessDeniedError, NotFoundError
from backend.document.domain.access import require_can_read
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Document, DocumentPart
from backend.project.domain.access import is_member
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


@dataclass(frozen=True)
class DocumentContext:
    """A document the caller is allowed to see, with the project it was authorized against.

    The project is returned rather than dropped because the callers that need a *second*
    decision — publishing requires ownership, not mere membership — would otherwise have
    to load it again.
    """

    project: Project
    document: Document


@dataclass(frozen=True)
class PartContext:
    """A part the caller is allowed to see, plus the chain it was authorized through."""

    project: Project
    document: Document
    part: DocumentPart


class DocumentAccess:
    """Fetch-and-authorize for documents and their parts.

    Constructed with repositories rather than a session so it can be exercised against
    fakes: every method takes the session it should read through.
    """

    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()

    # --- Composite entry points: what an endpoint should ask for ---

    async def require_project(self, session: AsyncSession, user: User, project_id: UUID) -> Project:
        """The project, if ``user`` is a member of it.

        Typed ``User`` rather than ``User | None``: there is no anonymous read that stops
        at a project. Published *documents* are public; the project they live in is not.
        """
        project = await self._load_project(session, project_id)
        if not is_member(project, user.id):
            raise AccessDeniedError("You do not have access to this project")
        return project

    async def require_document(
        self,
        session: AsyncSession,
        user: User | None,
        project_id: UUID,
        document_id: UUID,
    ) -> DocumentContext:
        """The document at ``project_id/document_id``, if this caller may read it."""
        if user is None:
            project = await self._load_project(session, project_id)
            document = await self.document_in_project(session, project, document_id)
            # Ordered after the containment check on purpose: a document filed under a
            # different project must read as absent, not as forbidden.
            require_can_read(document, project, None)
            return DocumentContext(project=project, document=document)
        project = await self.require_project(session, user, project_id)
        document = await self.document_in_project(session, project, document_id)
        return DocumentContext(project=project, document=document)

    async def require_part(
        self,
        session: AsyncSession,
        user: User | None,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> PartContext:
        """The part at ``project_id/document_id/part_id``, if this caller may read it."""
        context = await self.require_document(session, user, project_id, document_id)
        part = await self.part_in_document(session, context.document, part_id)
        return PartContext(project=context.project, document=context.document, part=part)

    async def require_part_by_id(
        self, session: AsyncSession, user: User | None, part_id: UUID
    ) -> PartContext:
        """The part named by ``part_id`` alone, if this caller may read it.

        The media routes address a part directly, so the project and document have to be
        derived from it rather than trusted from the path. Note the asymmetry this
        creates, and it is deliberate: an authenticated caller must be a *member*, so a
        published document's image is reachable at ``/public/media/parts/{id}`` and not at
        ``/media/parts/{id}`` for a non-member. The private route is a member route; it
        does not double as an authenticated view of the public one.
        """
        part = await self._documents.get_part_row(session, part_id)
        if part is None:
            raise NotFoundError("Part not found")
        document = await self._documents.get_by_id_for_authz(session, part.document_id)
        if document is None:
            raise NotFoundError("Document not found")
        if user is None:
            project = await self._load_project(session, document.project_id)
            require_can_read(document, project, None)
        else:
            project = await self.require_project(session, user, document.project_id)
        return PartContext(project=project, document=document, part=part)

    # --- Individual steps, for callers that already hold part of the chain ---

    async def document_in_project(
        self, session: AsyncSession, project: Project, document_id: UUID
    ) -> Document:
        """A document that must belong to ``project``.

        A document under a different project is reported as missing rather than
        forbidden: the caller supplied the pairing, and a 403 would confirm that the
        document exists somewhere else.
        """
        document = await self._documents.get_by_id(session, document_id)
        if document is None or document.project_id != project.id:
            raise NotFoundError("Document not found")
        return document

    async def part_in_document(
        self, session: AsyncSession, document: Document, part_id: UUID
    ) -> DocumentPart:
        """A part that must belong to ``document`` — same containment rule as above."""
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")
        return part

    async def _load_project(self, session: AsyncSession, project_id: UUID) -> Project:
        project = await self._projects.get_by_id(session, project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return project
