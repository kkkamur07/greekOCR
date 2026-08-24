"""The documents in a project: their existence, their names, and their workflow state.

One responsibility: a document's life from creation to deletion, plus the reads that
expose it. Page images, geometry and transcription text are three other modules; this one
only ever touches the ``documents`` and ``transcriptions`` rows.

The public (unauthenticated) reads live here rather than in a separate "public service"
because they are the *same* lifecycle reads with a different audience, and the audience is
already a parameter of :class:`DocumentAccess`. Splitting them apart would put the
published-workflow rule in two places again.
"""

from __future__ import annotations

from typing import NamedTuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import PageCursor
from backend.core.exceptions import AccessDeniedError, ValidationError
from backend.document.application.document_access import DocumentAccess, PartContext
from backend.document.application.patch_fields import (
    DOCUMENT_UPDATE_FIELDS,
    reject_unknown_fields,
)
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    Transcription,
)
from backend.project.domain.access import is_owner
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


class PublicLayoutPage(NamedTuple):
    """One page of anonymous layout, and whether the blocks on it are all of them.

    ``lines`` carries the probe row the caller needs to build a continuation cursor.
    Blocks have no cursor, so their overflow is reported as a flag instead - a client
    that cannot see it has no way to tell a truncated page from a complete one.
    """

    blocks: list[Block]
    blocks_truncated: bool
    lines: list[Line]


class DocumentCatalog:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)

    async def list_documents(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        include_archived: bool = False,
        limit: int = 50,
        cursor=None,
    ) -> list[Document]:
        await self._access.require_project(session, user, project_id)
        return await self._documents.list_for_project(
            session,
            project_id,
            include_archived=include_archived,
            limit=limit,
            cursor=cursor,
        )

    async def create_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        name: str,
    ) -> Document:
        await self._access.require_project(session, user, project_id)
        return await self._documents.create(session, project_id=project_id, name=name)

    async def get_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> Document:
        context = await self._access.require_document(session, user, project_id, document_id)
        return context.document

    async def get_document_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
    ) -> Document:
        # ``None`` is the whole difference: the public router has no authentication
        # dependency, so there is no caller to check membership for, and the seam falls
        # through to the published-workflow rule.
        context = await self._access.require_document(session, None, project_id, document_id)
        return context.document

    async def get_published_part(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> DocumentPart:
        context = await self._access.require_part(session, None, project_id, document_id, part_id)
        return context.part

    async def get_published_part_context(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> PartContext:
        """The published part *and* its document, for callers that need both.

        Exports name their files after the document, and a second ``get_document_public``
        would repeat the access check this one already made.
        """
        return await self._access.require_part(session, None, project_id, document_id, part_id)

    async def update_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        **fields: object,
    ) -> Document:
        reject_unknown_fields(fields, DOCUMENT_UPDATE_FIELDS, "document update")
        context = await self._access.require_document(session, user, project_id, document_id)
        project, document = context.project, context.document
        if "workflow" in fields and fields["workflow"] is not None:
            workflow = fields["workflow"]
            if not isinstance(workflow, DocumentWorkflow):
                raise ValidationError("Invalid workflow value")
            # Publishing drops the whole document - page images and
            # transcriptions - to unauthenticated readers. That is a decision for
            # whoever owns the project, not for anyone they shared it with.
            if workflow is DocumentWorkflow.published and not is_owner(project, user.id):
                raise AccessDeniedError("Only the project owner can publish a document")
        return await self._documents.update(session, document, **fields)

    async def delete_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> None:
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        image_keys = [part.image_key for part in document.parts]
        await self._documents.delete_with_media_intents(session, document, image_keys)

    async def list_transcriptions(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[Transcription]:
        document = await self.get_document(session, user, project_id, document_id)
        return await self._documents.list_transcriptions(session, document.id)

    async def list_transcriptions_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
    ) -> list[Transcription]:
        document = await self.get_document_public(session, project_id, document_id)
        return await self._documents.list_transcriptions(session, document.id)

    async def list_document_layout_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        *,
        limit: int,
        cursor: PageCursor | None = None,
    ) -> PublicLayoutPage:
        """Bounded layout read for anonymous callers.

        ``limit`` is the page size. Both reads fetch one row beyond it, which is what
        lets the caller tell "this is everything" from "there is more": for lines the
        extra row becomes the continuation cursor, and for blocks - which have no cursor
        - it becomes ``blocks_truncated``.

        Blocks accompany the first page only, so repeating them per page cannot unbound
        the read. Reporting the truncation matters because a bare ``LIMIT`` returning
        exactly ``limit`` rows is indistinguishable from a document that has exactly
        that many blocks: a published page silently lost the rest of its regions with
        nothing in the response a client could notice.
        """
        document = await self.get_document_public(session, project_id, document_id)
        blocks: list[Block] = []
        blocks_truncated = False
        if cursor is None:
            probed = await self._documents.list_blocks_for_document(
                session, document.id, limit=limit + 1
            )
            blocks_truncated = len(probed) > limit
            blocks = probed[:limit]
        lines = await self._documents.list_lines_for_document(
            session, document.id, limit=limit + 1, cursor=cursor
        )
        return PublicLayoutPage(blocks=blocks, blocks_truncated=blocks_truncated, lines=lines)
