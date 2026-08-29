"""Which pages a whole-document export covers, and what the download is called.

Three export shapes hang off one document - PAGE XML with the page images, a
transcription PDF, and plain text - and all three have to answer the same two questions
before they render anything: which parts are in, and what should the saved file be
named. Answering them once, here, is what keeps ``reviewed_only`` from drifting between
the three endpoints, and it is the single place that decides what an export with nothing
in it means.

That decision is a 404, and it is the same answer for a document with no parts at all as
for ``reviewed_only=true`` matching none of them. The alternative, an empty archive, is
worse in the only place it would ever be seen: a browser download that produces a
zero-entry zip or a zero-page PDF looks like a success and reads as corruption, and the
caller cannot tell "nothing is reviewed yet" from "the export broke". A 404 says which.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.document.application.document_access import DocumentAccess
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Document, DocumentPart
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User

#: Width of the zero-padded page number an archive entry is named by. Four digits sorts
#: correctly in every file manager for any manuscript anybody is going to upload here.
PAGE_NUMBER_DIGITS = 4


@dataclass(frozen=True)
class DocumentExportSelection:
    """The document, the parts that are in the export, and where each one sits.

    ``page_numbers`` covers *every* part of the document, not only the selected ones, so
    that a ``reviewed_only`` export of pages 2 and 5 names them 2 and 5 rather than
    renumbering them 1 and 2. A page number that changes meaning depending on a query
    parameter is a page number nobody can cite.
    """

    document: Document
    parts: list[DocumentPart]
    page_numbers: dict[UUID, int]

    def page_number(self, part: DocumentPart) -> int:
        return self.page_numbers[part.id]


class DocumentExportSelector:
    """Authorize a document export and resolve the pages it covers."""

    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)

    async def select(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        reviewed_only: bool = False,
    ) -> DocumentExportSelection:
        """The pages of ``document_id`` this caller may export, in reading order.

        Membership is the same gate the per-page exports use, so a document-level export
        can never be a way around it. ``published`` plays no part: these are member
        routes, and the per-page holdback flag governs the anonymous surface only.
        """
        context = await self._access.require_document(session, user, project_id, document_id)
        # Position, not the stored ``order`` value: orders are allowed to have gaps
        # (``next_part_order`` hands out max+1, so deleting a middle page leaves a hole),
        # and the reader numbers pages by where they sit in the list. Same rule as
        # ``DocumentRepository.part_page_number``, computed once for the whole document
        # rather than once per part, because an export reads every part anyway.
        ordered = sorted(context.document.parts, key=lambda part: part.order)
        page_numbers = {part.id: number for number, part in enumerate(ordered, start=1)}
        parts = [part for part in ordered if part.reviewed] if reviewed_only else ordered
        if not parts:
            raise NotFoundError("No pages to export")
        return DocumentExportSelection(
            document=context.document, parts=parts, page_numbers=page_numbers
        )


def document_download_name(document_name: str, suffix: str) -> str:
    """``<document-name-slug>-<suffix>``: the name a whole-document download saves as.

    Hyphenated and lowercased rather than the underscored ``export_file_stem`` a single
    page carries, and deliberately so: that stem is shared by the XML and the image
    inside one page's zip, where it exists to make two files read as one unit. A
    document download has no sibling to match, so it takes the plain slug shape the
    clients ask for instead of borrowing a convention that means something else.

    Characters no common filesystem accepts are dropped and runs of whitespace or
    underscores collapse to one hyphen. Non-ASCII letters stay: a Greek title is a Greek
    title, and ``attachment_disposition`` carries it through the header intact.
    """
    return f"{_document_slug(document_name)}-{suffix}"


# Path separators, the characters Windows refuses, and the Unicode Cc (control) block:
# the same set ``export_file_stem`` strips, for the same reason.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f-\x9f]+')
_SLUG_SEPARATORS = re.compile(r"[\s_]+")
_SLUG_HYPHEN_RUNS = re.compile(r"-{2,}")
# Bounded so a long title cannot push the download name past a path limit once the
# suffix and the client's download directory are added to it.
_MAX_SLUG_CHARS = 80


def _document_slug(document_name: str) -> str:
    safe = _UNSAFE_FILENAME_CHARS.sub("", document_name).strip()
    safe = _SLUG_HYPHEN_RUNS.sub("-", _SLUG_SEPARATORS.sub("-", safe))
    return safe[:_MAX_SLUG_CHARS].strip("-.").lower() or "document"
