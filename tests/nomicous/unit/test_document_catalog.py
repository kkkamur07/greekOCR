"""``DocumentCatalog`` — a document's life from creation to deletion, without HTTP.

The catalog is the half of the old ``DocumentService`` that owns the ``documents`` and
``transcriptions`` rows. Its interface is small but it carries three things that are not
obvious from the signatures and that only break loudly if something pins them:

* *every* entry point authorizes before it touches a row, and the member entry points
  authorize before they touch the repository at all;
* the ``_public`` reads are the same lifecycle reads with ``user=None``, so the
  published-only rule applies to them without being restated per method;
* the writes are single repository calls, so the row change and its consequences
  (media-deletion intents) land in one transaction.

The seam underneath is tested in ``test_document_access_seam``; what is covered here is
the catalog's use of it, so the real :class:`DocumentAccess` is constructed over fake
repositories rather than stubbed out — "a non-member is refused" should be an observable
outcome, not an assertion that a method with a particular name was called.

Not covered here, deliberately: the publish-requires-ownership rule, which already has
unit coverage in ``test_auth_hardening``; the route-level pagination that wraps
``list_documents`` and ``list_document_layout_public``, covered in
``test_document_input_limits``; and the SQL the repository emits, covered against Postgres
in ``tests/nomicous/integration``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from backend.core.api.pagination import PageCursor
from backend.core.exceptions import AccessDeniedError, NotFoundError, ValidationError
from backend.document.application.document_access import DocumentAccess
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    Transcription,
    TranscriptionKind,
)
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


class _Session:
    """The catalog never reads through the session; every load goes via a repository."""


class _ProjectRepository:
    def __init__(self, project: Project | None) -> None:
        self._project = project

    async def get_by_id(self, _session, project_id):
        if self._project is None or self._project.id != project_id:
            return None
        return self._project


class _DocumentRepository:
    """Records what the catalog asked for, and answers from in-memory rows.

    ``calls`` exists to prove *ordering* (nothing is loaded before the caller is
    authorized), not to count queries: the assertions below only ever ask whether a
    particular call happened, never how many times.
    """

    def __init__(
        self,
        document: Document | None = None,
        part: DocumentPart | None = None,
        *,
        listed: list[Document] | None = None,
        transcriptions: list[Transcription] | None = None,
        blocks: list[Block] | None = None,
        lines: list[Line] | None = None,
    ) -> None:
        self._document = document
        self._part = part
        self._listed = listed or []
        self._transcriptions = transcriptions or []
        self._blocks = blocks or []
        self._lines = lines or []
        self.calls: list[str] = []
        self.list_kwargs: dict = {}
        self.created: list[dict] = []
        self.updates: list[dict] = []
        self.deleted: list[tuple[Document, list[str]]] = []
        self.transcription_lookups: list[uuid.UUID] = []
        self.block_limits: list[int] = []
        self.line_reads: list[dict] = []

    async def get_by_id(self, _session, document_id):
        self.calls.append("get_by_id")
        if self._document is None or self._document.id != document_id:
            return None
        return self._document

    async def get_part(self, _session, part_id):
        self.calls.append("get_part")
        if self._part is None or self._part.id != part_id:
            return None
        return self._part

    async def list_for_project(self, _session, project_id, **kwargs):
        self.calls.append("list_for_project")
        self.list_kwargs = {"project_id": project_id, **kwargs}
        return list(self._listed)

    async def create(self, _session, *, project_id, name):
        self.calls.append("create")
        self.created.append({"project_id": project_id, "name": name})
        return Document(id=uuid.uuid4(), project_id=project_id, name=name)

    async def update(self, _session, document, **fields):
        self.calls.append("update")
        self.updates.append(dict(fields))
        for key, value in fields.items():
            setattr(document, key, value)
        return document

    async def delete_with_media_intents(self, _session, document, image_keys):
        self.calls.append("delete_with_media_intents")
        self.deleted.append((document, list(image_keys)))

    async def list_transcriptions(self, _session, document_id):
        self.calls.append("list_transcriptions")
        self.transcription_lookups.append(document_id)
        return list(self._transcriptions)

    async def list_blocks_for_document(self, _session, document_id, *, limit):
        self.calls.append("list_blocks_for_document")
        self.block_limits.append(limit)
        return list(self._blocks)[:limit]

    async def list_lines_for_document(self, _session, document_id, *, limit, cursor=None):
        self.calls.append("list_lines_for_document")
        self.line_reads.append({"limit": limit, "cursor": cursor})
        rows = self._lines
        if cursor is not None:
            rows = [
                line for line in rows if (line.created_at, line.id) > (cursor.created_at, cursor.id)
            ]
        return list(rows)[:limit]


def _user(user_id=None) -> User:
    return User(
        id=user_id or uuid.uuid4(),
        email="reader@example.org",
        username="reader",
        hashed_password="x",
    )


def _fixture(
    *,
    workflow: DocumentWorkflow = DocumentWorkflow.draft,
    owner_id: uuid.UUID | None = None,
    parts: int = 0,
    **repo_kwargs,
):
    """A project the ``owner_id`` owns, with one document and ``parts`` page images."""
    owner_id = owner_id or uuid.uuid4()
    project = Project(id=uuid.uuid4(), name="Codices", owner_id=owner_id)
    project.shared_users = []
    document = Document(id=uuid.uuid4(), project_id=project.id, name="MS 1", workflow=workflow)
    document.parts = [
        DocumentPart(
            id=uuid.uuid4(), document_id=document.id, order=index, image_key=f"page-{index}.webp"
        )
        for index in range(parts)
    ]
    first_part = document.parts[0] if document.parts else None
    documents = _DocumentRepository(document, first_part, **repo_kwargs)
    projects = _ProjectRepository(project)
    catalog = DocumentCatalog(
        documents=documents,
        projects=projects,
        access=DocumentAccess(documents=documents, projects=projects),
    )
    return catalog, project, document, documents


def _line(created_at: datetime) -> Line:
    line = Line(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0, baseline={})
    line.created_at = created_at
    return line


# --- Listing and creating: the project gate comes first ---
# Tests that membership is decided before any document row is read or written, and that
# the caller's paging arguments survive the trip. Does not test the SQL that applies them.


async def test_member_lists_documents_and_paging_arguments_are_passed_through() -> None:
    owner_id = uuid.uuid4()
    listed = [Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="A")]
    catalog, project, _document, repo = _fixture(owner_id=owner_id, listed=listed)
    cursor = PageCursor(created_at=datetime(2026, 1, 1, tzinfo=UTC), id=uuid.uuid4())

    result = await catalog.list_documents(
        _Session(),
        _user(owner_id),
        project.id,
        include_archived=True,
        limit=7,
        cursor=cursor,
    )

    assert result == listed
    # The router adds the +1 probe row and decodes the cursor; the catalog must not
    # reinterpret either, or keyset paging silently changes shape.
    assert repo.list_kwargs == {
        "project_id": project.id,
        "include_archived": True,
        "limit": 7,
        "cursor": cursor,
    }


async def test_listing_a_project_you_are_not_in_is_forbidden_and_reads_nothing() -> None:
    catalog, project, _document, repo = _fixture()

    with pytest.raises(AccessDeniedError):
        await catalog.list_documents(_Session(), _user(), project.id)
    assert repo.calls == []


async def test_listing_a_missing_project_is_not_found() -> None:
    catalog, _project, _document, _repo = _fixture()

    with pytest.raises(NotFoundError, match="Project not found"):
        await catalog.list_documents(_Session(), _user(), uuid.uuid4())


async def test_create_writes_into_the_project_the_caller_belongs_to() -> None:
    owner_id = uuid.uuid4()
    catalog, project, _document, repo = _fixture(owner_id=owner_id)

    created = await catalog.create_document(_Session(), _user(owner_id), project.id, name="Codex B")

    assert created.name == "Codex B"
    assert created.project_id == project.id
    assert repo.created == [{"project_id": project.id, "name": "Codex B"}]


async def test_non_member_cannot_create_a_document_and_nothing_is_written() -> None:
    catalog, project, _document, repo = _fixture()

    with pytest.raises(AccessDeniedError):
        await catalog.create_document(_Session(), _user(), project.id, name="Smuggled")
    assert repo.created == []


# --- Reads: the same lifecycle read, two audiences ---
# Tests that the ``_public`` variants really do fall through to the anonymous rule rather
# than trusting their caller. Does not re-test the 404/403 split itself, which belongs to
# the access seam.


async def test_member_read_returns_the_document() -> None:
    owner_id = uuid.uuid4()
    catalog, project, document, _repo = _fixture(owner_id=owner_id)

    assert (
        await catalog.get_document(_Session(), _user(owner_id), project.id, document.id) is document
    )


async def test_public_read_serves_published_and_hides_draft() -> None:
    catalog, project, document, _repo = _fixture(workflow=DocumentWorkflow.published)
    assert await catalog.get_document_public(_Session(), project.id, document.id) is document

    catalog, project, document, _repo = _fixture(workflow=DocumentWorkflow.draft)
    with pytest.raises(NotFoundError):
        await catalog.get_document_public(_Session(), project.id, document.id)


async def test_published_part_is_reachable_and_a_draft_part_is_not() -> None:
    """``get_published_part`` takes no user at all: it is the anonymous path by construction.

    The annotation context calls it for PDF and PAGE-XML downloads, so if it ever stopped
    applying the workflow rule an unpublished manuscript would leak as a rendered artifact.
    """
    catalog, project, document, _repo = _fixture(workflow=DocumentWorkflow.published, parts=1)
    part = document.parts[0]

    assert await catalog.get_published_part(_Session(), project.id, document.id, part.id) is part

    catalog, project, document, _repo = _fixture(workflow=DocumentWorkflow.draft, parts=1)
    with pytest.raises(NotFoundError):
        await catalog.get_published_part(_Session(), project.id, document.id, document.parts[0].id)


async def test_transcription_layers_are_listed_for_the_authorized_document() -> None:
    """The id handed to the repository is the *loaded* document's, not the caller's path value.

    They agree here, and they must: taking the path value on trust would make the
    containment check in the access seam decorative for this read.
    """
    owner_id = uuid.uuid4()
    layer = Transcription(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        name="Ground truth",
        kind=TranscriptionKind.ground_truth,
    )
    catalog, project, document, repo = _fixture(owner_id=owner_id, transcriptions=[layer])

    result = await catalog.list_transcriptions(_Session(), _user(owner_id), project.id, document.id)

    assert result == [layer]
    assert repo.transcription_lookups == [document.id]


async def test_public_transcription_listing_refuses_a_draft() -> None:
    """The layer names and kinds are still document content; the workflow rule owns them."""
    catalog, project, document, repo = _fixture(workflow=DocumentWorkflow.draft)

    with pytest.raises(NotFoundError):
        await catalog.list_transcriptions_public(_Session(), project.id, document.id)
    assert "list_transcriptions" not in repo.calls


# --- Updating: what a PATCH may say, and in what order it is judged ---
# Tests field admission and workflow typing. The publish-requires-ownership rule is
# already pinned in test_auth_hardening and is not repeated here.


async def test_unknown_field_is_refused_loudly_and_nothing_is_written() -> None:
    owner_id = uuid.uuid4()
    catalog, project, document, repo = _fixture(owner_id=owner_id)

    with pytest.raises(ValidationError, match="workflowe"):
        await catalog.update_document(
            _Session(), _user(owner_id), project.id, document.id, workflowe="published"
        )
    # The repository applies fields with ``setattr``, so a key that slipped through would
    # be written verbatim onto the ORM row.
    assert repo.updates == []


async def test_unknown_fields_are_reported_together_and_sorted() -> None:
    owner_id = uuid.uuid4()
    catalog, project, document, _repo = _fixture(owner_id=owner_id)

    with pytest.raises(ValidationError) as raised:
        await catalog.update_document(
            _Session(), _user(owner_id), project.id, document.id, zeta=1, alpha=2
        )

    assert "alpha, zeta" in str(raised.value)


async def test_the_field_whitelist_is_applied_before_anything_is_loaded() -> None:
    """A malformed PATCH is a 422 regardless of who sent it, so it need not cost a load."""
    catalog, project, document, repo = _fixture()

    with pytest.raises(ValidationError):
        await catalog.update_document(
            _Session(), _user(), project.id, document.id, owner_id=uuid.uuid4()
        )
    assert repo.calls == []


async def test_a_string_workflow_cannot_smuggle_a_publish_past_the_owner_check() -> None:
    """The owner gate keys off ``DocumentWorkflow.published``, so the value must be that enum.

    ``DocumentWorkflow`` subclasses ``str``, which makes ``"published" == workflow`` true
    while ``isinstance("published", DocumentWorkflow)`` is false. Accepting the bare string
    would therefore write a published row without ever asking who the owner is.
    """
    catalog, project, document, repo = _fixture(owner_id=uuid.uuid4())
    collaborator = _user()
    project.shared_users = [collaborator]

    with pytest.raises(ValidationError, match="Invalid workflow"):
        await catalog.update_document(
            _Session(), collaborator, project.id, document.id, workflow="published"
        )
    assert repo.updates == []
    assert document.workflow is DocumentWorkflow.draft


async def test_a_member_renames_a_document_and_the_field_reaches_the_row() -> None:
    owner_id = uuid.uuid4()
    catalog, project, document, repo = _fixture(owner_id=owner_id)

    updated = await catalog.update_document(
        _Session(), _user(owner_id), project.id, document.id, name="Renamed"
    )

    assert updated is document
    assert document.name == "Renamed"
    assert repo.updates == [{"name": "Renamed"}]


async def test_a_non_member_cannot_update_even_a_well_formed_patch() -> None:
    catalog, project, document, repo = _fixture()

    with pytest.raises(AccessDeniedError):
        await catalog.update_document(_Session(), _user(), project.id, document.id, name="Hijacked")
    assert repo.updates == []


# --- Deletion: the row and the bytes it points at ---
# Tests that every page image is queued for removal in the same call that drops the
# document. Does not test the media reaper that consumes those intents.


async def test_delete_queues_an_intent_for_every_page_image() -> None:
    owner_id = uuid.uuid4()
    catalog, project, document, repo = _fixture(owner_id=owner_id, parts=3)

    await catalog.delete_document(_Session(), _user(owner_id), project.id, document.id)

    deleted_document, image_keys = repo.deleted[0]
    assert deleted_document is document
    # One call, so the row deletion and the intents commit together: an intent without a
    # deletion would erase a live page, a deletion without an intent would orphan bytes.
    assert image_keys == ["page-0.webp", "page-1.webp", "page-2.webp"]


async def test_deleting_a_document_with_no_parts_still_deletes_it() -> None:
    owner_id = uuid.uuid4()
    catalog, project, document, repo = _fixture(owner_id=owner_id, parts=0)

    await catalog.delete_document(_Session(), _user(owner_id), project.id, document.id)

    assert repo.deleted == [(document, [])]


async def test_a_non_member_cannot_delete_and_no_intent_is_queued() -> None:
    catalog, project, document, repo = _fixture(parts=2)

    with pytest.raises(AccessDeniedError):
        await catalog.delete_document(_Session(), _user(), project.id, document.id)
    assert repo.deleted == []


# --- Anonymous layout: bounded on both axes ---
# Tests that one unauthenticated request cannot fan out to a whole manuscript's geometry.
# The cursor encoding and the +1 probe row live in the router, tested next door in
# test_document_input_limits.


async def test_first_page_carries_blocks_capped_at_the_same_bound_as_lines() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    blocks = [Block(id=uuid.uuid4(), part_id=uuid.uuid4(), order=index) for index in range(5)]
    lines = [_line(base) for _ in range(5)]
    catalog, project, document, repo = _fixture(
        workflow=DocumentWorkflow.published, blocks=blocks, lines=lines
    )

    returned_blocks, returned_lines = await catalog.list_document_layout_public(
        _Session(), project.id, document.id, limit=2
    )

    assert len(returned_blocks) == 2
    assert len(returned_lines) == 2
    # Blocks are not paginated, so the only thing standing between an anonymous caller and
    # every block in the manuscript is that they share the line bound.
    assert repo.block_limits == [2]


async def test_a_resumed_page_carries_lines_only() -> None:
    """Blocks accompany the first page; repeating them on every page would unbound them."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    blocks = [Block(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0)]
    lines = [_line(base) for _ in range(3)]
    catalog, project, document, repo = _fixture(
        workflow=DocumentWorkflow.published, blocks=blocks, lines=lines
    )
    cursor = PageCursor(created_at=base, id=uuid.UUID(int=0))

    returned_blocks, _returned_lines = await catalog.list_document_layout_public(
        _Session(), project.id, document.id, limit=10, cursor=cursor
    )

    assert returned_blocks == []
    assert "list_blocks_for_document" not in repo.calls
    assert repo.line_reads == [{"limit": 10, "cursor": cursor}]


async def test_layout_of_a_draft_is_not_found_and_no_geometry_is_read() -> None:
    catalog, project, document, repo = _fixture(
        workflow=DocumentWorkflow.draft,
        blocks=[Block(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0)],
        lines=[_line(datetime(2026, 1, 1, tzinfo=UTC))],
    )

    with pytest.raises(NotFoundError):
        await catalog.list_document_layout_public(_Session(), project.id, document.id, limit=10)
    assert repo.line_reads == []
