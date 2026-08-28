"""The ``DocumentAccess`` seam, exercised without HTTP.

``domain/access.py`` is tested next door in ``test_document_access``; that covers the
*predicate*. What is covered here is the seam that wraps it — which rows get loaded, in
what order, and which exception each failure produces. The 404-versus-403 split is the
point: 403 admits that a thing exists, so it is only ever used where the caller has
already been told the project exists.
"""

from __future__ import annotations

import uuid

import pytest

from backend.core.exceptions import AccessDeniedError, NotFoundError
from backend.document.application.document_access import DocumentAccess
from backend.document.infrastructure.orm_models import Document, DocumentPart, DocumentWorkflow
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


class _Session:
    """Nothing here reads through the session; the fake repositories answer directly."""


class _ProjectRepository:
    def __init__(self, project: Project | None) -> None:
        self._project = project

    async def get_by_id(self, _session, project_id):
        if self._project is None or self._project.id != project_id:
            return None
        return self._project


class _DocumentRepository:
    def __init__(
        self,
        document: Document | None = None,
        part: DocumentPart | None = None,
    ) -> None:
        self._document = document
        self._part = part
        self.calls: list[str] = []

    async def get_by_id(self, _session, document_id):
        self.calls.append("get_by_id")
        if self._document is None or self._document.id != document_id:
            return None
        return self._document

    async def get_by_id_for_authz(self, _session, document_id):
        self.calls.append("get_by_id_for_authz")
        if self._document is None or self._document.id != document_id:
            return None
        return self._document

    async def get_part(self, _session, part_id):
        self.calls.append("get_part")
        if self._part is None or self._part.id != part_id:
            return None
        return self._part

    async def get_part_row(self, _session, part_id):
        self.calls.append("get_part_row")
        if self._part is None or self._part.id != part_id:
            return None
        return self._part


#: The fixture document's share secret. Every anonymous-and-published test below passes
#: it explicitly, the same way a real caller would carry it on the query string - the
#: fixture setting it is not itself proof that the seam checks it.
TOKEN = "s3cr3t-share-token"


def _fixture(*, workflow=DocumentWorkflow.draft, owner_id=None, shared=(), part_published=True):
    owner_id = owner_id or uuid.uuid4()
    project = Project(id=uuid.uuid4(), name="Codices", owner_id=owner_id)
    project.shared_users = list(shared)
    document = Document(
        id=uuid.uuid4(),
        project_id=project.id,
        name="MS 1",
        workflow=workflow,
        public_share_token=TOKEN,
    )
    part = DocumentPart(
        id=uuid.uuid4(),
        document_id=document.id,
        order=0,
        image_key="k.webp",
        published=part_published,
    )
    documents = _DocumentRepository(document, part)
    access = DocumentAccess(documents=documents, projects=_ProjectRepository(project))
    return access, project, document, part, documents


def _user(user_id=None) -> User:
    return User(
        id=user_id or uuid.uuid4(),
        email="reader@example.org",
        username="reader",
        hashed_password="x",
    )


# --- Member reads: the project must exist and the caller must belong to it ---


@pytest.mark.asyncio
async def test_member_gets_document_and_the_project_it_was_authorized_against() -> None:
    owner_id = uuid.uuid4()
    access, project, document, _part, _repo = _fixture(owner_id=owner_id)

    context = await access.require_document(_Session(), _user(owner_id), project.id, document.id)

    assert context.document is document
    # The project rides along so ``update_document`` can ask the owner question without
    # a second load.
    assert context.project is project


@pytest.mark.asyncio
async def test_shared_collaborator_is_a_member() -> None:
    collaborator = _user()
    access, project, document, _part, _repo = _fixture(shared=[collaborator])

    context = await access.require_document(_Session(), collaborator, project.id, document.id)

    assert context.document is document


@pytest.mark.asyncio
async def test_non_member_is_forbidden_not_hidden() -> None:
    access, project, document, _part, _repo = _fixture()

    with pytest.raises(AccessDeniedError):
        await access.require_document(_Session(), _user(), project.id, document.id)


@pytest.mark.asyncio
async def test_missing_project_is_not_found_before_membership_is_considered() -> None:
    access, _project, document, _part, repo = _fixture()

    with pytest.raises(NotFoundError, match="Project not found"):
        await access.require_document(_Session(), _user(), uuid.uuid4(), document.id)
    # The document was never loaded: the chain stops at the first failure.
    assert repo.calls == []


@pytest.mark.asyncio
async def test_document_filed_under_another_project_reads_as_missing() -> None:
    owner_id = uuid.uuid4()
    access, project, document, _part, _repo = _fixture(owner_id=owner_id)
    document.project_id = uuid.uuid4()

    with pytest.raises(NotFoundError, match="Document not found"):
        await access.require_document(_Session(), _user(owner_id), project.id, document.id)


# --- Anonymous reads: published *and* the matching token, or it must look absent ---


@pytest.mark.asyncio
async def test_anonymous_reads_a_published_document_with_the_right_token() -> None:
    access, project, document, _part, _repo = _fixture(workflow=DocumentWorkflow.published)

    context = await access.require_document(_Session(), None, project.id, document.id, token=TOKEN)

    assert context.document is document


@pytest.mark.asyncio
async def test_anonymous_draft_is_not_found_rather_than_forbidden() -> None:
    """403 would confirm the document exists. The public surface must not do that."""
    access, project, document, _part, _repo = _fixture(workflow=DocumentWorkflow.draft)

    with pytest.raises(NotFoundError):
        await access.require_document(_Session(), None, project.id, document.id, token=TOKEN)


@pytest.mark.asyncio
async def test_anonymous_archived_document_is_not_found() -> None:
    access, project, document, _part, _repo = _fixture(workflow=DocumentWorkflow.archived)

    with pytest.raises(NotFoundError):
        await access.require_document(_Session(), None, project.id, document.id, token=TOKEN)


@pytest.mark.asyncio
async def test_anonymous_published_read_with_the_wrong_token_is_not_found() -> None:
    """A wrong guess must read exactly like a document that was never published."""
    access, project, document, _part, _repo = _fixture(workflow=DocumentWorkflow.published)

    with pytest.raises(NotFoundError, match="Document not found"):
        await access.require_document(
            _Session(), None, project.id, document.id, token="wrong-token"
        )


@pytest.mark.asyncio
async def test_anonymous_published_read_with_no_token_is_not_found() -> None:
    access, project, document, _part, _repo = _fixture(workflow=DocumentWorkflow.published)

    with pytest.raises(NotFoundError, match="Document not found"):
        await access.require_document(_Session(), None, project.id, document.id)


@pytest.mark.asyncio
async def test_anonymous_published_read_with_no_token_minted_yet_is_not_found() -> None:
    """A document can be published with ``public_share_token`` still ``None`` in theory
    (a row written before this feature, or a bug elsewhere) - the token comparison must
    fail closed rather than treat "no secret to check" as "anyone may read this".
    """
    access, project, document, _part, _repo = _fixture(workflow=DocumentWorkflow.published)
    document.public_share_token = None

    with pytest.raises(NotFoundError, match="Document not found"):
        await access.require_document(_Session(), None, project.id, document.id, token=TOKEN)


# --- Parts reached through the project path ---


@pytest.mark.asyncio
async def test_require_part_returns_the_whole_chain() -> None:
    owner_id = uuid.uuid4()
    access, project, document, part, _repo = _fixture(owner_id=owner_id)

    context = await access.require_part(
        _Session(), _user(owner_id), project.id, document.id, part.id
    )

    assert (context.project, context.document, context.part) == (project, document, part)


@pytest.mark.asyncio
async def test_part_filed_under_another_document_reads_as_missing() -> None:
    owner_id = uuid.uuid4()
    access, project, document, part, _repo = _fixture(owner_id=owner_id)
    part.document_id = uuid.uuid4()

    with pytest.raises(NotFoundError, match="Part not found"):
        await access.require_part(_Session(), _user(owner_id), project.id, document.id, part.id)


@pytest.mark.asyncio
async def test_require_part_authorizes_before_it_loads_the_part() -> None:
    access, project, document, part, repo = _fixture()

    with pytest.raises(AccessDeniedError):
        await access.require_part(_Session(), _user(), project.id, document.id, part.id)
    assert "get_part" not in repo.calls


# --- Parts reached by id alone: the media routes ---


@pytest.mark.asyncio
async def test_media_lookup_derives_the_project_from_the_part() -> None:
    owner_id = uuid.uuid4()
    access, project, document, part, repo = _fixture(owner_id=owner_id)

    context = await access.require_part_by_id(_Session(), _user(owner_id), part.id)

    assert context.part is part
    assert context.project is project
    # The row-only loaders, not the eager ones: serving bytes needs ``image_key`` alone.
    assert repo.calls == ["get_part_row", "get_by_id_for_authz"]


@pytest.mark.asyncio
async def test_media_lookup_for_a_missing_part_is_not_found() -> None:
    access, _project, _document, _part, _repo = _fixture()

    with pytest.raises(NotFoundError, match="Part not found"):
        await access.require_part_by_id(_Session(), _user(), uuid.uuid4())


@pytest.mark.asyncio
async def test_anonymous_media_lookup_allows_published_and_hides_draft() -> None:
    access, _project, _document, part, _repo = _fixture(workflow=DocumentWorkflow.published)
    assert (await access.require_part_by_id(_Session(), None, part.id, token=TOKEN)).part is part

    access, _project, _document, part, _repo = _fixture(workflow=DocumentWorkflow.draft)
    with pytest.raises(NotFoundError):
        await access.require_part_by_id(_Session(), None, part.id, token=TOKEN)


@pytest.mark.asyncio
async def test_anonymous_media_lookup_with_the_wrong_token_is_not_found() -> None:
    access, _project, _document, part, _repo = _fixture(workflow=DocumentWorkflow.published)

    with pytest.raises(NotFoundError):
        await access.require_part_by_id(_Session(), None, part.id, token="wrong-token")


# --- Held-back parts: a document can be public while one of its pages is not ---


@pytest.mark.asyncio
async def test_anonymous_reader_cannot_reach_an_unpublished_part() -> None:
    access, project, document, part, _repo = _fixture(
        workflow=DocumentWorkflow.published, part_published=False
    )

    with pytest.raises(NotFoundError, match="Part not found"):
        await access.require_part(_Session(), None, project.id, document.id, part.id, token=TOKEN)

    with pytest.raises(NotFoundError, match="Part not found"):
        await access.require_part_by_id(_Session(), None, part.id, token=TOKEN)


@pytest.mark.asyncio
async def test_member_reaches_an_unpublished_part_regardless_of_the_flag() -> None:
    """The flag drives the anonymous surface only - the owner's toggle UI needs every
    part back, published or not, or it would have nothing to render a switch for.
    """
    owner_id = uuid.uuid4()
    access, project, document, part, _repo = _fixture(
        owner_id=owner_id, workflow=DocumentWorkflow.published, part_published=False
    )

    context = await access.require_part(
        _Session(), _user(owner_id), project.id, document.id, part.id
    )

    assert context.part is part


@pytest.mark.asyncio
async def test_private_media_route_is_a_member_route_even_when_published() -> None:
    """A published document's image is public at ``/public/media``, not at ``/media``.

    The authenticated route checks membership and nothing else, so an authenticated
    non-member is refused here while an anonymous caller is served next door. That
    asymmetry is pre-existing behaviour and is pinned deliberately.
    """
    access, _project, _document, part, _repo = _fixture(workflow=DocumentWorkflow.published)

    with pytest.raises(AccessDeniedError):
        await access.require_part_by_id(_Session(), _user(), part.id)


# --- The project-only question ---


@pytest.mark.asyncio
async def test_require_project_checks_membership() -> None:
    owner_id = uuid.uuid4()
    access, project, _document, _part, _repo = _fixture(owner_id=owner_id)

    assert await access.require_project(_Session(), _user(owner_id), project.id) is project
    with pytest.raises(AccessDeniedError):
        await access.require_project(_Session(), _user(), project.id)
