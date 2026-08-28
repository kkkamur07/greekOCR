"""``domain/access.py`` — the bare predicate, without the seam around it.

``test_document_access_seam`` covers which rows get loaded and which exception each
failure produces; this file covers only the yes/no question those calls are built on:
does this caller, with this token, get to read this document. The
:func:`secrets.compare_digest` requirement is pinned here too, since a plain ``==`` would
still pass every other test in this file.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from backend.core.exceptions import NotFoundError
from backend.document.domain.access import can_read_document, require_can_read
from backend.document.infrastructure.orm_models import Document, DocumentWorkflow
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User

TOKEN = "s3cr3t-share-token"


def _project(*, owner_id=None, shared=()) -> Project:
    project = Project(id=uuid.uuid4(), name="Codices", owner_id=owner_id or uuid.uuid4())
    project.shared_users = list(shared)
    return project


def _document(*, workflow=DocumentWorkflow.draft, token: str | None = TOKEN) -> Document:
    return Document(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        name="MS 1",
        workflow=workflow,
        public_share_token=token,
    )


def _user(user_id=None) -> User:
    return User(
        id=user_id or uuid.uuid4(),
        email="reader@example.org",
        username="reader",
        hashed_password="x",
    )


# --- Members: the token never enters into it ---


def test_owner_reads_any_workflow_without_a_token() -> None:
    owner_id = uuid.uuid4()
    project = _project(owner_id=owner_id)
    document = _document(workflow=DocumentWorkflow.draft)

    assert can_read_document(document, project, _user(owner_id)) is True


def test_shared_collaborator_reads_any_workflow_without_a_token() -> None:
    collaborator = _user()
    project = _project(shared=[collaborator])
    document = _document(workflow=DocumentWorkflow.archived)

    assert can_read_document(document, project, collaborator) is True


# --- Anonymous: published, and the token must match exactly ---


def test_anonymous_with_the_right_token_reads_a_published_document() -> None:
    project = _project()
    document = _document(workflow=DocumentWorkflow.published)

    assert can_read_document(document, project, None, TOKEN) is True


@pytest.mark.parametrize("workflow", [DocumentWorkflow.draft, DocumentWorkflow.archived])
def test_anonymous_with_the_right_token_cannot_read_an_unpublished_document(workflow) -> None:
    project = _project()
    document = _document(workflow=workflow)

    assert can_read_document(document, project, None, TOKEN) is False


def test_anonymous_with_no_token_cannot_read_a_published_document() -> None:
    project = _project()
    document = _document(workflow=DocumentWorkflow.published)

    assert can_read_document(document, project, None) is False


def test_anonymous_with_the_wrong_token_cannot_read_a_published_document() -> None:
    project = _project()
    document = _document(workflow=DocumentWorkflow.published)

    assert can_read_document(document, project, None, "not-the-token") is False


def test_anonymous_against_a_document_with_no_token_minted_cannot_read_it() -> None:
    """A published row whose token is still ``None`` must fail closed, not open."""
    project = _project()
    document = _document(workflow=DocumentWorkflow.published, token=None)

    assert can_read_document(document, project, None, TOKEN) is False


def test_a_prefix_of_the_real_token_is_not_accepted() -> None:
    """Guards against a naive ``str.startswith``-style check creeping back in."""
    project = _project()
    document = _document(workflow=DocumentWorkflow.published)

    assert can_read_document(document, project, None, TOKEN[:-1]) is False


def test_the_comparison_goes_through_compare_digest_not_equality() -> None:
    """The whole point of the token check is that it cannot be timed. A ``==`` would
    still return the right true/false answer for every case above, so the only way to
    pin *how* the comparison is made is to watch the call itself.
    """
    project = _project()
    document = _document(workflow=DocumentWorkflow.published)

    with patch(
        "backend.document.domain.access.secrets.compare_digest", return_value=True
    ) as compare_digest:
        assert can_read_document(document, project, None, "whatever-was-sent") is True

    compare_digest.assert_called_once_with(document.public_share_token, "whatever-was-sent")


# --- require_can_read: always 404, never 403, on the anonymous path ---


def test_require_can_read_passes_a_readable_document_through() -> None:
    project = _project()
    document = _document(workflow=DocumentWorkflow.published)

    require_can_read(document, project, None, TOKEN)  # does not raise


def test_require_can_read_raises_not_found_for_a_wrong_token() -> None:
    project = _project()
    document = _document(workflow=DocumentWorkflow.published)

    with pytest.raises(NotFoundError, match="Document not found"):
        require_can_read(document, project, None, "wrong-token")


def test_require_can_read_raises_not_found_for_an_unpublished_document() -> None:
    project = _project()
    document = _document(workflow=DocumentWorkflow.draft)

    with pytest.raises(NotFoundError, match="Document not found"):
        require_can_read(document, project, None, TOKEN)
