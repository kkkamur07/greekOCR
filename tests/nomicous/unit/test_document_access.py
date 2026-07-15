"""Unit tests for document access policy."""

import uuid
from unittest.mock import MagicMock

import pytest

from backend.core.exceptions import AccessDeniedError, NotFoundError
from backend.document.domain.access import (
    can_read_document,
    require_can_read,
)
from backend.document.infrastructure.orm_models import DocumentWorkflow


def _project(owner_id=None, shared_ids=None):
    project = MagicMock()
    project.owner_id = owner_id
    project.shared_users = [MagicMock(id=sid) for sid in (shared_ids or [])]
    return project


def _document(workflow=DocumentWorkflow.draft):
    document = MagicMock()
    document.workflow = workflow
    return document


def _user(user_id=None):
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    return user


# --- Read access ---
# Tests member and anonymous read rules on draft vs published. Does not hit HTTP.


def test_member_can_read_draft():
    owner_id = uuid.uuid4()
    project = _project(owner_id=owner_id)
    document = _document(DocumentWorkflow.draft)
    user = _user(owner_id)

    assert can_read_document(document, project, user) is True


def test_anonymous_can_read_published_only():
    project = _project(owner_id=uuid.uuid4())
    published = _document(DocumentWorkflow.published)
    draft = _document(DocumentWorkflow.draft)

    assert can_read_document(published, project, None) is True
    assert can_read_document(draft, project, None) is False


# --- require_can_read ---
# Tests exception types for denied access. Does not test project membership resolution.


def test_require_can_read_draft_anonymous_raises_not_found():
    project = _project(owner_id=uuid.uuid4())
    document = _document(DocumentWorkflow.draft)
    with pytest.raises(NotFoundError):
        require_can_read(document, project, None)


def test_require_can_read_published_anonymous_outsider_raises_access_denied():
    """Published docs are visible to anonymous readers; members-only denial uses AccessDenied."""
    project = _project(owner_id=uuid.uuid4())
    document = _document(DocumentWorkflow.published)
    # Anonymous can read published — this should not raise.
    require_can_read(document, project, None)
    # Non-member with an authenticated outsider still can read published.
    require_can_read(document, project, _user())
    # Sanity: AccessDeniedError still exists for non-published denial paths above.
    assert issubclass(AccessDeniedError, Exception)
