"""Unit tests for document access policy."""

import uuid
from unittest.mock import MagicMock

import pytest

from backend.core.exceptions import AccessDeniedError, NotFoundError
from backend.document.domain.access import (
    can_mutate_document,
    can_read_document,
    require_can_mutate,
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


# --- require_can_read / require_can_mutate ---
# Tests exception types for denied access. Does not test project membership resolution.


def test_require_can_read_draft_anonymous_raises_not_found():
    project = _project(owner_id=uuid.uuid4())
    document = _document(DocumentWorkflow.draft)
    with pytest.raises(NotFoundError):
        require_can_read(document, project, None)


def test_require_can_mutate_anonymous_raises():
    project = _project(owner_id=uuid.uuid4())
    with pytest.raises(AccessDeniedError, match="Authentication required"):
        require_can_mutate(project, None)


# --- Mutate access ---
# Tests outsiders cannot mutate. Does not test collaborator vs owner distinctions.


def test_outsider_cannot_mutate():
    project = _project(owner_id=uuid.uuid4())
    user = _user()
    assert can_mutate_document(project, user) is False
