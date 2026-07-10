"""Unit tests for project membership rules."""

import uuid

from backend.project.domain.access import has_owner, is_member, is_owner
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


def _project(*, owner_id: uuid.UUID | None, shared_ids: list[uuid.UUID] | None = None) -> Project:
    project = Project(name="Test", slug="test", owner_id=owner_id)
    if shared_ids:
        project.shared_users = [
            User(id=sid, email=f"{sid}@t", username=f"u{sid}") for sid in shared_ids
        ]
    return project


# --- Orphan projects ---
# Tests projects without an owner. Does not test HTTP or database persistence.


def test_orphan_project_has_no_owner():
    project = _project(owner_id=None)
    assert not has_owner(project)
    assert not is_owner(project, uuid.uuid4())


# --- Shared membership ---
# Tests shared users count as members on ownerless projects. Does not test owner projects.


def test_orphan_project_shared_user_is_member():
    collaborator = uuid.uuid4()
    project = _project(owner_id=None, shared_ids=[collaborator])
    assert is_member(project, collaborator)
    assert not is_member(project, uuid.uuid4())
