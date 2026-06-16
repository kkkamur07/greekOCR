"""Project membership rules (owner + shared collaborators).

Orphan policy (``owner_id`` NULL after account deletion): shared collaborators
retain read/list access via :func:`is_member`; owner-only mutations require a
live owner (:func:`has_owner`). Preventing owner deletion while projects exist
is deferred to a later issue.
"""

from uuid import UUID

from backend.project.infrastructure.orm_models import Project


def has_owner(project: Project) -> bool:
    """True when the project still has an owning user."""
    return project.owner_id is not None


def is_owner(project: Project, user_id: UUID) -> bool:
    if project.owner_id is None:
        return False
    return project.owner_id == user_id


def is_member(project: Project, user_id: UUID) -> bool:
    if project.owner_id is not None and project.owner_id == user_id:
        return True
    return any(shared.id == user_id for shared in project.shared_users)
