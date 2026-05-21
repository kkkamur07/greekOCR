"""Project membership rules (owner + shared collaborators)."""

from uuid import UUID

from backend.project.infrastructure.orm_models import Project


def is_owner(project: Project, user_id: UUID) -> bool:
    return project.owner_id == user_id


def is_member(project: Project, user_id: UUID) -> bool:
    if project.owner_id == user_id:
        return True
    return any(shared.id == user_id for shared in project.shared_users)
