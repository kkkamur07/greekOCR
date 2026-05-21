---
name: lane-projects-sharing
description: AFK parallel lane for issue 002 (project CRUD + sharing). Use on branch feat/002-projects-sharing off feat/001-user-auth-jwt. Vertical-slice TDD only.
---

You own **lane D — projects** (`issues/002-projects-sharing.md`).

## Rules

- Branch: `feat/002-projects-sharing` from `feat/001-user-auth-jwt`
- **TDD**: one test → minimal code → green; TestClient + real Postgres
- **DDD**: `backend/project/{domain,application,infrastructure,api}` — replace stub in `backend/project/api/projects.py`
- Use `get_current_user`, `NotFoundError`, `AccessDeniedError`, `ConflictError` from 001
- `PYTHONPATH=.`; `requirements-platform.txt` only (no editable package install)

## Deliverables

1. Owner CRUD: create/read/update/delete (slug, name, guidelines)
2. Share/unshare collaborator by **username**
3. List projects: owned OR shared only
4. Non-member: 403 or 404 on read/mutate
5. Tests: owner CRUD, shared user access, non-member denial
6. Fixtures: two users + optional project seed in `tests/conftest.py`

## Done

- `pytest tests/test_projects.py -v` all green
- Issue `status: review`, `branch: feat/002-projects-sharing`
- Commit; push origin

Do not touch frontend/openapi (012) or inference jobs (004).
