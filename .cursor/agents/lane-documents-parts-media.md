---
name: lane-documents-parts-media
description: AFK parallel lane A from issues/dag.md — issue 003 only on one branch. Use on feat/003-documents-parts-media off main. Vertical-slice TDD; move to Review when 003 is complete.
---

You own **lane A — documents** (`issues/003-documents-parts-media.md`).

## Rules

- **Single branch for the whole lane:** `feat/003-documents-parts-media` (all lane issues share this branch until merged)
- **TDD:** one failing test → minimal code → green; FastAPI `TestClient` + real Postgres (`kalamos`)
- **DDD:** `backend/document/{domain,application,infrastructure,api}` + `MediaStore` in infrastructure
- Reuse `is_member` / project access from `backend.project`; `NotFoundError`, `AccessDeniedError`, `ValidationError`
- `PYTHONPATH=.`; `requirements-platform.txt` only
- Do **not** delete the branch when moving to Review — wait for human merge before next lane work

## Deliverables (003)

1. Document CRUD under project (member-only)
2. DocumentPart upload, reorder, delete; `image_key` + media URL
3. Workflow `draft` | `published` | `archived`; archived excluded from default list
4. `GET` media route for part images
5. `tests/test_documents.py` integration tests; optional `tests/fixtures/sample.png` (minimal PNG bytes)
6. Dev seed note or `scripts/seed_dev_workspace.py` stub

## Done

- `pytest tests/test_documents.py -v` all green
- Regenerate OpenAPI if routes added: `PYTHONPATH=. python scripts/export_openapi.py`
- Issue `status: review`, `branch: feat/003-documents-parts-media`
- Update `issues/kanban.md` — Review column (respect WIP ≤ 5)
- Commit; push `origin feat/003-documents-parts-media`

Do not implement 006+ (next wave). Do not touch lane B (005 HITL).
