# Core (FastAPI platform)

| Path | Role |
|------|------|
| `app.py` | `create_app()` - middleware + `include_router` for all contexts |
| `api/` | Platform routes (e.g. `health.py`) |
| `schemas/` | HTTP DTOs (`*Response`, `*Edit`, `*Request`) |
| `settings/` | Split Pydantic settings (`InfrastructureSettings`, `AuthSettings`, `AppSettings`, `MLSettings`, `StorageSettings`) |
| `exceptions.py` | Shared exception types (`NotFoundError`, `AccessDeniedError`, …) - HTTP mapping in later issues |
| `.env` | Copy from `.env.example` (not committed; DB password **`dev`** in dev) |

**Router wiring:** `app.py` composes routers only; handlers live in `api/` or `backend/<context>/api/`.

**DTO convention:** domain edits use `*Edit` / `*Request`; reads use `*Response` under `schemas/` or `<context>/api/schemas.py`.
