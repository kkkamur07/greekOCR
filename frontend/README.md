# greekOCR frontend

Vite + React client for the greekOCR platform API.

## API type codegen

Types are generated from the FastAPI OpenAPI schema so the client stays aligned with backend DTOs (auth, projects, etc.).

### Prerequisites

- Python env with platform dependencies (`requirements-platform.txt`)
- Node.js 20+

### Regenerate types

From the repository root:

```bash
PYTHONPATH=. python scripts/export_openapi.py
cd frontend
npm run codegen:api
```

This writes:

- `frontend/openapi/openapi.json` — exported schema from `create_app()`
- `frontend/src/api/schema.d.ts` — TypeScript types (`openapi-typescript`)

Both files are **committed** so CI and reviewers see API contract changes in diffs.

### Smoke check

```bash
cd frontend
npm run typecheck:api   # generated API types only
npm run build           # typecheck:api + Vite production bundle
```

Legacy OCR UI components still have open `tsc` debt; use `npm run typecheck` for the full app when fixing those files.

`src/api/client.ts` is the JWT-aware fetch wrapper; pages under `src/pages/` implement projects and documents (issue 013).

## Local dev

1. Copy env: `cp .env.local.example .env.local` (set `VITE_API_BASE_URL` if the API is not on port 8000).
2. Seed a dev user (from repo root, DB running):

   ```bash
   PYTHONPATH=. python scripts/seed_dev_user.py
   ```

   Default credentials: `dev@kalamos.local` / `dev-pass-123` (override with `DEV_USER_EMAIL`, `DEV_USER_PASSWORD`).

3. Create a project via the UI after login, or use the API / optional `scripts/seed_dev_workspace.py` when available.

4. Start the API and frontend:

   ```bash
   cd frontend && npm run dev
   ```

### When to run

- After changing FastAPI routes or Pydantic schemas under `backend/`
- Before opening a PR that touches API contracts
