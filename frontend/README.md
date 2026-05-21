# greekOCR frontend

Vite + React client for the greekOCR platform API. Editor UI ports concepts and components from [eScriptorium](https://github.com/PSL-Paris-Saclay/escriptorium) (`_support_repo/escriptorium/front/vue/`) into React — not a line-for-line Vue port, but the same canvas, transcription panel, and workflow patterns.

## API type codegen

Types are generated from the FastAPI OpenAPI schema so the client stays aligned with backend DTOs.

### Prerequisites

- Python env with platform dependencies (`requirements-platform.txt`)
- Node.js 20+
- PostgreSQL running (`docker compose up db -d`)

### Regenerate types

From the repository root:

```bash
PYTHONPATH=. python scripts/export_openapi.py
cd frontend
npm run codegen:api
```

Committed outputs:

- `frontend/openapi/openapi.json`
- `frontend/src/api/schema.d.ts`

### Smoke check

```bash
cd frontend
npm run typecheck:api   # API + pages typecheck
npm run build           # typecheck:api + Vite production bundle
```

Use `npm run typecheck` for the full app (includes legacy `/demo` OCR components).

## Local dev

1. **Env**

   ```bash
   cd frontend
   cp .env.local.example .env.local
   ```

   Set `VITE_API_BASE_URL` if the API is not on `http://localhost:8000`.

2. **Dev user** (repo root, DB up):

   ```bash
   PYTHONPATH=. python scripts/seed_dev_user.py
   ```

   Default: `dev@kalamos.local` / `dev-pass-123`.

3. **API + frontend**

   ```bash
   # Terminal 1 — API
   uvicorn backend.core.app:create_app --factory --reload

   # Terminal 2 — Vite
   cd frontend && npm install && npm run dev
   ```

   App: `http://localhost:5173` — login → projects → documents.

## Public published view

Anonymous users read **published** documents only:

```text
/public/projects/{projectId}/documents/{documentId}
```

Example: `http://localhost:5173/public/projects/<uuid>/documents/<uuid>`

### Publish for testing

1. Sign in, create a document, upload a part image.
2. PATCH workflow to `published`:

   ```bash
   curl -X PATCH "http://localhost:8000/projects/{project_id}/documents/{document_id}" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"workflow": "published"}'
   ```

3. Open the public URL in a private window (no JWT). Drafts return **404**.

Logged-in members see **Open in editor** on the public page; the editor shows **View public page** when workflow is `published`.

## eScriptorium → React component map

| eScriptorium (Vue) | React port | Role |
|--------------------|------------|------|
| `VisuPanel` / canvas | `ImageCanvas/` | Folio image, regions, zoom (`react-zoom-pan-pinch`) |
| `VisuLine` | `ImageCanvas/components/RegionOveraly.tsx` | Line/box overlays |
| Transcription UI | `TrascriptionPanel/` | Layer list + detail (folder typo kept) |
| `TaskDashboard` | `JobsPanel/` | Job enqueue + poll (test jobs until 006/009) |
| Document workflow | `WorkflowBadge.tsx` | draft / published / archived |
| Toolbar / modes | `ControlBar/` | Legacy demo at `/demo` only |
| Authenticated media | `AuthenticatedImage.tsx` | JWT for `/media/parts/...` |
| Public media | `RemoteImage.tsx` + `publicPartMediaUrl()` | No JWT; `/public/media/parts/{id}` |

Platform routes live in `src/pages/`; HTTP in `src/api/client.ts` (`skipAuth` for public routes).

## Issue tracking

Frontend lanes (014–017) are tracked in `issues/` but **do not count toward kanban WIP** limits used for backend/platform lanes.

### When to run codegen

- After changing FastAPI routes or Pydantic schemas under `backend/`
- Before opening a PR that touches API contracts

## Jobs panel (issue 016)

The document editor shows a **Jobs** card (`JobsPanel`) that tracks jobs enqueued from the UI. Each row polls `GET /jobs/{id}` every ~1.5s until status is `done` or `failed`. Failed jobs surface the API `error` string via antd `notification` (and inline on the row).

### Dev smoke: noop test job

1. API: `ENABLE_TEST_JOB_ROUTES=true` in `backend/core/.env` (see `backend/core/.env.example`).
2. Frontend: `VITE_ENABLE_TEST_JOBS=true` in `.env.local` (see `.env.local.example`).
3. Start API (`uvicorn backend.core.app:create_app --factory --reload` from repo root).
4. Open a document → **Run test job** → row moves `pending` → `running` → `done`.

Segment/transcribe enqueue buttons arrive with issues 006/009; the panel already supports multiple concurrent job rows.
