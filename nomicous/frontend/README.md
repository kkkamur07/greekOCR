# Nomicous Frontend

Vite + React client for the Nomicous production platform API. The app provides
authentication, Project/Document navigation, public published-document views,
and the Page editor used for layout, Pairing, Review status, Annotation
history, Export, Transcription editing, jobs, and PDF artifacts.

Run frontend commands from `nomicous/frontend/`.

## Quick Start

```bash
cd nomicous/frontend
npm install
cp .env.local.example .env.local
npm run dev
```

The dev app runs at `http://localhost:5173`. By default it talks to
`http://localhost:8000`; override with `VITE_API_BASE_URL` in `.env.local`.

Start the backend separately:

```bash
cd ../..
docker compose up db -d
cd nomicous
PYTHONPATH=. alembic -c infrastructure/alembic.ini upgrade head
PYTHONPATH=. uvicorn backend.core.app:create_app --factory --reload
```

## Directory Map

```text
frontend/
  package.json                 # scripts and dependencies
  vite.config.ts               # Vite + React
  vitest.setup.ts              # test DOM setup and global mocks
  openapi/openapi.json         # exported FastAPI schema
  src/
    main.tsx                   # React entrypoint
    App.tsx                    # route tree and shell
    LegacyDemoApp.tsx          # old OCR/editor demo route
    api/
      client.ts                # typed fetch wrapper and API helpers
      schema.d.ts              # generated OpenAPI TypeScript types
      errors.ts                # ApiError
    auth/                      # token storage and redirect helpers
    components/                # shared UI and editor components
    pages/                     # route-level pages and tests
    services/                  # legacy/demo service helpers
    types/                     # legacy/demo types
```

## Main Libraries

| Library | Role |
|---------|------|
| React 19 | UI framework |
| React Router 7 | Client routing |
| Ant Design 6 | Forms, cards, buttons, alerts, notifications |
| Vite 7 | Dev server and production build |
| Vitest + Testing Library | Component and route behavior tests |
| openapi-typescript | Generates `src/api/schema.d.ts` |
| react-zoom-pan-pinch | Image canvas zoom/pan |

## Scripts

```bash
npm run dev             # Vite dev server on 5173
npm run build           # typecheck:api + production bundle
npm run preview         # serve built bundle
npm run test            # Vitest run
npm run typecheck       # full TypeScript project references
npm run typecheck:api   # API-focused typecheck used by build
npm run codegen:api     # regenerate schema.d.ts from openapi/openapi.json
npm run lint            # ESLint
```

For workflow PRs, the current focused gate is usually:

```bash
npm run test -- PageEditorPlaceholderPage.test.tsx
npm run build
```

`npm run typecheck`, `npm run lint`, and all-test runs may also surface legacy
demo files; fix or scope those separately when touching that surface.

## API Contract Flow

Backend routes are the source of truth. When FastAPI schemas or routes change:

```bash
# from repository root
python scripts/platform/export_openapi.py
cd nomicous/frontend
npm run codegen:api
npm run typecheck:api
```

Committed API artifacts:

- `frontend/openapi/openapi.json`
- `frontend/src/api/schema.d.ts`
- hand-authored helpers and aliases in `frontend/src/api/client.ts`

`apiRequest()` attaches the JWT from `src/auth/storage.ts` unless a request sets
`skipAuth: true`. Public routes use `skipAuth` and unauthenticated media helpers.

## Route Pages

| Page | Purpose |
|------|---------|
| `LoginPage.tsx`, `RegisterPage.tsx` | Authentication |
| `ProjectsPage.tsx` | Project list/create entry |
| `ProjectDashboardPage.tsx` | Project-level navigation |
| `DocumentDetailPage.tsx` | Document metadata, parts, workflow actions |
| `PublicDocumentPage.tsx` | Anonymous read-only view for published Documents |
| `PageEditorPlaceholderPage.tsx` | Main Page/Document part editor |

Each route page should have behavior tests next to it. Prefer Testing Library
queries that describe user-visible behavior rather than component internals.

## Page Editor Workflows

`PageEditorPlaceholderPage.tsx` is the central editor route. It coordinates:

- Loading the Document, selected Document part, layout Blocks/Lines, Transcription layers, and Pairing state.
- Drawing/editing Segment geometry on the Page image.
- Importing Page transcription text and showing candidate Text lines.
- Pairing a selected Segment to a candidate Text line.
- Editing approved Ground truth Line transcription text directly.
- Showing Pairing progress as paired Lines over total Lines.
- Marking the Page reviewed/unreviewed independently from Pairing progress.
- Working with Transcription layers: Ground truth is editable; model layers are read-only and copyable into Ground truth.
- Triggering Export/PDF artifact behavior when exposed by the active branch.
- Tracking background jobs via SSE (`GET /jobs/{id}/events`) with poll fallback
  (`jobPolling.ts`, `useJobPolling`, `BackgroundJobsProvider`). See backend
  **Job status notifications** in [`backend/README.md`](../backend/README.md).

Domain language follows `nomicous/CONTEXT.md`: Page, Document part, Segment, Text
line, Page transcription, Line transcription, Pairing, Human review, Export,
Transcription PDF.

## Component Map

| Component | Role |
|-----------|------|
| `ProtectedRoute` | Redirects unauthenticated users |
| `AuthenticatedImage` | Fetches protected media with JWT |
| `RemoteImage` | Displays public unauthenticated media |
| `WorkflowBadge` | Document workflow display |
| `document/JobsNotice` | Document-level job status notices |
| `ImageCanvas/` | Page image, overlays, drawing, zoom/pan |
| `page-editor/` | Page editor strips, panes, hooks, and job queue |
| `ControlBar/` | Legacy demo toolbar |

The editor borrows ideas from eScriptorium's manuscript workflow, but it is not
a line-for-line Vue port.

## Public Published View

Anonymous users can read published Documents only:

```text
/public/projects/{projectId}/documents/{documentId}
```

Draft Documents return 404 through the public API. Authenticated project members
see editor navigation affordances on the public page.

## Jobs Notice Smoke Test

1. Set `ENABLE_TEST_JOB_ROUTES=true` in `backend/core/.env`.
2. Set `VITE_ENABLE_TEST_JOBS=true` in `frontend/.env.local`.
3. Start the API from `nomicous/`.
4. Start Vite from `nomicous/frontend/`.
5. Open a Document and click **Run test job**.

The notice should move `pending` -> `running` -> `done`, or show the API error if
the job fails.

## Special Notes

- `LegacyDemoApp.tsx`, `services/`, and some `types/` are retained for old OCR
  demo behavior. Do not use them as patterns for production platform pages.
- API-generated types may mark fields optional even when current backend
  responses always include them; UI code should be robust where feasible.
- Production build warns about large chunks today; that is a bundle-splitting
  follow-up, not a build failure.
