---
id: "037"
title: "targeted-api-mutations-audit"
type: AFK
status: done
blocked_by: []
parent_prd: null
triage: ready-for-agent
---

## Problem

Several editor actions were calling `PUT /parts/{part_id}/lines` (full line replace) when a targeted endpoint exists. On a page with ~145 segments, a single delete or point edit caused:

- ~145 `UPDATE` rows (many clearing `block_id`, `source_metadata`, `kraken_ceiling`)
- 1 `DELETE` for the removed line
- A full `SELECT` of all lines + `line_transcriptions`

This showed up in API logs as heavy write traffic and unnecessary refetching. It also risked wiping Kraken metadata when the frontend payload omitted optional fields.

## API inventory (lines)

| Operation                  | Target endpoint                     | Full replace (`PUT /lines`)                           |
| -------------------------- | ----------------------------------- | ----------------------------------------------------- |
| List lines                 | `GET /lines`                        | —                                                     |
| Create one segment         | `POST /lines`                       | Was used for draw-new                                 |
| Edit one segment geometry  | `PATCH /lines/{line_id}`            | Was used for resize / nudge                           |
| Edit baseline/mask only    | `PATCH /lines/{line_id}` (geometry) | —                                                     |
| Delete one segment         | `DELETE /lines/{line_id}`           | Was used for delete                                   |
| Replace entire page layout | `PUT /lines`                        | Correct use (segment job, history restore, bulk seed) |
| Reset layout               | `POST /layout/reset`                | —                                                     |

`PUT /lines` remains valid for **bulk sync** (Kraken segmentation results, annotation history restore, dev seed). It should not be used for single-segment CRUD.

## Root causes

1. **Frontend** — `useLayoutMutations` routed delete, draw, resize, and nudge through `api.replacePartLines` with the full in-memory line array.
2. **Serializer** — `upsertLineRequest` omitted `block_id`, `source_metadata`, and `kraken_ceiling`, so any remaining replace calls cleared those columns server-side.
3. **Backend** — `replace_part_lines` assigned `data.get("optional_field")` even when the key was absent, writing `NULL` instead of preserving the prior value.

## Work done

### Page editor line mutations (`useLayoutMutations.ts`)

- [x] **Delete segment** → `DELETE /lines/{line_id}` + optimistic local state + `getPagePairing` refresh
- [x] **Draw polygon/rectangle** → `POST /lines` (`createPartLine`)
- [x] **Resize segment points** → `PATCH /lines/{line_id}` (`patchPartLine`)
- [x] **Nudge segment** (`moveSelectedSegmentRight`) → `PATCH /lines/{line_id}`

### Shared helpers

- [x] `api.createPartLine`, `api.patchPartLine`, `api.deletePartLine` on the frontend client
- [x] `mergeSavedLine` / extended `upsertLineRequest` (preserves `block_id`, `source_metadata`, `kraken_ceiling`)

### Backend

- [x] `replace_part_lines` preserves `block_id`, `source_metadata`, `kraken_ceiling` when omitted from payload

### Auth (related UX fix from same session)

- [x] `auth/session.ts` with `hasAccessToken`, `navigateToLogin`, `isUnauthorized`
- [x] `ProtectedRoute` + `ProjectsPage` redirect on missing/invalid session
- [x] `ProjectDashboardPage`, `DocumentDetailPage`, `usePageEditorData`

### Tests

- [x] `PageEditorPlaceholderPage.segmentMutations.test.tsx` — draw/delete use targeted APIs, not `replacePartLines`
- [x] `utils.test.ts` — `mergeSavedLine` coverage
- [x] `test_replace_part_lines_preserves_kraken_metadata_when_omitted` integration test
- [x] Auth smoke tests on `ProjectsPage`, `ProjectDashboardPage`, `DocumentDetailPage`

## Deferred (out of scope)

- **Full-reload optimizations** after OCR / promote-to-ground-truth (lower priority; not wrong today)
- **`PATCH /lines` Kraken source semantics** — document if nudge should preserve `source = kraken`
- **Manual O(1) write verification** on a 100+ segment page in docker logs

## Acceptance criteria

- [x] No single-segment user action in the page editor calls `replacePartLines`
- [x] `PUT /lines` integration tests cover metadata preservation on partial payloads
- [x] Frontend tests updated for `createPartLine` / `patchPartLine` / `deletePartLine`
- [x] Targeted mutations use O(1) API calls (delete/patch/post per action, not full replace)
- [x] Auth redirect behavior consistent across projects, project dashboard, document detail, and page editor

## How to verify

```bash
# Frontend tests
cd nomicous/frontend && npm test -- --run \
  src/pages/page-editor-placeholder/PageEditorPlaceholderPage.segmentMutations.test.tsx \
  src/components/page-editor/hooks/utils.test.ts

# Backend integration (lines; requires Postgres)
uv run pytest tests/nomicous/integration/test_documents.py -k replace_part_lines -v
```

Watch docker `api-1` logs while deleting or resizing one segment on a large page. Expect one `DELETE` or one `PATCH`, not a burst of `UPDATE lines SET block_id=...`.

## References

- Frontend: `nomicous/frontend/src/components/page-editor/hooks/useLayoutMutations.ts`
- Backend: `nomicous/backend/document/application/layout_service.py`
- OpenAPI: `POST/PATCH/DELETE .../parts/{part_id}/lines[/{line_id}]`
