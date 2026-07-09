---
id: "037"
title: "targeted-api-mutations-audit"
type: AFK
status: backlog
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

| Operation | Target endpoint | Full replace (`PUT /lines`) |
|-----------|----------------|----------------------------|
| List lines | `GET /lines` | — |
| Create one segment | `POST /lines` | Was used for draw-new |
| Edit one segment geometry | `PATCH /lines/{line_id}` | Was used for resize / nudge |
| Edit baseline/mask only | `PATCH /lines/{line_id}` (geometry) | — |
| Delete one segment | `DELETE /lines/{line_id}` | Was used for delete |
| Replace entire page layout | `PUT /lines` | Correct use (segment job, history restore, bulk seed) |
| Reset layout | `POST /layout/reset` | — |

`PUT /lines` remains valid for **bulk sync** (Kraken segmentation results, annotation history restore, dev seed). It should not be used for single-segment CRUD.

## Root causes

1. **Frontend** — `useLayoutMutations` routed delete, draw, resize, and nudge through `api.replacePartLines` with the full in-memory line array.
2. **Serializer** — `upsertLineRequest` omitted `block_id`, `source_metadata`, and `kraken_ceiling`, so any remaining replace calls cleared those columns server-side.
3. **Backend** — `replace_part_lines` assigned `data.get("optional_field")` even when the key was absent, writing `NULL` instead of preserving the prior value.

## Work done (in current branch, needs review + tests)

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
- [x] Partial: `ProjectDashboardPage`, `DocumentDetailPage`, `usePageEditorData`

## Remaining work

### 1. Finish and verify line-mutation slice

- [ ] Update `PageEditorPlaceholderPage.test.tsx` — polygon draw test still expects `replacePartLines`; should expect `createPartLine`
- [ ] Add/extend unit tests for `mergeSavedLine` and targeted mutation paths
- [ ] Add integration test: partial `PUT /lines` without `kraken_ceiling` / `source_metadata` preserves existing values (backend)
- [ ] Manual check on a 100+ segment page: delete / resize should log 1 write, not 100+

### 2. Other full-reload patterns (lower priority)

These are not wrong, but worth revisiting if API volume is still high:

| Location | Current behavior | Possible improvement |
|----------|------------------|----------------------|
| `usePairingState.refreshAfterOcr` | `listPartLines` + `listTranscriptions` after OCR | Merge `job.result.lines` locally (already partially done in `applyTranscribeResult`), skip full line reload when job payload is complete |
| `usePairingState.promoteSelectedSegmentToGroundTruth` | `listPartLines` + `getPagePairing` | Patch local `line_transcriptions` from `copyToGroundTruth` response |
| `useLayoutMutations.runAutoSegment` | `listPartLines` + `getPartLayout` + `getPagePairing` after job | Keep full reload (segment job replaces layout server-side) |
| `usePageEditorData` initial load | Parallel fetch of layout, lines, transcriptions, pairing | OK for page mount; consider deduping with `getCache` where repeated |

### 3. Auth redirect consistency

- [ ] Apply `hasAccessToken` / `isUnauthorized` pattern to any remaining protected loaders that still show error banners on 401
- [ ] Add smoke tests for `/projects/:id` and `/projects/:id/documents/:id` unauthenticated redirect

### 4. `PATCH /lines/{line_id}` semantics

`patch_part_line` always sets `source = manual` and `manual_geometry = True`. Document or adjust if we need to preserve `source = kraken` when only nudging geometry without user intent to override.

### 5. Correct uses of `PUT /lines` (do not change)

- `AnnotationHistoryService.restore_snapshot` — intentional full restore
- `scripts/platform/seed_dev_annotated_data.py` — bulk seed
- Kraken segmentation callback (backend job path) — replaces machine-generated lines

## Acceptance criteria (done when)

- [ ] No single-segment user action in the page editor calls `replacePartLines`
- [ ] `PUT /lines` integration tests cover metadata preservation on partial payloads
- [ ] Frontend tests updated for `createPartLine` / `patchPartLine` / `deletePartLine`
- [ ] API logs on delete/resize of one segment show O(1) writes, not O(n) line updates
- [ ] Auth redirect behavior consistent across projects, project dashboard, document detail, and page editor

## How to verify

```bash
# Frontend tests
cd nomicous/frontend && npm test -- --run src/pages/PageEditorPlaceholderPage.test.tsx

# Backend integration (lines)
uv run pytest tests/nomicous/integration/test_documents.py -k replace_part_lines -v
```

Watch docker `api-1` logs while deleting or resizing one segment on a large page. Expect one `DELETE` or one `PATCH`, not a burst of `UPDATE lines SET block_id=...`.

## References

- Frontend: `nomicous/frontend/src/components/page-editor/hooks/useLayoutMutations.ts`
- Backend: `nomicous/backend/document/application/layout_service.py`
- OpenAPI: `POST/PATCH/DELETE .../parts/{part_id}/lines[/{line_id}]`
