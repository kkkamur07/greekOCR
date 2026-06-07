---
id: "010"
title: "annotation-history"
type: AFK
status: done
blocked_by:
  - "009-page-lock.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Annotation history: timed snapshots, pairing milestones, lock/unlock events, retention, list/restore UI

## What to build

Filesystem **annotation history** per page: write snapshots on a configurable interval during editing, at 50% and 100% **pairing progress**, and on lock/unlock. Enforce retention (five rolling timed snapshots; protected milestone and lock/unlock snapshots). Expose list and restore via API; add a history panel in the page editor.

- Nested **`HistorySettings`**: `snapshot_interval_minutes` (default 5), `max_timed_snapshots` (default 5), `pairing_milestones` (default `[50, 100]`).
- Snapshots stored under `data/annotations/history/<stem>/` with metadata (timestamp, reason, pairing progress at capture).
- **Restore** replaces current annotation on disk; editor reloads; history entries are retained.
- Restore allowed when page is unlocked (or unlock-first flow documented in UI if locked).

Does not include transcription PDF share mode (slice 011).

## Acceptance criteria

- [x] Editing a page creates timed snapshots at the configured interval (debounced server-side or on save with last-snapshot timestamp).
- [x] Crossing 50% and 100% pairing progress writes protected milestone snapshots.
- [x] Lock and unlock (slice 009) each write a protected snapshot.
- [x] No more than five non-protected timed snapshots per page; older timed snapshots are pruned.
- [x] Protected snapshots are never pruned by the timed cap.
- [x] `GET /pages/{stem}/history` returns ordered list with id, timestamp, reason label, pairing progress.
- [x] `POST /pages/{stem}/history/{snapshot_id}/restore` restores annotation; editor reflects restored segments and pairings.
- [x] History UI lists snapshots and confirms before restore.
- [x] `HistorySettings` nested in `Settings` with env overrides.
- [x] Backend tests cover retention, milestone capture, and restore round-trip.

## Blocked by

- `issues/009-page-lock.md` (lock/unlock event snapshots and lock-aware restore policy)

## User stories addressed

- User story 62
- User story 63
- User story 64
- User story 65
- User story 66
- User story 67
- User story 68
- User story 69
- User story 70
- User story 75 (HistorySettings portion)
