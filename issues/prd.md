# PRD: Bug-fix and annotation polish (grilling 2026-07-13)

**Status:** ready-for-agent
**Parent glossary:** `nomicous/CONTEXT.md`, `inference/CONTEXT.md`
**ADRs:** 0001 (browser auth memory + cookie + CSRF), 0002 (job SSE + polling fallback), 0003 (image canvas archival boundary)

## Problem Statement

Researchers using Nomicous hit friction that blocks trust in production: public Document links flash empty or “not available” while loading; reload dumps them back to login despite a valid session; Product jobs cannot be cancelled and sticky “✓ completed” alerts clutter the Page editor; Segment editing fights the canvas (pan vs vertex drag, no Escape finalize, weak delete/undo); pairing copy is confusing; Transcription PDF fails in production with no usable explanation; helper install does not prefetch models and download links are not always “latest”; errors are opaque; observability is incomplete; GitHub still carries obsolete PRs/branches; and misleading `folio*` naming plus mixed-script fixtures make live OCR tests unreliable.

## Solution

Ship a focused polish batch so the app behaves as researchers expect: keep chrome and spin only the waiting content; restore sessions on reload; simplify helper download and prefetch local-eligible weights on install; cancel Product jobs and use the job panel + auto-dismiss toasts; make Segment vertex edit, Escape, and Edit undo/redo feel natural; simplify pairing Save; fix Transcription PDF with a bundled Unicode font and plain-language failures; add logging-first full-stack observability with a frontend failure beacon; keep `greek-calamari-v1` out of production until Hub-ready and use Syriac live fixtures for OCR tests; drop `folio*` naming and clean storage of orphaned objects; close stale GitHub PRs/branches and ensure Quality, Security, and Deployment checks run on every pull request.

## User Stories

1. As a visitor on a public Document link, I want a spinner in the document content region while the Document loads, so that I never see a false “Document not available” flash.
2. As a visitor on a public Document link, I want the nav and public chrome to stay visible during load, so that the page does not feel like a full-route blank reload.
3. As a researcher anywhere in the app, I want the same “chrome stays, content spins” loading rule, so that waiting feels consistent.
4. As a visitor, I want a plain-language error only after a real public fetch failure, so that I know whether to retry or that the link is wrong.
5. As a logged-in researcher, I want reload to restore my session via the session cookie and return me to the same page, so that I am not bounced to login while the session is valid.
6. As a researcher, I want the access token to remain in memory only, so that auth stays aligned with the agreed security model.
7. As a researcher installing the Inference helper, I want the primary CTA to always point at GitHub `releases/latest`, so that I never download a stale build from a hardcoded release URL.
8. As a researcher, I want the helper install CTA to prefer my OS when detection is cheap, so that download has as few steps as possible.
9. As a researcher, I want the app to auto-detect a running helper without a manual refresh, so that local inference becomes available as soon as the helper is up.
10. As a researcher installing or first-launching the Inference helper, I want local-eligible model weights to prefetch then, so that OCR does not stall mid-run on first download.
11. As a researcher, I want `greek-calamari-v1` hidden from the model picker until Hub provenance exists, so that I cannot select a Greek OCR model that is not actually available.
12. As a researcher running OCR/segmentation, I want in-flight progress in the existing job panel at the bottom-left, so that I can see what is running without sticky page alerts.
13. As a researcher, I want a Cancel control on each active Product job in that panel, so that I can stop a job I started by mistake.
14. As a researcher who cancels a Product job, I want the job to become cancelled and any partial Segment or Model transcription results discarded, so that the Page is not left half-applied.
15. As a researcher, I want success and error outcomes as auto-dismiss toasts, so that “✓ OCR completed…” messages do not persist on the Page.
16. As a researcher editing a Segment, I want clicking a vertex then Delete to remove only that point, so that I can refine geometry without deleting the Segment.
17. As a researcher, I want clicking a Segment edge to add a point, so that I can densify the boundary where needed.
18. As a researcher, I want confirmation before deleting a whole Segment, so that I do not wipe a line by accident.
19. As a researcher dragging a vertex, I want the Page not to pan at the same time, so that pointer motion edits the point, not the viewport.
20. As a researcher, I want Escape to commit the current Segment geometry and deselect (hide selection chrome and pairing strip), so that I return to a normal Page view after editing.
21. As a researcher, I want Ctrl/Cmd+Z Edit undo and matching redo for in-session canvas edits, so that I can reverse vertex/segment mistakes without using Annotation history.
22. As a researcher pairing a Segment, I want one text field and a Save control that writes Ground truth, so that the flow is obvious and I can re-edit later by re-selecting the Segment.
23. As a researcher generating a Transcription PDF, I want generation to succeed with a bundled Greek-capable Unicode font, so that PDF review works in production (including Vercel).
24. As a researcher, when PDF generation still fails, I want a plain-language error explaining what went wrong, so that I am not stuck on “load failed.”
25. As a researcher, when any user-facing action fails, I want a clean “this failed because…” message with no stack traces, so that I understand the failure without developer noise.
26. As an operator, I want structured logs with a correlation/ref across Platform API, worker, Inference helper, and frontend failure beacons, so that I can diagnose production issues end-to-end.
27. As an operator, I want Prometheus deferred to a later self-hosted phase, so that this batch ships logging-first observability without blocking on metrics infra.
28. As a developer running OCR tests, I want Syriac fixtures paired with `syriac-calamari-v1` (live, not mocked), so that CI exercises real transcription without exposing unfinished Greek models.
29. As a developer, I want manuscript fixtures under `manuscripts/{script}/` folders, so that Script and model always match.
30. As a developer, I want no special `folio.webp` artefact and no `folio*` upload stems in tests/code, so that naming matches normal Document-part WebP storage.
31. As an operator, I want orphaned or test-named objects cleaned from storage buckets, so that production storage is not littered with misleading `folio.webp`-style objects.
32. As a developer, I want job SSE heartbeats about every 45s with clients ignoring them and polling as fallback, so that long job watches remain reliable on Vercel.
33. As a maintainer, I want Quality, Security, and Deployment and Images workflows to actually run on every pull request, so that merges are gated by real checks, not only workflow file definitions.
34. As a maintainer, I want obsolete open PRs #41, #23, and #27 closed and stale remote `issue/*` branches deleted, so that GitHub hygiene matches current `main` architecture.
35. As a maintainer, I want new work branched only from current `main` (no rebase of old `annote/`/`ml/` stacks), so that we do not revive obsolete architecture.

## Implementation Decisions

- **Loading UX (app-wide):** Keep surrounding chrome. Spinner only in the waiting content region. Never empty/error copy while fetch in flight.
- **Session restore:** Valid session cookie → refresh in-memory access token → same route. No login flash. ADR 0001.
- **Helper download:** Always GitHub `releases/latest`; OS-detect primary CTA; auto-detect helper.
- **Helper weight prefetch:** On install/first launch for local-eligible weights; not mid-OCR.
- **Greek model gating:** Keep `greek-calamari-v1` out until Hub revision + artifact SHA.
- **Job panel:** Bottom-left; Cancel per active job; discard partials on cancel; auto-dismiss toasts.
- **SSE heartbeats:** ~45s; clients ignore; polling fallback (ADR 0002). Done.
- **Annotation edit:** Vertex delete/add, confirm Segment delete, no pan-while-drag, Escape commit+deselect, Edit undo/redo.
- **Pairing strip:** One field + Save → Ground truth.
- **Transcription PDF:** Bundled Unicode font; plain-language failures.
- **Observability:** Logging-first + correlation/ref + frontend beacon; Prometheus later.
- **Fixtures / folio:** Script folders; drop `folio*`; live Syriac OCR tests.
- **CI:** Quality, Security, Deployment on every PR.
- **GitHub hygiene:** Close #41/#23/#27; delete stale `issue/*` branches.

## Testing Decisions

- External behavior only; highest existing seams; live fixtures/models for OCR; no mocked OCR.
- Seams: public loading, session restore, jobs/cancel/toasts/SSE, annotation+pairing, PDF font, helper download/prefetch, errors/observability, fixtures/folio, CI, GitHub hygiene.

## Out of Scope

- Broad codebase green-up beyond this batch; helper UI beyond download/prefetch; ONNX/Core ML; re-enable Greek before Hub-ready; Prometheus; rebasing obsolete stacks.

## Further Notes

- Ship as one consolidated PR from `feat/bugfix-polish-batch`.
- Prod evidence: PDF font 500; SSE timeouts (mitigated).
