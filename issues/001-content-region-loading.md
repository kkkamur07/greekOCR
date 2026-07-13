---
id: "001"
title: "content-region-loading"
type: AFK
status: review
blocked_by: []
parent_prd: "issues/prd.md"
---

## Parent

[issues/prd.md](prd.md)

## What to build

App-wide loading rule: keep chrome (nav, public shell, editor chrome); show a spinner only in the content region that is waiting. Never show empty-state or error copy while that region’s fetch is in flight. Canonical path: public Document view must not flash “Document not available” then come alive — apply the same pattern elsewhere content is fetched.

## Acceptance criteria

- [x] Public Document view keeps chrome visible and spins only inside the document content region while loading
- [x] Error / empty copy appears only after a real failed or empty result (not during in-flight fetch)
- [x] Same chrome-stays / content-spins rule applied for other primary content fetches in the app shell
- [x] Tests cover public view loading behavior at the existing PublicDocumentPage / public access seam

## Blocked by

None - can start immediately
