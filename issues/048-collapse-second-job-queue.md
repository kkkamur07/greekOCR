---
id: "048"
title: "collapse-second-job-queue"
type: AFK
status: in_progress
tracker: "https://github.com/kkkamur07/greekOCR/issues/48"
blocked_by: []
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Delete the second job queue. Today a cloud job crosses two databases and two workers over seven steps, while the path a laptop takes is four steps across one database. Remove the inference service's own queue so there is one queue, owned by the platform.

Cloud inference is currently off, so this is a deletion rather than a migration — no live path to preserve, no cutover, no compatibility window. It is the first slice deliberately: it shrinks the surface every later slice touches.

What goes: the `inference_jobs` table and its repository, the inference service's database and settings layer, its jobs API, the queue half of its worker, its queue-admission advisory locks, and the HTTP hop on the platform that fed it. `psycopg2` and `sqlalchemy` leave the inference dependency group, and the `inference-api` container is removed — the registry endpoint the agent syncs from is served by the platform.

The synchronous run path and the model execution code stay. Only the queue goes.

Per ADR 0003.

## Acceptance criteria

- [ ] No `inference_jobs` table, repository, or migration path remains
- [ ] The inference dependency group installs with no Postgres driver or ORM
- [ ] The `inference-api` container and its compose/deployment entries are gone
- [ ] The platform no longer submits jobs over HTTP to the inference service
- [ ] Tests covering the deleted queue are deleted, not migrated
- [ ] Full platform test suite green against live Postgres
- [ ] Model execution through the synchronous run path still works against real ONNX artifacts

## Blocked by

None - can start immediately
