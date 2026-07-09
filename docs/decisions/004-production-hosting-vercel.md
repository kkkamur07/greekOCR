# Architecture Decision Record: Production hosting on Vercel

**Status:** Accepted  
**Date:** 2026-07-09  
**Supersedes:** —  
**Related:** [003](003-supabase-hosted-postgres-and-storage.md), [002](002-local-inference-helper.md), [`docs/deployment/production.md`](../deployment/production.md)

## Context

Nomicous is moving to production at:

- `nomicous.com` — marketing landing page
- `app.nomicous.com` — annotation SPA
- `api.nomicous.com` — platform REST API
- `inference.nomicous.com` — cloud OCR / segmentation

The team prefers **Vercel** for hosting and already uses **Supabase** for Postgres + Storage.

## Problem

The monorepo contains static assets, a Vite SPA, a FastAPI platform API with background workers and SSE, and a PyTorch-based inference stack with a separate job worker. Not every component fits Vercel’s serverless model.

## Decision

| Surface | Host | Rationale |
|---------|------|-----------|
| Landing | Vercel static (`landing/`) | Pure HTML/CSS; zero server logic |
| App SPA | Vercel (`nomicous/frontend/`) | Vite build + SPA rewrites |
| Platform API | Vercel Python (`deploy/platform/`) | Request/response API; Supabase for state |
| Inference API + workers | **Docker on Railway/Fly** | PyTorch size, long job runtime, Postgres `LISTEN` |
| Platform job worker | **Same persistent host as inference** | Must poll/claim jobs continuously |
| Database + page images | Supabase | Already integrated |
| Local OCR | Inference Helper on user machine | Browser → `localhost:8001`, results via platform API |

### Serverless API constraints

On the Vercel platform API deployment:

- `JOB_WORKER_ENABLED=false` — dispatch runs on `platform-worker`
- `JOB_SSE_NOTIFICATIONS_ENABLED=false` — NOTIFY listener needs a persistent process
- `BEHIND_PROXY=true` — Vercel terminates TLS
- Frontend **polls** job status when SSE is unavailable (existing fallback)

### Domain map

```text
nomicous.com      → Vercel (landing)
app.nomicous.com  → Vercel (frontend)  → api.nomicous.com
api.nomicous.com  → Vercel (FastAPI)   → Supabase
inference.*       → Railway/Fly        → Supabase + callback → api.nomicous.com
localhost:8001    → Inference Helper (optional, user machine)
```

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| All services on Vercel | Inference bundle exceeds size limits; jobs exceed timeout; workers impossible |
| All services on one VM | Works, but team wants Vercel for web surfaces |
| Supabase Edge Functions for API | Would rewrite auth, jobs, and media layers |
| Serverless framework for workers | Adds complexity; Docker worker is already implemented |

## Consequences

**Positive**

- Fast deploys for landing, app, and API from Git
- Supabase eliminates filesystem coupling on the API
- Local helper path unchanged for power users
- Clear split: Vercel for HTTP surfaces, persistent host for ML

**Negative**

- Two hosting providers (Vercel + Railway/Fly) plus Supabase
- Job SSE disabled on serverless API (polling only)
- Platform worker is an extra process to monitor
- Inference weights cache needs persistent volume on inference host

## Implementation checklist

- [x] `landing/vercel.json`, `nomicous/frontend/vercel.json`
- [x] `deploy/platform/` Vercel FastAPI bundle + `build.sh`
- [x] `JOB_SSE_NOTIFICATIONS_ENABLED` setting
- [x] `backend.jobs.worker_main` standalone worker entry
- [x] `docs/deployment/production.md` runbook
- [x] `.env.production.example` for API and frontend
- [ ] Create Vercel projects and attach domains
- [ ] Deploy inference + workers to Railway/Fly
- [ ] Production smoke test (auth, upload, cloud job, helper modal)
