# ADR-002: Local inference via Inference Helper

## Status

**Accepted** (2026-07-09) — implemented in issues 038–041.

## Summary

Researchers use a **hosted** web app. OCR and segmentation can run on **their laptop CPU** instead of our servers if they install a small background app once — the **Inference Helper**. The browser talks to the helper on `localhost`, then saves results through the normal hosted API. If the helper is missing or the user prefers it, **cloud inference** (existing job queue) still works.

---

## Context

### The problem

Today, segment and transcribe jobs run on **remote inference** servers. That works, but:

- We pay for CPU the researcher already has.
- Every run adds queue + network latency.
- Model weights live on our servers, not on the researcher's machine.

Researchers need a one-time install path — no Docker, no terminal — that lets the hosted editor use local CPU when a model allows it.

### The constraint

A browser **cannot** start a Python process on the user's machine. Something must be installed once. After that, the **browser** can call it over `http://127.0.0.1`.

Our cloud API also **cannot** call `localhost` on a researcher's laptop. So local inference cannot use the same job-queue path as cloud jobs — the browser has to orchestrate it.

---

## Decision

Ship an **Inference Helper**: a slim native background service (Windows, macOS, Linux) that exposes:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Lets the browser detect if the helper is running |
| `GET /inference/v1/catalog` | Lists models and `host_eligibility` |
| `POST /inference/v1/run` | Runs segment or transcribe synchronously |

The hosted SPA probes health, runs inference locally when eligible, then **persists** results via new authenticated API routes (same Postgres data as cloud jobs).

**Default:** use local inference when the helper is healthy and the model is `host_eligibility: local`.  
**Fallback:** user toggles **"Use cloud inference"**, or has no helper installed → existing remote job path.

---

## How it works

```
┌─────────────┐     HTTPS      ┌──────────────┐     ┌──────────┐
│   Browser   │ ─────────────► │  Hosted API  │ ──► │ Postgres │
│  (hosted    │   persist      │  + auth JWT  │     │          │
│   SPA)      │ ◄───────────── │              │     └──────────┘
└──────┬──────┘                └──────────────┘
       │
       │ HTTP (localhost only)
       ▼
┌─────────────────────┐
│  Inference Helper   │  127.0.0.1:8001
│  Calamari + Kraken  │  weights → ~/.nomicous/hf/cache/
└─────────────────────┘
```

**Cloud path (unchanged):** browser enqueues a **product job** → inference worker → webhook callback → merge into Postgres.

**Local path (new):** browser → helper `/run` → browser → `POST …/local-inference/{transcribe,segment}` → merge into Postgres. No product job is created for the inference step itself.

---

## Nuances worth getting right

| Topic | What it means |
|-------|----------------|
| **`host_eligibility`** | Where a model *may* run: `local`, `remote`, or `any`. Declared per model in `inference/registry.yaml`. |
| **`device: cpu \| cuda`** | Hardware hint only — **not** the same as "run on laptop." A `cuda` model can still be `host_eligibility: remote`. |
| **Registry sync** | Deployed API serves `GET /inference/v1/registry`. Helper fetches it on startup so **new models do not require reinstalling the helper**. Weights still download on first use. |
| **Helper is slim** | No Postgres, no async job router, no platform code — only health, catalog, and sync `/run`. |
| **Always-on (v1)** | Helper auto-starts after install (LaunchAgent / Scheduled Task / systemd). Idle model unload deferred. |
| **CORS** | Helper allows the hosted SPA origin (and dev origins). Set `HELPER_CORS_ORIGINS` at build time for production. |
| **Bind address** | `127.0.0.1` only — not exposed on the LAN. |

---

## Security (v1)

Helper `/run` has **no auth token** in v1. Rationale: open-source stack, inference-only, localhost-only. Any local process could already call it.

**Accepted risk:** shared/library machines where another user on the same OS could hit `localhost:8001`.

**Deferred:** browser ↔ helper token handshake after login (v1.1).

Persist endpoints on the hosted API **do** require normal user JWT — only logged-in researchers write results.

---

## Alternatives considered

| Option | Why not (for v1) |
|--------|------------------|
| **Desktop app (Tauri/Electron)** | Duplicates the hosted SPA for no gain. |
| **Cloud inference only** | Ignores researcher CPU; higher server cost. |
| **In-browser WASM / ONNX** | Cannot reuse existing Calamari / Kraken / PyTorch stack. |
| **Cloud pushes jobs to laptop** | Unified queue, but overkill — API still cannot reach localhost without a persistent agent protocol. |
| **Transcribe-only helper** | Rejected — all `host_eligibility: local` models (segment + transcribe) run locally from day one. |

---

## Consequences

- **Larger install** than transcribe-only: helper bundles Calamari transcribe **and** Kraken segment.
- **Two orchestration paths** to maintain: browser-local vs cloud job queue. Persist logic must stay aligned (same merge rules, same layer shapes).
- **New API surface:** `POST …/local-inference/transcribe` and `…/segment` (authenticated).
- **Registry schema change:** `host_eligibility` on every model entry.
- **Cache location:** `HF_CACHE_ROOT=~/.nomicous/hf/cache/` for Hub weights.
- **Packaging:** PyInstaller per OS; see [`packaging/helper/README.md`](../../packaging/helper/README.md).

---

## Implementation checklist

- [x] `inference/helper/` slim app (`health`, catalog, `/run`, CORS, `127.0.0.1`)
- [x] `host_eligibility` on registry entries + catalog endpoint
- [x] Registry sync from hosted `GET /inference/v1/registry`
- [x] Frontend: health probe, local run (segment + transcribe), cloud toggle
- [x] Backend: persist local inference results (JWT-protected)
- [x] Packaging: macOS, Windows, Linux installers with auto-start

---

## See also

- [Launch readiness](../local-inference-helper-launch-readiness.md) — rollout, blockers, go/no-go
- [Adding an inference model](../adding-inference-models.md) — registry, Hub, deploy, helper sync
- [`inference/CONTEXT.md`](../../inference/CONTEXT.md) — registry vocabulary
- [`nomicous/CONTEXT.md`](../../nomicous/CONTEXT.md) — platform domain terms
