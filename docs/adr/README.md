# Architecture decision records (ADRs)

Short, durable decisions that are hard to infer from code alone. New ADRs use
the `docs/adr/` layout (replacing the older `docs/decisions/` tree on legacy
branches).

| ADR | Status | Topic |
|-----|--------|--------|
| [0001-browser-auth-memory-cookie-csrf.md](0001-browser-auth-memory-cookie-csrf.md) | Accepted | In-memory JWT + HttpOnly refresh cookie + CSRF |
| [0002-job-status-sse-with-polling-fallback.md](0002-job-status-sse-with-polling-fallback.md) | Accepted | SSE with serverless-safe polling fallback |
| [0003-image-canvas-archival-boundary.md](0003-image-canvas-archival-boundary.md) | Accepted | ImageCanvas removal boundary |

**When to add an ADR:** auth, job transport, storage, or cross-cutting frontend
changes that affect multiple services.

Back to [docs index](../README.md).
