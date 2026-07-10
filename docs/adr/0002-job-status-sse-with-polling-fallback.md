# Job status delivery: SSE with polling fallback

## Status

Accepted

## Context

Job-status updates are useful immediately, but the platform API is deployed as a
Vercel serverless function in production. It cannot keep the Postgres `LISTEN`
connection required for server-side notification fan-out. The browser must still
show terminal job states when a stream is unavailable or interrupted.

## Decision

Use Server-Sent Events (SSE) when `JOB_SSE_NOTIFICATIONS_ENABLED=true`. A
persistent API process listens for Postgres `NOTIFY` events and fans them out on
`GET /jobs/{job_id}/events`.

The frontend treats SSE as an optimization, not a correctness dependency. It
falls back to authenticated `GET /jobs/{job_id}` polling when the stream cannot
open, closes, or becomes idle. Vercel deployments set
`JOB_SSE_NOTIFICATIONS_ENABLED=false`; persistent Docker deployments can enable
it.

## Consequences

- Job completion remains visible on both serverless and persistent deployments.
- Persistent deployments reduce unchanged polling requests while a job runs.
- SSE clients and their fallback polling loop require test coverage.
