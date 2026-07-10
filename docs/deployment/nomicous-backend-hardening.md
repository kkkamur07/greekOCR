# Nomicous backend hardening

Migration `022_public_boundary_state_integrity` must run before deploying this
backend revision.

- Public API failures return stable `{ "error": { "code", "message" } }`
  payloads. They do not include Pydantic details, exception strings, callback
  payloads, or job-handler exception text. The `X-Error-ID` response header
  identifies the corresponding server-side log event.
- Pagination cursors are URL-safe base64 JSON, capped at 1024 characters.
  Malformed or oversized values return the stable `422 VALIDATION_ERROR`
  response.
- Inference callbacks must match the persisted inference job ID exactly and
  may only claim a `waiting` product job. A durable callback claim prevents a
  replay from merging output twice; terminal replays are idempotent.
- `document_parts` now has a database uniqueness constraint on
  `(document_id, order)`. Upload allocation locks the parent document and
  reordering uses a temporary sequence inside one transaction.
- Part and document deletion commits a `media_deletion_intents` outbox record
  with the relational deletion. The application retries pending object-store
  deletes asynchronously on startup and every minute. Failed upload
  compensation also records an intent when its immediate delete fails.
- Authentication throttling serializes each client/path window with a
  transaction-scoped PostgreSQL advisory lock. Passwords exceeding 72 UTF-8
  bytes are rejected before bcrypt is invoked.

The media intent table is intentionally retained after a successful delete for
auditability. Completed intents are not retried.
