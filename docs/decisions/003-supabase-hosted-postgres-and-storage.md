# ADR 003: Supabase as hosted Postgres + Storage

**Status:** Accepted  
**Date:** 2026-07-09  
**Context:** Fast shared test/staging deploy without operating our own Postgres or image volume.

## Problem

Local Docker Postgres + `MEDIA_ROOT` is fast for development but does not give a shared, always-on environment for demos, remote testing, or eventual hosted API deploys. We need a managed database and durable page image storage without adopting a full BaaS client stack.

## Decision

Use **Supabase** for:

1. **Postgres** — schema via existing **Alembic** migrations (not Supabase CLI migrations)
2. **Storage** — private bucket for document part page images behind a `MediaStore` abstraction (`local` | `supabase`)

Do **not** use Supabase Auth, Data API (PostgREST), or automatic RLS. Application JWT and FastAPI authorization remain unchanged.

## Alternatives considered

| Option | Pros | Cons | Why not |
|--------|------|------|---------|
| **Supabase full BaaS** (Auth + Data API + RLS) | Less backend code | Conflicts with existing JWT + SQLAlchemy; RLS removed in `021` | Wrong fit for current architecture |
| **Hosted Postgres only** (Neon, RDS) + S3 | Fine-grained control | Two vendors; more wiring | Supabase bundles Postgres + Storage simply |
| **Keep Docker only** | Simplest dev | No shared remote env | Does not solve test deploy |
| **Store images in Postgres bytea** | One datastore | DB bloat; bad for OCR page scans | Already rejected (`image_key` pattern) |

## Consequences

### Positive

- Same Alembic history local and remote
- `STORAGE_BACKEND` switch preserves local dev speed
- WebP normalization reduces Storage size with OCR-safe lossless default
- Secret key stays server-side; publishable key unused

### Negative / trade-offs

- **Three connection URLs** (direct migrator + pooler runtime) — operational overhead
- **Transaction pooler** requires `statement_cache_size=0` for asyncpg
- **Password URL-encoding** required for special characters
- **Migration `015–021`** create then drop RLS roles — must handle Supabase default privileges (fixed in `021`)
- Storage uploads go through API bandwidth when proxying images (acceptable for test deploy)

## Implementation checklist

- [x] `MediaStore` local + Supabase backends
- [x] `.env.supabase.example`, `migrate_supabase.sh`, `docs/deployment/supabase.md`
- [x] Settings fallback `.env` → `.env.supabase`
- [x] asyncpg `sslmode` / pooler prepared-statement fixes in `db.py`
- [x] Alembic URL encoding for `%` in passwords
- [ ] Optional: `docker-compose.supabase.yml` override (API without local `db`)
- [ ] Optional: signed URL image serving

## References

- [docs/deployment/supabase.md](../deployment/supabase.md) — operational guide and pros/cons tables
- Issue tracker: `docs/todo.md` (Supabase profile)
