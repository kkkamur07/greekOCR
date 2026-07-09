# Testing

Pytest layout and commands for the platform, inference service, and Hugging Face helpers.

---

## Prerequisites

| Lane | Requires |
|------|----------|
| Unit (`not integration and not ml`) | Python deps only |
| Integration | Postgres on `localhost:5433` (or env URLs from `.env`) |
| ML | Postgres + inference weights under `inference/weights/` |

**Tip:** Stop the Docker API and inference-worker before integration tests if they contend for DB advisory locks:

```bash
docker compose stop api inference-worker
```

---

## Commands

From the repository root (`uv` / `poe` tasks in `pyproject.toml`):

```bash
# Fast unit tests (no Postgres, no weights)
uv run poe test-fast

# Platform integration (Postgres)
uv run poe test-integration

# Platform ML jobs (real Kraken/Calamari — slow)
uv run poe test-ml

# Inference service tests
uv run poe test-inference

# Hugging Face cache / URI tests
uv run poe test-hf
```

Direct `pytest` examples:

```bash
uv run --group platform --group inference pytest tests/nomicous -m "not integration and not ml" -q
uv run --group platform --group inference pytest tests/nomicous -m integration -q
uv run --group platform --group inference pytest tests/nomicous -m ml -q
uv run --group platform --group inference pytest tests/inference -q
```

Frontend:

```bash
cd nomicous/frontend && npm test
```

---

## Markers

Defined in `pyproject.toml`:

| Marker | Meaning |
|--------|---------|
| `integration` | Needs Postgres |
| `ml` | Runs real model weights (slow) |

---

## Full Docker suite

With Compose running (`docker compose up`), hit the live API at `http://localhost:8000` and inference at `http://localhost:8010`. Integration tests use their own DB sessions and do not require the API container unless noted in a specific test.

For Supabase-backed local testing, see [deployment/supabase.md](../deployment/supabase.md) — same pytest commands; set `DATABASE_URL` / `SYNC_DATABASE_URL` in env or `conftest.py` overrides.

---

## Known pitfalls

| Symptom | Fix |
|---------|-----|
| Tests hang on DB lock | Stop `api` / `inference-worker` containers; terminate stale Postgres sessions |
| `DuplicatePreparedStatementError` with Supabase pooler | Handled in `nomicous/infrastructure/db.py` (`statement_cache_size=0`) |
| ML tests skip or fail | Ensure weights exist and `inference/weights/` is populated per [`inference/README.md`](../../inference/README.md) |
