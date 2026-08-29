# Infrastructure

Everything that describes how this repository is *run and shipped*, rather than
what it computes. Before this directory existed these files sat loose at the
repository root; nothing about their behaviour changed in the move, but every
path inside them is now relative to this folder.

## Directory Map

```text
infrastructure/
  docker-compose.yml           # development stack: db, api, platform-worker, frontend
  docker-compose.supabase.yml  # override: Supabase Postgres + Storage, no local db
  docker-bake.hcl              # parallel image builds with a persistent local cache
  .env.compose.example         # template for the ignored infrastructure/.env
  gitleaks.toml                # secret-scanning ruleset, shared by CI and pre-commit
  platform/                    # Vercel bundle for the FastAPI platform API
```

Two pieces of deployment configuration deliberately live elsewhere:

- `nomikos/Dockerfile` and `nomikos/frontend/Dockerfile` stay beside the code
  they build.
- `nomikos/Dockerfile.dockerignore` stays beside its Dockerfile because Docker
  only reads a `.dockerignore` from the *build context* root, which for the API
  image is the repository root. Its header explains the sidecar naming.

`nomikos/infrastructure/` is a different thing entirely: it is the app's
database and Alembic package, not deployment configuration.

## Compose

The Compose files use paths relative to *this* directory (`../nomikos`,
`../inference`, `../data`), and Compose resolves those against the file's own
location rather than your shell's. Both of these therefore work and are
equivalent:

```bash
# from the repository root
docker compose -f infrastructure/docker-compose.yml up --build

# or from here
cd infrastructure && docker compose up --build
```

The documentation uses the first form throughout, because most Compose commands
appear next to commands that must run from the repository root anyway.

The project name is pinned to `nomikos` in the file, so the container and volume
names (`nomikos-db-1`, `nomikos_postgres_data`) do not depend on which directory
you invoke Compose from.

### Environment

Compose reads `.env` from the directory holding the Compose file, so the
development credentials live in `infrastructure/.env`:

```bash
cp infrastructure/.env.compose.example infrastructure/.env
```

Then fill in `POSTGRES_PASSWORD`, `JWT_SECRET`, and `INFERENCE_WEBHOOK_SECRET`.
That file is ignored by `.gitignore` and matched by `gitleaks.toml`; the
repository has leaked a file at exactly this path once before (`e81a50c`,
carrying `DATABASE_URL` and `JWT_SECRET`), which is why both guards name it.

The Supabase profile additionally reads `nomikos/backend/core/.env.supabase`:

```bash
docker compose -f infrastructure/docker-compose.yml \
  -f infrastructure/docker-compose.supabase.yml up --build
```

See [`docs/deployment/supabase.md`](../docs/deployment/supabase.md).

## Images

```bash
docker buildx bake -f infrastructure/docker-bake.hcl
```

Bake resolves build contexts against the bake file, so `..` is the repository
root. The local cache paths (`.docker-cache/`) are resolved against your working
directory instead, so run bake from the repository root to keep the cache where
`.gitignore` expects it.

## Platform bundle

`platform/` is the Vercel deployment of the FastAPI platform API
(`api.nomikos.app`). `platform/build.sh` copies the backend and the parts of
`inference/` the API needs into `platform/nomikos/` and `platform/inference/`;
both are build artifacts and are gitignored.

Vercel's **Root Directory** project setting must point at `infrastructure/platform`.
See [`docs/deployment/vercel-platform-api.md`](../docs/deployment/vercel-platform-api.md).

## Secret scanning

`gitleaks.toml` is read by two callers, and both name it explicitly because it
is no longer at the default root path:

- `.pre-commit-config.yaml` runs `gitleaks protect --staged --config infrastructure/gitleaks.toml`
- `.github/workflows/security.yml` sets `GITLEAKS_CONFIG: infrastructure/gitleaks.toml`

If either loses its path, gitleaks falls back to the default ruleset. That does
not fail open, it fails *noisily* on a test fixture this config allowlists, so a
broken path shows up as a red CI job rather than a missed secret.
