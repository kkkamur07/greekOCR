# Catalog SQL changes

One SQL file per production catalog change, named `<date>-<what>.sql`.

Rules:

- Every file is idempotent (safe to run twice) and wrapped in `begin;`/`commit;`.
- Raw inserts must pass `gen_random_uuid()` for `id` (the PK has an app-side default only).
- Display name lives in `name`; dispatch reads `artifact_ref` (`registry://<registry id>?tag=stable`), so names are free to change while registry ids stay fixed.

Apply with `scripts/platform/apply_catalog_sql.sh`:

```bash
bash scripts/platform/apply_catalog_sql.sh scripts/platform/catalog/<file>.sql
bash scripts/platform/apply_catalog_sql.sh scripts/platform/catalog/<file>.sql --apply
```

The first run only looks (shows the database, the current rows, the SQL).
The second run sends the file over stdin and shows the rows again.
