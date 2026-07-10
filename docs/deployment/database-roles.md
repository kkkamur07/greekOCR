# Database service roles

Nomicous uses four PostgreSQL **NOLOGIN group roles**. Provider-managed login
principals receive one group role each; their connection URLs live only in the
provider's secret store. No role password belongs in Git, an Alembic revision,
or a Docker image.

| Group role | Used by | Database scope |
|---|---|---|
| `nomicous_migrator` | one-off Alembic operator | schema usage/create plus application tables, sequences, and types |
| `nomicous_api` | platform API | application-table CRUD, sequence usage, and type usage |
| `nomicous_platform_worker` | persistent platform worker | claim/update `jobs`; read job-input tables |
| `nomicous_inference_worker` | inference API and worker | read/update `inference_jobs` only |

The platform API does not need the migrator connection at runtime. The
inference containers do not receive the API, storage, JWT, or migration
credentials.

## Bootstrap

1. Open a SQL session with a provider/operator principal that has `CREATEROLE`
   and owns the `public` schema objects.
2. Apply `scripts/platform/provision_database_roles.sql`.
3. Create or select provider-managed LOGIN principals. The exact command and
   password lifecycle is provider-specific; do not create password-bearing
   `CREATE ROLE ... LOGIN PASSWORD ...` statements in this repository.
4. Grant exactly one service group to each LOGIN principal. For example:

   ```sql
   GRANT nomicous_api TO <provider-managed-api-login>;
   GRANT nomicous_platform_worker TO <provider-managed-platform-worker-login>;
   GRANT nomicous_inference_worker TO <provider-managed-inference-worker-login>;
   GRANT nomicous_migrator TO <provider-managed-migrator-login>;
   ```

5. Store the four connection URLs in the appropriate provider services:
   - Alembic runner: `MIGRATOR_DATABASE_URL`
   - platform API: `DATABASE_URL`, `SYNC_DATABASE_URL`
   - platform worker: `DATABASE_URL`, `SYNC_DATABASE_URL`
   - inference API and worker: `INFERENCE_DATABASE_URL`
6. Run `alembic upgrade head` through the migrator/operator connection.

Migration `025_service_roles` performs the same grants when the current
operator can create roles. On providers that forbid `CREATEROLE`, it emits a
notice and leaves the existing operator setup running; apply the bootstrap
script once to enforce the policy. It intentionally does not transfer table
ownership or revoke the provider operator, because both can break Supabase
maintenance and future DDL.

## Supabase operator notes

Use the Database dashboard or a direct operator connection for bootstrap and
migrations. Keep the Data API disabled for this application, or keep `public`
tables unexposed: these group roles are for server-side SQLAlchemy processes,
not browser/PostgREST access.

Supabase may restrict user and password creation. In that case, create the
available managed database users in the dashboard, grant the group roles there,
and place each generated URI directly in the corresponding service secret.
Keep the Supabase `postgres` operator URI only for bootstrap/migrations, never
in runtime API or worker configuration.

## Verification and rollback

Run these checks as the operator after rollout:

```sql
SELECT rolname, rolcanlogin, rolsuper, rolcreaterole
FROM pg_roles
WHERE rolname LIKE 'nomicous_%'
ORDER BY rolname;

SELECT grantee, privilege_type
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND grantee LIKE 'nomicous_%'
ORDER BY grantee, table_name, privilege_type;
```

Smoke-test each service with its own runtime URL. A platform worker should
claim a job, and an inference worker should see and update `inference_jobs`;
neither should be able to create tables or read unrelated runtime tables.

`alembic downgrade -1` removes grants but preserves the group roles and login
memberships. Remove memberships and roles manually only after every service has
been moved to a replacement role.
