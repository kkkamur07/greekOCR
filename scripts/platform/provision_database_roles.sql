-- One-time database-role bootstrap for a provider/operator connection.
--
-- Run with psql (or the provider SQL console) before deploying the runtime
-- services. This script creates NOLOGIN groups only; create provider-managed
-- LOGIN users separately and assign each one exactly one group. It contains no
-- passwords and does not change the current schema owner.

DO $$
DECLARE
  role_name text;
BEGIN
  FOREACH role_name IN ARRAY ARRAY[
    'nomicous_migrator',
    'nomicous_api',
    'nomicous_platform_worker',
    'nomicous_inference_worker'
  ]
  LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
      EXECUTE format(
        'CREATE ROLE %I NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION',
        role_name
      );
    END IF;
  END LOOP;
END
$$;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;

GRANT USAGE, CREATE ON SCHEMA public TO nomicous_migrator;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nomicous_migrator;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nomicous_migrator;
GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_migrator;

GRANT USAGE ON SCHEMA public TO nomicous_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nomicous_api;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nomicous_api;
GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_api;

GRANT USAGE ON SCHEMA public TO nomicous_platform_worker;
GRANT SELECT, UPDATE ON TABLE jobs TO nomicous_platform_worker;
GRANT SELECT ON TABLE documents, document_parts, blocks, lines, model_bindings, inference_models
  TO nomicous_platform_worker;
GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_platform_worker;

GRANT USAGE ON SCHEMA public TO nomicous_inference_worker;
GRANT SELECT, UPDATE ON TABLE inference_jobs TO nomicous_inference_worker;
GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_inference_worker;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT ALL PRIVILEGES ON TABLES TO nomicous_migrator;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT ALL PRIVILEGES ON SEQUENCES TO nomicous_migrator;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE ON TYPES TO nomicous_migrator;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO nomicous_api;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO nomicous_api;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE ON TYPES TO nomicous_api;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  REVOKE ALL ON TYPES FROM PUBLIC;
