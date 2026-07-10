"""Static regressions for deployment and database-privilege hardening."""

from __future__ import annotations

from tests.fixtures.paths import REPO_ROOT


def test_inference_image_reincludes_only_required_hub_resolver_sources() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "inference" / "Dockerfile").read_text(encoding="utf-8")

    assert "!src/" in dockerignore
    assert "!src/hf/resolve/**" in dockerignore
    assert "COPY src/hf/__init__.py src/hf/paths.py /app/src/hf/" in dockerfile
    assert "COPY src/hf/resolve /app/src/hf/resolve" in dockerfile
    assert "COPY src/hf /app/src/hf" not in dockerfile


def test_runtime_images_are_non_root_and_have_import_and_health_checks() -> None:
    platform_dockerfile = (REPO_ROOT / "nomicous" / "Dockerfile").read_text(encoding="utf-8")
    inference_dockerfile = (REPO_ROOT / "inference" / "Dockerfile").read_text(encoding="utf-8")

    for dockerfile, import_surface in (
        (platform_dockerfile, "from backend.core.main import app"),
        (inference_dockerfile, "from inference.api.main import app"),
    ):
        assert "USER appuser" in dockerfile
        assert "HEALTHCHECK" in dockerfile
        assert import_surface in dockerfile


def test_development_compose_ports_are_loopback_only_and_secrets_are_interpolated() -> None:
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for mapping in (
        '"127.0.0.1:5433:5432"',
        '"127.0.0.1:8000:8000"',
        '"127.0.0.1:8010:8001"',
        '"127.0.0.1:5173:5173"',
    ):
        assert mapping in compose
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?" in compose
    assert "postgres:dev@" not in compose


def test_role_migration_defines_service_boundaries_without_passwords() -> None:
    migration = (
        REPO_ROOT
        / "nomicous"
        / "infrastructure"
        / "alembic"
        / "versions"
        / "025_service_database_roles.py"
    ).read_text(encoding="utf-8")

    for role in (
        "nomicous_migrator",
        "nomicous_api",
        "nomicous_platform_worker",
        "nomicous_inference_worker",
    ):
        assert role in migration
    assert "NOLOGIN" in migration
    assert "PASSWORD" not in migration
    assert "GRANT SELECT, UPDATE ON TABLE inference_jobs TO nomicous_inference_worker" in migration
    assert "GRANT SELECT, UPDATE ON TABLE jobs TO nomicous_platform_worker" in migration


def test_release_workflow_refuses_asset_replacement_and_generates_evidence() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "release-helper.yml").read_text(
        encoding="utf-8"
    )

    assert "--clobber" not in workflow
    assert "SHA256SUMS" in workflow
    assert "actions/attest-build-provenance@" in workflow
    assert "anchore/sbom-action@e22c389904149dbc22b58101806040fa8d37a610" in workflow
    assert "aquasecurity/trivy-action@57a97c7e7821a5776cebc9bb87c984fa69cba8f1" in workflow
    assert "overwrite: false" in workflow
