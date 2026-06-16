"""Relocated platform smoke tests that do not require Postgres."""

from backend.core.app import create_app


def test_relocated_platform_app_imports_with_expected_surfaces() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/health" in paths
    assert "/auth/register" in paths
    assert "/projects" in paths
    assert any(path.startswith("/projects/{project_id}/documents") for path in paths)
    assert any(path.startswith("/media/parts") for path in paths)
    assert "/openapi.json" in paths
