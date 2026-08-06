"""Static regressions for deployment and database-privilege hardening."""

from __future__ import annotations

import re
import tomllib

import yaml

from tests.fixtures.paths import REPO_ROOT


def _workflow_body(name: str) -> str:
    """Parse a workflow and render it back without comments.

    Asserting on raw workflow text makes a comment *about* a deleted mechanism
    indistinguishable from the mechanism itself. Round-tripping through the YAML
    parser drops comments, and failing to parse is itself the assertion: a
    workflow that does not load is a broken CI run nobody sees until release day.
    """
    document = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text("utf-8"))
    assert isinstance(document, dict), f"{name} did not parse as a workflow document"
    return yaml.safe_dump(document, default_flow_style=False)


def _flatten_group(groups: dict[str, list], name: str) -> list[str]:
    """Resolve a PEP 735 dependency group, following `include-group` entries."""
    resolved: list[str] = []
    for entry in groups[name]:
        if isinstance(entry, dict):
            resolved.extend(_flatten_group(groups, entry["include-group"]))
        else:
            resolved.append(entry)
    return resolved


def test_published_package_ships_the_onnx_runtime_and_nothing_else() -> None:
    """ADR 0006: one runtime, CPU only, and no training stack in the wheel.

    This used to read `packaging/helper/pyinstaller.spec` and check its hidden
    imports and excludes, because the frozen installer decided by hand what
    reached a laptop. The **published package** decides it by construction:
    `[project].dependencies` *is* the closure that reaches a researcher, so
    that is what this holds.

    ADR 0004 put Torch here and this test held it. ADR 0006 reversed that: the
    runtime is ONNX Runtime again and Torch only builds the artifact, so Torch
    and `safetensors` must now be *absent* from the published closure. That is
    the whole point of #65 - a plain `pip install` pulling 4.8 GB of CUDA wheels
    is unreachable if Torch is not in the closure at all.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = pyproject["project"]["dependencies"]
    assert any(dependency.startswith("onnxruntime") for dependency in project_dependencies)
    assert not any(dependency.startswith("torch") for dependency in project_dependencies)
    assert not any(dependency.startswith("safetensors") for dependency in project_dependencies)
    # Nothing that trains a model is part of what runs one.
    for forbidden in ("transformers", "accelerate", "torchvision", "kraken"):
        assert not any(dependency.startswith(forbidden) for dependency in project_dependencies), (
            f"{forbidden} reaches every researcher who installs the package"
        )

    groups = pyproject["dependency-groups"]
    assert "parity" not in groups, "the kraken parity group outlived its second runtime"
    # ADR 0006 keeps Torch on a maintainer's machine only, behind `--group export`,
    # which is what exports the `.onnx` artifact the runtime then loads.
    assert any(dependency.startswith("torch") for dependency in _flatten_group(groups, "export"))
    assert not any(
        "kraken" in dependency
        for group in groups.values()
        for dependency in group
        if isinstance(dependency, str)
    )
    # The `helper` group died with `inference/api` and `inference/helper` (#60).
    # It named what the loopback HTTP surfaces needed on top of the runtime, and
    # there is no longer anything under `inference/` that serves HTTP.
    assert "helper" not in groups
    assert not any(
        dependency.startswith("onnxruntime") for dependency in _flatten_group(groups, "inference")
    )

    # CUDA wheels must not be reachable: PyPI serves them on Linux and Windows.
    sources = pyproject["tool"]["uv"]["sources"]["torch"]
    markers = {entry["marker"] for entry in sources if entry["index"] == "pytorch-cpu"}
    assert markers == {"sys_platform == 'linux' or sys_platform == 'win32'"}


def test_runtime_images_are_non_root_and_have_import_and_health_checks() -> None:
    dockerfile = (REPO_ROOT / "nomicous" / "Dockerfile").read_text(encoding="utf-8")

    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "from backend.core.main import app" in dockerfile


def test_runtime_images_uninstall_vulnerable_system_packaging_tools() -> None:
    dockerfile = (REPO_ROOT / "nomicous" / "Dockerfile").read_text(encoding="utf-8")
    assert "pip uninstall -y pip setuptools wheel" in dockerfile


def test_development_compose_ports_are_loopback_only_and_secrets_are_interpolated() -> None:
    """README tells every evaluator to run this file; the bindings are the blast radius.

    Widening any of these to 0.0.0.0 puts an unauthenticated Postgres and the
    platform API on whatever network the researcher's laptop is attached to, and
    dropping the `:?` interpolation re-introduces a committed default password.
    """
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for mapping in (
        '"127.0.0.1:5433:5432"',
        '"127.0.0.1:8000:8000"',
        '"127.0.0.1:5173:5173"',
    ):
        assert mapping in compose
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?" in compose
    assert "postgres:dev@" not in compose
    assert 'CLOUD_INFERENCE_ENABLED: "true"' in compose
    assert compose.count("\n      INFERENCE_WEBHOOK_SECRET:") == 1
    assert "INFERENCE_SERVICE_SECRET" not in compose


def test_role_migration_defines_service_boundaries_without_passwords() -> None:
    migration = (
        REPO_ROOT / "nomicous" / "infrastructure" / "alembic" / "versions" / "002_service_roles.py"
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
    assert "GRANT SELECT, UPDATE ON TABLE jobs TO nomicous_platform_worker" in migration


def test_inference_worker_role_reaches_nothing_after_the_queue_collapse() -> None:
    """ADR 0003 left the group with no table to read.

    The squashed chain expresses that as an absence rather than as a grant that a
    later revision revokes: the group is still created so 002 and the bootstrap
    script find all four, and nothing is ever granted to it.
    """
    migration = (
        REPO_ROOT / "nomicous" / "infrastructure" / "alembic" / "versions" / "002_service_roles.py"
    ).read_text(encoding="utf-8")
    bootstrap = (REPO_ROOT / "scripts" / "platform" / "provision_database_roles.sql").read_text(
        encoding="utf-8"
    )

    # Created, so the "all four groups exist" check in 002 passes.
    assert "nomicous_inference_worker" in migration
    for source in (migration, bootstrap):
        assert not [
            line
            for line in source.splitlines()
            if "nomicous_inference_worker" in line and line.lstrip().startswith("GRANT")
        ]


def test_platform_backend_ships_bundled_unicode_pdf_font() -> None:
    font = REPO_ROOT / "nomicous" / "backend" / "core" / "assets" / "fonts" / "NotoSans-Regular.ttf"
    assert font.is_file()
    assert font.stat().st_size > 100_000

    fonts_module = (REPO_ROOT / "nomicous" / "backend" / "core" / "fonts.py").read_text(
        encoding="utf-8"
    )
    assert "assets" in fonts_module
    assert "NotoSans-Regular.ttf" in fonts_module

    # deploy/platform/build.sh copytree of nomicous/backend includes assets/fonts.
    build_script = (REPO_ROOT / "deploy" / "platform" / "build.sh").read_text(encoding="utf-8")
    assert '"nomicous" / "backend"' in build_script


def test_vercel_frontend_connect_src_permits_no_loopback_origin() -> None:
    """`connect-src` reaches the app and the API, and nothing on the laptop.

    ADR 0002 deleted the browser-to-loopback call (#60), so the entry that used
    to permit `http://127.0.0.1:8001` permits nothing that exists. It is
    asserted absent rather than merely narrowed: a hosted HTTPS page calling
    `127.0.0.1` is the fragility the redesign removed, and a grant left behind
    is the thing a future change would build on.
    """
    vercel = (REPO_ROOT / "nomicous" / "frontend" / "vercel.json").read_text(encoding="utf-8")

    assert "connect-src 'self' https://api.nomicous.com;" in vercel
    for origin in (
        "127.0.0.1",
        "localhost",
        "[::1]",
    ):
        assert origin not in vercel


def test_landing_csp_uses_json_ld_hash_instead_of_unsafe_inline() -> None:
    import base64
    import hashlib
    import re

    html = (REPO_ROOT / "landing" / "index.html").read_text(encoding="utf-8")
    vercel = (REPO_ROOT / "landing" / "vercel.json").read_text(encoding="utf-8")
    match = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
    assert match is not None
    digest = base64.b64encode(hashlib.sha256(match.group(1).encode("utf-8")).digest()).decode()
    assert f"'sha256-{digest}'" in vercel
    assert "'unsafe-inline'" not in vercel


def test_workflows_pin_every_action_to_a_commit_sha() -> None:
    """A floating tag in a job holding `contents: write` is a release-signing risk."""
    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows

    floating: list[str] = []
    for workflow in workflows:
        for line in workflow.read_text(encoding="utf-8").splitlines():
            match = re.search(r"uses:\s*([^\s@]+)@([^\s]+)", line)
            if match and not re.fullmatch(r"[0-9a-f]{40}", match.group(2)):
                floating.append(f"{workflow.name}: {match.group(0)}")
    assert not floating, f"unpinned action references: {floating}"


def test_release_publishes_with_no_long_lived_credential() -> None:
    """Publishing must not reintroduce a secret that has to be rotated.

    The signing pipelines this replaced held five repository secrets between
    them plus a GPG manifest key, all long-lived, all revocable only by hand.
    Trusted Publishing mints a token per run from the workflow's OIDC identity,
    so there is nothing in repository settings to leak.
    """
    body = _workflow_body("release.yml")

    assert "--trusted-publishing always" in body
    assert "id-token: write" in body
    assert "secrets." not in body, "the release path grew a long-lived credential again"

    # Provenance attestation must not be skippable by flipping repo visibility.
    assert "!github.event.repository.private" not in body

    # The scan input must be the closure that actually reaches a researcher.
    assert "scan-ref: release-scan" in body
    assert "uv export --locked --no-default-groups --no-hashes" in body
    assert re.search(r"^\s+path: release-scan\s*$", body, re.M) is not None
