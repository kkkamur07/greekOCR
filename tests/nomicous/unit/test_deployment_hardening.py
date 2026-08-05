"""Static regressions for deployment and database-privilege hardening."""

from __future__ import annotations

import re
import subprocess
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


def test_published_package_ships_the_torch_runtime_and_nothing_else() -> None:
    """ADR 0004: one runtime, CPU only, and no training stack in the wheel.

    This used to read `packaging/helper/pyinstaller.spec` and check its hidden
    imports and excludes, because the frozen installer decided by hand what
    reached a laptop. The **published package** decides it by construction:
    `[project].dependencies` *is* the closure that reaches a researcher, so
    that is what this holds.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = pyproject["project"]["dependencies"]
    assert any(dependency.startswith("torch") for dependency in project_dependencies)
    assert any(dependency.startswith("safetensors") for dependency in project_dependencies)
    assert not any(dependency.startswith("onnxruntime") for dependency in project_dependencies)
    # Nothing that trains a model is part of what runs one.
    for forbidden in ("transformers", "accelerate", "torchvision", "kraken"):
        assert not any(dependency.startswith(forbidden) for dependency in project_dependencies), (
            f"{forbidden} reaches every researcher who installs the package"
        )

    groups = pyproject["dependency-groups"]
    assert "parity" not in groups, "the kraken parity group outlived its second runtime"
    assert "export" not in groups
    assert not any(
        "kraken" in dependency
        for group in groups.values()
        for dependency in group
        if isinstance(dependency, str)
    )
    helper = _flatten_group(groups, "helper")
    assert not any(dependency.startswith("onnxruntime") for dependency in helper)

    # CUDA wheels must not be reachable: PyPI serves them on Linux and Windows.
    sources = pyproject["tool"]["uv"]["sources"]["torch"]
    markers = {entry["marker"] for entry in sources if entry["index"] == "pytorch-cpu"}
    assert markers == {"sys_platform == 'linux' or sys_platform == 'win32'"}


def test_no_tray_icon_library_is_reachable() -> None:
    """A CLI has no business growing a tray icon.

    ADR 0001 decision 5 justified this with "zero terminal use is the product
    constraint"; ADR 0002 retired that constraint and kept the assertion
    anyway. The **inference agent** is a foreground process a researcher starts
    and stops - there is no window, no menu bar item, and no daemon to
    represent in one.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "pystray" not in pyproject
    assert "pystray" not in lockfile


def test_no_native_packaging_survives() -> None:
    """A release is a PyPI publish; no per-OS artifact is built anywhere.

    Frozen installers made this project the distributor of a vendored
    dependency tree with no update channel: a CVE in `protobuf`, `Pillow`, or
    `scipy` meant a four-platform rebuild, re-sign, and re-notarize, with no way
    to make anyone install the result. Patching is now a dependency bump plus a
    **version floor** bump, which the platform enforces on the claim path. If
    any of this grows back, that property is gone with it.
    """
    # Tracked content, not directory existence: a checkout that once ran the
    # PyInstaller build still has `packaging/helper/{build,dist}` on disk, and
    # those are gitignored leftovers of the mechanism rather than the mechanism.
    # Testing `.exists()` fails on a developer's machine while passing in CI,
    # which is the wrong way round for a guard.
    tracked = subprocess.run(
        ["git", "ls-files", "packaging", ".github/workflows/release-helper.yml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == []

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "packaging" not in pyproject["dependency-groups"]
    assert not any(
        dependency.lower().startswith("pyinstaller")
        for group in pyproject["dependency-groups"].values()
        for dependency in group
        if isinstance(dependency, str)
    )

    # No workflow may build, sign, or notarize a per-OS bundle again. Nor may
    # one name a signing secret: they are being revoked, so a reference is a
    # release that fails at the point of publishing.
    forbidden = (
        "pyinstaller",
        "build-dmg",
        "build-tarball",
        "build-installer",
        "codesign",
        "notarytool",
        "signtool",
        "Get-AuthenticodeSignature",
        "MACOS_CERTIFICATE_P12",
        "MACOS_CERTIFICATE_PASSWORD",
        "MACOS_CODESIGN_IDENTITY",
        "MACOS_NOTARY_PROFILE",
        "WINDOWS_SIGNING_CERT_BASE64",
        "WINDOWS_SIGNING_CERT_PASSWORD",
        "RELEASE_SIGNING_GPG_KEY",
        "RELEASE_SIGNING_GPG_PASSPHRASE",
    )
    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows
    for workflow in workflows:
        body = _workflow_body(workflow.name)
        for token in forbidden:
            assert token not in body, f"{workflow.name} still references {token}"


def test_runtime_images_are_non_root_and_have_import_and_health_checks() -> None:
    dockerfile = (REPO_ROOT / "nomicous" / "Dockerfile").read_text(encoding="utf-8")

    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "from backend.core.main import app" in dockerfile


def test_no_inference_service_image_is_built() -> None:
    """ADR 0003 removed the inference-api container; 050 ships a package instead."""
    assert not (REPO_ROOT / "inference" / "Dockerfile").exists()

    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    supabase = (REPO_ROOT / "docker-compose.supabase.yml").read_text(encoding="utf-8")
    bake = (REPO_ROOT / "docker-bake.hcl").read_text(encoding="utf-8")

    for text in (compose, supabase, bake):
        assert "inference-api" not in text
        assert "inference-worker" not in text


def test_development_compose_ports_are_loopback_only_and_secrets_are_interpolated() -> None:
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


def test_platform_bundle_includes_contract_dependencies() -> None:
    build_script = (REPO_ROOT / "deploy" / "platform" / "build.sh").read_text(encoding="utf-8")

    assert '"inference" / "admission.py"' in build_script
    assert '"inference" / "settings.py"' in build_script


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


def test_vercel_frontend_permits_only_the_one_helper_loopback_origin() -> None:
    """`connect-src` names the single helper origin the app actually calls.

    A port wildcard let any page on the app origin reach every service listening
    on the visitor's machine, which is a far larger grant than talking to the
    helper. `nomicous/frontend/src/inference/constants.ts` builds exactly one
    URL (`HELPER_BASE_URL`), defaulting to `http://127.0.0.1:8001`, so the
    wildcards protected nothing that is reachable today.
    """
    vercel = (REPO_ROOT / "nomicous" / "frontend" / "vercel.json").read_text(encoding="utf-8")

    assert "http://127.0.0.1:8001" in vercel
    assert "http://127.0.0.1:*" not in vercel
    assert "http://localhost:8001" not in vercel
    assert "http://localhost:*" not in vercel
    assert "http://[::1]:8001" not in vercel
    assert "http://[::1]:*" not in vercel
    assert "connect-src" in vercel


def test_vercel_frontend_csp_records_why_inline_scripts_are_still_allowed() -> None:
    """`'unsafe-inline'` in script-src is a known gap; it must stay documented.

    Next.js 16's App Router emits per-render inline `<script>` blocks carrying the
    RSC flight payload, with no nonce attribute, so neither a hash allowlist nor
    a static header can admit them. Removing the keyword here would ship a page
    that renders but never hydrates. See docs/security/frontend-csp.md.
    """
    vercel = (REPO_ROOT / "nomicous" / "frontend" / "vercel.json").read_text(encoding="utf-8")
    rationale = (REPO_ROOT / "docs" / "security" / "frontend-csp.md").read_text(encoding="utf-8")

    assert "script-src 'self' 'unsafe-inline'" in vercel
    assert "middleware" in rationale
    assert "nonce" in rationale
    assert "self.__next_f" in rationale


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


def test_runtime_images_uninstall_vulnerable_system_packaging_tools() -> None:
    dockerfile = (REPO_ROOT / "nomicous" / "Dockerfile").read_text(encoding="utf-8")
    assert "pip uninstall -y pip setuptools wheel" in dockerfile


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
    """ADR 0003 left the group with no table to read; 006 revokes what remains."""
    drop = (
        REPO_ROOT
        / "nomicous"
        / "infrastructure"
        / "alembic"
        / "versions"
        / "006_drop_inference_jobs.py"
    ).read_text(encoding="utf-8")
    bootstrap = (REPO_ROOT / "scripts" / "platform" / "provision_database_roles.sql").read_text(
        encoding="utf-8"
    )

    assert 'op.drop_table("inference_jobs")' in drop
    assert "REVOKE ALL ON SCHEMA public FROM nomicous_inference_worker" in drop
    assert not [
        line
        for line in bootstrap.splitlines()
        if "nomicous_inference_worker" in line and line.lstrip().startswith("GRANT")
    ]


def test_release_workflow_refuses_asset_replacement_and_generates_evidence() -> None:
    document = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    body = _workflow_body("release.yml")

    assert "--clobber" not in body
    assert "actions/attest-build-provenance@" in body
    assert "anchore/sbom-action@e22c389904149dbc22b58101806040fa8d37a610" in body
    # 57a97c7e... was pinned with a `# v0.36.0` comment but is in fact the v0.35.0
    # commit; ed142fd0... is the real v0.36.0.
    assert "aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25" in body
    assert document[True]["release"]["types"] == ["published"]
    assert 'gh release view "$RELEASE_TAG"' in body

    # One build, on one runner. There is no per-OS matrix: the wheel is
    # pure-Python and the platform-specific bytes are PyPI's problem now.
    jobs = document["jobs"]
    assert list(jobs) == ["publish"]
    assert jobs["publish"]["runs-on"] == "ubuntu-latest"
    assert "strategy" not in jobs["publish"]


def test_inference_group_carries_no_postgres_driver_or_orm() -> None:
    """ADR 0003: nothing under `inference/` talks to a database."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    inference = _flatten_group(pyproject["dependency-groups"], "inference")

    forbidden = ("psycopg", "sqlalchemy", "asyncpg", "alembic")
    assert not [dependency for dependency in inference if dependency.lower().startswith(forbidden)]

    # ADR 0004: the runtime ships with the package, not with a container. The
    # image-level Torch checks 049 wrote here went with inference/Dockerfile,
    # which ADR 0003 deleted; this is the half that does not depend on an image.
    assert any(
        dependency.startswith("torch") for dependency in pyproject["project"]["dependencies"]
    )


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


def test_platform_requirements_are_gated_against_the_lockfile() -> None:
    deployment = (REPO_ROOT / ".github" / "workflows" / "deployment.yml").read_text(
        encoding="utf-8"
    )
    requirements = (REPO_ROOT / "deploy" / "platform" / "requirements.txt").read_text(
        encoding="utf-8"
    )

    export = "uv export --locked --only-group platform-prod --no-hashes -o deploy/platform/requirements.txt"
    assert export in requirements, "requirements.txt header no longer records its export command"
    assert export in deployment
    assert "git diff --exit-code -- deploy/platform/requirements.txt" in deployment
