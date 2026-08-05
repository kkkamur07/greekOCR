"""Static regressions for deployment and database-privilege hardening."""

from __future__ import annotations

import re
import tomllib

from tests.fixtures.paths import REPO_ROOT


def _flatten_group(groups: dict[str, list], name: str) -> list[str]:
    """Resolve a PEP 735 dependency group, following `include-group` entries."""
    resolved: list[str] = []
    for entry in groups[name]:
        if isinstance(entry, dict):
            resolved.extend(_flatten_group(groups, entry["include-group"]))
        else:
            resolved.append(entry)
    return resolved


def test_inference_image_reincludes_only_required_hub_resolver_sources() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "inference" / "Dockerfile").read_text(encoding="utf-8")

    assert "!src/" in dockerignore
    assert "!src/hf/resolve/**" in dockerignore
    assert ".docker-cache/" in dockerignore
    assert "COPY src/hf/__init__.py src/hf/paths.py /app/src/hf/" in dockerfile
    assert "COPY src/hf/resolve /app/src/hf/resolve" in dockerfile
    assert "COPY src/hf /app/src/hf" not in dockerfile


def test_helper_freeze_ships_the_torch_runtime_and_nothing_else() -> None:
    """ADR 0004: one runtime, CPU only, and no training stack in the installer.

    The inverse of this test used to enforce a Torch denylist through
    `excludes.txt` and a release-time bundle verifier. Both were deleted with
    the second runtime; what still has to hold is that the frozen helper can
    actually load the models and does not drag Kraken or the training stack in
    with them.
    """
    spec = (REPO_ROOT / "packaging" / "helper" / "pyinstaller.spec").read_text(encoding="utf-8")

    assert not (REPO_ROOT / "packaging" / "helper" / "excludes.txt").exists()
    assert not (REPO_ROOT / "packaging" / "helper" / "scripts" / "verify-bundle.py").exists()

    assert 'collect_submodules("kraken")' not in spec
    assert '"kraken.blla"' not in spec
    assert '"kraken.lib.vgsl"' not in spec
    # The frozen helper runs the same modules as the hosted service.
    for module in (
        '"inference.architectures.blla.blla"',
        '"inference.architectures.blla.blla_model"',
        '"inference.architectures.blla.blla_decoder"',
        '"inference.architectures.calamari.checkpoint"',
        '"inference.architectures.calamari.model"',
        '"safetensors.torch"',
        '"torch"',
    ):
        assert module in spec
    # Excludes are size pruning now, not a runtime denylist.
    for excluded in ('"kraken"', '"transformers"', '"nomicous"', '"sqlalchemy"'):
        assert excluded in spec

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = pyproject["project"]["dependencies"]
    assert any(dependency.startswith("torch") for dependency in project_dependencies)
    assert any(dependency.startswith("safetensors") for dependency in project_dependencies)
    assert not any(dependency.startswith("onnxruntime") for dependency in project_dependencies)

    groups = pyproject["dependency-groups"]
    assert "parity" not in groups, "the kraken parity group outlived its second runtime"
    assert "export" not in groups
    assert not any(
        "kraken" in dependency
        for group in groups.values()
        for dependency in group
        if isinstance(dependency, str)
    )
    # The helper group is what the frozen installer installs; it must resolve to
    # a runtime that can actually open both artifacts.
    helper = _flatten_group(groups, "helper")
    assert not any(dependency.startswith("onnxruntime") for dependency in helper)

    # CUDA wheels must not be reachable: PyPI serves them on Linux and Windows.
    sources = pyproject["tool"]["uv"]["sources"]["torch"]
    markers = {entry["marker"] for entry in sources if entry["index"] == "pytorch-cpu"}
    assert markers == {"sys_platform == 'linux' or sys_platform == 'win32'"}


def test_helper_packaging_uses_one_foreground_server() -> None:
    launcher = (REPO_ROOT / "packaging" / "helper" / "tray_launcher.py").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    windows_build = (
        REPO_ROOT / "packaging" / "helper" / "windows" / "build-installer.ps1"
    ).read_text(encoding="utf-8")
    shared_build = (
        REPO_ROOT / "packaging" / "helper" / "scripts" / "build-pyinstaller.sh"
    ).read_text(encoding="utf-8")
    smoke_test = (
        REPO_ROOT / "packaging" / "helper" / "scripts" / "smoke-test.py"
    ).read_text(encoding="utf-8")

    assert "multiprocessing" not in launcher
    assert "pystray" not in pyproject
    for build in (shared_build, windows_build):
        assert "--isolated --no-dev --group helper --group packaging" in build
        assert "--group inference" not in build
        assert "smoke-test.py" in build
    assert "& bash" not in windows_build
    # The freeze is still proven to serve, even though it is no longer proven
    # to be Torch-free.
    assert "/health" in smoke_test
    assert "/inference/v1/info" in smoke_test


def test_helper_installers_are_user_scoped_and_verify_startup() -> None:
    linux_install = (REPO_ROOT / "packaging" / "helper" / "linux" / "install-helper.sh").read_text(
        encoding="utf-8"
    )
    mac_install = (REPO_ROOT / "packaging" / "helper" / "macos" / "install-helper.sh").read_text(
        encoding="utf-8"
    )
    windows_install = (
        REPO_ROOT / "packaging" / "helper" / "windows" / "install-helper.ps1"
    ).read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github" / "workflows" / "release-helper.yml").read_text(
        encoding="utf-8"
    )

    # HELPER_REGISTRY_URL is templated into sed/heredoc content, so both shell
    # installers must reject values that could escape those templates.
    for installer in (linux_install, mac_install):
        assert "HELPER_REGISTRY_URL must be https://" in installer
        assert "tr -d 'A-Za-z0-9._~:/?#@%=+[]-'" in installer
    assert "HELPER_CORS_ORIGINS" not in linux_install
    assert "AUTOSTART_FILE" in linux_install
    assert "RUNNER=" in linux_install
    assert "STAGE_ROOT=" in linux_install
    assert "BACKUP_ROOT=" in linux_install
    assert "restore_previous_install" in linux_install
    assert "curl --fail --silent --max-time 2 http://127.0.0.1:8001/health" in linux_install
    assert 'APP_DST="$HOME/Applications/' in mac_install
    assert 'cp -R "$APP_SRC" /Applications/' not in mac_install
    assert "APP_STAGE=" in mac_install
    assert "APP_BACKUP=" in mac_install
    assert "restore_previous_install" in mac_install
    assert "s|__INSTALL_DIR__|$INSTALL_DIR|g" in mac_install
    assert "s|__INSTALL_DIR__|$INSTALL_DIR/nomicous-inference-helper|g" not in mac_install
    assert 'launchctl bootstrap "gui/$(id -u)"' in mac_install
    assert "Invoke-WebRequest" in windows_install
    assert "Stop-ScheduledTask" in windows_install
    assert '"HF_CACHE_ROOT" = $CacheDir' in windows_install
    assert "$StageRoot" in windows_install
    assert "$BackupRoot" in windows_install
    assert "$PreviousTaskExisted" in windows_install
    assert "$PreviousUserEnvironment" in windows_install
    assert "Stop-HelperTaskAndWait" in windows_install
    assert "Wait-InstallUnlocked" in windows_install
    assert windows_install.index("Stop-ScheduledTask") < windows_install.index(
        "Move-Item -LiteralPath $InstallRoot"
    )
    assert "ubuntu-22.04" in workflow


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
    assert 'CLOUD_INFERENCE_ENABLED: "true"' in compose
    assert compose.count("\n      INFERENCE_WEBHOOK_SECRET:") == 2
    assert compose.count("\n      INFERENCE_SERVICE_SECRET:") == 2


def test_platform_bundle_includes_contract_dependencies() -> None:
    build_script = (REPO_ROOT / "deploy" / "platform" / "build.sh").read_text(encoding="utf-8")

    assert '"inference" / "admission.py"' in build_script
    assert '"inference" / "infrastructure" / "settings.py"' in build_script


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
    for relative in ("nomicous/Dockerfile", "inference/Dockerfile"):
        dockerfile = (REPO_ROOT / relative).read_text(encoding="utf-8")
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
    # 57a97c7e... was pinned with a `# v0.36.0` comment but is in fact the v0.35.0
    # commit; ed142fd0... is the real v0.36.0.
    assert "aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25" in workflow
    assert "overwrite: false" in workflow
    assert "types: [published]" in workflow
    assert 'gh release view "$RELEASE_TAG"' in workflow
    assert "macos-15" in workflow
    assert "macos-15-intel" in workflow
    assert "nomicous-inference-helper-macos.dmg" in workflow
    assert "nomicous-inference-helper-macos-intel.dmg" in workflow
    assert "expected four installer assets" in workflow


def test_inference_image_installs_the_model_runtime() -> None:
    """`--only-group inference` drops [project].dependencies, i.e. torch.

    Model imports are lazy, so an image built that way still starts, still serves
    /health, and fails every real inference request. Assert both the install flag
    and a build-time check that actually loads the runtime - and that the wheel
    it loads is the CPU build (ADR 0004).
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dockerfile = (REPO_ROOT / "inference" / "Dockerfile").read_text(encoding="utf-8")

    assert any(
        dependency.startswith("torch") for dependency in pyproject["project"]["dependencies"]
    )
    assert "--only-group inference" not in dockerfile
    assert "uv sync --frozen --no-default-groups --group inference" in dockerfile
    assert "import torch" in dockerfile
    assert "torch.version.cuda" in dockerfile

    deployment = (REPO_ROOT / ".github" / "workflows" / "deployment.yml").read_text(
        encoding="utf-8"
    )
    assert "import torch" in deployment
    assert "inference.architectures.calamari.adapter" in deployment
    assert "torch.version.cuda" in deployment


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


def test_release_signing_is_mandatory_not_best_effort() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "release-helper.yml").read_text(
        encoding="utf-8"
    )

    # Provenance attestation must not be skippable by flipping repo visibility.
    assert "!github.event.repository.private" not in workflow
    assert "Require code-signing credentials" in workflow
    assert "Require release manifest signing key" in workflow
    assert "Refusing to publish unsigned installers" in workflow
    assert "SHA256SUMS.asc" in workflow
    # The scan inputs must be the shipped bytes, not the build-script directory.
    assert "scan-ref: release-scan" in workflow
    assert "scan-ref: packaging/helper" not in workflow
    # The sbom-action `path:` input (not `subject-path:`) must not be the
    # build-script directory.
    assert re.search(r"^\s+path: packaging/helper\s*$", workflow, re.M) is None


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
