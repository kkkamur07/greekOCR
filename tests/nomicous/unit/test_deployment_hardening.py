"""Static regressions for deployment and database-privilege hardening."""

from __future__ import annotations

import tomllib

from tests.fixtures.paths import REPO_ROOT


def test_inference_image_reincludes_only_required_hub_resolver_sources() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "inference" / "Dockerfile").read_text(encoding="utf-8")

    assert "!src/" in dockerignore
    assert "!src/hf/resolve/**" in dockerignore
    assert ".docker-cache/" in dockerignore
    assert "COPY src/hf/__init__.py src/hf/paths.py /app/src/hf/" in dockerfile
    assert "COPY src/hf/resolve /app/src/hf/resolve" in dockerfile
    assert "COPY src/hf /app/src/hf" not in dockerfile


def test_helper_freeze_is_onnx_only() -> None:
    spec = (REPO_ROOT / "packaging" / "helper" / "pyinstaller.spec").read_text(encoding="utf-8")
    excludes = (REPO_ROOT / "packaging" / "helper" / "excludes.txt").read_text(encoding="utf-8")

    assert 'collect_submodules("kraken")' not in spec
    assert '"kraken.blla"' not in spec
    assert '"kraken.lib.vgsl"' not in spec
    assert '"inference.architectures.blla.blla"' not in spec
    assert '"inference.architectures.blla.blla_model"' not in spec
    assert '"safetensors.torch"' not in spec
    assert '"inference.architectures.blla.blla_decoder"' in spec
    assert '"inference.architectures.blla.onnx"' in spec
    for dependency in ("torch", "torchvision", "safetensors", "kraken"):
        assert f"\n{dependency}\n" in excludes

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = pyproject["dependency-groups"]
    assert not any(dependency.startswith(("torch", "safetensors")) for dependency in groups["inference"])
    assert any(dependency.startswith("torch") for dependency in groups["export"])
    assert any(dependency.startswith("safetensors") for dependency in groups["export"])


def test_helper_packaging_uses_one_foreground_server() -> None:
    launcher = (REPO_ROOT / "packaging" / "helper" / "tray_launcher.py").read_text(
        encoding="utf-8"
    )
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    windows_build = (
        REPO_ROOT / "packaging" / "helper" / "windows" / "build-installer.ps1"
    ).read_text(encoding="utf-8")
    shared_build = (
        REPO_ROOT / "packaging" / "helper" / "scripts" / "build-pyinstaller.sh"
    ).read_text(encoding="utf-8")
    bundle_verifier = (
        REPO_ROOT / "packaging" / "helper" / "scripts" / "verify-bundle.py"
    ).read_text(encoding="utf-8")

    assert "multiprocessing" not in launcher
    assert "pystray" not in pyproject
    for build in (shared_build, windows_build):
        assert "--isolated --no-dev --group helper --group packaging" in build
        assert "--group inference" not in build
        assert "verify-bundle.py" in build
    assert "& bash" not in windows_build
    assert '"PYZ-00.toc"' in bundle_verifier
    assert '"COLLECT-00.toc"' in bundle_verifier
    assert '"src.model.inference_export"' in bundle_verifier


def test_helper_installers_are_user_scoped_and_verify_startup() -> None:
    linux_install = (
        REPO_ROOT / "packaging" / "helper" / "linux" / "install-helper.sh"
    ).read_text(encoding="utf-8")
    mac_install = (
        REPO_ROOT / "packaging" / "helper" / "macos" / "install-helper.sh"
    ).read_text(encoding="utf-8")
    windows_install = (
        REPO_ROOT / "packaging" / "helper" / "windows" / "install-helper.ps1"
    ).read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github" / "workflows" / "release-helper.yml").read_text(
        encoding="utf-8"
    )

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
    assert 's|__INSTALL_DIR__|$INSTALL_DIR|g' in mac_install
    assert 's|__INSTALL_DIR__|$INSTALL_DIR/nomicous-inference-helper|g' not in mac_install
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


def test_vercel_frontend_permits_helper_loopback_origins() -> None:
    vercel = (REPO_ROOT / "nomicous" / "frontend" / "vercel.json").read_text(encoding="utf-8")

    assert "http://127.0.0.1:8001" in vercel
    assert "http://localhost:8001" in vercel
    assert "http://[::1]:8001" in vercel
    assert "connect-src" in vercel


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
    assert "aquasecurity/trivy-action@57a97c7e7821a5776cebc9bb87c984fa69cba8f1" in workflow
    assert "overwrite: false" in workflow
    assert "types: [published]" in workflow
    assert 'gh release view "$RELEASE_TAG"' in workflow
    assert "macos-15" in workflow
    assert "macos-15-intel" in workflow
    assert "nomicous-inference-helper-macos.dmg" in workflow
    assert "nomicous-inference-helper-macos-intel.dmg" in workflow
    assert "expected four installer assets" in workflow
