"""Verify that a frozen helper is Torch-free and can serve its HTTP API."""

from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen


FORBIDDEN_PREFIXES = (
    "functorch",
    "libtorch",
    "safetensors",
    "torch",
    "torchgen",
    "torchvision",
)
FORBIDDEN_MODULE_PREFIXES = (
    "inference.architectures.blla.blla",
    "inference.architectures.blla.blla_model",
    "inference.architectures.calamari.config",
    "inference.architectures.calamari.layers",
    "inference.architectures.calamari.model",
    "kraken",
    "safetensors",
    "src.model.inference_export",
    "torch",
    "torchgen",
    "torchvision",
)


def _is_forbidden_name(name: str) -> bool:
    normalized = name.lower()
    return any(
        normalized == prefix
        or normalized.startswith(f"{prefix}.")
        or normalized.startswith(f"{prefix}_")
        or normalized.startswith(f"{prefix}-")
        for prefix in FORBIDDEN_PREFIXES
    )


def _assert_onnx_only_bundle(bundle_root: Path) -> None:
    leaked = sorted(
        str(path.relative_to(bundle_root))
        for path in bundle_root.rglob("*")
        if _is_forbidden_name(path.name)
    )
    if leaked:
        raise RuntimeError(f"native Torch files leaked into helper bundle: {leaked[:20]}")


def _toc_entry_names(path: Path) -> set[str]:
    try:
        entries = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as error:
        raise RuntimeError(f"unable to inspect PyInstaller manifest: {path}") from error
    names: set[str] = set()

    def collect(value: object) -> None:
        if not isinstance(value, (list, tuple)):
            return
        if (
            len(value) >= 3
            and isinstance(value[0], str)
            and isinstance(value[-1], str)
            and value[-1] in {"BINARY", "DATA", "EXECUTABLE", "EXTENSION", "PYMODULE"}
        ):
            names.add(value[0])
            return
        for item in value:
            collect(item)

    collect(entries)
    return names


def _assert_onnx_only_manifests(build_root: Path) -> None:
    manifest_paths = (build_root / "PYZ-00.toc", build_root / "COLLECT-00.toc")
    missing = [str(path) for path in manifest_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing PyInstaller manifests: {missing}")

    names = set().union(*(_toc_entry_names(path) for path in manifest_paths))
    leaked_modules = sorted(
        name
        for name in names
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_MODULE_PREFIXES
        )
    )
    if leaked_modules:
        raise RuntimeError(f"native modules leaked into helper archive: {leaked_modules[:20]}")


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _request_json(url: str) -> object:
    with urlopen(url, timeout=2) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.loads(response.read())


def _smoke_test(executable: Path, bundle_root: Path) -> None:
    registry_paths = sorted(bundle_root.rglob("registry.yaml"))
    if not registry_paths:
        raise RuntimeError("frozen helper bundle does not contain inference/registry.yaml")

    port = _reserve_port()
    with tempfile.TemporaryDirectory(prefix="nomicous-helper-smoke-") as temporary:
        home = Path(temporary)
        env = os.environ.copy()
        for name in (
            "HELPER_AUTH_SECRET",
            "HELPER_REGISTRY_URL",
            "HELPER_SECURE_MODE",
            "INFERENCE_REGISTRY_PATH",
        ):
            env.pop(name, None)
        env.update(
            {
                "HELPER_BUNDLED_REGISTRY_PATH": str(registry_paths[0]),
                "HELPER_CACHED_REGISTRY_PATH": str(home / "registry.yaml"),
                "HELPER_CACHED_REGISTRY_ETAG_PATH": str(home / "registry.etag"),
                "HELPER_HOST": "127.0.0.1",
                "HELPER_PORT": str(port),
                "HF_CACHE_ROOT": str(home / "hf-cache"),
                "HOME": str(home),
                "NO_PROXY": "127.0.0.1,localhost",
                "USERPROFILE": str(home),
            }
        )
        process = subprocess.Popen(
            [str(executable)],
            cwd=bundle_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        error: Exception | None = None
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"helper exited during startup with code {process.returncode}")
                try:
                    health = _request_json(f"{base_url}/health")
                    if not isinstance(health, dict) or health.get("status") != "ok":
                        raise RuntimeError(f"unexpected health response: {health!r}")
                    catalog = _request_json(f"{base_url}/inference/v1/catalog")
                    if not isinstance(catalog, dict) or not catalog.get("models"):
                        raise RuntimeError(f"unexpected catalog response: {catalog!r}")
                    return
                except Exception as request_error:
                    error = request_error
                    time.sleep(0.25)
            raise RuntimeError(f"helper did not become ready: {error}")
        finally:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate(timeout=5)
            if process.returncode not in (0, -15, 1) and output:
                print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "pyinstaller",
    )
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    executable = args.executable.resolve()
    if not bundle_root.is_dir():
        raise SystemExit(f"bundle root does not exist: {bundle_root}")
    if not executable.is_file():
        raise SystemExit(f"helper executable does not exist: {executable}")

    _assert_onnx_only_bundle(bundle_root)
    _assert_onnx_only_manifests(args.build_root.resolve())
    _smoke_test(executable, bundle_root)
    print(f"Verified ONNX-only helper bundle: {bundle_root}")


if __name__ == "__main__":
    main()
