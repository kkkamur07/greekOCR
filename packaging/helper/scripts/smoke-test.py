"""Launch a frozen helper and prove it answers its HTTP API.

This is the half of the deleted `verify-bundle.py` that was never about Torch.
That script did two jobs: enforce the `excludes.txt` denylist against the
bundle and both PyInstaller manifests, and start the frozen binary to check it
actually serves. ADR 0004 made Torch the runtime, so the denylist and its
enforcement are gone; the startup check is not, because a bundle that freezes
cleanly and then cannot answer `/health` is the failure mode this catches, and
it is only reachable at release time.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen


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
            "HELPER_REGISTRY_URL",
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
                    raise RuntimeError(
                        f"helper exited during startup with code {process.returncode}"
                    )
                try:
                    health = _request_json(f"{base_url}/health")
                    if not isinstance(health, dict) or health.get("status") != "ok":
                        raise RuntimeError(f"unexpected health response: {health!r}")
                    info = _request_json(f"{base_url}/inference/v1/info")
                    if not isinstance(info, dict) or not info.get("models"):
                        raise RuntimeError(f"unexpected info response: {info!r}")
                    if info.get("service") != "nomicous-inference-helper":
                        raise RuntimeError(f"info document does not identify the helper: {info!r}")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    executable = args.executable.resolve()
    if not bundle_root.is_dir():
        raise SystemExit(f"bundle root does not exist: {bundle_root}")
    if not executable.is_file():
        raise SystemExit(f"helper executable does not exist: {executable}")

    _smoke_test(executable, bundle_root)
    print(f"Frozen helper bundle serves its API: {bundle_root}")


if __name__ == "__main__":
    main()
