"""Helper startup weight prefetch (no live Hub network in unit tests).

Hub I/O is replaced only at the Hub client seam (same pattern as
``tests/hf/test_resolve.py``). Prefetch still runs real registry load +
``resolve_weights_source`` / cache write.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml
from src.hf.resolve import set_default_hub_client

from inference.helper.prefetch import prefetch_local_eligible_weights

REPO_ROOT = Path(__file__).resolve().parents[3]
MOCK_WEIGHTS = REPO_ROOT / "src/hf/local/syriac/calamari/v1/stable/best.pt"
HUB_REVISION = "a" * 40
ARTIFACT_SHA256 = hashlib.sha256(MOCK_WEIGHTS.read_bytes()).hexdigest()


class MockHubClient:
    """Stand in for Hugging Face Hub only — not for registry/resolve logic."""

    def __init__(self) -> None:
        self.downloads: list[tuple[str, str, Path]] = []

    def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None:
        self.downloads.append((repo_id, revision, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MOCK_WEIGHTS, local_dir / "best.pt")


def test_prefetch_resolves_local_hf_models_only(monkeypatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "syriac-calamari-v1": {
                        "task": "transcribe",
                        "architecture": "calamari",
                        "device": "cpu",
                        "host_eligibility": "local",
                        "versions": {
                            "stable": {
                                "weights_source": "hf://example/syriac@stable",
                                "hub_revision": HUB_REVISION,
                                "artifact_sha256": ARTIFACT_SHA256,
                            }
                        },
                    },
                    "remote-only": {
                        "task": "transcribe",
                        "architecture": "calamari",
                        "device": "cpu",
                        "host_eligibility": "remote",
                        "versions": {
                            "stable": {
                                "weights_source": "hf://example/remote@stable",
                                "hub_revision": "c" * 40,
                                "artifact_sha256": "d" * 64,
                            }
                        },
                    },
                    "kraken-segment": {
                        "task": "segment",
                        "architecture": "kraken-segment",
                        "device": "cpu",
                        "host_eligibility": "local",
                        "versions": {
                            "stable": {
                                "weights_source": "package://kraken/blla.mlmodel",
                            }
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    cache_root = tmp_path / "hf-cache"
    client = MockHubClient()
    set_default_hub_client(client)
    monkeypatch.setenv("HF_CACHE_ROOT", str(cache_root))
    try:
        prefetch_local_eligible_weights(registry_path)
    finally:
        set_default_hub_client(None)

    assert len(client.downloads) == 1
    assert client.downloads[0][0] == "example/syriac"
    assert (cache_root / "syriac-calamari-v1" / "stable" / "best.pt").is_file()
