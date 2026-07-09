"""Essential hf:// resolution, cache, and fetch_model coverage."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from inference.weights import resolve_weights_source
from src.hf.resolve import resolve_hf_weights_source, set_default_hub_client
from src.hf.resolve.manifest import load_manifest
from src.hf.resolve.uri import parse_hf_weights_uri

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_WEIGHTS = REPO_ROOT / "src/hf/local/syriac/calamari/v1/stable/best.pt"


@dataclass
class MockHubClient:
    revision_sha: str = "abc123"
    downloads: list[tuple[str, str, Path]] = field(default_factory=list)
    resolve_error: Exception | None = None

    def resolve_revision_sha(self, repo_id: str, revision: str) -> str:
        if self.resolve_error is not None:
            raise self.resolve_error
        return self.revision_sha

    def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None:
        self.downloads.append((repo_id, revision, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MOCK_WEIGHTS, local_dir / "best.pt")


@pytest.fixture(autouse=True)
def reset_hub_client():
    set_default_hub_client(None)
    yield
    set_default_hub_client(None)


def test_parse_hf_weights_uri():
    parsed = parse_hf_weights_uri("hf://nomicous/greek-htr-calamari@stable")
    assert parsed.repo_id == "nomicous/greek-htr-calamari"
    assert parsed.registry_tag == "stable"


def test_resolve_downloads_then_reuses_cache(tmp_path: Path):
    client = MockHubClient(revision_sha="sha-v1")
    uri = "hf://nomicous/greek-htr-calamari@stable"
    kwargs = dict(
        uri=uri,
        registry_model_id="greek-calamari-v1",
        registry_tag="stable",
        architecture="calamari",
        hub_client=client,
        cache_root=tmp_path,
    )

    first = resolve_hf_weights_source(**kwargs)
    second = resolve_hf_weights_source(**kwargs)

    assert first == second
    assert len(client.downloads) == 1
    assert load_manifest(tmp_path / "greek-calamari-v1" / "stable") is not None


def test_resolve_surfaces_missing_repo_error(tmp_path: Path):
    class RepositoryNotFoundError(Exception):
        pass

    with pytest.raises(ValueError, match="Hub model repo not found"):
        resolve_hf_weights_source(
            "hf://nomicous/missing-htr-calamari@stable",
            registry_model_id="greek-calamari-v1",
            registry_tag="stable",
            architecture="calamari",
            hub_client=MockHubClient(resolve_error=RepositoryNotFoundError("missing")),
            cache_root=tmp_path,
        )


def test_inference_delegate_requires_hf_context():
    with pytest.raises(ValueError, match="registry_model_id and registry_tag"):
        resolve_weights_source("hf://nomicous/greek-htr-calamari@stable")


def test_fetch_model_warms_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_root = tmp_path / "cache"
    client = MockHubClient(revision_sha="sha-fetch")
    set_default_hub_client(client)
    monkeypatch.setenv("HF_CACHE_ROOT", str(cache_root))

    module_path = REPO_ROOT / "scripts/hf/fetch_model.py"
    spec = importlib.util.spec_from_file_location("fetch_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_model.py", "greek-calamari-v1", "--registry-tag", "stable"],
    )

    assert module.main() == 0
    assert len(client.downloads) == 1
    assert (cache_root / "greek-calamari-v1" / "stable" / "best.pt").is_file()
