"""Helper startup weight prefetch (no Hub network in unit tests)."""

from __future__ import annotations

from pathlib import Path

import yaml

from inference.helper.prefetch import prefetch_local_eligible_weights


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
                                "hub_revision": "a" * 40,
                                "artifact_sha256": "b" * 64,
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
                    "greek-kraken-segment-v1": {
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

    calls: list[tuple[str, str]] = []

    def fake_resolve(uri, **kwargs):  # noqa: ANN003
        calls.append((kwargs["registry_model_id"], kwargs["registry_tag"]))
        return tmp_path / "weights.pt"

    monkeypatch.setattr("inference.helper.prefetch.resolve_weights_source", fake_resolve)
    prefetch_local_eligible_weights(registry_path)

    assert calls == [("syriac-calamari-v1", "stable")]
