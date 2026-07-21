"""Runtime-policy regressions for the frozen ONNX-only helper."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.contracts.common import InferenceTask, RegistryArchitecture
from inference.jobs.runner import run_model


@pytest.mark.parametrize(
    ("task", "architecture", "artifact"),
    [
        (InferenceTask.segment, RegistryArchitecture.blla, Path("blla.safetensors")),
        (InferenceTask.transcribe, RegistryArchitecture.calamari, Path("best.pt")),
    ],
)
def test_onnx_only_runner_rejects_native_artifacts_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    task: InferenceTask,
    architecture: RegistryArchitecture,
    artifact: Path,
) -> None:
    monkeypatch.setattr("inference.jobs.runner.validate_image_bytes", lambda *_args: None)
    monkeypatch.setattr("inference.jobs.runner.validate_request_params", lambda *_args: None)
    monkeypatch.setattr(
        "inference.jobs.runner.get_inference_settings",
        lambda: SimpleNamespace(inference_registry_path=Path("registry.yaml")),
    )
    monkeypatch.setattr(
        "inference.jobs.runner.resolve_registry_entry",
        lambda **_kwargs: SimpleNamespace(
            architecture=architecture,
            versions={
                "stable": SimpleNamespace(
                    weights_source="file://unused",
                    hub_revision=None,
                    artifact_sha256=None,
                )
            },
        ),
    )
    monkeypatch.setattr(
        "inference.jobs.runner.resolve_weights_source",
        lambda *_args, **_kwargs: artifact,
    )
    monkeypatch.setattr(
        "inference.jobs.runner.run_blla_segment",
        lambda *_args, **_kwargs: pytest.fail("native BLLA must not be dispatched"),
    )
    monkeypatch.setattr(
        "inference.jobs.runner.run_calamari_transcribe",
        lambda *_args, **_kwargs: pytest.fail("native Calamari must not be dispatched"),
    )

    with pytest.raises(RuntimeError, match="ONNX-only runtime cannot load"):
        run_model(
            task=task,
            registry_model_id="model",
            registry_tag="stable",
            image_bytes=b"validated",
            onnx_only=True,
        )
