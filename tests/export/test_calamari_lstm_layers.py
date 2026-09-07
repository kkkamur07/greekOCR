"""The loader reads the recurrent depth a checkpoint declares.

Calamari checkpoints come in two shapes now. The published models carry
``lstm_layers: 2`` and a state dict with BiLSTMs at ``layers.4`` and
``layers.6``; everything published before them carries no ``lstm_layers`` key
at all and a single BiLSTM at ``layers.4``. Both must load, and the loader must
refuse a declared depth it cannot build rather than fail later inside
``load_state_dict`` where the message points at the weights instead of the
metadata.

These tests build their own checkpoints instead of pinning a fixture: a
two-BiLSTM ``best.pt`` is ten megabytes, and what is under test is the
topology-selection logic, not any particular set of trained weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.model.inference_export.calamari import (  # noqa: E402
    export_calamari_onnx,
    load_calamari_checkpoint,
)
from src.model.inference_export.calamari.checkpoint import (  # noqa: E402
    CalamariCheckpointMetadataError,
    CalamariCheckpointStateDictError,
)
from src.model.inference_export.calamari.config import default_model_config  # noqa: E402
from src.model.inference_export.calamari.model import CalamariTorchModel  # noqa: E402

LINE_HEIGHT = 48
CLASSES = 11


def _write_checkpoint(
    destination: Path,
    *,
    lstm_layers: int | None,
    classes: int = CLASSES,
    seed: int = 0,
) -> Path:
    """Write a checkpoint whose weights fit a stack of ``lstm_layers`` BiLSTMs.

    ``lstm_layers=None`` omits the key entirely, which is exactly the shape of
    every checkpoint published before the two-layer models existed.
    """
    torch.manual_seed(seed)
    depth = 1 if lstm_layers is None else lstm_layers
    model = CalamariTorchModel(default_model_config(classes=classes, lstm_layers=depth))
    model.eval()
    # The Lazy* modules only acquire their parameters once something has flowed
    # through them, so the state dict is empty until a forward pass has run.
    with torch.no_grad():
        model(
            torch.zeros((1, 8, LINE_HEIGHT, 1), dtype=torch.float32),
            image_lengths=torch.tensor([8]),
        )

    payload: dict[str, Any] = {
        "format": "calamari-pytorch-v1",
        "classes": classes,
        "line_height": LINE_HEIGHT,
        "charset": [""] + [chr(ord("a") + index) for index in range(classes - 1)],
        "blank_index": 0,
        "temperature": -1.0,
        "state_dict": model.state_dict(),
    }
    if lstm_layers is not None:
        payload["lstm_layers"] = lstm_layers
    torch.save(payload, destination)
    return destination


def test_two_bilstm_checkpoint_loads_and_round_trips_through_export(tmp_path: Path) -> None:
    """The shape the published Greek, Armenian and Syriac models are in.

    Loading is the half that used to fail: the loader built a one-BiLSTM graph
    from a private config and ``strict=True`` then rejected the
    ``layers.6.lstm.*`` weights. Exporting and replaying through ONNX Runtime is
    the other half, because a graph that loads but does not trace is no more
    publishable than one that does not load.
    """
    checkpoint = _write_checkpoint(tmp_path / "two.pt", lstm_layers=2)

    model, metadata = load_calamari_checkpoint(checkpoint)
    assert metadata.lstm_layers == 2
    # Two BiLSTMs, and the second one is fed the concatenated 400-wide output of
    # the first rather than the flattened convolution stack.
    recurrent = [layer for layer in model.layers if hasattr(layer, "lstm")]
    assert len(recurrent) == 2
    assert recurrent[0].lstm.input_size == 720
    assert recurrent[1].lstm.input_size == 400

    destination = tmp_path / "two.onnx"
    exported = export_calamari_onnx(checkpoint, destination)
    assert exported.lstm_layers == 2
    embedded = {entry.key: entry.value for entry in onnx.load(destination).metadata_props}
    assert embedded["lstm_layers"] == "2"
    assert embedded["classes"] == str(CLASSES)

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    for width in (7, 8, 17, 18, 129):
        image = np.random.default_rng(width).random((1, width, LINE_HEIGHT, 1), np.float32) * 255
        lengths = np.asarray([width], dtype=np.int64)
        with torch.no_grad():
            expected = model(torch.from_numpy(image), image_lengths=torch.from_numpy(lengths))
        actual_logits, actual_lengths = session.run(
            ["logits", "out_len"],
            {"image": image, "image_lengths": lengths},
        )
        # The same tolerance the single-BiLSTM export test uses. Measured drift
        # on the published two-layer models is ~1e-4 at worst over real
        # manuscript lines, well inside it.
        np.testing.assert_allclose(actual_logits, expected["logits"].numpy(), rtol=1e-4, atol=2e-4)
        np.testing.assert_array_equal(actual_lengths, expected["out_len"].numpy())
        # The claim that matters for a reader: every CTC frame decision agrees.
        np.testing.assert_array_equal(
            np.argmax(actual_logits, axis=-1),
            np.argmax(expected["logits"].numpy(), axis=-1),
        )


def test_checkpoint_without_lstm_layers_loads_as_one_layer(tmp_path: Path) -> None:
    """Backward compatibility with every checkpoint published before the key.

    Confirmed against the real artifact: the Syriac checkpoint at Hub revision
    ``5ff715e8`` has payload keys ``format``/``classes``/``line_height``/
    ``charset``/``state_dict`` and one BiLSTM at ``layers.4``. Defaulting an
    absent key to 2 would break it, and erroring on an absent key would break
    it louder.
    """
    checkpoint = _write_checkpoint(tmp_path / "one.pt", lstm_layers=None)

    model, metadata = load_calamari_checkpoint(checkpoint)
    assert metadata.lstm_layers == 1
    assert len([layer for layer in model.layers if hasattr(layer, "lstm")]) == 1
    assert not any(name.startswith("layers.6.") for name in model.state_dict())


@pytest.mark.parametrize(
    "value",
    [
        0,
        3,
        -1,
        True,  # ``bool`` is an ``int`` subclass, and ``True == 1`` would pass a naive check
        "2",
        2.0,
        None,
    ],
)
def test_invalid_lstm_layers_is_a_metadata_error(tmp_path: Path, value: object) -> None:
    """A depth the loader cannot build is a metadata defect, not a weights one.

    The subclass is the assertion. ``CalamariCheckpointStateDictError`` would
    send a deployment looking at the weights, which in this case are fine; what
    is wrong is the shape the checkpoint claims to have.
    """
    checkpoint = _write_checkpoint(tmp_path / "bad.pt", lstm_layers=1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["lstm_layers"] = value
    torch.save(payload, checkpoint)

    with pytest.raises(CalamariCheckpointMetadataError, match="lstm_layers"):
        load_calamari_checkpoint(checkpoint)


def test_declared_depth_that_contradicts_the_weights_is_a_state_dict_error(
    tmp_path: Path,
) -> None:
    """Depth is only ever *selected* from metadata, never inferred from weights.

    A checkpoint claiming one layer while carrying two sets of BiLSTM weights is
    a real corruption, and the loader has to reject it rather than silently drop
    the second layer. This is the failure that ``strict=True`` exists for, so it
    surfaces as a state-dict error.
    """
    checkpoint = _write_checkpoint(tmp_path / "mismatch.pt", lstm_layers=2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["lstm_layers"] = 1
    torch.save(payload, checkpoint)

    with pytest.raises(CalamariCheckpointStateDictError):
        load_calamari_checkpoint(checkpoint)
