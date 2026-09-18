"""The trainer-side exporter must produce a servable artifact.

`src/models/calamari/export.py::export_calamari_onnx` is the entry point the
cluster export path calls. It once wrote 7 metadata keys and traced a graph
whose LSTM time axis froze at the example width 8, which is exactly the
`best.onnx` that shipped unservable at `coptic-htr-calamari@bdaa22d3`. This
test exports a tiny random-weight checkpoint through that entry point and holds
the result to the serving contract: the real adapter opens it, every
representative width runs, logits match the Torch model, and the metadata key
set is complete. Fast and local: random weights, CPU only, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from nomikos_inference.architectures.calamari.adapter import _load_session  # noqa: E402
from src.models.calamari.checkpoint import save_calamari_checkpoint  # noqa: E402
from src.models.calamari.config import default_model_config  # noqa: E402
from src.models.calamari.export import export_calamari_onnx  # noqa: E402
from src.models.calamari.model import CalamariTorchModel  # noqa: E402

LINE_HEIGHT = 48
CHARSET = ["", " ", "a", "b", "c", "d", "e"]
WIDTHS = [8, 64, 517, 1200]

REQUIRED_METADATA_KEYS = frozenset(
    {
        "format",
        "architecture",
        "input_layout",
        "classes",
        "line_height",
        "charset",
        "blank_index",
        "temperature",
        "lstm_layers",
        "preprocessing",
        "input_name",
        "output_names",
    }
)


def _tiny_checkpoint(path: Path) -> CalamariTorchModel:
    """A random-weight training model with the production topology, saved to disk."""
    torch.manual_seed(7)
    model = CalamariTorchModel(
        default_model_config(classes=len(CHARSET), temperature=-1.0, lstm_layers=2)
    )
    with torch.no_grad():
        model(
            torch.zeros((1, 8, LINE_HEIGHT, 1), dtype=torch.float32),
            image_lengths=torch.tensor([8]),
        )
    model.eval()
    save_calamari_checkpoint(path, model, charset=CHARSET, line_height=LINE_HEIGHT)
    return model


def _line_tensor(width: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + width)
    canvas = rng.integers(0, 256, size=(LINE_HEIGHT, width)).astype(np.uint8)
    canvas[8:40, 2 : max(3, width - 2)] = 0
    return canvas.reshape(1, width, LINE_HEIGHT, 1).astype(np.float32)


def test_trainer_export_produces_a_servable_artifact(tmp_path: Path) -> None:
    model = _tiny_checkpoint(tmp_path / "best.pt")
    destination = tmp_path / "best.onnx"
    export_calamari_onnx(destination.parent / "best.pt", destination)

    embedded = {entry.key: entry.value for entry in onnx.load(str(destination)).metadata_props}
    assert frozenset(embedded) == REQUIRED_METADATA_KEYS

    session, charset, line_height = _load_session(str(destination), None)
    assert line_height == LINE_HEIGHT
    assert len(charset) == len(CHARSET)

    out_lens: list[int] = []
    with torch.no_grad():
        for width in WIDTHS:
            tensor = _line_tensor(width)
            feed = {
                "image": tensor,
                "image_lengths": np.asarray([tensor.shape[1]], dtype=np.int64),
            }
            logits, out_len = (
                np.asarray(part) for part in session.run(["logits", "out_len"], feed)
            )
            expected = model(
                torch.from_numpy(tensor), image_lengths=torch.tensor([tensor.shape[1]])
            )
            assert int(np.asarray(out_len)[0]) == int(expected["out_len"].numpy()[0])
            assert float(np.max(np.abs(logits - expected["logits"].numpy()))) < 1e-4
            out_lens.append(int(np.asarray(out_len)[0]))
    assert out_lens == sorted(out_lens) and len(set(out_lens)) == len(out_lens)
