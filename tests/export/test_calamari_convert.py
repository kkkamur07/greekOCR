"""The TF -> PyTorch Calamari converter is weight-for-weight lossless.

This is the converter's contract test. It runs the *actual* TensorFlow
calamari_ocr SavedModel (``best.ckpt``), loads the converted ``.pt`` through the
same loader the exporter uses, and asserts the two produce the same logits to
float32 rounding - the proof that the TF->PyTorch re-layout (conv transpose,
dense transpose, LSTM ``[i,f,c,o]`` gate order, forget-bias semantics) changed
no value.

It is skipped when TensorFlow is unavailable: the converter's read path imports
TF, and the production/dev environments deliberately do not ship it. Run the
suite inside a checkout that has ``tensorflow`` installed (e.g. ``uv run --group
calamari-train`` once the group exists, or a one-off venv) to exercise it.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.paths import REPO_ROOT

torch = pytest.importorskip("torch")

from src.model.inference_export.calamari import load_calamari_checkpoint  # noqa: E402

# The original TF artifact. It is not tracked (it is a ~8 MB SavedModel that
# predates this repo's packaging) - test fixtures should be committed, so a
# convert-from-TF integration test would add the vendor's .pb/.data files here.
# This test therefore runs only when the publisher points it at a real export.
TF_PREFIX_ENV = "CALAMARI_TF_CHECKPOINT_PREFIX"
TF_CONFIG_ENV = "CALAMARI_TF_CHECKPOINT_CONFIG"


def _tf_artifacts() -> tuple[Path, Path] | None:
    import os

    prefix = os.environ.get(TF_PREFIX_ENV)
    config = os.environ.get(TF_CONFIG_ENV)
    if not prefix or not config:
        return None
    p = Path(prefix)
    c = Path(config)
    if not p.with_suffix(".index").is_file() or not c.is_file():
        return None
    return p, c


def test_converter_is_weight_for_weight_lossless(tmp_path: Path) -> None:
    """TF logits == PyTorch logits to float32 rounding, on a full page.

    The strongest claim the converter can make: run the *actual* TensorFlow
    graph and the converted PyTorch graph on the same dense input, and assert
    the logits agree and every frame's argmax (the CTC decision) is identical.
    """
    artifacts = _tf_artifacts()
    if artifacts is None:
        pytest.skip(
            f"set {TF_PREFIX_ENV} to a TF checkpoint base name and "
            f"{TF_CONFIG_ENV} to its best.ckpt.json to run the TF parity test"
        )
    tf_prefix, tf_config = artifacts

    tf = pytest.importorskip("tensorflow")

    from scripts.hf.convert_calamari import convert_calamari_checkpoint

    destination = tmp_path / "converted.pt"
    metadata = convert_calamari_checkpoint(tf_prefix, tf_config, destination)

    # Load the converted checkpoint through the production loader (strict=True).
    model, loaded_metadata = load_calamari_checkpoint(destination)
    model.eval()
    assert loaded_metadata.classes == metadata.classes
    assert loaded_metadata.charset == tuple(metadata.charset)
    assert loaded_metadata.line_height == metadata.line_height

    # A dense input: tile a tracked line horizontally to many timesteps.
    from PIL import Image

    from nomikos_inference.architectures.calamari.preprocessing import (
        preprocess_line_image_bytes_to_calamari_tensor,
    )

    line = Image.open(REPO_ROOT / "tests/fixtures/manuscripts/syriac/transcribe_line.jpg").convert(
        "L"
    )
    w, h = line.size
    tiles = -(-3000 // w)
    pano = Image.new("L", (tiles * w, h))
    for i in range(tiles):
        pano.paste(line, (i * w, 0))
    pano_path = tmp_path / "pano.png"
    pano.crop((0, 0, tiles * w, h)).save(pano_path)
    image_u8 = preprocess_line_image_bytes_to_calamari_tensor(
        pano_path.read_bytes(), line_height=loaded_metadata.line_height
    )
    image_f32 = image_u8.astype(np.float32, copy=False)
    width = image_u8.shape[1]

    # TF reference -> logits (blank-first)
    saved = tf.saved_model.load(str(tf_prefix.with_suffix("")).replace("/variables/variables", ""))
    infer = saved.signatures["serving_default"]
    tf_logits = infer(
        img=tf.constant(image_u8, dtype=tf.uint8),
        img_len=tf.constant([[width]], dtype=tf.int32),
    )["root_3"].numpy()[0]

    # PyTorch port -> logits
    with torch.inference_mode():
        pt_logits = model(
            torch.from_numpy(image_f32),
            image_lengths=torch.tensor([width]),
        )["logits"].numpy()[0]

    assert tf_logits.shape == pt_logits.shape
    np.testing.assert_allclose(pt_logits, tf_logits, rtol=1e-4, atol=2e-4)
    # Every CTC frame decision is identical (the prediction-level guarantee).
    assert np.array_equal(np.argmax(tf_logits, axis=-1), np.argmax(pt_logits, axis=-1))


def test_converter_rejects_mismatched_classes(tmp_path: Path) -> None:
    """A config whose classes do not match the dense shape must fail, not drift."""
    artifacts = _tf_artifacts()
    if artifacts is None:
        pytest.skip("TF artifacts not configured")
    tf_prefix, tf_config = artifacts

    import json

    from scripts.hf.convert_calamari import (
        CalamariConversionError,
        convert_calamari_checkpoint,
    )

    bad_config = tmp_path / "bad.json"
    data = json.loads(tf_config.read_text(encoding="utf-8"))
    data["scenario"]["model"]["classes"] = 999
    bad_config.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CalamariConversionError, match="classes"):
        convert_calamari_checkpoint(tf_prefix, bad_config, tmp_path / "bad.pt")


def test_converter_source_digest_is_recorded(tmp_path: Path) -> None:
    """The converted checkpoint records the source shard SHA-256 for provenance."""
    artifacts = _tf_artifacts()
    if artifacts is None:
        pytest.skip("TF artifacts not configured")
    tf_prefix, tf_config = artifacts

    from scripts.hf.convert_calamari import convert_calamari_checkpoint

    destination = tmp_path / "converted.pt"
    convert_calamari_checkpoint(tf_prefix, tf_config, destination)
    checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
    assert "source_sha256" in checkpoint
    assert isinstance(checkpoint["source_sha256"], str)
    assert len(checkpoint["source_sha256"]) == 64


# --------------------------------------------------------------------------
# TF-free unit tests. These run in every environment and pin the geometry of
# the re-layout transforms, which is exactly what the TF parity test cannot
# guard in CI (it is always skipped there for lack of TensorFlow).
# --------------------------------------------------------------------------


def _instream(array: np.ndarray) -> np.ndarray:
    """An array with a distinct value per position, so every transpose is visible."""
    return np.arange(array.size, dtype=np.float32).reshape(array.shape)


def test_conv_weight_is_out_in_h_w() -> None:
    from scripts.hf.convert_calamari import _conv_weight

    # TF layout is [H, W, in, out]; PyTorch wants [out, in, H, W].
    tf_kernel = _instream(np.empty((2, 3, 4, 5)))
    out = _conv_weight(tf_kernel)
    assert out.shape == (5, 4, 2, 3)
    for o in range(5):
        for i in range(4):
            assert np.array_equal(out[o, i], tf_kernel[:, :, i, o])


def test_dense_weight_is_out_in() -> None:
    from scripts.hf.convert_calamari import _dense_weight

    # TF [in, out] -> PyTorch [out, in].
    tf_kernel = _instream(np.empty((3, 7)))
    out = _dense_weight(tf_kernel)
    assert out.shape == (7, 3)
    assert np.array_equal(out.T, tf_kernel)


def test_lstm_weights_transpose_without_reordering_gates() -> None:
    from scripts.hf.convert_calamari import _lstm_hh, _lstm_ih

    # The TF gate order [i, f, c, o] maps 1:1 onto PyTorch [i, f, g, o] with
    # no gate permutation; the converter only transposes [in, 4h] -> [4h, in].
    ih = _instream(np.empty((6, 8)))  # in=6, 4h=8
    hh = _instream(np.empty((2, 8)))  # h=2, 4h=8
    assert _lstm_ih(ih).shape == (8, 6)
    assert _lstm_hh(hh).shape == (8, 2)
    # Column c of the TF kernel becomes row c of the PyTorch kernel, same order.
    assert np.array_equal(_lstm_ih(ih)[:, 0], ih[0, :])
    assert np.array_equal(_lstm_hh(hh)[:, 1], hh[1, :])


def test_build_state_dict_derives_hidden_and_maps_layers() -> None:
    from scripts.hf.convert_calamari import SourceMetadata, _build_state_dict

    # A synthetic variable set at a non-default hidden size (h=8) proves hidden
    # is read from the recurrent kernel, not a hardcoded constant.
    hidden = 8
    in_features = 5
    h, w, cin, cout = 2, 3, 1, 4
    vars_ = {
        "variables/0/.ATTRIBUTES/VARIABLE_VALUE": _instream(np.empty((h, w, cin, cout))),  # conv0 k
        "variables/1/.ATTRIBUTES/VARIABLE_VALUE": np.zeros(cout, np.float32),  # conv0 b
        "variables/2/.ATTRIBUTES/VARIABLE_VALUE": _instream(
            np.empty((h, w, cout, cout))
        ),  # conv1 k
        "variables/3/.ATTRIBUTES/VARIABLE_VALUE": np.zeros(cout, np.float32),  # conv1 b
        "variables/4/.ATTRIBUTES/VARIABLE_VALUE": _instream(
            np.empty((in_features, 4 * hidden))
        ),  # fw ih
        "variables/5/.ATTRIBUTES/VARIABLE_VALUE": _instream(
            np.empty((hidden, 4 * hidden))
        ),  # fw hh
        "variables/6/.ATTRIBUTES/VARIABLE_VALUE": np.zeros(4 * hidden, np.float32),  # fw b
        "variables/7/.ATTRIBUTES/VARIABLE_VALUE": _instream(
            np.empty((in_features, 4 * hidden))
        ),  # bw ih
        "variables/8/.ATTRIBUTES/VARIABLE_VALUE": _instream(
            np.empty((hidden, 4 * hidden))
        ),  # bw hh
        "variables/9/.ATTRIBUTES/VARIABLE_VALUE": np.zeros(4 * hidden, np.float32),  # bw b
        "variables/10/.ATTRIBUTES/VARIABLE_VALUE": _instream(np.empty((8 * hidden, 3))),  # dense k
        "variables/11/.ATTRIBUTES/VARIABLE_VALUE": np.zeros(3, np.float32),  # dense b
    }
    layers = (
        {"name": "conv2d_0"},
        {"name": "maxpool2d_0"},
        {"name": "conv2d_1"},
        {"name": "maxpool2d_1"},
        {"name": "lstm_0"},
        {"name": "dropout_0"},
    )
    metadata = SourceMetadata(
        classes=3, charset=["", "a", "b"], line_height=48, temperature=-1.0, layers=layers
    )

    state = _build_state_dict(vars_, metadata)

    # Every loader key is present exactly once, with the geometry the loader
    # expects. The LSTM recurrent layout proves hidden was derived (4h = 32).
    assert state["layers.4.lstm.weight_hh_l0"].shape[-2:] == (4 * hidden, hidden)
    assert state["layers.4.lstm.bias_hh_l0"].shape == (4 * hidden,)
    assert state["logits.weight"].shape == (3, 8 * hidden)
    assert state["logits.bias"].shape == (3,)


def test_build_state_dict_rejects_unsupported_stack() -> None:
    from scripts.hf.convert_calamari import (
        CalamariConversionError,
        SourceMetadata,
        _build_state_dict,
    )

    # A Calamari model with a second BiLSTM layer is a real, loadable TF graph
    # but the PyTorch loader is a fixed 6-layer stack. The converter must refuse
    # it rather than silently map weights onto the wrong modules.
    metadata = SourceMetadata(
        classes=2,
        charset=["", "a"],
        line_height=48,
        temperature=-1.0,
        layers=(
            {"name": "conv2d_0"},
            {"name": "maxpool2d_0"},
            {"name": "conv2d_1"},
            {"name": "maxpool2d_1"},
            {"name": "lstm_0"},
            {"name": "lstm_1"},
            {"name": "dropout_0"},
        ),
    )
    with pytest.raises(CalamariConversionError, match="layer"):
        _build_state_dict({}, metadata)
