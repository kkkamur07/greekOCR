"""Export a kraken PP-OCRv6 recognition checkpoint as a self-contained ONNX artifact.

The checkpoint is a kraken ``PPOCRv6Model`` (``kraken.lib.ppocr.PPOCRv6Model``,
kraken 7.1.1): a PPLCNetV4 backbone, a LightSVTR neck whose attention is masked
by ``seq_lens``, and a CTC head. ``forward(x[N,3,96,W], seq_lens)`` returns
``(logits[N,1623,1,T], out_lens)`` with class 0 the CTC blank and no softmax
inside the graph (`T` follows the width-to-time formula below, nominally
`W / 8`).

The exported graph serves batch size 1. It takes one input ``image``
(NCHW ``[1, 3, 96, W]``, already preprocessed by the kraken recipe) and returns
one output ``logits`` (``[1, T, classes]`` with time on axis 1). The batch is
static; only the width axis is dynamic. There is no ``seq_lens`` input: the
graph serves the unmasked path. Kraken's own inference does pass per-line
widths, and its float32 floor can mask the last frame off on some widths;
that masking quirk is kraken's behavior, measured and recorded by the parity
script, not reproduced here (see the parity doc).

Two export-time rewrites, both on a deepcopy (the checkpoint bytes and the
loaded model are untouched):

1. The backbone ends in ``F.avg_pool2d(x, kernel_size=(h, 2))`` where ``h`` is
   read off the feature map (``kraken/lib/ppocr/backbone.py``, ``PPLCNetV4``).
   ``h`` depends only on the input height, which this contract fixes at 96, so
   it is probed once and frozen to that integer. The legacy exporter cannot
   trace a shape value used as a kernel size; freezing it is exactly equivalent
   for every width. This mirrors the ``_with_export_group_norm`` precedent in
   ``nomikos_inference/export/blla/export.py``: swap on a copy, prove parity.
2. The wrapper squeezes the singleton height and permutes the kraken
   ``[N, C, 1, W']`` layout to ``[N, W', C]`` so time sits on axis 1.

Kraken is a publish-time dependency only: it is imported lazily inside
:func:`export_ppocr_rec_onnx` and the caller gets a clear error when it is
missing. Nothing else in this package may import kraken.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: ONNX ``metadata_props["format"]`` for this contract.
FORMAT = "ppocr-rec-onnx-v1"
#: Architecture name a serving adapter would register under.
ARCHITECTURE = "ppocr_rec"
#: Provenance label for the checkpoint side of the export.
SOURCE_FORMAT = "kraken-safetensors"
#: Graph input: already-preprocessed NCHW line image.
INPUT_NAME = "image"
#: Graph output: raw CTC logits, time on axis 1.
OUTPUT_NAME = "logits"
#: House default, shared with the Calamari exporter.
DEFAULT_OPSET_VERSION = 17
#: Example width traced at export time. Must be a multiple of 8; only the
#: traced rank is kept, the width itself stays dynamic.
DEFAULT_EXAMPLE_WIDTH = 320
#: Nominal width stride of the recognizer (stem stride 4, final pooling 2).
SUBSAMPLING = 8

#: Exact width-to-time rule of the recognizer: the stem downsamples width by 4
#: with SAME-padded strided convolutions (``floor((W - 1) / 2) + 1`` twice) and
#: the final pooling halves once more with flooring. Verified against Torch for
#: every integer width 5..512 and the parity sweep (see the parity doc); the
#: exporter re-probes it for the model at hand and refuses to write a formula
#: the model disagrees with.
TIME_FORMULA = "T = (((W + 1) // 2 + 1) // 2) // 2"


def _time_steps(width: int) -> int:
    return (((width + 1) // 2 + 1) // 2) // 2


@dataclass(frozen=True)
class PPOCRRecExportReport:
    """Provenance for one export; also the operator's half of a registry entry."""

    classes: int
    variant: str
    line_height: int
    subsampling: int
    blank_index: int
    opset: int
    exporter: str
    example_width: int
    source_path: str
    source_sha256: str
    artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "classes": self.classes,
            "variant": self.variant,
            "line_height": self.line_height,
            "subsampling": self.subsampling,
            "blank_index": self.blank_index,
            "opset": self.opset,
            "exporter": self.exporter,
            "example_width": self.example_width,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


class _PPOCRRecONNXWrapper(nn.Module):
    """Serve ``[1, T, C]`` logits from the kraken ``[N, C, 1, W']`` graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> Tensor:
        logits, _ = self.model(image)
        return logits.squeeze(2).permute(0, 2, 1)


def _with_static_final_pool(model: nn.Module, line_height: int) -> nn.Module:
    """Copy ``model`` with the backbone's final pooling kernel frozen.

    ``PPLCNetV4.forward`` pools with ``kernel_size=(h, 2)`` where ``h`` is the
    live feature height. Every convolution above it preserves or strides the
    fixed input height deterministically, so for a fixed ``line_height`` the
    value is a constant; probing it once and inlining the integer traces the
    identical arithmetic while giving the legacy exporter the static kernel it
    requires. Width never enters the kernel, so no width is favoured.
    """
    exportable = copy.deepcopy(model)
    backbone = exportable.nn.backbone
    probe = torch.zeros((1, 3, line_height, 64), dtype=torch.float32)
    with torch.no_grad():
        features = backbone.conv1(probe)
        for stage in (
            backbone.blocks2,
            backbone.blocks3,
            backbone.blocks4,
            backbone.blocks5,
            backbone.blocks6,
        ):
            features = stage(features)
        pool_height = int(features.shape[2])

    def forward(self: nn.Module, inputs: Tensor) -> Tensor:
        values = backbone.conv1(inputs)
        values = backbone.blocks2(values)
        values = backbone.blocks3(values)
        values = backbone.blocks4(values)
        values = backbone.blocks5(values)
        values = backbone.blocks6(values)
        return F.avg_pool2d(values, kernel_size=(pool_height, 2))

    backbone.forward = forward.__get__(backbone)  # type: ignore[method-assign]
    return exportable.eval()


def _load_ppocr_rec_model(source: Path) -> tuple[nn.Module, dict[str, object]]:
    """Open a kraken PP-OCRv6 recognition checkpoint; kraken stays lazy here."""
    try:
        from kraken.lib.ppocr.model import PPOCRv6Model
        from kraken.models import load_models
    except ImportError as error:
        raise RuntimeError(
            "kraken 7.1.0 or newer is required to export a PP-OCRv6 recognition "
            "checkpoint; install it in the export environment"
        ) from error
    try:
        kraken_version = importlib.metadata.version("kraken")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("kraken is installed but its version is unreadable") from error
    try:
        models = load_models(str(source))
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"unable to load PP-OCRv6 checkpoint: {source}") from error
    if len(models) != 1:
        raise ValueError(f"expected exactly one model in {source}, found {len(models)}")
    model = models[0]
    if not isinstance(model, PPOCRv6Model):
        raise ValueError(f"expected a PPOCRv6Model in {source}, found {type(model).__name__}")
    if model.codec is None:
        raise ValueError(f"model in {source} has no codec; cannot build the charset")
    model.eval()
    info: dict[str, object] = {
        "variant": model.variant,
        "num_classes": model.num_classes,
        "line_height": int(model.input[2]),
        "seg_type": model.seg_type,
        "kraken_version": kraken_version,
    }
    return model, info


def _charset_list(model: nn.Module, num_classes: int) -> list[str]:
    """Index-to-grapheme list with the Calamari convention: index 0 is blank.

    Kraken labels are 1-indexed with 0 reserved for the CTC blank
    (``kraken/lib/codec.py``, ``PytorchCodec``), so entry 0 is the empty
    string and entry ``i`` is the grapheme labelled ``i``. A grapheme may hold
    several codepoints (it is still one string); a codec entry spanning several
    labels has no single index and is rejected rather than silently dropped.
    """
    c2l = model.codec.c2l
    multi = {grapheme: labels for grapheme, labels in c2l.items() if len(labels) != 1}
    if multi:
        raise ValueError(
            f"codec entries with several labels cannot be indexed by class: {sorted(multi)[:5]}"
        )
    charset = [""] * num_classes
    for grapheme, (label,) in c2l.items():
        if not 1 <= label < num_classes:
            raise ValueError(f"codec label {label} for {grapheme!r} is out of range")
        if charset[label]:
            raise ValueError(f"duplicate codec label {label}")
        charset[label] = grapheme
    if any(entry == "" for entry in charset[1:]):
        raise ValueError("codec labels are not contiguous from 1")
    return charset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_time_formula(model: nn.Module, line_height: int) -> None:
    """Refuse to stamp a width-to-time formula the model disagrees with."""
    with torch.no_grad(), torch.inference_mode():
        for width in range(8, 49):
            probe = torch.zeros((1, 3, line_height, width), dtype=torch.float32)
            logits, _ = model(probe)
            if logits.shape[3] != _time_steps(width):
                raise RuntimeError(
                    f"width-to-time formula {TIME_FORMULA!r} mispredicts width "
                    f"{width}: model gives T={logits.shape[3]}"
                )


def export_ppocr_rec_onnx(
    source: Path,
    destination: Path,
    *,
    opset_version: int = DEFAULT_OPSET_VERSION,
    example_width: int = DEFAULT_EXAMPLE_WIDTH,
) -> PPOCRRecExportReport:
    """Export a kraken PP-OCRv6 recognition checkpoint and return provenance."""
    if source.suffix != ".safetensors":
        raise ValueError("PP-OCRv6 recognition export requires a .safetensors checkpoint")
    if example_width < 8 or example_width % SUBSAMPLING:
        raise ValueError("example_width must be a multiple of 8 and at least 8")
    if opset_version < 17:
        raise ValueError("opset_version must be at least 17")

    model, info = _load_ppocr_rec_model(source)
    num_classes = int(info["num_classes"])
    line_height = int(info["line_height"])
    variant = str(info["variant"])
    seg_type = str(info["seg_type"])
    kraken_version = str(info["kraken_version"])
    if num_classes < 2:
        raise ValueError(f"model in {source} has {num_classes} classes")
    if line_height <= 0:
        raise ValueError(f"model in {source} has invalid input height")

    charset = _charset_list(model, num_classes)
    exportable = _with_static_final_pool(model, line_height)
    _check_time_formula(exportable, line_height)
    wrapper = _PPOCRRecONNXWrapper(exportable).eval()
    example = torch.zeros((1, 3, line_height, example_width), dtype=torch.float32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            example,
            destination,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                # Batch is static: serving submits one line at a time, and a
                # padded batch gives different logits than single lines in
                # kraken itself, so parity is defined at batch size 1.
                INPUT_NAME: {3: "width"},
                OUTPUT_NAME: {1: "time"},
            },
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"unable to export PP-OCRv6 recognition ONNX artifact: {destination}"
        ) from error

    try:
        import onnx

        onnx_model = onnx.load(destination)
        del onnx_model.metadata_props[:]
        metadata = _metadata_values(
            variant=variant,
            line_height=line_height,
            num_classes=num_classes,
            charset=charset,
            seg_type=seg_type,
            opset_version=opset_version,
            source=source,
            kraken_version=kraken_version,
        )
        for key, value in metadata.items():
            prop = onnx_model.metadata_props.add()
            prop.key = key
            prop.value = value
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, destination)
    except Exception as error:
        raise RuntimeError(
            f"unable to embed PP-OCRv6 recognition ONNX metadata: {destination}"
        ) from error

    return PPOCRRecExportReport(
        classes=num_classes,
        variant=variant,
        line_height=line_height,
        subsampling=SUBSAMPLING,
        blank_index=0,
        opset=opset_version,
        exporter=f"torch.onnx.export legacy dynamo=False (torch {torch.__version__})",
        example_width=example_width,
        source_path=str(source),
        source_sha256=_sha256(source),
        artifact_sha256=_sha256(destination),
    )


def _metadata_values(
    *,
    variant: str,
    line_height: int,
    num_classes: int,
    charset: list[str],
    seg_type: str,
    opset_version: int,
    source: Path,
    kraken_version: str,
) -> dict[str, str]:
    # Padding and temperature are kraken inference-config defaults
    # (``kraken/configs/base.py``, ``RecognitionInferenceConfig``): padding 16
    # is the blank margin on each side of a line, temperature 1.0 leaves the
    # softmax unchanged. Both are recorded so serving can reproduce inference;
    # neither lives in the graph.
    return {
        "format": FORMAT,
        "architecture": ARCHITECTURE,
        "variant": variant,
        "input_layout": "NCHW",
        "input_name": INPUT_NAME,
        "output_names": json.dumps([OUTPUT_NAME]),
        "input_channels": "3",
        "line_height": str(line_height),
        "subsampling": str(SUBSAMPLING),
        "time_formula": TIME_FORMULA,
        "classes": str(num_classes),
        "blank_index": "0",
        "charset": json.dumps(charset, ensure_ascii=False),
        "pad": "16",
        "pad_fill": "255",
        "temperature": "1.0",
        "seg_type": seg_type,
        "preprocessing": (
            "kraken recognition recipe: RGB, fixed-height resize to 96, "
            "horizontal white padding, scaled to [0, 1], inverted"
        ),
        "opset_version": str(opset_version),
        "source_format": SOURCE_FORMAT,
        "source_sha256": _sha256(source),
        "kraken_version": kraken_version,
        "torch_version": torch.__version__,
    }


__all__ = ["PPOCRRecExportReport", "export_ppocr_rec_onnx"]
