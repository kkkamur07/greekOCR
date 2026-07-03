"""Shared enums for ML task contracts and registry."""

from __future__ import annotations

import base64
from enum import Enum
from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator


class MLTask(str, Enum):
    segment = "segment"
    transcribe = "transcribe"
    binarize = "binarize"


class MLJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class RegistryArchitecture(str, Enum):
    calamari = "calamari"
    trocr = "trocr"
    kraken_segment = "kraken-segment"


class ComputeDevice(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    any = "any"


def _coerce_image_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return base64.b64decode(value)
    raise TypeError("image_bytes must be bytes or base64 string")


def _serialize_image_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode()


ImageBytes = Annotated[
    bytes,
    PlainValidator(_coerce_image_bytes),
    PlainSerializer(_serialize_image_bytes),
]
