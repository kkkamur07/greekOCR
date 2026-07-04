"""Shared schemas and helpers."""

from __future__ import annotations

import base64
import binascii
from enum import StrEnum
from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator


class MLTask(StrEnum):
    segment = "segment"
    transcribe = "transcribe"
    binarize = "binarize"


class MLJobStatus(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class RegistryArchitecture(StrEnum):
    calamari = "calamari"
    trocr = "trocr"
    kraken_segment = "kraken-segment"


class ComputeDevice(StrEnum):
    cpu = "cpu"
    cuda = "cuda"
    any = "any"

def _coerce_image_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value

    if isinstance(value, str):
        normalized = "".join(value.split())

        try:
            return base64.b64decode(normalized, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_bytes must be valid base64") from exc

    raise TypeError("image_bytes must be bytes or base64 string")


def _serialize_image_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode()


ImageBytes = Annotated[
    bytes,
    PlainValidator(_coerce_image_bytes),
    PlainSerializer(_serialize_image_bytes),
]

__all__ = [
    "ComputeDevice",
    "ImageBytes",
    "MLJobStatus",
    "MLTask",
    "RegistryArchitecture",
]
