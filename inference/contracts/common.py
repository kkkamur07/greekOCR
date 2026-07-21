"""Shared schemas and helpers."""

from __future__ import annotations

import base64
import binascii
from enum import StrEnum
from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator

from inference.admission import CLIENT_INPUT_ERROR, validate_encoded_image
from inference.infrastructure.settings import get_inference_settings


class InferenceTask(StrEnum):
    segment = "segment"
    transcribe = "transcribe"
    binarize = "binarize"


class InferenceJobStatus(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class RegistryArchitecture(StrEnum):
    calamari = "calamari"
    blla = "blla"
    blla_segment = "blla-segment"


class ComputeDevice(StrEnum):
    cpu = "cpu"
    cuda = "cuda"
    any = "any"


class HostEligibility(StrEnum):
    local = "local"
    remote = "remote"
    any = "any"


def _coerce_image_bytes(value: Any) -> bytes:
    settings = get_inference_settings()
    raw_bytes = validate_encoded_image(value, settings)
    if isinstance(value, bytes):
        return raw_bytes

    if isinstance(value, str):
        normalized = "".join(value.split())

        try:
            decoded = base64.b64decode(normalized, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(CLIENT_INPUT_ERROR) from exc
        if len(decoded) > settings.inference_max_decoded_image_bytes:
            raise ValueError(CLIENT_INPUT_ERROR)
        return decoded

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
    "HostEligibility",
    "ImageBytes",
    "InferenceJobStatus",
    "InferenceTask",
    "RegistryArchitecture",
]
