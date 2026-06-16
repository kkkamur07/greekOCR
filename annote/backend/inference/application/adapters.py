"""Model adapter registry stubs for future segment/transcribe runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ModelAdapter(Protocol):
    key: str

    async def run(self, payload: dict, *, params: dict | None = None) -> dict:
        """Run a model and return a canonical DTO."""


@dataclass(frozen=True)
class NoopModelAdapter:
    key: str

    async def run(self, payload: dict, *, params: dict | None = None) -> dict:
        return {
            "adapter": self.key,
            "payload": payload,
            "params": params or {},
            "status": "not_implemented",
        }


class ModelAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.key] = adapter

    def get(self, key: str) -> ModelAdapter:
        try:
            return self._adapters[key]
        except KeyError:
            raise KeyError(f"model adapter {key!r} is not registered") from None

    def keys(self) -> list[str]:
        return sorted(self._adapters)


def default_adapter_registry() -> ModelAdapterRegistry:
    registry = ModelAdapterRegistry()
    for key in (
        "kraken:segment",
        "kraken:transcribe",
        "trocr:transcribe",
        "huggingface:transcribe",
    ):
        registry.register(NoopModelAdapter(key=key))
    return registry
