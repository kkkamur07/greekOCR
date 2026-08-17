"""Deterministic character codec and CTC decoding for Calamari."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class CharacterCodec:
    """A blank-first, character-level CTC vocabulary."""

    charset: tuple[str, ...]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> CharacterCodec:
        return cls(("", *sorted({character for text in texts for character in text})))

    def __post_init__(self) -> None:
        if len(self.charset) < 2 or self.charset[0] != "" or len(set(self.charset)) != len(self.charset):
            raise ValueError("A codec must have a unique blank entry at index zero.")

    @property
    def classes(self) -> int:
        return len(self.charset)

    def encode(self, text: str) -> Tensor:
        ids = {character: index for index, character in enumerate(self.charset)}
        try:
            return torch.tensor([ids[character] for character in text], dtype=torch.long)
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is absent from the codec.") from error

    def decode_ctc(self, token_ids: Sequence[int]) -> str:
        output: list[str] = []
        previous = -1
        for token_id in token_ids:
            if token_id != 0 and token_id != previous:
                output.append(self.charset[token_id])
            previous = token_id
        return "".join(output)

    def decode_logits(self, logits: Tensor, lengths: Tensor) -> list[str]:
        predictions = logits.argmax(dim=-1).detach().cpu()
        return [
            self.decode_ctc(row[: int(length)].tolist())
            for row, length in zip(predictions, lengths.detach().cpu(), strict=True)
        ]
