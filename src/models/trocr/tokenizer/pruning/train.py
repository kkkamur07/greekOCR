"""Train a compact Greek/Syriac byte-level BPE tokenizer from GPT's base bytes.

The existing GPT tokenizer has a 50k-token vocabulary.  BPE merges only add
tokens, so it cannot be extended and then reduced to 1,700 entries.  This
script instead reuses its GPT-2 byte-level pre-tokenization, byte fallback, and
special-token configuration, then learns a fresh set of merges from the 256
base byte tokens on all processed pretraining and finetuning transcriptions.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from .. import bundled_tokenizer_path
from ..builder import load_tokenizer


DEFAULT_CORPUS_DIRECTORIES = (
    Path("data/trocr_processed/greek/pretraining"),
    Path("data/trocr_processed/greek/finetuning"),
    Path("data/trocr_processed/syriac/pretraining"),
    Path("data/trocr_processed/syriac/finetuning"),
)


@dataclass(frozen=True)
class TokenizationStatistics:
    """Summary statistics for validating a trained tokenizer."""

    records: int
    tokens: int
    unknown_tokens: int
    maximum_length: int


def iter_transcriptions(corpus_directories: Sequence[Path]) -> Iterator[str]:
    """Yield every transcription from ``gt_*.txt`` corpus manifests."""
    for corpus_directory in corpus_directories:
        corpus_directory = corpus_directory.expanduser().resolve()
        manifests = sorted(corpus_directory.glob("gt_*.txt"))
        if not manifests:
            raise FileNotFoundError(f"No gt_*.txt manifests found in {corpus_directory}")
        for manifest in manifests:
            for line_number, line in enumerate(
                manifest.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                try:
                    _, transcription = line.split("\t", maxsplit=1)
                except ValueError as error:
                    raise ValueError(
                        f"{manifest}:{line_number} must contain image<TAB>transcription"
                    ) from error
                yield transcription


def train_byte_bpe(
    corpus_directories: Sequence[Path],
    *,
    source_tokenizer_directory: Path,
    vocabulary_size: int = 2000,
) -> PreTrainedTokenizerBase:
    """Train byte-level GPT BPE merges while retaining its base byte alphabet."""
    if vocabulary_size < 259:
        raise ValueError("vocabulary_size must allow 256 byte tokens and three special tokens")

    source_tokenizer = load_tokenizer(
        str(source_tokenizer_directory),
        use_fast=True,
        pad_token="<|pad|>",
    )
    trained_tokenizer = source_tokenizer.train_new_from_iterator(
        iter_transcriptions(corpus_directories),
        vocab_size=vocabulary_size,
    )
    if len(trained_tokenizer) != vocabulary_size:
        raise RuntimeError(
            f"Expected a {vocabulary_size}-token vocabulary, got {len(trained_tokenizer)}"
        )
    if trained_tokenizer.pad_token_id is None or trained_tokenizer.unk_token_id is None:
        raise RuntimeError("Trained tokenizer is missing required PAD or UNK token")
    return trained_tokenizer


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    corpus_directories: Sequence[Path],
) -> TokenizationStatistics:
    """Verify exact round trips and report sequence-length/unknown-token totals."""
    records = 0
    token_count = 0
    unknown_count = 0
    maximum_length = 0
    for transcription in iter_transcriptions(corpus_directories):
        token_ids = tokenizer.encode(transcription, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != transcription:
            raise ValueError(
                "Tokenizer does not round-trip a transcription exactly: "
                f"{transcription!r} became {decoded!r}"
            )
        records += 1
        token_count += len(token_ids)
        unknown_count += token_ids.count(tokenizer.unk_token_id)
        maximum_length = max(maximum_length, len(token_ids))
    return TokenizationStatistics(records, token_count, unknown_count, maximum_length)


def train_and_save(
    corpus_directories: Sequence[Path],
    *,
    source_tokenizer_directory: Path,
    output_directory: Path,
    vocabulary_size: int = 1700,
    overwrite: bool = False,
) -> TokenizationStatistics:
    """Train, validate, and save a compact byte-level BPE tokenizer."""
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    tokenizer = train_byte_bpe(
        corpus_directories,
        source_tokenizer_directory=source_tokenizer_directory,
        vocabulary_size=vocabulary_size,
    )
    statistics = validate_tokenizer(tokenizer, corpus_directories)
    if statistics.unknown_tokens:
        raise ValueError(f"Byte-level tokenizer emitted {statistics.unknown_tokens} unknown tokens")
    tokenizer.save_pretrained(output_directory)
    return statistics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BPE tokenizer training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CORPUS_DIRECTORIES),
        help="Processed Greek/Syriac pretraining and finetuning directories.",
    )
    parser.add_argument(
        "--source-tokenizer",
        type=Path,
        default=bundled_tokenizer_path("gpt_tokenizer"),
        help="Current GPT tokenizer used as the byte-level template.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for the trained tokenizer.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1700,
        help="Total vocabulary size including base bytes and special tokens (default: 1700).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --output.")
    return parser.parse_args()


def main() -> None:
    """Train and save the compact Greek/Syriac GPT BPE tokenizer."""
    args = parse_args()
    statistics = train_and_save(
        args.corpus,
        source_tokenizer_directory=args.source_tokenizer,
        output_directory=args.output,
        vocabulary_size=args.vocab_size,
        overwrite=args.overwrite,
    )
    print(
        f"Saved {args.vocab_size:,}-token byte-level BPE tokenizer to {args.output}; "
        f"{statistics.records:,} records, {statistics.tokens:,} tokens, "
        f"{statistics.unknown_tokens} unknowns, max length {statistics.maximum_length}"
    )


if __name__ == "__main__":
    main()
