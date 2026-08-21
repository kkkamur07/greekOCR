"""Tokenizer loading and processor assembly."""

from pathlib import Path

from .builder import build_processor, load_tokenizer


_BUNDLED_TOKENIZER_DIRECTORIES = {
    "gpt": "gpt_tokenizer",
    "gpt_tokenizer": "gpt_tokenizer",
    "gpt_armenian_500": "gpt_armenian_500",
    "gpt_greek_500": "gpt_greek_500",
    "gpt_syriac_500": "gpt_syriac_500",
    "trocr": "trocr",
}


def bundled_tokenizer_path(name: str) -> Path:
    """Resolve a tokenizer bundled alongside this module."""
    try:
        directory = _BUNDLED_TOKENIZER_DIRECTORIES[name]
    except KeyError as error:
        available = ", ".join(sorted(_BUNDLED_TOKENIZER_DIRECTORIES))
        raise ValueError(f"Unknown bundled tokenizer {name!r}; choose one of: {available}.") from error

    path = Path(__file__).parent / directory
    if not path.is_dir():
        raise FileNotFoundError(f"Bundled tokenizer directory is missing: {path}")
    return path


def resolve_tokenizer_path(value: str | Path) -> Path:
    """Resolve a bundled tokenizer or a caller-provided tokenizer directory."""
    value_as_string = str(value)
    if value_as_string.startswith("bundled:"):
        return bundled_tokenizer_path(value_as_string.removeprefix("bundled:"))
    return Path(value).expanduser().resolve()


__all__ = ["build_processor", "load_tokenizer", "resolve_tokenizer_path"]
