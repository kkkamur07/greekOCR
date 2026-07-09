"""Hub staging tree layout, validation, and card generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.hf.paths import DEFAULT_STAGING_ROOT
from src.hf.resolve.artifacts import find_hub_artifact


@dataclass(frozen=True, slots=True)
class ModelStagingRef:
  script: str
  architecture: str
  model_version: str
  registry_tag: str

  @property
  def registry_model_id(self) -> str:
    return f"{self.script}-{self.architecture}-{self.model_version}"

  @property
  def hub_repo_slug(self) -> str:
    return hub_repo_slug(self.script, self.architecture)

  def weights_source(self, *, namespace: str) -> str:
    return f"hf://{namespace}/{self.hub_repo_slug}@{self.registry_tag}"


def hub_repo_slug(script: str, architecture: str) -> str:
  return f"{script}-htr-{architecture}"


# Hub model card `language` metadata requires ISO 639 codes, not script slugs.
_SCRIPT_LANGUAGE_CODES: dict[str, str] = {
  "greek": "grc",
  "syriac": "syr",
}


def hub_language_code(script: str) -> str:
  return _SCRIPT_LANGUAGE_CODES.get(script, script)


def model_staging_dir(
  ref: ModelStagingRef,
  *,
  staging_root: Path | None = None,
) -> Path:
  root = staging_root or DEFAULT_STAGING_ROOT
  return (
    root
    / "models"
    / ref.script
    / ref.architecture
    / ref.model_version
    / ref.registry_tag
  )


def validate_model_staging(
  staging_dir: Path,
  *,
  architecture: str,
) -> None:
  if not staging_dir.is_dir():
    raise ValueError(f"model staging directory not found: {staging_dir}")

  try:
    find_hub_artifact(staging_dir, architecture=architecture)
  except FileNotFoundError as exc:
    raise ValueError(
      f"model staging directory {staging_dir} is missing a supported Hub artifact "
      f"for architecture {architecture!r}"
    ) from exc


def build_model_card(
  ref: ModelStagingRef,
  *,
  namespace: str,
  task: str,
  registry_model_id: str | None = None,
) -> str:
  model_id = registry_model_id or ref.registry_model_id
  weights_source = f"hf://{namespace}/{ref.hub_repo_slug}@{ref.registry_tag}"
  title_script = ref.script.capitalize()

  language_code = hub_language_code(ref.script)
  return f"""---
language:
- {language_code}
tags:
- handwritten-text-recognition
- {ref.architecture}
- ocr
library_name: {ref.architecture}
---

# {title_script} HTR ({ref.architecture})

Handwritten text recognition checkpoint for **{ref.script}** manuscripts, published from the nomicous **Hub staging tree**.

| Field | Value |
|-------|-------|
| **registry model id** | `{model_id}` |
| **registry tag** | `{ref.registry_tag}` |
| **script** | `{ref.script}` |
| **architecture** | `{ref.architecture}` |
| **model version** | `{ref.model_version}` |
| **task** | `{task}` |
| **weights source** | `{weights_source}` |

## Usage

Resolve this checkpoint through the inference **Registry** with:

```yaml
weights_source: {weights_source}
```

Prefetch into the **Hub cache** without running inference:

```bash
PYTHONPATH=. python scripts/hf/fetch_model.py {model_id} --registry-tag {ref.registry_tag}
```
"""


@dataclass(frozen=True, slots=True)
class DatasetStagingRef:
  dataset_slug: str

  def repo_id(self, *, namespace: str) -> str:
    return f"{namespace}/{self.dataset_slug}"


def dataset_staging_dir(
  ref: DatasetStagingRef,
  *,
  staging_root: Path | None = None,
) -> Path:
  root = staging_root or DEFAULT_STAGING_ROOT
  return root / "datasets" / ref.dataset_slug


def validate_dataset_staging(staging_dir: Path) -> None:
  if not staging_dir.is_dir():
    raise ValueError(f"dataset staging directory not found: {staging_dir}")

  data_markers = (
    "data",
    "train",
    "images",
    "metadata.json",
    "metadata.yaml",
    "labels.csv",
    "labels.jsonl",
  )
  for name in data_markers:
    candidate = staging_dir / name
    if candidate.exists():
      return

  if any(staging_dir.iterdir()):
    return

  raise ValueError(
    f"dataset staging directory {staging_dir} has no recognised data layout "
    f"(expected one of {', '.join(data_markers)} or non-empty files)"
  )


def build_dataset_readme(
  ref: DatasetStagingRef,
  *,
  namespace: str,
  script: str,
  crop_format: str = "PNG line crops with UTF-8 transcriptions",
  provenance: str = "Curated from nomicous annotation exports.",
) -> str:
  return f"""---
language:
- {script}
tags:
- handwritten-text-recognition
- manuscript
- line-crops
task_categories:
- image-to-text
---

# {ref.dataset_slug}

Labelled manuscript line crops for training and evaluating **{script}** handwritten text recognition models.

| Field | Value |
|-------|-------|
| **Hub dataset slug** | `{ref.dataset_slug}` |
| **script** | `{script}` |
| **crop format** | {crop_format} |
| **provenance** | {provenance} |

## Relationship to inference models

This **Hub dataset repo** holds training/evaluation material only. Inference loads weights from separate **Hub model repos** (for example `{script}-htr-calamari`) via `hf://` **weights sources** in the inference **Registry**.

After publishing a new model generation, update `src/hf/publish/collection.yaml` so the `nomos` **Hub collection** lists both the dataset and the corresponding **Hub model repo**.

## Staging layout

Stage publish-ready files under:

```
src/hf/staging/datasets/{ref.dataset_slug}/
```

Then publish with:

```bash
PYTHONPATH=. python scripts/hf/publish_dataset.py {ref.dataset_slug} --script {script} --upload
```
"""
