# Models and datasets

Nomicous separates training and data preparation from production inference.
Training tools live in `src/`; the deployable runtime reads the verified
catalog in [`inference/registry.yaml`](../../inference/registry.yaml).

## Runtime models

### BLLA segmentation

`blla-segment` uses an inference-owned BLLA topology exported as ONNX to identify
line candidates on a page. Nomicous converts those candidates into editable
geometry, preserves the legacy `kraken_ceiling` field, simplifies polygons,
and can optionally refine or split candidates with Otsu-based processing.

### Calamari HTR

`syriac-calamari-v1` uses the ONNX Runtime Calamari graph for line
transcription:

```text
CNN → max pooling → CNN → max pooling
   → 200-unit bidirectional LSTM → dropout → linear CTC logits
   → greedy text decoding + character confidences
```

The conversion path validates tensor-only `calamari-pytorch-v1` checkpoints,
then exports the runtime artifact as ONNX. It uses safe `weights_only` loading
and verifies the configured artifact digest.
The vendored TensorFlow Calamari tree is used for training, not shipped in the
inference image.

Calamari and BLLA are capable enough for the current manuscript workflow while
remaining practical for CPU-first local execution. The helper does not require
CUDA, a GPU, or a training environment.

TrOCR appears in research experiments only. It has no runtime adapter, registry
entry, packaged checkpoint, or platform catalog path and is not supported by
the product.

## Current catalog

| ID                   | Task       | Architecture     | Artifact                                                                                         |
| -------------------- | ---------- | ---------------- | ------------------------------------------------------------------------------------------------ |
| `blla-segment`     | Segment    | BLLA ONNX       | `blla.onnx` from [segmentation repo](https://huggingface.co/kkkamur07/segmentation-blla) |
| `syriac-calamari-v1` | Transcribe | Calamari ONNX | [Hugging Face checkpoint](https://huggingface.co/kkkamur07/syriac-htr-calamari), pinned revision |

Greek Calamari is commented out because its Hub repository and verified
artifact are unavailable. Coptic, Armenian, and additional Greek models are
expansion targets rather than shipped runtime models.

New public models need a compatible adapter, immutable Hub revision, SHA-256
digest, registry entry, platform catalog metadata, tests, and a declared host
eligibility.

## From correction to training data

The intended expert-in-the-loop pipeline is:

1. Annotate page geometry.
2. Pair each segment with a transcription.
3. Correct and review the text as ground truth.
4. Export processed line images and transcription files.
5. Stage labelled crops under `src/hf/staging/datasets/`.
6. Validate and publish a separate Hugging Face dataset repository.
7. Train or fine-tune a script-specific model.
8. Publish verified weights and add the model to the registry.

The staging layout is:

```text
src/hf/staging/datasets/<dataset-slug>/
  images/
    manuscript-001/line-0001.png
  labels.csv
```

`labels.csv` is UTF-8 CSV with `image,transcription` columns. Each crop needs
one non-empty transcription. Source crops remain independent of
model-specific resizing and normalization.

Preparation tools support PAGE XML and eScriptorium/Transkribus-style inputs.
This repository provides the preparation and publishing workflow, but it does
not establish the size, rights, license, provenance, or publication status of
every planned corpus. Call a dataset expert-curated only when its release
documents the experts, review process, provenance, rights, and license.

## Publish a model

Dry-run validation:

```bash
PYTHONPATH=. python scripts/hf/publish_model.py \
  --script greek \
  --architecture calamari \
  --model-version v1 \
  --registry-tag stable \
  --task transcribe
```

After setting a write-capable `HF_TOKEN`, add `--upload`. Then pin the
immutable revision and SHA-256 in `inference/registry.yaml`, run tests,
prefetch the weights, update platform catalog metadata, and deploy the
matching runtime.

Dataset validation:

```bash
PYTHONPATH=. python scripts/hf/publish_dataset.py \
  greek-manuscript-lines --script greek
```

Use `--upload` only for a reviewed release. See
[`adding-inference-models.md`](adding-inference-models.md) and the
[Hub publishing reference](../../scripts/hf/README.md).
