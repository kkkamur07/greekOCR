# Hub dataset staging

This directory holds publish-ready labelled manuscript line crops for **Hub
dataset repos**. It does not hold inference checkpoints: those belong in the
separate model staging tree at `src/hf/staging/models/`.

Create one directory per **Hub dataset slug**:

```text
src/hf/staging/datasets/{script}-manuscript-lines/
src/hf/staging/datasets/{script}-{corpus}-htr-lines/
```

For example:

```text
src/hf/staging/datasets/greek-manuscript-lines/
  images/
    ms-001/
      line-0001.png
  labels.csv
```

`labels.csv` is UTF-8 CSV with required `image` and `transcription` columns:

```csv
image,transcription
images/ms-001/line-0001.png,λόγος
```

Each `image` path is relative to the dataset root and must begin with
`images/`. Every image crop must have exactly one non-empty transcription row,
and every row must reference an existing crop. Use PNG line crops where
possible, retain original dimensions, and keep model-specific preprocessing out
of the source dataset.

Publish a validated dataset with:

```bash
PYTHONPATH=. python scripts/hf/publish_dataset.py \
  greek-manuscript-lines --script greek
```

The command is a dry-run by default. Add `--upload` with an `HF_TOKEN` write
token to create or update the corresponding **Hub dataset repo**.
