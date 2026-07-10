# Hugging Face Hub scripts

Operator tooling for publishing inference weights and datasets from the **Hub staging tree**, and for syncing the `nomos` **Hub collection**. Domain vocabulary is defined in [`inference/CONTEXT.md`](../../inference/CONTEXT.md).

## Hub staging tree

Publish-ready artifacts live under `src/hf/staging/` before upload:

| Kind | Path |
|------|------|
| **Hub model repo** weights | `src/hf/staging/models/{script}/{architecture}/{model_version}/{registry_tag}/` |
| **Hub dataset repo** crops | `src/hf/staging/datasets/{hub-dataset-slug}/` |

### Model layout (Calamari)

Place architecture-native **Hub artifact** files at the staging leaf, for example:

```
src/hf/staging/models/greek/calamari/v1/stable/
  best.pt             # Calamari PyTorch checkpoint
```

The publish script maps this to **Hub repo slug** `{script}-htr-{architecture}` (e.g. `greek-htr-calamari`) and tags the uploaded revision with the **registry tag** (e.g. `stable`). After publish, the checkpoint is addressable as:

```
hf://<namespace>/<hub-repo-slug>@<registry-tag>
```

### Dataset layout

Stage labelled line crops under a searchable **Hub dataset slug** (`{script}-manuscript-lines` or `{script}-{corpus}-htr-lines`):

```
src/hf/staging/datasets/greek-manuscript-lines/
  images/
    ms-001/line-0001.png
  labels.csv
```

`labels.csv` must be UTF-8 CSV with `image,transcription` columns. Each image
path is relative to the dataset root, starts with `images/`, and pairs exactly
one crop with a non-empty transcription; every crop needs one row. Keep
model-specific resizing and normalization out of these source crops.

Datasets are published to separate **Hub dataset repos** — not mixed with model
weights or addressed through `hf://` weights sources. A dataset can train
multiple future **registry model ids**; a **Hub model repo** supplies the
inference weights for each registry id.

## Authentication

| Operation | `HF_TOKEN` required? |
|-----------|----------------------|
| Read public model/dataset repos | No |
| `fetch_model.py` (public repos) | No |
| `publish_model.py`, `publish_dataset.py`, `sync_collection.py` | Yes (write token) |

Set a write token only when performing live uploads:

```bash
export HF_TOKEN=hf_...
```

Default CI and local dry-runs do **not** call the Hub upload API unless `--upload` or `HF_PUBLISH=1` is set.

## First-time model publish runbook

1. **Copy training output** into the **Hub staging tree** path for the target script, architecture, model version, and registry tag.
2. **Validate locally** (dry-run prints the model card and target repo):

   ```bash
   PYTHONPATH=. python scripts/hf/publish_model.py \
     --script greek \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe
   ```

3. **Create the Hub model repo** (public) by uploading with a write token:

   ```bash
   export HF_TOKEN=hf_...
   PYTHONPATH=. python scripts/hf/publish_model.py \
     --script greek \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe \
     --upload
   ```

4. **Point the Registry** at the published revision in [`inference/registry.yaml`](../inference/registry.yaml), for example:

   ```yaml
   weights_source: hf://nomicous/greek-htr-calamari@stable
   hub_revision: <40-character-commit-created-by-the-upload>
   artifact_sha256: <sha256-of-best.pt>
   ```

   Resolve `stable` once after upload and copy its immutable commit plus the
   Hub artifact SHA-256 into the Registry. Inference downloads by that commit
   rather than the mutable tag, and rejects cache or artifact digest changes.
   Then complete the platform catalog and deploy steps in
   [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

5. **Prefetch** to warm the **Hub cache**:

   ```bash
   PYTHONPATH=. python scripts/hf/fetch_model.py greek-calamari-v1 --registry-tag stable
   ```

6. **Update the collection** — add the model slug to `src/hf/publish/collection.yaml` and run `sync_collection.py` after setting `hub_slug` (see below).

Override `--namespace` if the **Hub namespace** is not yet `nomicous`. Use `--registry-model-id` when the legacy id differs from the default `{script}-{architecture}{model_version}` derivation.

## Dataset publish

1. Create a directory under `src/hf/staging/datasets/` using
   `{script}-manuscript-lines` or `{script}-{corpus}-htr-lines`.
2. Add the required `images/` and `labels.csv` pairing layout above.
3. Run the command below without `--upload` to validate the staging layout and
   print the generated dataset README. This is safe in CI and does not contact
   the Hub.
4. Set a write-capable `HF_TOKEN` and add `--upload` to create or update the
   target **Hub dataset repo** at `<namespace>/<dataset-slug>`.

```bash
PYTHONPATH=. python scripts/hf/publish_dataset.py greek-manuscript-lines --script greek
# live upload:
HF_PUBLISH=1 PYTHONPATH=. python scripts/hf/publish_dataset.py greek-manuscript-lines --script greek --upload
```

## Hub collection (`nomos`)

Source of truth: [`src/hf/publish/collection.yaml`](../src/hf/publish/collection.yaml).

**When to update the collection**

- After publishing a new **Hub model repo** or **Hub dataset repo** that should appear on the project landing page.
- When renaming collection metadata (title/description).
- Commit collection changes in git before syncing so membership is reviewed in PRs.

**First-time collection setup**

1. Create the `nomos` collection on Hugging Face (or note the slug returned by the Hub UI).
2. Set `hub_slug` in `src/hf/publish/collection.yaml` to the full slug (e.g. `nomicous/nomos-abc123`).
3. List model and dataset slugs under `models:` / `datasets:`.
4. Sync:

   ```bash
   PYTHONPATH=. python scripts/hf/sync_collection.py
   HF_PUBLISH=1 PYTHONPATH=. python scripts/hf/sync_collection.py --upload
   ```

## Scripts

| Script | Purpose |
|--------|---------|
| `fetch_model.py` | Prefetch `hf://` weights into **Hub cache** |
| `publish_model.py` | Staging → **Hub model repo** + model card + revision tag |
| `publish_dataset.py` | Staging → **Hub dataset repo** + dataset README |
| `sync_collection.py` | `collection.yaml` → **Hub collection** membership |

## Tests

```bash
uv run --group inference pytest tests/hf
```

Publish/sync tests use `MockPublishClient` and default to dry-run — no Hub credentials required in CI.
