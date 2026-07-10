# Adding an inference model

End-to-end checklist for shipping a new **segment** or **transcribe** model to production. Three stores must stay aligned:

| Store | Purpose | Updated by |
|-------|---------|------------|
| **`inference/registry.yaml`** | Runtime catalog: task, architecture, weights URI, `host_eligibility` | Git + API deploy |
| **Postgres `inference_models`** | Editor model picker (`GET /inference/models`) | Seed script or DB insert |
| **Hub (or bundled weights)** | Checkpoint bytes | `publish_model.py` or `src/hf/local/` |

Vocabulary: [`inference/CONTEXT.md`](../../inference/CONTEXT.md).

## Overview

```
Train / export checkpoint
    → stage under src/hf/staging/…
    → publish to Hub (hf://…)
    → add entry to inference/registry.yaml
    → add InferenceModel row in Postgres
    → deploy platform API
    → installed helpers pull registry on startup (no reinstall)
    → first local /run downloads weights into ~/.nomicous/hf/cache/
```

**Registry sync:** deployed API serves `GET /inference/v1/registry` (public YAML + `ETag`). The Inference Helper fetches it into `~/.nomicous/registry.yaml` on startup when `HELPER_REGISTRY_URL` is set. The bundled copy in the installer is an offline fallback only.

**Weights:** resolved lazily from `weights_source` on first inference run — same path for cloud inference and the helper. See [`scripts/hf/README.md`](../../scripts/hf/README.md).

---

## 1. Choose identifiers

Follow the **registry model id** convention:

```
{script}-{architecture}-{model_version}
```

Examples: `greek-calamari-v1`, `greek-kraken-segment-v1`.

| Field | Meaning | Example |
|-------|---------|---------|
| **script** | Writing system / language family | `greek`, `syriac` |
| **architecture** | Runtime adapter | `calamari`, `kraken-segment` |
| **model_version** | Family generation | `v1`, `v2` |
| **registry tag** | Named weight snapshot | `stable` (default) |

**Hub repo slug** (separate from registry model id): `{script}-htr-{architecture}` → `greek-htr-calamari`.

**Platform `artifact_ref`** (Postgres): `registry://<registry_model_id>?tag=<registry_tag>`
Example: `registry://greek-calamari-v1?tag=stable`.

---

## 2. Publish weights

### Calamari transcribe (Hub)

1. Copy the converted `.pt` checkpoint into the **Hub staging tree**:

   ```
   src/hf/staging/models/{script}/calamari/{model_version}/{registry_tag}/
     best.pt
   ```

2. Dry-run, then upload:

   ```bash
   PYTHONPATH=. python scripts/hf/publish_model.py \
     --script greek \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe

   export HF_TOKEN=hf_...
   PYTHONPATH=. python scripts/hf/publish_model.py \
     --script greek \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe \
     --upload
   ```

3. Optional: warm cache locally:

   ```bash
   PYTHONPATH=. python scripts/hf/fetch_model.py greek-calamari-v1 --registry-tag stable
   ```

4. Optional: add the Hub repo to [`src/hf/publish/collection.yaml`](../../src/hf/publish/collection.yaml) and run `sync_collection.py`.

Full publish runbook: [`scripts/hf/README.md`](../../scripts/hf/README.md).

### Record immutable provenance for new Hub entries

Before adding an `hf://` source to the Registry, resolve the public Hub tag to
its 40-character commit and read the SHA-256 for the architecture-native Hub
artifact. Record both values in the same Registry version entry. The runtime
uses `hub_revision` when present and verifies `artifact_sha256` before loading
or reusing cache contents.

Existing `hf://` entries without both provenance fields are accepted for
migration compatibility and resolve from their `weights_source` tag. Do not add
new unpinned entries; update legacy entries with both fields when their
published artifact provenance is available.

For a public model repo, `huggingface_hub` can report both values without
`HF_TOKEN`:

```bash
uv run --group inference python -c \
  "from huggingface_hub import HfApi; print(HfApi().model_info('<namespace>/<repo>', revision='stable', files_metadata=True))"
```

`HF_TOKEN` is only required by the publish command, not by model download or
metadata lookup for public repositories.

### Kraken segment (bundled)

Kraken BLLA ships via `package://kraken/blla.mlmodel` inside the `kraken` Python package — no Hub upload for the default segment model. For a custom Kraken checkpoint, use `file://` (bundled under `src/hf/local/`) or `hf://` like Calamari.

### Local / offline dev

Point `weights_source` at bundled files under `src/hf/local/`:

```yaml
weights_source: file://local/greek/calamari/v1/stable/best.pt
```

(`file://` paths are relative to `src/hf/`.)

---

## 3. Register in `inference/registry.yaml`

Add a model block under `models:`:

```yaml
models:
  greek-calamari-v1:
    task: transcribe          # transcribe | segment
    architecture: calamari    # calamari | kraken-segment
    device: cpu               # compute hint (cpu | cuda)
    host_eligibility: local   # local | remote | any
    versions:
      stable:
        weights_source: hf://<namespace>/greek-htr-calamari@stable
        hub_revision: <40-character-resolved-Hub-commit>
        artifact_sha256: <sha256-of-best.pt>
```

When present, the two provenance fields must be supplied together for an
`hf://` **weights source**. `weights_source` retains the **registry tag** for
readability; a pinned `hub_revision` prevents a changed Hub tag from changing
the bytes inference loads until a reviewed Registry update replaces the
revision and digest. For packaged Kraken assets, record `artifact_sha256`
without `hub_revision` so the packaged `.mlmodel` is also verified before
Kraken loads it.

**`host_eligibility`**

| Value | Behaviour |
|-------|-----------|
| `local` | May run on the Inference Helper when installed |
| `remote` | Cloud inference only (GPU / large models) |
| `any` | Helper or cloud, user preference decides |

Run unit tests:

```bash
uv run --group inference pytest tests/inference/unit/test_registry.py -q
```

---

## 4. Register in the platform catalog (Postgres)

The editor lists models from **`inference_models`**, not directly from `registry.yaml`. Each row needs:

| Column | Value |
|--------|-------|
| `name` | Same as **registry model id** (unique) |
| `provider` | e.g. `kraken`, `calamari`, `huggingface` |
| `task` | `segment` or `transcribe` |
| `artifact_ref` | `registry://<registry_model_id>?tag=stable` |
| `default_params` | JSON, e.g. `{"device": "cpu"}` |

### Development

Extend [`scripts/platform/seed_dev_inference.py`](../../scripts/platform/seed_dev_inference.py) and run:

```bash
uv run --group platform python scripts/platform/seed_dev_inference.py
```

### Production

Insert the row (migration, admin script, or one-off SQL). `name` must match the registry model id so `artifact_ref` resolution stays consistent with cloud jobs.

Optional: create a **ModelBinding** at project, document, or part scope so the new model is the default for a workspace.

---

## 5. Tests and CI

Update or add coverage as needed:

| Area | Tests |
|------|-------|
| Registry parsing | `tests/inference/unit/test_registry.py` |
| Hosted registry endpoint | `tests/nomicous/integration/test_inference_registry.py` |
| Helper sync | `tests/inference/unit/test_registry_sync.py` |
| Hub resolve / prefetch | `tests/hf/` |
| Platform catalog | `tests/nomicous/integration/test_inference_catalog.py` |
| ML integration (optional) | `tests/nomicous/integration/ml/` |

```bash
uv run --group inference pytest tests/inference tests/hf -q
uv run --group platform --group inference pytest tests/nomicous/integration/test_inference_registry.py -q
```

---

## 6. Deploy

1. **Deploy platform API** — ships the updated `inference/registry.yaml` in the container (`/app/inference/registry.yaml`). The new endpoint is live:

   ```bash
   curl -s https://api.example.com/inference/v1/registry
   ```

2. **Deploy cloud inference** (`inference-api` / `inference-worker`) if remote inference should serve the model — same `registry.yaml` mount.

3. **Inference Helper** — **no new installer required**. On next helper start (login / reboot), it fetches the registry when `HELPER_REGISTRY_URL` points at your API. Weights download on first local OCR/segment run.

   Verify locally:

   ```bash
   HELPER_REGISTRY_URL=http://localhost:8000/inference/v1/registry \
     uv run --group inference python -m inference.helper

   curl -s http://127.0.0.1:8001/inference/v1/catalog
   ```

4. **New helper releases** are only needed for **code** or **packaging** changes — not for new models.

Set the production registry URL when building installers:

```bash
HELPER_REGISTRY_URL=https://api.example.com/inference/v1/registry \
  bash packaging/helper/macos/build-dmg.sh
```

See [`packaging/helper/README.md`](../../packaging/helper/README.md).

---

## 7. Verify end-to-end

- [ ] `curl …/inference/v1/registry` returns the new model id
- [ ] Authenticated `GET /inference/models` lists the new **InferenceModel**
- [ ] Helper `GET /inference/v1/catalog` shows correct `host_eligibility`
- [ ] Cloud job: enqueue segment/transcribe → job completes with expected output
- [ ] Local path: helper up, cloud toggle off → pairing assist / auto-segment works; weights appear under `~/.nomicous/hf/cache/<registry_model_id>/stable/`

---

## Quick reference: new Calamari transcribe model

1. Stage `best.pt` → `src/hf/staging/models/{script}/calamari/v1/stable/`
2. `publish_model.py … --upload`
3. Add block to `inference/registry.yaml` with `hf://…` **weights_source**
4. Upsert `InferenceModel` with `artifact_ref: registry://{id}?tag=stable`
5. Run tests → deploy API (+ inference workers for cloud path)
6. Helpers pick up registry automatically; weights on first use

## Related docs

- [`inference/README.md`](../../inference/README.md) — inference service and helper
- [`scripts/hf/README.md`](../../scripts/hf/README.md) — Hub publish and prefetch
- [`README.md`](../../README.md#local-inference-helper) — local vs remote inference architecture
- [`issues/done/huggingface/036-hf-registry-id-migration.md`](../../issues/done/huggingface/036-hf-registry-id-migration.md) — registry id naming migration (historical)
