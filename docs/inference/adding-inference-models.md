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
    → release the published package so agents carry the new entry
    → first run downloads weights into ~/.nomicous/hf/cache/
```

**Where the Registry lives at runtime:** an **inference agent** reads the
`registry.yaml` that ships inside its own **published package**
(`INFERENCE_REGISTRY_PATH` overrides it, which is how a source checkout runs an
unreleased entry). It does not sync one from the platform - the claim it
receives names a **registry model id** and **registry tag**, and the agent
resolves both locally. The deployed API still serves the file at
`GET /inference/v1/registry` (public YAML + `ETag`), which is a convenient way
to confirm a deploy shipped the entry you expect.

**Weights:** resolved lazily from `weights_source` on the first run that needs
them - the same path on a researcher's laptop and on a hosted worker, because
both run the same package. See [`scripts/hf/README.md`](../../scripts/hf/README.md).

---

## 1. Choose identifiers

Follow the **registry model id** convention:

```
{script}-{architecture}-{model_version}
```

Examples: `syriac-calamari-v1`, `blla-segment`.

| Field | Meaning | Example |
|-------|---------|---------|
| **script** | Writing system / language family | `syriac`, `coptic` |
| **architecture** | Runtime adapter | `calamari`, `blla-segment` |
| **model_version** | Family generation | `v1`, `v2` |
| **registry tag** | Named weight snapshot | `stable` (default) |

**Hub repo slug** (separate from registry model id): `{script}-htr-{architecture}` → `syriac-htr-calamari`.

**Platform `artifact_ref`** (Postgres): `registry://<registry_model_id>?tag=<registry_tag>`
Example: `registry://syriac-calamari-v1?tag=stable`.

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
     --script syriac \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe

   export HF_TOKEN=hf_...
   PYTHONPATH=. python scripts/hf/publish_model.py \
     --script syriac \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe \
     --upload
   ```

3. Optional: warm cache locally:

   ```bash
   PYTHONPATH=. python scripts/hf/fetch_model.py syriac-calamari-v1 --registry-tag stable
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

### BLLA segment

The BLLA runtime loads `blla.safetensors` from the registry-pinned
`segmentation-blla` Hub artifact. The inference image does not install the
Kraken Python package; nothing in the repository does since ADR 0004 retired
the ONNX runtime and the parity harness Kraken was the oracle for.

### Local / offline dev

Point `weights_source` at bundled files under `src/hf/local/`:

```yaml
weights_source: file://local/syriac/calamari/v1/stable/best.pt
```

(`file://` paths are relative to `src/hf/`.)

---

## 3. Register in `inference/registry.yaml`

Add a model block under `models:`:

```yaml
models:
  syriac-calamari-v1:
    task: transcribe          # transcribe | segment
    architecture: calamari    # calamari | blla-segment
    device: cpu               # compute hint (cpu | cuda)
    host_eligibility: local   # local | remote | any
    versions:
      stable:
        weights_source: hf://<namespace>/syriac-htr-calamari@stable
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
| `local` | May run on an **inference agent** on the researcher's machine |
| `remote` | Hosted worker only (GPU / large models) |
| `any` | Either host; **host preference** and **capacity** fix one at submission |

Run unit tests:

```bash
uv run --group inference --group export pytest tests/inference/unit/test_registry.py -q
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
| Hub resolve / prefetch | `tests/hf/` |
| Platform catalog | `tests/nomicous/integration/test_inference_catalog.py` |
| ML integration (optional) | `tests/nomicous/integration/ml/` |

```bash
uv run --group inference --group export pytest tests/inference tests/hf -q
uv run --group platform --group inference pytest tests/nomicous/integration/test_inference_registry.py -q
```

---

## 6. Deploy

1. **Deploy platform API** - ships the updated `inference/registry.yaml` in the container (`/app/inference/registry.yaml`). The new endpoint is live:

   ```bash
   curl -s https://api.example.com/inference/v1/registry
   ```

2. **Publish the package.** A registry entry only reaches an **inference agent**
   inside the wheel it ships in, so a new model is a PyPI release - one wheel,
   one runner, no per-OS installer build. See
   [`inference/README.md`](../../inference/README.md#releasing-and-what-that-changed-about-security-patching).

3. **Roll out the hosted agent**, if cloud inference should serve the model, by
   upgrading the package on its host. It runs the same wheel a laptop does.

4. **Local inference** - researchers pick the model up with
   `uv tool upgrade nomicous-inference`, or automatically at the next **launch
   check** if you raise the platform's **version floor**
   (`INFERENCE_AGENT_MIN_VERSION`) past the last release without it. Weights
   download on the first run that needs them.

   Verify from a source checkout before releasing, with the registry read from
   the tree rather than from an installed wheel:

   ```bash
   NOMICOUS_API_URL=http://localhost:8000 \
     uv run --group inference python -m inference.cli run --exit-when-empty
   ```

---

## 7. Verify end-to-end

- [ ] `curl …/inference/v1/registry` returns the new model id
- [ ] Authenticated `GET /inference/models` lists the new **InferenceModel**
- [ ] `host_eligibility` in the shipped `registry.yaml` matches what the model can actually run on
- [ ] Cloud job: hosted agent claims a segment/transcribe page → job completes with expected output
- [ ] Local path: with **host preference** on and a paired machine running `nomicous run`, the job announces the local host, completes, and leaves weights under `~/.nomicous/hf/cache/<registry_model_id>/stable/`

---

## Quick reference: new Calamari transcribe model

1. Stage `best.pt` → `src/hf/staging/models/{script}/calamari/v1/stable/`
2. `publish_model.py … --upload`
3. Add block to `inference/registry.yaml` with `hf://…` **weights_source**
4. Upsert `InferenceModel` with `artifact_ref: registry://{id}?tag=stable`
5. Run tests → deploy API → publish the package
6. Agents carry the entry once upgraded; weights download on first use

## Related docs

- [`inference/README.md`](../../inference/README.md) - published package, CLI, and registry
- [`scripts/hf/README.md`](../../scripts/hf/README.md) - Hub publish and prefetch
- [`README.md`](../../README.md#self-hosting-and-local-inference) - local vs cloud inference architecture
- Issue **036** - registry id naming migration (historical; see git history on branches with `issues/done/`)
