# Registry model id migration (issue 036)

Legacy **registry model ids** were migrated to the `{script}-{architecture}-{model_version}` convention from `inference/CONTEXT.md`. This was a single cutover — legacy ids are no longer accepted by `inference/registry.yaml`.

## Old → new mapping

| Legacy id | New registry model id | Task | Hub repo slug (if remote) |
|-----------|----------------------|------|---------------------------|
| `greek-calamariv1` | `greek-calamari-v1` | transcribe | `nomicous/greek-htr-calamari` |
| `syriac-calamariv1` | `syriac-calamari-v1` | transcribe | (local bundled: `src/hf/local/syriac/calamari/v1/stable/best.pt`) |
| `kraken-blla` | `greek-kraken-segment-v1` | segment | (Kraken package: `package://kraken/blla.mlmodel`) |

## Operator actions

1. **Platform `InferenceModel` rows** — Re-seed or update `artifact_ref` values:
   - `registry://greek-calamariv1?tag=stable` → `registry://greek-calamari-v1?tag=stable`
   - `registry://syriac-calamariv1?tag=stable` → `registry://syriac-calamari-v1?tag=stable`
   - `registry://kraken-blla?tag=stable` → `registry://greek-kraken-segment-v1?tag=stable`

   Dev seed: `python scripts/platform/seed_dev_inference.py`

2. **Hub cache** — Cached artifacts now live under `src/hf/cache/<registry_model_id>/<registry_tag>/`. Remove stale directories such as `src/hf/cache/greek-calamariv1/` if present.

3. **Prefetch** — Use the new id:
   ```bash
   PYTHONPATH=. python scripts/hf/fetch_model.py greek-calamari-v1 --registry-tag stable
   ```

4. **Environment overrides** (platform dev defaults):
   - `DEFAULT_SEGMENT_MODEL=greek-kraken-segment-v1`
   - `DEFAULT_TRANSCRIBE_MODEL=syriac-calamari-v1`

## Unchanged paths

- **Interim weight folders** under `inference/weights/calamari/{greek,syriac}-calamariv1/` retain legacy directory names until those checkpoints are relocated to `src/hf/local/`.
- **Hub repo slugs** (`greek-htr-calamari`, etc.) were already on the new convention and did not change.
