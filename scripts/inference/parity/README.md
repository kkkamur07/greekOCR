# Parity: local run vs platform job

Compares `run_model` locally with the finished platform job for the
same page and model, covering `segment` and `transcribe`.

## Use (read only by default)

```bash
python scripts/inference/parity/compare_local_vs_platform.py \
  --task segment --model ppocr-segment --job-id <uuid> \
  --project-id <p> --document-id <d> --part-id <part> \
  --out /tmp/nmk-parity/run1
```

With email login, set `NOMIKOS_EMAIL` plus `NOMIKOS_PASSWORD`, or
`NOMIKOS_TOKEN`, or pass `--env-file PATH` (also accepts
`LOCUST_EMAIL`/`LOCUST_PASSWORD`). Secrets never reach the report.

## Safety

`--job-id` only reads. `--enqueue` creates a platform job; for
`segment` it additionally requires
`--i-know-segment-replaces-the-lines` and first prints the part line
count, because a segment job replaces the part lines.

## Output

`report.json` plus `report.md` under `--out`. Exit 0 on IDENTICAL,
1 on NUMERIC, CONFIDENCE_ONLY or MISMATCH, 2 on error or failed job.
Verdicts: segment IDENTICAL, NUMERIC (at most 1.0 px, same flags),
MISMATCH; transcribe IDENTICAL, CONFIDENCE_ONLY, MISMATCH.
```

## Local run fidelity

Image bytes come from `GET /media/parts/{part_id}` (the stored
original, no width). Params come from the job payload `ml_params`;
transcribe line regions are rebuilt from the part lines in
`(order, created_at)` order with verbatim float points. See the
script module docstring for the exact backend references.
