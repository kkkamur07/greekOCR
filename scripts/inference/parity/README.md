# Parity: local run vs platform job

Compares `run_model` locally with the finished platform job for the
same page and model (`segment` and `transcribe`). `--api`,
`--project-id`, `--document-id` and `--part-id` are required, so
production is never an accidental target.

## Use (read only by default)

```bash
python scripts/inference/parity/compare_local_vs_platform.py \
  --api https://api.nomikos.app --task segment --model ppocr-segment \
  --job-id <uuid> --project-id <p> --document-id <d> --part-id <part> \
  --out /tmp/nmk-parity/run1
```

With email login, set `NOMIKOS_EMAIL` plus `NOMIKOS_PASSWORD`, or
`NOMIKOS_TOKEN`, or pass `--env-file PATH` (also accepts
`LOCUST_EMAIL`/`LOCUST_PASSWORD`). Secrets never reach the report.

## Safety

`--job-id` only reads. `--enqueue` creates a platform job; for
`segment` it requires `--i-know-segment-replaces-the-lines` and
first prints the part line count (a segment job replaces them).

## Output

`report.json` plus `report.md` under `--out`. Exit 0 on IDENTICAL,
1 on NUMERIC, CONFIDENCE_ONLY or MISMATCH, 2 on error or failed job,
3 on EMPTY or NOT_COMPARABLE. Segment compares the stored part
lines (`job.result` is only the merge summary); protected lines
force NOT_COMPARABLE unless `--allow-merged` is given. Worker
version: not exposed by the API.

## Local run fidelity

Image bytes come from `GET /media/parts/{part_id}` (stored
original). Params come from job payload `ml_params`; transcribe
regions are rebuilt from part lines in `(order, created_at)` order.
