# Parity: local run vs platform job

Compares `run_model` locally with the finished platform job for the
same page and model (`segment` and `transcribe`). `--api` and
the three ids are required, so production is never an accidental
target. Reports go under `--out` (dir 0o700, files 0o600).

## Use (read only by default)

```bash
python scripts/inference/parity/compare_local_vs_platform.py \
  --api https://api.nomikos.app --task segment --model ppocr-segment \
  --job-id <uuid> --project-id <p> --document-id <d> --part-id <part> \
  --out /tmp/nmk-parity/run1
```

Credentials: `NOMIKOS_TOKEN`, or `NOMIKOS_EMAIL` plus
`NOMIKOS_PASSWORD`, or `--env-file PATH` (also `LOCUST_*`).
Secrets never reach the report.

## Safety

`--job-id` only reads. The job must target the requested document
and part. `--enqueue` creates a platform job; for `segment` it
requires `--i-know-segment-replaces-the-lines` (a segment job
replaces the part lines). Unverifiable models give NOT_COMPARABLE
unless `--trust-job-model` is passed.

## Output

Exit 0 on IDENTICAL, 1 on NUMERIC, CONFIDENCE_ONLY or MISMATCH,
2 on error or failed job, 3 on EMPTY or NOT_COMPARABLE. Segment
compares stored part lines, never `job.result` (merge summary
only). Staleness rule: every model-produced stored line must name
the selected job in `source_metadata.job_id`; newer foreign lines
give NOT_COMPARABLE, older ones fall under the protected gate
(`--allow-merged` overrides). `--line-limit N` compares only the
first N transcribe lines on both sides (worker version is not
exposed by the API).
