# Hosted API load testing

This suite uses Locust to measure the platform API and its database-backed
read paths. It covers the public document viewer, authenticated project
dashboard, document/page-editor hydration, model lookup, media, and job
status paths. It intentionally does not upload documents, enqueue jobs, or
modify documents.

## Run a smoke test

Verify the deployment first:

```bash
curl --fail "$LOCUST_HOST/health"
```

Then run a small headless test:

```bash
LOCUST_HOST=https://api.nomicous.com \
uv run --group load-testing locust \
  -f tests/load/locustfile.py \
  --headless \
  -u 2 \
  -r 1 \
  -t 30s
```

Use the Locust dashboard when tuning a test:

```bash
LOCUST_HOST=https://api.nomicous.com \
LOCUST_PROJECT_ID="$TEST_PROJECT_ID" \
LOCUST_DOCUMENT_ID="$TEST_DOCUMENT_ID" \
LOCUST_PART_ID="$TEST_PART_ID" \
LOCUST_ACCESS_TOKEN="$ACCESS_TOKEN" \
uv run --group load-testing locust \
  -f tests/load/locustfile.py \
  --web-host 127.0.0.1 \
  --web-port 8089
```

Open http://127.0.0.1:8089, set the user count and spawn rate, then start the
test from the dashboard. Keep the dashboard bound to loopback and begin with
low concurrency.

## Authenticated traffic

Authenticate a dedicated load-test user without printing the JWT:

```bash
# Alternatively, put these two values in tests/load/.env (ignored by git).
export LOCUST_EMAIL="load-test@example.com"
export LOCUST_PASSWORD="your-password"
source tests/load/get-token.sh
```

Prefer a pre-created, dedicated load-test user's token:

```bash
LOCUST_HOST=https://api.nomicous.com \
LOCUST_ACCESS_TOKEN="$ACCESS_TOKEN" \
LOCUST_PROJECT_ID="$TEST_PROJECT_ID" \
uv run --group load-testing locust \
  -f tests/load/locustfile.py \
  --headless -u 10 -r 2 -t 2m
```

`LOCUST_PROJECT_ID` is optional and enables read-only project and job-list
requests. To enable the full authenticated/document path set, also provide
`LOCUST_DOCUMENT_ID` and `LOCUST_PART_ID`. Set `LOCUST_JOB_ID` to include
repeated job-status reads.

`LOCUST_EMAIL` and `LOCUST_PASSWORD` are supported for a small authentication
smoke test, but repeated login traffic is subject to the production auth rate
limit and should not be used for the main load test.

The public document, layout, media, transcription, PAGE XML, and PDF routes
reuse the same project/document/part IDs. Use IDs for a published test
document when exercising those paths.

Do not commit tokens, passwords, cookies, exported Locust reports, or customer
data. Start with low user counts, agree on an upper bound before increasing
load, and monitor API function errors, database connections, 429 responses,
latency percentiles, and Vercel/Supabase logs.
