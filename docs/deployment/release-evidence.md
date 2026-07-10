# Production release evidence

Use one copy of this record for every production rollout. Do not place secret
values, raw authorization headers, submitted payloads, or customer data here.

## Release

- Release identifier:
- Commit SHA:
- Operator:
- Start time (UTC):
- Previous deployment identifier:
- Rollback owner:

## Required gates

| Gate | Evidence link or identifier | Result | Verified by |
| --- | --- | --- | --- |
| Frontend, Python, contract, secret, dependency, and Docker CI checks |  |  |  |
| Generated Vercel bundle inspection |  |  |  |
| Model revision and artifact-digest verification |  |  |  |
| Migration and least-privilege role verification |  |  |  |
| Provider-managed secret rotation/revocation confirmation |  |  |  |
| Preview smoke test |  |  |  |
| Production smoke test |  |  |  |

## Baseline and rollout metrics

| Measurement | Before deployment | One-hour check | 24-hour check |
| --- | --- | --- | --- |
| European API p50 |  |  |  |
| European API p95 |  |  |  |
| European API p99 |  |  |  |
| API error rate |  |  |  |
| Request volume |  |  |  |
| Job completion health |  |  |  |

## Critical-flow verification

- [ ] `/health` returned 200 from the production domain.
- [ ] Login and token refresh worked with the production cookie settings.
- [ ] Upload stored the expected WebP media object.
- [ ] Job submission, polling/SSE fallback, callback, and representative export completed.
- [ ] Vercel, worker, and database logs showed no sensitive values.
- [ ] Vercel function region was `fra1`.
- [ ] The previous deployment remained available for rollback.

## Decision

- [ ] Advance: error rate is within 10% of baseline and API p95 is within 20%.
- [ ] Hold and investigate: a non-critical rollout threshold was exceeded.
- [ ] Roll back: a security issue, data-integrity incident, cross-user exposure,
  critical-flow failure, error rate above 2× baseline, or API p95 more than 50%
  above baseline occurred.

Decision, timestamp, and notes:
