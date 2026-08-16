# Hosted cloud inference worker

How to run a persistent **inference agent** that claims `cloud` work from
`api.nomicous.com`. It is the same **published package** a researcher runs on a
laptop ([ADR 0002](../adr/0002-inference-cli-replaces-loopback-helper.md)),
differing only by the credential it presents: a **service credential**
(`NOMICOUS_SERVICE_TOKEN`) instead of a **device token**
([ADR 0003](../adr/0003-single-job-queue-cloud-worker-claims-like-a-device.md)).

There is no inference *service* to deploy. The worker installs
`nomicous-inference`, runs `nomicous run`, reaches the platform **outbound**,
and listens on nothing. It opens no port, no tunnel, and no DNS record.

Terminology: [`inference/CONTEXT.md`](../../inference/CONTEXT.md).

---

## Requirements

- A persistent Linux host. CPU-only ONNX Runtime is the shipped runtime
  ([ADR 0006](../adr/0006-onnx-runtime-is-the-inference-runtime.md)); no GPU is
  required. GPU acceleration is a follow-up, not part of this runbook.
- Outbound HTTPS to `api.nomicous.com` and `huggingface.co` (for weight
  download).
- Enough memory for N workers. Measured on an 8-core / 15 GiB box: one page
  peaks ~3 GB and segmentation is **segment-bound** at ~48 s/page, single-threaded
  decode. **Two workers is the sweet spot**; three-to-four only if memory stays
  comfortable. Throughput scales by adding workers, not by tuning threads.

---

## 1. Install the agent

The agent must be an **installed distribution**, not a source checkout: a
checkout reports `0+unknown` to the **version floor**, and the platform refuses
an unparseable version with `426` before it authenticates.

```bash
# uv (the same installer the documented laptop path uses)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12        # requires-python is >=3.11,<3.13

# Option A — from PyPI once a release exists:
uv tool install --python 3.12 nomicous-inference

# Option B — build the wheel from a source checkout (before the first PyPI release):
cd /path/to/greekOCR
uv build                                    # -> dist/nomicous_inference-<version>-py3-none-any.whl
uv tool install --python 3.12 ./dist/nomicous_inference-<version>-py3-none-any.whl
```

Verify the version is real. This is what the claim endpoint reads:

```bash
nomicous version     # must print a version, NOT "0+unknown"
```

---

## 2. Generate the service credential

The **service credential** claims `cloud` work for *any* account, so it has the
same floor as a device-token key: 32+ characters, non-placeholder
([ADR 0005](../adr/0005-agent-claim-endpoint-and-the-inference-service-account.md)).

```bash
openssl rand -hex 32    # -> INFERENCE_WORKER_SERVICE_TOKEN  (also NOMICOUS_SERVICE_TOKEN on the box)
openssl rand -hex 32    # -> DEVICE_TOKEN_HMAC_SECRET        (platform-side only, never on the box)
```

The **service token must match on both sides**: `INFERENCE_WORKER_SERVICE_TOKEN`
on the platform equals `NOMICOUS_SERVICE_TOKEN` on the worker. A mismatch is a
`401`, not a `404`.

---

## 3. Platform (Vercel) configuration

On the `nomicous-api` Vercel project, in the **Production** environment:

| Variable | Value | Why |
| ---------- | ------- | ----- |
| `DEVICE_PAIRING_ENABLED` | `true` | Unlocks the whole device layer, including the claim endpoint. Off by default in production. |
| `DEVICE_TOKEN_HMAC_SECRET` | 32+ chars, ≠ `JWT_SECRET` | Required to boot once pairing is on. Platform-side only. |
| `INFERENCE_WORKER_SERVICE_TOKEN` | the token from §2 | Authenticates the hosted worker. |
| `CLOUD_INFERENCE_ENABLED` | `true` | Fail-closed guard: the platform requires a real `INFERENCE_WEBHOOK_SECRET`. |
| `INFERENCE_WEBHOOK_SECRET` | real secret | Already required in production; keep it. |
| `INFERENCE_AGENT_MIN_VERSION` | *(default `0.1.0` is fine)* | Refuses agents below this. Leave unless pinning a newer build. |

Vercel environment variables do **not** hot-apply: after changing them you must
**redeploy** the `nomicous-api` project, and the variables must be set under the
**Production** environment (not Preview/Development).

---

## 4. Configure the worker

```bash
sudo mkdir -p /etc/nomicous
sudo tee /etc/nomicous/worker.env > /dev/null <<'EOF'
NOMICOUS_API_URL=https://api.nomicous.com
NOMICOUS_SERVICE_TOKEN=<the service token from §2>
EOF
sudo chmod 600 /etc/nomicous/worker.env
```

`NOMICOUS_WORKER_NAME` is required by the platform (a worker that cannot name
itself is refused the same way as a bad token) and must be distinct per
process, so one worker can never report another worker's page. It comes from
the systemd unit below, not the env file.

Warm the Hub cache once so the first live page does not pay a weight download:

```bash
cd /path/to/greekOCR
uv sync --group inference
PYTHONPATH=. uv run --group inference python scripts/hf/fetch_model.py syriac-calamari-v1 --registry-tag stable
PYTHONPATH=. uv run --group inference python scripts/hf/fetch_model.py blla-segment --registry-tag stable
```

Weights land in `~/.nomicous/hf/cache` (override with `HF_CACHE_ROOT`). The
installed agent reads the same cache at claim time.

---

## 5. Run it supervised

One `nomicous run` = one claim loop = one page in flight. Scale by running N
processes, each with its own worker name.

`/etc/systemd/system/nomicous-worker@.service`:

```ini
[Unit]
Description=Nomicous cloud inference worker %i (claims from api.nomicous.com)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
EnvironmentFile=/etc/nomicous/worker.env
Environment=NOMICOUS_WORKER_NAME=<hostname>-%i
ExecStart=/root/.local/bin/nomicous run
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Start two instances:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nomicous-worker@1.service nomicous-worker@2.service
```

---

## 6. Verify

```bash
journalctl -u nomicous-worker@1.service -f
```

A healthy worker prints the launch-check verdict and then:

```
Claiming  cloud work for the platform, as <hostname>-1
Waiting for work. Press Ctrl-C to stop.
```

Submit a segment or transcribe job and watch it run on **cloud**. The first
page is slower than later ones: the ONNX session loads into memory on the
worker's first claim. The weights were pre-downloaded; the session was not. A
successful run logs `ran in …s` and then `reported done`.

---

## 7. Operations

Scale up by adding instances:

```bash
sudo systemctl enable --now nomicous-worker@3.service
```

Each instance is a distinct device row and reports its own capacity. Keep
memory in mind: ~3 GB peak per worker, and segmentation peak memory is tracked
as a known follow-up (issue #62).

| Symptom | Cause | Fix |
| --------- | ------- | ----- |
| `not serving the claim endpoint … disabled by default in production` (404) | `DEVICE_PAIRING_ENABLED` not `true`, or not redeployed | Set it under Production and redeploy |
| `401` on claim | service token mismatch | Make `INFERENCE_WORKER_SERVICE_TOKEN` equal `NOMICOUS_SERVICE_TOKEN` |
| `426` at launch | installed version below the **version floor** | `uv tool upgrade nomicous-inference`, or lower `INFERENCE_AGENT_MIN_VERSION` |
| Jobs stuck `waiting` | no worker reporting capacity for `cloud` | Start/check the workers; a worker's first claim is what registers capacity |

The agent self-upgrades at launch against the platform's version floor. A
worker below the floor is refused and told to upgrade; a worker merely behind
the newest release is served normally and told it is outdated.

## Related

- [`inference/README.md`](../../inference/README.md): the published package, CLI, registry, and limits
- [production.md](production.md): full hosted topology and rollback
- [`adding-inference-models.md`](../inference/adding-inference-models.md): shipping a new model to this worker
- ADRs [0002](../adr/0002-inference-cli-replaces-loopback-helper.md),
  [0003](../adr/0003-single-job-queue-cloud-worker-claims-like-a-device.md),
  [0005](../adr/0005-agent-claim-endpoint-and-the-inference-service-account.md)
