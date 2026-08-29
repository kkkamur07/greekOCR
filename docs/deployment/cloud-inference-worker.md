# Hosted cloud inference worker

How to run a persistent **inference agent** that claims `cloud` work from
`api.nomikos.app`. It is the same **published package** a researcher runs on a
laptop ([ADR 0002](../adr/0002-inference-cli-replaces-loopback-helper.md)),
differing only by the credential it presents: a **service credential**
(`NOMIKOS_SERVICE_TOKEN`) instead of a **device token**
([ADR 0003](../adr/0003-single-job-queue-cloud-worker-claims-like-a-device.md)).

There is no inference *service* to deploy. The worker installs
`nomikos-inference`, runs `nomikos run`, reaches the platform **outbound**,
and listens on nothing. It opens no port, no tunnel, and no DNS record.

Terminology: [`nomikos_inference/CONTEXT.md`](../../nomikos_inference/CONTEXT.md).

---

## Requirements

- A persistent Linux host. CPU-only ONNX Runtime is the shipped runtime
  ([ADR 0006](../adr/0006-onnx-runtime-is-the-inference-runtime.md)); no GPU is
  required. GPU acceleration is a follow-up, not part of this runbook.
- Outbound HTTPS to `api.nomikos.app` and `huggingface.co` (for weight
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
uv tool install --python 3.12 nomikos-inference

# Option B — build the wheel from a source checkout (before the first PyPI release):
cd /path/to/greekOCR
uv build                                    # -> dist/nomikos_inference-<version>-py3-none-any.whl
uv tool install --python 3.12 ./dist/nomikos_inference-<version>-py3-none-any.whl
```

Verify the version is real. This is what the claim endpoint reads:

```bash
nomikos version     # must print a version, NOT "0+unknown"
```

---

## 2. Generate the service credential

The **service credential** claims `cloud` work for *any* account, so it has the
same floor as a device-token key: 32+ characters, non-placeholder
([ADR 0005](../adr/0005-agent-claim-endpoint-and-the-inference-service-account.md)).

```bash
openssl rand -hex 32    # -> INFERENCE_WORKER_SERVICE_TOKEN  (also NOMIKOS_SERVICE_TOKEN on the box)
openssl rand -hex 32    # -> DEVICE_TOKEN_HMAC_SECRET        (platform-side only, never on the box)
```

The **service token must match on both sides**: `INFERENCE_WORKER_SERVICE_TOKEN`
on the platform equals `NOMIKOS_SERVICE_TOKEN` on the worker. A mismatch is a
`401`, not a `404`.

---

## 3. Platform (Vercel) configuration

On the `nomikos-api` Vercel project, in the **Production** environment:

| Variable | Value | Why |
| ---------- | ------- | ----- |
| `DEVICE_PAIRING_ENABLED` | `true` | Unlocks the whole device layer, including the claim endpoint. Off by default in production. |
| `DEVICE_TOKEN_HMAC_SECRET` | 32+ chars, ≠ `JWT_SECRET` | Required to boot once pairing is on. Platform-side only. |
| `INFERENCE_WORKER_SERVICE_TOKEN` | the token from §2 | Authenticates the hosted worker. |
| `CLOUD_INFERENCE_ENABLED` | `true` | Fail-closed guard: the platform requires a real `INFERENCE_WEBHOOK_SECRET`. |
| `INFERENCE_WEBHOOK_SECRET` | real secret | Already required in production; keep it. |
| `INFERENCE_AGENT_MIN_VERSION` | *(default `0.1.0` is fine)* | Refuses agents below this. Leave unless pinning a newer build. |

Vercel environment variables do **not** hot-apply: after changing them you must
**redeploy** the `nomikos-api` project, and the variables must be set under the
**Production** environment (not Preview/Development).

---

## 4. Configure the worker

```bash
sudo mkdir -p /etc/nomikos
sudo tee /etc/nomikos/worker.env > /dev/null <<'EOF'
NOMIKOS_API_URL=https://api.nomikos.app
NOMIKOS_SERVICE_TOKEN=<the service token from §2>
EOF
sudo chmod 600 /etc/nomikos/worker.env
```

`NOMIKOS_WORKER_NAME` is required by the platform (a worker that cannot name
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

Weights land in `~/.nomikos/hf/cache` (override with `HF_CACHE_ROOT`). The
installed agent reads the same cache at claim time.

---

## 5. Run it supervised

One `nomikos run` = one claim loop = one page in flight. Scale by running N
processes, each with its own worker name.

`/etc/systemd/system/nomikos-worker@.service`:

```ini
[Unit]
Description=Nomikos cloud inference worker %i (claims from api.nomikos.app)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
EnvironmentFile=/etc/nomikos/worker.env
Environment=NOMIKOS_WORKER_NAME=<hostname>-%i
ExecStart=/root/.local/bin/nomikos run
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Start two instances:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nomikos-worker@1.service nomikos-worker@2.service
```

---

## 6. Verify

```bash
journalctl -u nomikos-worker@1.service -f
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
sudo systemctl enable --now nomikos-worker@3.service
```

Each instance is a distinct device row and reports its own capacity. Keep
memory in mind: ~3 GB peak per worker, and segmentation peak memory is tracked
as a known follow-up (issue #62).

| Symptom | Cause | Fix |
| --------- | ------- | ----- |
| `not serving the claim endpoint … disabled by default in production` (404) | `DEVICE_PAIRING_ENABLED` not `true`, or not redeployed | Set it under Production and redeploy |
| `401` on claim | service token mismatch | Make `INFERENCE_WORKER_SERVICE_TOKEN` equal `NOMIKOS_SERVICE_TOKEN` |
| `426` at launch | installed version below the **version floor** | `uv tool upgrade nomikos-inference`, or lower `INFERENCE_AGENT_MIN_VERSION` |
| Jobs stuck `waiting` | no worker reporting capacity for `cloud` | Start/check the workers; a worker's first claim is what registers capacity |
| Worker pulls weights from an old namespace after a migration | `registry.yaml` ships in the wheel; `git pull` does not touch it, and the Hub redirect hides it | Reinstall the wheel and restart, then verify the resolved namespace (below) |

The agent self-upgrades at launch against the platform's version floor. A
worker below the floor is refused and told to upgrade; a worker merely behind
the newest release is served normally and told it is outdated.

### Changing which model repo the worker pulls

`registry.yaml` is **packaged inside the wheel**, not read from a checkout.
`[tool.hatch.build.targets.wheel]` sets `packages = ["nomikos_inference"]` and does not
exclude it, so the installed worker resolves weights from
`site-packages/nomikos_inference/registry.yaml`.

A `git pull` on the box therefore does **not** change what the worker
downloads. Updating the registry means reinstalling:

```bash
# From PyPI: name the version that actually contains the change. No client-side
# command can install a registry that was never released. `uv tool upgrade` is a
# no-op once the installed version matches, and `--reinstall` refetches that
# same version's wheel, old registry and all. Both report success either way.
uv tool install --reinstall "nomikos-inference==<version containing the change>"

# Or from a source checkout, where the version usually has not moved at all.
# --force is what makes a same-version wheel replace the installed one:
uv build && uv tool install --force --python 3.12 ./dist/nomikos_inference-<version>-py3-none-any.whl

sudo systemctl restart 'nomikos-worker@*'  # a running worker holds the old registry
```

Then run the check below. It is the only step that distinguishes a reinstall
that changed the registry from one that reinstalled the same bytes.

Alternatively point `INFERENCE_REGISTRY_PATH` at a checkout the box does pull,
which trades the packaged default for a file you have to keep current yourself.

The reason this is worth stating rather than leaving to inference: **nothing
looks wrong when you get it wrong.** A Hub repo that has been transferred to a
new namespace keeps serving its old path as a redirect, indefinitely and
silently. So a worker still pinned to the pre-migration namespace downloads the
same bytes and runs normally, and the pull that was supposed to move it appears
to have worked. There is no error to notice and no symptom to chase, which is
what makes it survive to the point where whoever debugs it has never heard of
the migration.

Do not check this by confirming that a download succeeds; the redirect makes
that pass either way. Check the namespace the worker actually resolved:

```bash
# INFERENCE_REGISTRY_PATH, if the worker sets it, wins over the packaged copy.
# The sed strips surrounding quotes: systemd removes them when it reads the
# EnvironmentFile, so a quoted value that the worker resolves perfectly well
# would otherwise reach grep with literal quote characters still attached.
registry=$(sudo grep -hE '^INFERENCE_REGISTRY_PATH=' /etc/nomikos/worker.env \
             | tail -1 | cut -d= -f2- \
             | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//")

if [ -z "$registry" ]; then
  worker_bin=/root/.local/bin/nomikos   # match ExecStart in your unit file
  tool_root=$(dirname "$(dirname "$(sudo readlink -f "$worker_bin")")")
  registry=$(sudo sh -c "ls $tool_root/lib/python*/site-packages/nomikos_inference/registry.yaml")
fi

echo "worker registry: $registry"
sudo grep weights_source "$registry"
```

It prints the path before the contents on purpose. The whole difficulty here is
knowing *which* `registry.yaml` you are looking at, so a check that shows only
the namespace is one you have to trust rather than read.

Three things this deliberately avoids, all of which produce a confident wrong
answer rather than an error.

**Do not assume the packaged copy is the live one.** Setting
`INFERENCE_REGISTRY_PATH` (the alternative offered above) repoints the worker
at a file outside the wheel, and a check hardcoded to `site-packages` would
then report a registry nothing resolves. Read the override first, and fall back
to the package only when it is unset.

**Do not import the package to locate the file.** `import nomikos_inference` resolves
against `sys.path`, and the current directory comes first, so run from a source
checkout it imports the checkout's `nomikos_inference/` and prints whatever
`registry.yaml` says **there**. That is the one copy the worker does not use,
and it is the copy that always looks correct right after a `git pull`, which is
the mistake this section exists to catch.

**Do not use `uv tool dir` from your own shell.** It resolves per user. The
unit above runs `/root/.local/bin/nomikos`, so an operator checking as
themselves inspects their own tool environment and either finds nothing or
reads an unrelated install, while the worker's registry goes unexamined.
Deriving `tool_root` from the binary the unit actually names is what keeps the
answer tied to the process being diagnosed. If your `ExecStart` differs, change
`worker_bin` to match it.

## Related

- [`nomikos_inference/README.md`](../../nomikos_inference/README.md): the published package, CLI, registry, and limits
- [production.md](production.md): full hosted topology and rollback
- [`adding-inference-models.md`](../inference/adding-inference-models.md): shipping a new model to this worker
- ADRs [0002](../adr/0002-inference-cli-replaces-loopback-helper.md),
  [0003](../adr/0003-single-job-queue-cloud-worker-claims-like-a-device.md),
  [0005](../adr/0005-agent-claim-endpoint-and-the-inference-service-account.md)
