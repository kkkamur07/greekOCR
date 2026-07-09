---
id: "041"
title: "inference-helper-packaging"
type: AFK
status: done
blocked_by:
  - "038-inference-helper-local-transcribe-tracer.md"
  - "039-local-segment-tracer.md"
parent_prd: "issues/done/inference/prd-local-inference-helper.md"
triage: ready-for-agent
---

## Parent

`issues/done/inference/prd-local-inference-helper.md` — Local inference via **Inference helper** (browser-orchestrated).

## What to build

Package the **Inference helper** for non-technical researchers on macOS, Windows, and Linux. One install per machine; no terminal or Docker. Bundles Python, PyTorch CPU, `inference/` (Calamari + Kraken), `src/hf/resolve/`, and bundled **Registry**. Auto-start on login; tray icon indicating running state.

Deliverables under `packaging/helper/`:

| OS | Installer | Auto-start |
|----|-----------|------------|
| macOS | `.dmg` | LaunchAgent |
| Windows | `.msi` or `.exe` | Task Scheduler / Run key |
| Linux | `.deb` or AppImage | systemd user unit |

Helper defaults: `127.0.0.1:8001`, `HF_CACHE_ROOT=~/.nomicous/hf/cache/`, CORS for hosted origin. Document build commands in inference README or packaging README.

## Acceptance criteria

- [x] macOS installer installs helper, auto-starts on login, tray shows running state; hosted site health probe succeeds after install.
- [x] Windows installer meets same behavioral bar.
- [x] Linux installer meets same behavioral bar.
- [x] Installed helper runs both transcribe and segment `/run` against bundled or Hub-resolved weights without developer `python -m`.
- [x] Build scripts are repeatable from CI or documented local build steps (installer smoke may be manual/out of default CI).
- [x] Install link in 040 banner points at real download location or release page.

## Blocked by

- [038-inference-helper-local-transcribe-tracer.md](038-inference-helper-local-transcribe-tracer.md)
- [039-local-segment-tracer.md](039-local-segment-tracer.md)

## User stories covered

- 1, 2, 3, 4, 5, 6, 8, 10, 15, 42, 44, 45
