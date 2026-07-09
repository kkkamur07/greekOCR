# Local Inference via Inference Helper — Product Requirements Document

> **Status: completed** (issues 038–041). ADR: [`docs/decisions/002-local-inference-helper.md`](../docs/decisions/002-local-inference-helper.md).

## Problem Statement

Researchers use a **hosted** annotation platform (browser SPA, API, Postgres). Today, segment and transcribe models run on **remote inference** servers the platform calls via **Product jobs**. That means:

- Server compute bills scale with every OCR and segmentation run, even when the researcher's laptop has spare CPU.
- Latency includes queue claim, callback round-trips, and network hops for work that could finish on-machine in seconds.
- Model weights are cached on inference servers (`src/hf/cache/` in Docker), not on the researcher's machine — repeated sessions re-download or re-fetch through the cloud path.
- Non-technical researchers cannot run Docker Compose; there is no productized way to use **local inference** against the hosted app.

Researchers need models marked for local CPU in the **Registry** to run on their own machine via a one-time install, with **remote inference** as an explicit fallback when the helper is missing, slow, or the user prefers cloud.

## Solution

Ship an **Inference helper** — a small native background app (Windows, macOS, Linux) that runs a slim **Inference sidecar** on `127.0.0.1:8001`. The hosted browser probes `GET /health`, calls `POST /inference/v1/run` for segment and transcribe when **Inference preference** is local, then persists results through the hosted API into Postgres. **Remote inference** remains the existing **Product job** path and is used when the helper is absent, the user toggles **Use cloud inference**, or the selected model has `host_eligibility: remote`.

This follows ADR [`docs/decisions/002-local-inference-helper.md`](../docs/decisions/002-local-inference-helper.md): browser-orchestrated localhost inference because the hosted API cannot reach a researcher's `localhost`.

## User Stories

### Inference helper install and lifecycle

1. As a non-technical researcher, I want to install the **Inference helper** once with a normal installer (no terminal, no Docker), so that I can run OCR on my CPU without IT setup.
2. As a researcher on macOS, I want a `.dmg` installer that adds the helper to Applications and auto-starts it on login, so that local inference is ready whenever I open the hosted site.
3. As a researcher on Windows, I want an `.msi` or `.exe` installer with auto-start on login, so that I get the same experience as macOS users.
4. As a researcher on Linux, I want a `.deb` or AppImage with a systemd user service, so that the helper starts automatically after install.
5. As a researcher, I want a tray icon showing whether the **Inference sidecar** is running, so that I know local inference is available without opening a terminal.
6. As a researcher, I want the helper to stay always-on after install (v1), so that the first OCR after opening the browser is fast without cold-start delays.
7. As a researcher on a shared library computer with no helper installed, I want the hosted site to use **remote inference** automatically with no error, so that I can still annotate.
8. As a researcher, I want the helper to bind only to `127.0.0.1`, so that inference is not exposed on the local network.
9. As an operator, I want the helper to reuse the existing `inference/` Calamari and Kraken adapters, so that local and remote paths produce the same canonical output shapes.
10. As an operator, I want the helper slim app to exclude Postgres, async job routers, and cloud callback wiring, so that the install stays small and maintainable.

### Model download and cache

11. As a researcher, I want models downloaded from **Hub model repos** (via `hf://` **weights sources**) into `~/.nomicous/hf/cache/` on first use, so that weights persist across browser sessions.
12. As a researcher, I want subsequent runs to reuse cached weights when the **Hub revision** has not changed, so that I am not re-downloading on every OCR.
13. As a researcher running pairing assist for the first time, I want a clear "Downloading model (~40MB)…" message, so that I understand why OCR is slow initially.
14. As a researcher waiting on a model download, I want a **Use cloud instead** action, so that I can skip the download and fall back to **remote inference**.
15. As an operator, I want `HF_CACHE_ROOT` (or equivalent) to point the helper at `~/.nomicous/hf/cache/`, so that cache location is consistent across OS installs.
16. As a developer, I want to run the helper with `python -m inference.helper` before installers ship, so that Phases 1–4 are testable without packaging.

### Registry and host eligibility

17. As a platform admin, I want each **registry model id** to declare `host_eligibility: local | remote | any`, so that the product knows which models may run on the **Inference helper**.
18. As a platform admin, I want `device: cpu | cuda` to remain a **Compute device** hint distinct from host placement, so that `device: cpu` is not misread as "run on researcher's laptop."
19. As a researcher, I want all current CPU-sized models (`greek-calamari-v1`, `syriac-calamari-v1`, `greek-kraken-segment-v1`) marked `host_eligibility: local`, so that both transcribe and segment can run locally from v1.
20. As a platform admin, I want future large or GPU-only models marked `host_eligibility: remote`, so that they never download to laptops and always use **remote inference**.
21. As a frontend developer, I want a catalog endpoint exposing `host_eligibility` per model, so that the UI can decide local vs cloud without hardcoding model ids.
22. As a researcher selecting a `host_eligibility: remote` model, I want the UI to route to **remote inference** even when the helper is present, so that I cannot accidentally trigger an unsupported local run.

### Local transcribe (pairing assist and page OCR)

23. As a researcher with the helper running, I want **Pairing assist** on a selected segment to call the helper's `/inference/v1/run` transcribe path, so that single-line OCR uses my CPU.
24. As a researcher, I want local transcribe results persisted to hosted Postgres via the API, so that my work is saved regardless of where inference ran.
25. As a researcher, I want local transcribe to return `text`, `confidence`, and `character_confidences` like **remote inference**, so that the editor heatmap and provenance UX stay consistent.
26. As a researcher running page-level transcribe with the helper available, I want lines batched to local `/run` (or an equivalent local batch path) when models are `local`-eligible, so that full-page OCR also saves server cost.
27. As a researcher, I want local transcribe to create/update the same **Model transcription** layer shape that job callbacks produce, so that **Copy to ground truth** and layer comparison keep working.

### Local segment

28. As a researcher with the helper running, I want **Auto-segment** on a page to call the helper's `/run` segment path, so that Kraken segmentation uses my CPU.
29. As a researcher, I want local segment results merged into page layout via the hosted API using the same **Segment merge** rules as cloud jobs, so that manual geometry is respected.
30. As a researcher re-running segment locally, I want machine lines updated, added, and pruned consistently with the cloud path, so that layout behavior does not depend on inference host.
31. As a researcher, I want local segment to support Otsu refinement parameters already exposed in the editor, so that local and cloud segment options match.

### Inference preference and fallback

32. As a researcher, I want a single **Use cloud inference** toggle in settings, so that I can force **remote inference** when my machine is slow without uninstalling the helper.
33. As a researcher, I want **Inference preference** to default to local when the helper health probe succeeds and the model is `host_eligibility: local`, so that server cost is minimized by default.
34. As a researcher without the helper, I want pairing assist and auto-segment to use existing **Product jobs** and **remote inference**, so that the product works unchanged.
35. As a researcher who toggles cloud inference on, I want the next OCR/segment action to enqueue a cloud job immediately, so that fallback is one click.
36. As a researcher who toggles cloud inference off with the helper present, I want the next action to use local `/run`, so that I can switch back without reinstalling.
37. As a researcher, I want my **Inference preference** to persist across browser sessions (user settings or `localStorage`), so that I do not re-toggle every visit.

### Helper discovery and security (v1)

38. As a researcher, I want the browser to discover the helper via `GET localhost:8001/health` with a short timeout, so that the UI knows quickly whether local inference is available.
39. As a researcher without the helper, I want a non-blocking banner with an install link, so that I can optionally install the helper without blocking annotation.
40. As a security-conscious operator, I accept v1 helper `/run` without auth (health probe only) because the stack is open source and inference-only on localhost; token handshake is deferred.
41. As a developer, I want CORS on the helper to allow the hosted production origin (and local dev origin), so that browser `fetch` to `localhost:8001` succeeds.

### Operations and developer experience

42. As a developer, I want helper logs written to a predictable location under `~/.nomicous/`, so that support can diagnose failed local runs.
43. As a CI maintainer, I want helper and persist-path tests to run without installers or GPU, so that regressions are caught in default CI.
44. As an operator, I want helper and cloud inference to share one **Registry** schema, so that model versions do not diverge between hosts.
45. As a project maintainer, I want packaging scripts under a dedicated `packaging/helper/` tree, so that macOS, Windows, and Linux builds are repeatable.

### Persistence and data model

46. As a researcher, I want annotations, projects, and transcriptions to remain in hosted Postgres whether I use local or remote inference, so that switching hosts never loses work.
47. As a platform developer, I want a dedicated API to persist local transcribe results (line id, model id, text, confidence, character confidences), so that the browser does not fake internal job callbacks.
48. As a platform developer, I want a dedicated API to persist local segment results (blocks, lines, merge metadata), so that segment merge rules stay server-authoritative.
49. As a platform developer, I want local persist endpoints to require normal user auth (JWT), so that only logged-in researchers write results.

### Future (out of v1 scope but informed by design)

50. As a security-conscious operator, I want a future token handshake between browser and helper after login, so that `/run` is not an open localhost API on shared machines.
51. As a researcher on an older laptop, I want optional model unload when idle if RAM complaints arise, so that the helper does not hold hundreds of MB when annotating elsewhere.
52. As a product owner, I want optional Chrome extension + native messaging later for on-demand helper launch, so that idle RAM can be reduced without always-on tray behavior.

## Implementation Decisions

- **Architecture**: Hosted SPA + hosted API + hosted Postgres; **Inference helper** on researcher machine. Browser orchestrates local `/inference/v1/run`; hosted job worker continues **remote inference** only. See ADR 002.
- **Rejected alternatives**: Full desktop app (Tauri/Electron), in-browser WASM/ONNX, cloud-default inference, outbound WebSocket agent (deferred), transcribe-only v1 (rejected — all `host_eligibility: local` models run locally).
- **Helper app**: New `inference/helper/` slim FastAPI app — `health` + `/inference/v1/run` (+ catalog) only; no jobs router, no Postgres. Entrypoint `python -m inference.helper`; bind `127.0.0.1:8001`.
- **Auth on helper v1**: No `inference_service_secret` on helper `/run` (differs from cloud inference API). CORS allows hosted origin. Token handshake deferred.
- **Cache**: `HF_CACHE_ROOT=~/.nomicous/hf/cache/`; reuse existing **Hub integration** (`src/hf/resolve/`) for `hf://` download and **Hub cache manifest** integrity.
- **Registry**: Add `host_eligibility: local | remote | any` to each model in `inference/registry.yaml`. All three current models are `local` in v1. Expose via `GET /inference/v1/catalog` on helper (and optionally mirror on hosted API).
- **Frontend routing**: New `useInferenceHost` (probe health, read preference, check `host_eligibility`). `usePairingState.runSegmentOcr` and `useLayoutMutations.runAutoSegment` branch: local `/run` + API persist vs existing `enqueueTranscribePart` / `segmentPart` + `trackJobAndWait`.
- **Inference preference**: Single "Use cloud inference" toggle; default local when helper healthy and model eligible. Persist in `localStorage` v1; user settings API optional enhancement.
- **Backend persist**: New authenticated endpoints to apply local transcribe and segment results using the same merge/persistence logic as inference job callbacks — without creating fake **Product jobs** for the local path.
- **Packaging**: PyInstaller (or equivalent) bundling Python + PyTorch CPU + `inference/` + `src/hf/`; per-OS installers with auto-start (LaunchAgent / Task Scheduler / systemd user unit). Tray icon for running state.
- **Product jobs**: Remain the cloud path only. Local path bypasses queue but must produce identical persisted artifacts.
- **Scope v1**: Both segment and transcribe for `host_eligibility: local` models; always-on helper; health-probe discovery; no extension, no desktop shell.

## Testing Decisions

Tests assert **external behavior** at the highest existing seams; do not test PyInstaller or OS installer internals in default CI.

| Seam | What to test | Prior art |
|------|----------------|-----------|
| **Helper HTTP** | `GET /health` returns OK; `POST /inference/v1/run` transcribe + segment return canonical JSON without service secret; CORS headers present for allowed origin | `tests/inference/integration/test_run.py` (cloud inference client with secret — adapt for helper app factory) |
| **Registry host eligibility** | Parsing `host_eligibility`; catalog endpoint lists models with correct eligibility; `device` vs `host_eligibility` independence | `tests/inference/unit/test_registry.py` |
| **Hub cache on helper** | Resolve `hf://` into `HF_CACHE_ROOT`; cache hit on second resolve (mock Hub at boundary) | `tests/hf/` / issue 030 patterns |
| **Backend local persist — transcribe** | Authenticated POST persists **Model transcription** row with text, confidence, character confidences; matches shape from job callback | `tests/nomicous/integration/test_jobs.py`, ML callback tests |
| **Backend local persist — segment** | Authenticated POST applies **Segment merge** (update/add/prune machine lines); manual geometry untouched | Segment merge integration tests |
| **Frontend routing** | When probe succeeds and preference local, pairing/segment calls localhost client mock; when probe fails or cloud toggled, calls existing enqueue APIs | Vitest with `fetch` mock; optional Playwright smoke later |

**Good test bar**: A developer with `python -m inference.helper` and mocked or bundled weights can curl `/run` and see the same output shape as cloud; persist integration tests prove Postgres rows match the cloud job path without enqueueing a **Product job**.

**Out of CI default**: Real installer smoke on three OSes; live Hub download without mock (optional marked integration test).

## Out of Scope

- Full desktop application (Tauri/Electron webview shell).
- In-browser WASM / WebGPU inference.
- Browser spawning Python without a prior install (browser security forbids this).
- Outbound agent / WebSocket job routing from cloud to laptop (deferred).
- Auth token handshake on helper `/run` (v1.1).
- Chrome extension + native messaging for on-demand launch (optional later).
- Model unload on idle / RAM optimization (until user complaints).
- Replacing **Product jobs** for cloud path or unifying local runs into the job queue.
- Automatic timeout-based fallback from local to cloud (manual toggle only in v1).
- Self-hosted full docker compose per institution as product UX (developer-only).
- New model tiers beyond `host_eligibility` flags in **Registry**.

## Further Notes

- Domain vocabulary: `nomicous/CONTEXT.md` (**Local inference**, **Remote inference**, **Inference sidecar**, **Inference helper**, **Inference preference**, **Host eligibility**) and `inference/CONTEXT.md` (**Inference host**, **Lite model tier**, **Server model tier**).
- ADR: [`docs/decisions/002-local-inference-helper.md`](../docs/decisions/002-local-inference-helper.md) (Accepted 2026-07-09).
- Supersedes the open HITL question in issue 028 (sync vs job-backed OCR) for hosted deployment: local = browser `/run` + API persist; cloud = **Product jobs** + **remote inference**.
- Grilling session decisions captured 2026-07-09: one install (not desktop app), health probe only, always-on helper, all `local` models (segment + transcribe), cloud fallback via toggle.
- Prefer thin vertical slices: first tracer proves local transcribe pairing assist end-to-end; second adds local segment; third polishes preference/install UX; fourth ships installers.
