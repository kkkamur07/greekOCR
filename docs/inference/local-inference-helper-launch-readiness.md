# Local Inference Helper — production launch readiness

Assessment of the **Inference Helper** feature (issues 038–041, ADR [`002-local-inference-helper`](../decisions/002-local-inference-helper.md)) against a standard shipping-and-launch pre-launch checklist.

**Verdict (2026-07-09):** **Ready for staged internal/beta rollout** after production URL configuration and a manual installer build. **Not ready for broad public launch** until installer distribution, code signing, and observability gaps below are closed.

---

## Summary

| Area | Status | Notes |
|------|--------|-------|
| Core product path | ✅ Done | Browser probe → local `/run` → authenticated API persist works for transcribe + segment |
| Registry sync | ✅ Done | `GET /inference/v1/registry` + helper ETag cache; no reinstall for new models |
| Frontend UX | ✅ Done | Cloud toggle, install banner, download banner, remote-only model handling |
| Backend persist | ✅ Done | JWT-protected endpoints; same merge rules as cloud jobs |
| Packaging scripts | ✅ Done | macOS `.dmg`, Windows zip, Linux tarball + auto-start |
| Installer distribution | ❌ Blocker | No GitHub release artifacts; banner links to empty releases page |
| Production config | ⚠️ Partial | Install scripts default to `*.nomicous.example` placeholders |
| Code signing | ❌ Blocker | PyInstaller `codesign_identity=None`; no notarization / SmartScreen path |
| CI / release automation | ❌ Gap | No workflow builds or publishes installers per commit/tag |
| Feature flag | ❌ Gap | Ships to all users on deploy — no kill switch |
| Observability | ⚠️ Partial | Helper logs to `~/.nomicous/logs/`; no client or API metrics for local path |
| E2E verification | ⚠️ Partial | Unit + integration tests exist; no Playwright helper smoke |
| Security (accepted v1) | ⚠️ Accepted risk | No auth on helper `/run`; CORS must allow production SPA origin |

---

## Pre-launch checklist

### Code quality

| Item | Status | Evidence / gap |
|------|--------|----------------|
| Tests pass | ⚠️ | Helper unit tests: 7/8 pass locally (`test_helper_catalog_lists_host_eligibility` fails when `~/.nomicous/registry.yaml` from prior runs shadows bundled registry). Frontend `inferenceHost.test.ts`: 6/6 pass. Integration tests (`test_local_inference_persist`, `test_inference_registry`) require Postgres on `:5433`. |
| Build succeeds | ⚠️ | PyInstaller scripts exist; **no committed `packaging/helper/dist/` artifacts** and no CI build verification in repo. |
| Lint / typecheck | ✅ | Ruff configured; frontend Vitest suite present. |
| Code reviewed | ⚠️ | Feature implemented; formal PR review not tracked here. |
| No stray debug logging | ✅ | No `console.log` in production frontend paths (only in dev capture script). |
| Error handling | ✅ | Local path: abort + cloud fallback, persist errors surfaced; registry sync fails open to cached/bundled YAML. |

**Action:** Fix catalog test isolation (unset `HELPER_REGISTRY_URL` / use temp `HELPER_CACHED_REGISTRY_PATH` in `test_helper_app.py`). Run full `uv run poe test-fast test-inference` + `npm test` before deploy.

### Security

| Item | Status | Evidence / gap |
|------|--------|----------------|
| No secrets in VCS | ✅ | JWT / DB secrets via env only. |
| Input validation on persist APIs | ✅ | Pydantic schemas with `min_length`, confidence bounds; auth via `get_current_user`. |
| Auth on persist endpoints | ✅ | `persist_local_transcribe` / `persist_local_segment` require JWT. |
| Helper binds localhost only | ✅ | Default `HELPER_HOST=127.0.0.1`. |
| CORS on helper | ⚠️ | Must set `HELPER_CORS_ORIGINS` to **production SPA origin** at PyInstaller build time (defaults include localhost dev only). |
| CORS on platform API | ✅ | Existing `CORSMiddleware` on hosted app. |
| Auth on helper `/run` | ⚠️ Accepted v1 | ADR 002: intentionally open on localhost; risk on shared/library machines. Token handshake deferred. |
| Public registry endpoint | ✅ | `GET /inference/v1/registry` is intentionally public (YAML catalog only, no secrets). |
| Rate limiting on registry | — | Not required for static YAML; optional CDN cache. |

### Performance

| Item | Status | Evidence / gap |
|------|--------|----------------|
| Helper bundle size | ⚠️ | CPU-only PyTorch + Kraken stack; manual review of `dist/` before signing documented in [`packaging/helper/README.md`](../../packaging/helper/README.md). No size budget or CI gate. |
| Model download UX | ✅ | Banner with model id + “Use cloud instead”. |
| Lazy weight load | ✅ | First `/run` resolves `hf://` into `~/.nomicous/hf/cache/`. |
| No N+1 on persist | ✅ | Batch persist for transcribe lines. |

### Accessibility

| Item | Status | Evidence / gap |
|------|--------|----------------|
| Install / download banners | ✅ | `role="status"`, `aria-live="polite"` on active download banner. |
| Settings toggle | ✅ | Native checkbox in settings panel; disabled when model is remote-only. |
| Keyboard / screen reader audit | ⚠️ | Not run for new inference UI; recommend quick axe pass on page editor with banner visible. |

### Infrastructure

| Item | Status | Evidence / gap |
|------|--------|----------------|
| `HELPER_REGISTRY_URL` in installers | ⚠️ | Injected at install time; defaults to `https://api.nomicous.example/inference/v1/registry` — **must override for production**. |
| `INFERENCE_REGISTRY_PATH` on API | ✅ | `MLSettings.inference_registry_path` → serves deployed `inference/registry.yaml`. |
| Database migrations | ✅ | Local persist uses existing tables; no new migration required for v1. |
| Health checks | ✅ | Helper `GET /health`; platform `GET /health` (DB-aware). |
| Logging | ✅ | LaunchAgent / systemd / Scheduled Task → `~/.nomicous/logs/inference-helper.log`. |
| Installer auto-start | ✅ | LaunchAgent (macOS), Scheduled Task (Windows), systemd user unit (Linux). |
| GitHub release artifacts | ❌ | `PageEditorInferenceBanner` links to `https://github.com/kkkamur07/greekOCR/releases` — **no releases published yet**. |
| Code signing / notarization | ❌ | macOS Gatekeeper and Windows SmartScreen will warn on unsigned binaries. |
| Windows installer format | ⚠️ | PRD asked for `.msi`/`.exe`; shipped approach is **zip + `install-helper.ps1`** (extra IT friction vs one-click installer). |

### Documentation

| Item | Status | Evidence / gap |
|------|--------|----------------|
| ADR | ✅ | [`docs/decisions/002-local-inference-helper.md`](../decisions/002-local-inference-helper.md) |
| Add-model runbook | ✅ | [`docs/inference/adding-inference-models.md`](adding-inference-models.md) |
| Packaging README | ✅ | [`packaging/helper/README.md`](../../packaging/helper/README.md) |
| Launch / rollback doc | ✅ | This file |
| Changelog | ❌ | No project CHANGELOG entry for 038–041 |

---

## What is production-ready today

These paths can be exercised **without** shipping installers to end users:

1. **Developers** — `python -m inference.helper` + hosted SPA on localhost (CORS includes `localhost:5173`).
2. **Staging API** — Deploy platform with `GET /inference/v1/registry`; point a dev helper at staging `HELPER_REGISTRY_URL`.
3. **Cloud-only users** — No helper installed: existing remote inference path unchanged; install banner is non-blocking.
4. **New models** — Update `inference/registry.yaml`, deploy API, restart helpers; weights download on first use (documented in adding-inference-models).

---

## Blockers before public launch

Priority order:

1. **Publish signed installers** — Build per OS, attach to a GitHub Release (or CDN); update install banner URL to a versioned asset or release latest page with real artifacts.
2. **Production URLs at build/install** — Pass real values when building installers:
   ```bash
   HELPER_CORS_ORIGINS=https://app.yourdomain.com \
   HELPER_REGISTRY_URL=https://api.yourdomain.com/inference/v1/registry \
   bash packaging/helper/macos/build-dmg.sh
   ```
   Windows: `build-installer.ps1 -CorsOrigin … -RegistryUrl …`
3. **Code signing** — macOS notarization + Windows Authenticode (or document enterprise sideload process).
4. **Feature flag or staged UI** — Hide install banner / local path until team validates in production (see rollout below).
5. **Observability** — At minimum: log local-vs-cloud routing decisions client-side (sampled) and alert on registry sync failure rate from helper logs.

---

## Recommended staged rollout

Follow incremental exposure; do **not** enable for 100% of users on first deploy.

```
1. DEPLOY API + frontend (local path code live, flag OFF or banner hidden)
   └── Verify GET /inference/v1/registry returns valid YAML + ETag

2. INTERNAL TEAM (flag ON for staff accounts)
   └── Install signed helper on 2–3 machines (macOS + Windows + Linux)
   └── Manual smoke: pairing assist, auto-segment, cloud toggle, model download

3. BETA (5–10% of projects or invite list)
   └── Monitor: persist API 4xx/5xx, client errors, support tickets
   └── Thresholds: roll back if persist error rate >2× baseline or new JS error types >0.1% sessions

4. GRADUAL (25% → 50% → 100%)
   └── Same monitoring window (24–48h per step)

5. PUBLISH INSTALLERS publicly when beta metrics green
```

### Feature flag suggestion

No flag exists today. Minimal option before launch:

```typescript
// e.g. VITE_ENABLE_LOCAL_INFERENCE=true in production when ready
const localInferenceEnabled = import.meta.env.VITE_ENABLE_LOCAL_INFERENCE === 'true';
```

- **Flag OFF:** Skip health probe; always use cloud jobs; hide install banner.
- **Flag ON:** Current behavior (`useInferenceHost`, banners, local `/run`).

Give the flag an owner and remove within two weeks of 100% rollout.

---

## Monitoring and observability

### What to watch

| Signal | Source | Alert threshold |
|--------|--------|-----------------|
| `POST …/local-inference/transcribe` 5xx rate | API logs / APM | >2× baseline |
| `POST …/local-inference/segment` 5xx rate | API logs / APM | >2× baseline |
| Registry sync warnings | `~/.nomicous/logs/inference-helper.log` | Spike after API deploy |
| Client fetch failures to `127.0.0.1:8001` | Frontend error reporting (not wired yet) | New error type >0.1% sessions |
| Model download duration | Manual / future metric | P95 >5 min without user abort |

### Post-deploy verification (first hour)

```
□ curl https://api…/inference/v1/registry → 200, valid YAML
□ curl -H "If-None-Match: \"<etag>\"" …/registry → 304
□ Installed helper: curl http://127.0.0.1:8001/health → ok
□ Installed helper: curl http://127.0.0.1:8001/inference/v1/catalog → models with host_eligibility
□ Browser: pairing assist with helper → transcription in Postgres
□ Browser: toggle “Use cloud inference” → cloud job enqueued
□ Browser: no helper → cloud path, no blocking error
```

---

## Rollback plan

### Trigger conditions

- Local persist 5xx rate **>2×** baseline for 15+ minutes
- Widespread reports of helper crashing on startup (auto-start loop)
- Registry sync serving invalid YAML (should be caught by `load_registry` on API — helper rejects bad payloads)

### Rollback steps

| Layer | Action | Time |
|-------|--------|------|
| **Frontend** | Set `VITE_ENABLE_LOCAL_INFERENCE=false` and redeploy SPA | < 5 min |
| **API** | Revert `ml_registry_router` include (optional; cloud path unaffected) | < 10 min |
| **Helpers** | Users keep old cached `~/.nomicous/registry.yaml`; cloud toggle forces remote inference | Immediate |
| **Installers** | Unpublish release assets or mark pre-release | < 5 min |

No database rollback required — local persist writes the same tables as cloud callbacks.

---

## Go / no-go criteria

### Go for **internal beta**

- [ ] Production `HELPER_REGISTRY_URL` and `HELPER_CORS_ORIGINS` configured in at least one signed/internal installer
- [ ] `GET /inference/v1/registry` live on production API
- [ ] Manual smoke on macOS + one other OS
- [ ] `test-fast`, `test-inference`, `npm test` green in CI or locally
- [ ] Team assigned to monitor first 24h

### Go for **public launch**

All internal beta items, plus:

- [ ] GitHub Release (or CDN) with signed macOS + Windows + Linux artifacts
- [ ] Install banner points to real download
- [ ] Feature flag graduated to 100% with clean metrics
- [ ] Code signing / notarization (or documented enterprise exception)
- [ ] Changelog / user-facing “Install helper” doc
- [ ] Rollback drill completed once

---

## Quick commands

```bash
# Platform tests (no Postgres)
uv run poe test-fast
uv run poe test-inference

# Platform integration (Postgres on :5433)
uv run poe test-integration

# Frontend
cd nomicous/frontend && npm test

# Build macOS installer (set production origins)
HELPER_CORS_ORIGINS=https://app.example.com \
HELPER_REGISTRY_URL=https://api.example.com/inference/v1/registry \
bash packaging/helper/macos/build-dmg.sh

# Dev helper with registry sync
HELPER_REGISTRY_URL=http://localhost:8000/inference/v1/registry \
python -m inference.helper
```

---

## Related docs

- [ADR 002 — Local inference helper](../decisions/002-local-inference-helper.md)
- [Adding an inference model](adding-inference-models.md)
- [Packaging README](../../packaging/helper/README.md)
- [PRD (completed)](../../issues/done/inference/prd-local-inference-helper.md)
