# Inference Helper packaging

Ship a **minimal** native install (`.dmg`, Windows zip + installer script, Linux tarball) containing only what local inference needs:

- `inference/helper` slim FastAPI app (`health`, `catalog`, `/inference/v1/run`)
- Calamari transcribe (PyTorch **CPU**)
- Kraken segment (`package://kraken/blla.mlmodel`)
- `src/hf/resolve` for `hf://` weight download into `~/.nomicous/hf/cache/`
- Bundled `inference/registry.yaml`
- Optional system tray via `pystray` (packaging-only dependency)

## Explicitly excluded from installers

- Platform API (`nomicous/`), Postgres drivers, Alembic, frontend assets
- Training stacks (`transformers`, `accelerate`, `datasets`, notebooks)
- GPU/CUDA torch builds (CPU wheel only)
- Desktop shells (Tauri/Electron) — helper is a background service + tray icon

## Layout

```
packaging/helper/
  excludes.txt              # PyInstaller exclude list (one module per line)
  pyinstaller.spec            # loads excludes.txt
  scripts/
    build-pyinstaller.sh      # shared uv + PyInstaller build
  macos/
    build-dmg.sh              # entry: build → .app → .dmg
    Info.plist
    install-helper.sh
    com.nomicous.inference-helper.plist
  linux/
    build-tarball.sh
    install-helper.sh
    nomicous-inference-helper.service
  windows/
    build-installer.ps1
    install-helper.ps1
```

## Build (PyInstaller)

```bash
bash packaging/helper/scripts/build-pyinstaller.sh
```

Or via a platform installer script (runs the shared build first):

```bash
bash packaging/helper/macos/build-dmg.sh
bash packaging/helper/linux/build-tarball.sh
powershell packaging/helper/windows/build-installer.ps1
```

Manual PyInstaller (same as the shared script):

```bash
cd packaging/helper
uv pip install pyinstaller pystray
uv run --group inference pyinstaller pyinstaller.spec
```

The spec entry point is `packaging/helper/tray_launcher.py` (server + tray). Use `--no-tray` for headless/systemd installs.

Review `dist/nomicous-inference-helper/` before signing — if unexpected large directories appear (e.g. `torch/test`), add them to `excludes.txt`. Do **not** exclude `scipy`, `torchvision`, `sklearn`, `skimage`, or `shapely`: Kraken segment requires them at runtime (importing `kraken.blla` pulls in `scipy`), and excluding them breaks the segment model. Do **not** exclude `httpx` either: `huggingface_hub` uses it as its HTTP backend, so excluding it breaks `hf://` weight downloads (Calamari transcribe).

The current release bundles are large because they carry the complete CPU
runtime: approximately 311.6 MiB for macOS, 301.8 MiB for Windows, and
432.2 MiB for Linux in the `inference-helper-v0.1.1` release. These figures are
compressed download sizes and can change with each build. Since the published
model repositories are public, a future size-reduction pass could replace
`huggingface_hub` and its HTTP stack with a minimal `wget`-based downloader.
That should only happen after preserving pinned revisions, artifact hashes,
multi-file snapshots, and the existing cache layout.

## Per-OS installers

| OS | Command | Auto-start |
|----|---------|------------|
| macOS | `bash packaging/helper/macos/build-dmg.sh` | LaunchAgent |
| Windows | `powershell packaging/helper/windows/build-installer.ps1` | Scheduled Task at logon |
| Linux | `bash packaging/helper/linux/build-tarball.sh` | systemd user unit |

Outputs land in `packaging/helper/dist/`:

- `nomicous-inference-helper-macos.dmg`
- `nomicous-inference-helper-windows.zip`
- `nomicous-inference-helper-linux.tar.gz`

Each installer runs `install-helper` which copies the PyInstaller bundle, creates `~/.nomicous/hf/cache/`, and registers auto-start.

The helper is loopback-only and intentionally has no browser-shipped secret. Its installer
sets `HELPER_CORS_ORIGINS=https://app.nomicous.com` at runtime. For a different hosted SPA
origin, set an explicit origin while installing (for example,
`HELPER_CORS_ORIGINS=https://app.example.com bash install-helper.sh` on macOS/Linux, or
`.\install-helper.ps1 -CorsOrigins https://app.example.com` on Windows), then restart the
helper. Wildcard origins are rejected.

## Code signing

Builds are **unsigned by default** (Gatekeeper / SmartScreen will warn). Provide credentials via env vars / params to produce a trusted, distributable build. All signing is skipped with a warning when the relevant credentials are absent.

### macOS (Developer ID + notarization)

Requires a "Developer ID Application" certificate in your keychain and a stored `notarytool` profile:

```bash
# One-time: store notarization credentials under a profile name
xcrun notarytool store-credentials nomicous-notary \
  --apple-id you@example.com --team-id TEAMID --password <app-specific-password>

MACOS_CODESIGN_IDENTITY="Developer ID Application: Your Org (TEAMID)" \
MACOS_NOTARY_PROFILE="nomicous-notary" \
  bash packaging/helper/macos/build-dmg.sh
```

- `MACOS_CODESIGN_IDENTITY` — deep-signs the `.app` (and inner dylibs) with the hardened runtime and signs the `.dmg`.
- `MACOS_ENTITLEMENTS` — override the default `macos/entitlements.plist` (relaxes library validation / JIT for the bundled Python + torch).
- `MACOS_NOTARY_PROFILE` — submits the `.dmg` to Apple, waits, and staples the ticket.

Signing only (no notarization) still leaves a Gatekeeper prompt on first launch — set the notary profile for a clean install.

### Windows (Authenticode)

```powershell
# Cert already in the local store:
powershell packaging/helper/windows/build-installer.ps1 -SigningThumbprint <sha1-thumbprint>

# Or a .pfx file:
$env:WINDOWS_SIGNING_CERT = "C:\path\cert.pfx"
$env:WINDOWS_SIGNING_CERT_PASSWORD = "…"
powershell packaging/helper/windows/build-installer.ps1
```

Signs the helper `.exe` (add `-SignAllBinaries` to sign every bundled `.exe`/`.dll`/`.pyd`) with an RFC-3161 timestamp before zipping. Requires `signtool.exe` (Windows SDK). New EV/OV certs still accrue SmartScreen reputation over time.

### Linux

No OS-level signing gate. Distribute the tarball over HTTPS; optionally publish a detached GPG signature / SHA-256 sum alongside it.

## Runtime defaults

- Bind: `127.0.0.1:8001`
- Cache: `HF_CACHE_ROOT=~/.nomicous/hf/cache/`
- Registry sync: `HELPER_REGISTRY_URL=https://api…/inference/v1/registry` → `~/.nomicous/registry.yaml` (bundled `registry.yaml` is offline fallback)
- No auth on `/run` (v1)
- Browser access: the hosted app calls `http://127.0.0.1:8001` directly; do not enable
  `HELPER_SECURE_MODE` for this browser flow because its secret must not be embedded in frontend code.

## Smoke test after install

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/inference/v1/catalog
```
