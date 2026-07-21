# Inference Helper packaging

Ship a **minimal** native install (`.dmg`, Windows zip + installer script, Linux tarball) containing only what local inference needs:

- `inference/helper` slim FastAPI app (`health`, `catalog`, `/inference/v1/run`)
- Calamari transcribe (ONNX Runtime **CPU**)
- BLLA page segmentation (`hf://kkkamur07/segmentation-blla@stable`, ONNX)
- `src/hf/resolve` for `hf://` weight download into `~/.nomicous/hf/cache/`
- Bundled `inference/registry.yaml`

## Explicitly excluded from installers

- Platform API (`nomicous/`), Postgres drivers, Alembic, frontend assets
- Training stacks (`transformers`, `accelerate`, `datasets`, notebooks)
- Torch, torchvision, safetensors, and all native model implementations
- GPU/CUDA runtime libraries
- Desktop shells (Tauri/Electron) - helper is a background service

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
    diagnose-helper.sh
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
# Run on Apple silicon:
MACOS_ARCH=arm64 bash packaging/helper/macos/build-dmg.sh

# Run on an Intel Mac:
MACOS_ARCH=x86_64 bash packaging/helper/macos/build-dmg.sh

bash packaging/helper/linux/build-tarball.sh
powershell packaging/helper/windows/build-installer.ps1
```

Manual PyInstaller (same as the shared script):

```bash
cd packaging/helper
uv run --isolated --no-dev --group helper --group packaging pyinstaller --clean pyinstaller.spec
```

The spec entry point is `packaging/helper/tray_launcher.py` and runs the server directly.

Every platform build runs `scripts/verify-bundle.py` before assembling its
installer. The verifier rejects leaked Torch/safetensors files, launches the
actual frozen executable (the `.app` executable on macOS), and checks both
`/health` and `/inference/v1/catalog`. Do **not** exclude `onnxruntime`,
`opencv`, `Pillow`, or `httpx`: those are required by inference, preprocessing,
and `hf://` downloads.

The native Python/Torch implementations remain available in the `train`,
`export`, and `parity` environments for training, export, and parity checks.
The production `inference` dependency set and frozen helper are ONNX-only; the
helper explicitly rejects non-`.onnx` artifacts.

## Per-OS installers

| OS | Command | Auto-start |
|----|---------|------------|
| macOS (Apple silicon) | `MACOS_ARCH=arm64 bash packaging/helper/macos/build-dmg.sh` | LaunchAgent |
| macOS (Intel) | `MACOS_ARCH=x86_64 bash packaging/helper/macos/build-dmg.sh` | LaunchAgent |
| Windows | `powershell packaging/helper/windows/build-installer.ps1` | Scheduled Task at logon |
| Linux | `bash packaging/helper/linux/build-tarball.sh` | systemd user unit or desktop autostart |

Outputs land in `packaging/helper/dist/`:

- `nomicous-inference-helper-macos.dmg` (Apple silicon; existing asset name)
- `nomicous-inference-helper-macos-intel.dmg`
- `nomicous-inference-helper-windows.zip`
- `nomicous-inference-helper-linux.tar.gz`

Each installer runs `install-helper` which copies the PyInstaller bundle, creates `~/.nomicous/hf/cache/`, and registers auto-start.
macOS builds must run natively on the architecture being packaged; the build
script rejects an `MACOS_ARCH`/runner mismatch instead of silently producing
an incompatible bundle. The release workflow uses `macos-15` for Apple silicon
and `macos-15-intel` for Intel.

The helper is loopback-only and accepts browser requests only from
`https://app.nomicous.com`. It intentionally has no browser-shipped secret.

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

- `MACOS_CODESIGN_IDENTITY` - deep-signs the `.app` (and inner dylibs) with the hardened runtime and signs the `.dmg`.
- `MACOS_ENTITLEMENTS` - override the default `macos/entitlements.plist` (relaxes library validation for bundled native Python/ONNX libraries).
- `MACOS_NOTARY_PROFILE` - submits the `.dmg` to Apple, waits, and staples the ticket.

Signing only (no notarization) still leaves a Gatekeeper prompt on first launch - set the notary profile for a clean install.

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

If Ubuntu reports `connection refused`, run the packaged diagnostic script:

```bash
"${XDG_DATA_HOME:-$HOME/.local/share}/nomicous/inference-helper/diagnose-helper.sh"
```

It checks the loopback endpoint, the systemd user service or desktop-autostart
fallback, listening sockets, and the helper log.
