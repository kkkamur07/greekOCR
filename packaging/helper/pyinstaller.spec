# PyInstaller spec for the Nomicous Inference Helper.
#
# Goal: ship only runtime needed for Calamari transcribe + Kraken segment on CPU.
# Exclude training stacks, notebooks, GUI toolkits, and unused torch backends.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO_ROOT = Path(SPECPATH).resolve().parents[1]

block_cipher = None

# macOS code signing (ignored on Windows/Linux). When MACOS_CODESIGN_IDENTITY
# is set, PyInstaller signs the bootloader and bundled binaries during the
# freeze; build-dmg.sh then re-signs the assembled .app and notarizes. Leaving
# it unset produces an unsigned build (Gatekeeper will warn).
_macos_identity = os.environ.get("MACOS_CODESIGN_IDENTITY") or None
_entitlements = None
if _macos_identity:
    _entitlements_override = os.environ.get("MACOS_ENTITLEMENTS")
    if _entitlements_override:
        _entitlements = str(Path(_entitlements_override).resolve())
    else:
        _default_entitlements = Path(SPECPATH) / "macos" / "entitlements.plist"
        _entitlements = str(_default_entitlements) if _default_entitlements.is_file() else None

# Data files bundled beside the binary (not Python deps).
datas = [
    (str(REPO_ROOT / "inference" / "registry.yaml"), "inference"),
]

# Kraken segment weights ship via package://kraken/blla.mlmodel (importlib
# resources), so its bundled data (blla.mlmodel, iso15924.json, ...) must be
# collected explicitly — PyInstaller does not gather package data by default.
datas += collect_data_files("kraken")

hiddenimports = [
    "inference.helper",
    "inference.helper.__main__",
    "inference.architectures.calamari",
    "inference.architectures.kraken",
    "src.hf.resolve",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.lifespan.on",
]

# Kraken, SciPy and scikit-image import large parts of their surface
# dynamically; collect submodules so the frozen segment path does not hit
# ModuleNotFoundError at runtime.
hiddenimports += collect_submodules("kraken")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("skimage")

# huggingface_hub loads its HTTP backend (httpx) lazily inside functions, so
# PyInstaller's static analysis never discovers it. Force-collect httpx (and
# huggingface_hub) or hf:// weight downloads fail with "No module named httpx"
# on the Calamari transcribe path.
hiddenimports += collect_submodules("httpx")
hiddenimports += collect_submodules("huggingface_hub")

_excludes_path = Path(SPECPATH) / "excludes.txt"
excludes = [
    line.strip()
    for line in _excludes_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

a = Analysis(
    [str(REPO_ROOT / "packaging" / "helper" / "tray_launcher.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="nomicous-inference-helper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=_macos_identity,
    entitlements_file=_entitlements,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="nomicous-inference-helper",
)

# On macOS, emit a real .app via BUNDLE so PyInstaller lays out the Python
# framework and shared libraries under Contents/Frameworks/. Hand-assembling a
# bundle by copying the onedir tree into Contents/MacOS/ leaves libpython in
# Contents/MacOS/_internal/, but the bootloader (once running from inside a
# .app) looks for it in Contents/Frameworks/ and the helper crashes on launch.
# BUNDLE is a no-op on non-macOS platforms, so guard it for clarity.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Nomicous Inference Helper.app",
        icon=None,
        bundle_identifier="com.nomicous.inference-helper",
        info_plist={
            "CFBundleName": "Nomicous Inference Helper",
            "CFBundleDisplayName": "Nomicous Inference Helper",
            "CFBundleExecutable": "nomicous-inference-helper",
            "CFBundlePackageType": "APPL",
            # Background agent: no Dock icon, no menu bar presence.
            "LSUIElement": True,
        },
    )
