#!/usr/bin/env bash
# Build a macOS .dmg for the Inference Helper (PyInstaller + LaunchAgent).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HELPER_DIR="$ROOT/packaging/helper"
DIST_DIR="$HELPER_DIR/dist"
APP_NAME="Nomicous Inference Helper"
APP_BUNDLE="$DIST_DIR/${APP_NAME}.app"

# PyInstaller must run on the target architecture. CI sets MACOS_ARCH
# explicitly; local builds default to the current machine architecture.
REQUESTED_ARCH="${MACOS_ARCH:-$(uname -m)}"
case "$REQUESTED_ARCH" in
  arm64|aarch64)
    MACOS_ARCH="arm64"
    DEFAULT_DMG_NAME="nomicous-inference-helper-macos.dmg"
    ;;
  x86_64|amd64)
    MACOS_ARCH="x86_64"
    DEFAULT_DMG_NAME="nomicous-inference-helper-macos-intel.dmg"
    ;;
  *)
    echo "ERROR: unsupported MACOS_ARCH '$REQUESTED_ARCH' (use arm64 or x86_64)." >&2
    exit 1
    ;;
esac

ACTUAL_ARCH="$(uname -m)"
if [ "$ACTUAL_ARCH" != "$MACOS_ARCH" ]; then
  echo "ERROR: requested macOS architecture '$MACOS_ARCH' but runner is '$ACTUAL_ARCH'." >&2
  echo "       PyInstaller builds must run natively on the target architecture." >&2
  exit 1
fi

DMG_NAME="${MACOS_DMG_NAME:-$DEFAULT_DMG_NAME}"
DMG_PATH="$DIST_DIR/$DMG_NAME"

# Code signing / notarization (all optional - unset = unsigned build).
#   MACOS_CODESIGN_IDENTITY  "Developer ID Application: Name (TEAMID)"
#   MACOS_ENTITLEMENTS       path to entitlements plist (default: macos/entitlements.plist)
#   MACOS_NOTARY_PROFILE     notarytool keychain profile (from `notarytool store-credentials`)
CODESIGN_IDENTITY="${MACOS_CODESIGN_IDENTITY:-}"
ENTITLEMENTS="${MACOS_ENTITLEMENTS:-$SCRIPT_DIR/entitlements.plist}"
NOTARY_PROFILE="${MACOS_NOTARY_PROFILE:-}"

export MACOS_CODESIGN_IDENTITY="$CODESIGN_IDENTITY"
export MACOS_ENTITLEMENTS="$ENTITLEMENTS"

"$HELPER_DIR/scripts/build-pyinstaller.sh"

# PyInstaller's BUNDLE target (see pyinstaller.spec) produces the .app with the
# correct macOS layout: the Python framework and shared libraries live under
# Contents/Frameworks/, which is where the bootloader looks when the executable
# runs from inside a .app. Do NOT hand-assemble the bundle by copying the onedir
# output into Contents/MacOS/ - that leaves libpython in Contents/MacOS/_internal/
# and the helper crashes on launch with "Failed to load Python shared library".
if [ ! -d "$APP_BUNDLE" ]; then
  echo "ERROR: expected PyInstaller app bundle at '$APP_BUNDLE'." >&2
  echo "       Ensure pyinstaller.spec defines a BUNDLE() target for macOS." >&2
  exit 1
fi

# Ship the LaunchAgent template inside the bundle so install-helper.sh can
# template it into ~/Library/LaunchAgents at install time.
mkdir -p "$APP_BUNDLE/Contents/Resources"
cp "$SCRIPT_DIR/com.nomicous.inference-helper.plist" "$APP_BUNDLE/Contents/Resources/"

mkdir -p "$DIST_DIR/payload/Applications"
rm -rf "$DIST_DIR/payload/Applications/${APP_NAME}.app"
cp -R "$APP_BUNDLE" "$DIST_DIR/payload/Applications/"

cp "$SCRIPT_DIR/install-helper.sh" "$DIST_DIR/payload/install-helper.sh"
chmod +x "$DIST_DIR/payload/install-helper.sh"

# --- Code sign the assembled .app (hardened runtime) --------------------------
if [ -n "$CODESIGN_IDENTITY" ]; then
  echo "Signing app bundle with: $CODESIGN_IDENTITY"
  # Sign every Mach-O inside the bundle from the inside out, then the bundle
  # itself. Signing nested binaries first avoids the pitfalls of --deep.
  find "$DIST_DIR/payload/Applications/${APP_NAME}.app" \
    \( -type f \( -name "*.dylib" -o -name "*.so" -o -perm -111 \) \) -print0 \
    | while IFS= read -r -d '' macho; do
        codesign --force --timestamp --options runtime \
          --entitlements "$ENTITLEMENTS" --sign "$CODESIGN_IDENTITY" "$macho" 2>/dev/null || true
      done
  codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" --sign "$CODESIGN_IDENTITY" \
    "$DIST_DIR/payload/Applications/${APP_NAME}.app"
  codesign --verify --deep --strict --verbose=2 "$DIST_DIR/payload/Applications/${APP_NAME}.app"
else
  echo "WARNING: MACOS_CODESIGN_IDENTITY not set - building UNSIGNED .app (Gatekeeper will warn on install)."
fi

hdiutil create -volname "Nomicous Inference Helper" -srcfolder "$DIST_DIR/payload" -ov -format UDZO "$DMG_PATH"

if [ -n "$CODESIGN_IDENTITY" ]; then
  codesign --force --timestamp --sign "$CODESIGN_IDENTITY" "$DMG_PATH"
fi

# --- Notarize + staple --------------------------------------------------------
if [ -n "$NOTARY_PROFILE" ]; then
  echo "Submitting $DMG_PATH for notarization (profile: $NOTARY_PROFILE)..."
  xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG_PATH"
  echo "Notarized and stapled $DMG_PATH"
elif [ -n "$CODESIGN_IDENTITY" ]; then
  echo "NOTE: signed but not notarized (MACOS_NOTARY_PROFILE unset). Gatekeeper may still block on first launch."
fi

echo "Built $DMG_PATH"
