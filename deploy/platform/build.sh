#!/usr/bin/env bash
# Bundle platform API sources into deploy/platform for Vercel.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"
uv export --only-group platform-prod --no-hashes -o "$DEST/requirements.txt"

rm -rf "$DEST/nomicous" "$DEST/inference"
mkdir -p "$DEST/nomicous" "$DEST/inference"

rsync -a \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='media' \
  "$ROOT/nomicous/backend/" "$DEST/nomicous/backend/"
rsync -a \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "$ROOT/nomicous/infrastructure/" "$DEST/nomicous/infrastructure/"
cp "$ROOT/inference/__init__.py" "$DEST/inference/"
cp "$ROOT/inference/registry.yaml" "$DEST/inference/"
cp -r "$ROOT/inference/contracts" "$DEST/inference/"
cp -r "$ROOT/inference/registry" "$DEST/inference/"
cp "$ROOT/nomicous/VERSION" "$DEST/nomicous/"
