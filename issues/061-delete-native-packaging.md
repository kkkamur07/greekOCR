---
id: "061"
title: "delete-native-packaging"
type: AFK
status: done
tracker: "https://github.com/kkkamur07/greekOCR/issues/61"
blocked_by:
  - "050-publish-inference-package.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Delete native packaging. Roughly 1,137 lines of per-OS packaging wrapped 516 lines of program; a release becomes a PyPI publish.

What goes: the PyInstaller spec, the three installer script sets, the four-way build matrix, both signing pipelines (Developer ID notarization and Authenticode), the signing credentials, and the release assets.

This changes the character of security patching, which is the real point. Frozen installers made us the distributor of a vendored dependency tree with no update channel — every CVE in `onnxruntime`, `protobuf`, `Pillow`, or `scipy` required a four-platform rebuild, re-sign, and re-notarize, with no way to make anyone install the result. It becomes a dependency bump plus a version-floor bump, enforced by #55.

The assertion that a tray-icon library must not appear stays, even though the constraint that motivated it is retired: a CLI has no business growing a tray icon.

While here, correct the packaging documentation that still references a deleted catalog route.

## Acceptance criteria

- [ ] No PyInstaller spec, installer scripts, or bundle-signing configuration remain
- [ ] No build matrix produces per-OS artifacts
- [ ] Signing credentials are removed from CI configuration and can be revoked
- [ ] The tray-icon exclusion assertion still passes
- [ ] Remaining packaging documentation describes install from PyPI and references no deleted routes
- [ ] CI green with the packaging jobs removed

## Notes from #50

Found while publishing the package; left alone deliberately, because deleting
them is this issue's job.

- `packaging/helper/pyinstaller.spec` had `"src.hf.resolve"` in `hiddenimports`.
  That module moved into the package as `inference.hub`, so the entry was
  retargeted rather than left dangling — the spec still builds until it goes.
- `release-helper.yml` builds its SBOM and Trivy input from
  `uv export --no-default-groups --group helper --group packaging`. The `helper`
  group was cut down to the loopback HTTP surface (the model runtime moved to
  `[project].dependencies`), but that export resolves byte-identically to before,
  so the scan inputs are unchanged. Both groups die with this issue and #60.
- `packaging/helper/scripts/smoke-test.py` and the Windows installer assertion in
  `tests/nomicous/unit/test_deployment_hardening.py:142` both pin `HF_CACHE_ROOT`.
  The runtime default moved to `~/.nomicous/hf/cache`, which is what those two
  were setting by hand; the env var still overrides.
- The tray-icon exclusion assertion this issue keeps lives in the same file and
  still passes.
- `docs/codebase-review-2026-08-04.md` still refers to `src/hf/resolve/` and
  `src/hf/cache/`. It is a dated review, left as a record rather than edited.

## Blocked by

- #50
