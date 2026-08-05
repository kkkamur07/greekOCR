---
id: "050"
title: "publish-inference-package"
type: AFK
status: backlog
tracker: "https://github.com/kkkamur07/greekOCR/issues/50"
blocked_by:
  - "048-collapse-second-job-queue.md 049-torch-free-runtime-boundary.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Publish the runtime as one package, `nomicous-inference`, carrying both the library and the CLI entry point. The hosted worker installs the same package; two packages would buy a version-compatibility matrix between components that always ship together.

The boundary moves `hf://` **weights source** resolution *into* the package — it is already on the ONNX runtime path despite living outside the inference tree — and leaves the Torch definitions outside it.

This is the slice that lets the packaging denylist die. `excludes.txt` and the bundle verifier exist to keep Torch out of a frozen bundle by denylist; a real package boundary makes it unleakable, so both are deleted rather than ported forward.

Only the console entry point is needed here; the subcommands land in later slices.

## Acceptance criteria

- [ ] A wheel builds from the repository and installs into a clean virtual environment
- [ ] Torch and Torchvision are absent from the installed dependency closure
- [ ] The installed package resolves an `hf://` **weights source**, verifies **artifact SHA-256**, and caches under the researcher's home directory
- [ ] A real page is segmented and transcribed *through the installed package*, not the repository tree
- [ ] The console entry point is present and executable after install
- [ ] `excludes.txt` and the bundle verifier are deleted
- [ ] Installed runtime footprint stays in the ~60 MB range, not the ~400 MB range

## Blocked by

- #48
- #49
