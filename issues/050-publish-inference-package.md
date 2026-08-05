---
id: "050"
title: "publish-inference-package"
type: AFK
status: in_progress
tracker: "https://github.com/kkkamur07/greekOCR/issues/50"
blocked_by:
  - "048-collapse-second-job-queue.md 049-torch-runtime-archive-onnx.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Publish the runtime as one package, `nomicous-inference`, carrying both the library and the CLI entry point. The hosted worker installs the same package; two packages would buy a version-compatibility matrix between components that always ship together.

The boundary moves `hf://` **weights source** resolution *into* the package — it is already on the ONNX runtime path despite living outside the inference tree — and leaves the Torch definitions outside it.

This is the slice that lets the packaging denylist die. `excludes.txt` and the bundle verifier exist to keep Torch out of a frozen bundle by denylist; a real package boundary makes it unleakable, so both are deleted rather than ported forward.

Only the console entry point is needed here; the subcommands land in later slices.

## Reversed by ADR 0004

Two criteria below were written when ONNX Runtime was the runtime and Torch was
the thing to keep out. [ADR 0004](../docs/adr/0004-pytorch-is-the-inference-runtime.md)
made PyTorch the runtime, so Torch *is* the package and the ~60 MB target is
gone. `excludes.txt` and the bundle verifier were already deleted by #49, which
killed them by construction rather than by boundary. The struck criteria are
replaced by the two below them.

## Acceptance criteria

- [x] A wheel builds from the repository and installs into a clean virtual environment
- [x] ~~Torch and Torchvision are absent from the installed dependency closure~~ → Torch is present and CPU-only; no CUDA wheel and no `onnxruntime` in the closure, on every target platform
- [x] The installed package resolves an `hf://` **weights source**, verifies **artifact SHA-256**, and caches under the researcher's home directory
- [x] A real page is segmented and transcribed *through the installed package*, not the repository tree
- [x] The console entry point is present and executable after install
- [x] `excludes.txt` and the bundle verifier are deleted (landed in #49)
- [x] ~~Installed runtime footprint stays in the ~60 MB range, not the ~400 MB range~~ → the real footprint and cold-install time are measured and recorded in ADR 0004: 969 MB installed, ~980 MB fetched

## Blocked by

- #48
- #49
