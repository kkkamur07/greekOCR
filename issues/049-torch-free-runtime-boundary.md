---
id: "049"
title: "torch-free-runtime-boundary"
type: AFK
status: ready
tracker: "https://github.com/kkkamur07/greekOCR/issues/49"
blocked_by: []
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Make the ONNX runtime path structurally Torch-free.

The "Torch-free helper" property is currently enforced only by a packaging denylist, and it is not actually true: Torch is imported at module scope on the BLLA path, so importing the runtime imports Torch. Before a real package boundary can exist, the boundary has to be real in the code.

Move the Torch model definitions — the BLLA model graph and the Calamari model/layers/config modules — into the export and parity tree, outside what will become the published package. The ONNX execution paths must import cleanly in an environment where Torch is not installed at all.

Conversion and parity scripts keep using Torch; they simply stop living inside the runtime tree.

Note that the runtime stays ONNX. Switching to Torch was considered during planning and rejected on measurement — see the Implementation Decisions section of #47.

## Acceptance criteria

- [ ] No module reachable from the ONNX execution paths imports Torch at module scope
- [ ] The runtime imports and executes a real page in an environment with Torch uninstalled
- [ ] Torch model definitions live in the export/parity tree and conversion scripts still run
- [ ] Both architectures still honour the shared artifact-preflight and per-line isolation contracts
- [ ] Segment and transcribe produce byte-identical output to before the move, on real bundled weights

## Blocked by

None - can start immediately
