---
id: "049"
title: "torch-runtime-archive-onnx"
type: AFK
status: in_progress
tracker: "https://github.com/kkkamur07/greekOCR/issues/49"
blocked_by: []
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
adr: "docs/adr/0004-pytorch-is-the-inference-runtime.md"
---

## Parent

#47

## What to build

**Supersedes the original scope of this issue, which was the exact opposite.** This issue previously required making the ONNX runtime path structurally Torch-free. The owner has since decided the runtime becomes PyTorch and ONNX is retired. See ADR 0004.

Replace ONNX Runtime with PyTorch as the execution runtime for both architectures, and archive the ONNX apparatus.

The reasoning that made ONNX worth its complexity was a frozen native bundle, where every megabyte shipped through a signing and notarization pipeline and a denylist kept Torch out. Distribution is now `uv tool install` from PyPI (ADR 0002), so that constraint is gone — and with it the justification for maintaining two runtimes for one set of models.

Training is already PyTorch. Making inference PyTorch too means one framework end to end, with no conversion step between what was trained and what runs.

### Archive rather than delete

The owner asked for this to be archived, not destroyed. Move the retired ONNX apparatus under `archive/onnx-runtime/` with a README explaining what it was, when it was retired, and pointing at ADR 0004 — git history alone is too hard to find later.

Archive: the two ONNX execution paths, the ONNX conversion scripts, and the parity tooling.

### What this deletes outright

Because Torch becomes the runtime rather than a forbidden dependency, several things stop having a reason to exist:

- `excludes.txt` and the bundle verifier — they exist to keep Torch out by denylist
- The ONNX↔Torch parity apparatus, and the `kraken` parity dependency, since there is no longer a second runtime to be at parity *with*
- Dual-format artifact publishing

### Weights

Native checkpoints are already published alongside the ONNX ones — `best.pt` for Calamari and `blla.safetensors` for BLLA are in the staging tree today, and the offline bundled weights are `.pt` only. Point **weights source** entries at the native **Hub artifact**s.

**Artifact SHA-256 verification must survive this change**, now covering the native checkpoints. Loading a Torch checkpoint is a code-execution surface in a way loading an ONNX graph is not, so use a safe loading path — prefer `safetensors` where available, and never `torch.load` with pickle on an unverified file.

### Preserve

Keep the two shared execution seams — `architectures/artifact.py` (artifact preflight ordering, which determines HTTP status) and `architectures/isolation.py` (per-line failure policy). Read `isolation.py`'s docstring: the exception *types* it re-raises are what map to 503 versus 422, and collapsing them would report a broken model as a successful transcription of a blank page. Its "must stay Torch-free" note is now obsolete and should be corrected, but the policy it enforces is not.

`tests/inference/unit/test_architecture_contract.py` exercises both architectures against both seams and must keep passing.

## Acceptance criteria

- [ ] Both segment and transcribe execute through PyTorch on CPU
- [ ] `model.eval()` and inference-mode/no-grad are used on every inference path
- [ ] No ONNX Runtime dependency remains in the runtime dependency closure
- [ ] Retired ONNX and parity code lives under `archive/` with a README pointing at ADR 0004
- [ ] `excludes.txt`, the bundle verifier, and the `kraken` parity dependency are gone
- [ ] **Artifact SHA-256** is still verified before an artifact is loaded
- [ ] Checkpoint loading does not unpickle unverified files
- [ ] Both shared execution seams still hold, with their HTTP status mapping intact
- [ ] Transcription and segmentation output is compared against the ONNX baseline on real weights, and any difference is reported rather than absorbed into a changed expectation
- [ ] Single-page CPU latency is recorded before and after, so the cost of the switch is on the record

## Blocked by

None - can start immediately
