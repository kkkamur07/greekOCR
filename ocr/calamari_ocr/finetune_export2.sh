#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --gres=gpu:1
# Fine-tune the Bible-trained Calamari OCR model on the line crops in "export 2".
#
# Expected raw input layout:
#   export 2/
#     some_line.jpg
#     some_line.txt
#
# Calamari expects paired files with the same basename and .gt.txt labels:
#   dataset/calamari_finetune_export2/train/some_line.jpg
#   dataset/calamari_finetune_export2/train/some_line.gt.txt
#
# This script builds that pack, then warm-starts from:
#   outputs/calamari-greek-bible/best.ckpt
#
# Optional environment overrides:
#   CALAMARI_FINETUNE_EXPORT_DIR     default: ./export 2
#   CALAMARI_FINETUNE_PACK           default: ./dataset/calamari_finetune_export2
#   CALAMARI_FINETUNE_OUTPUT         default: ./outputs/calamari-greek-export2-finetune
#   CALAMARI_FINETUNE_CHECKPOINT     default: ./outputs/calamari-greek-bible/best.ckpt
#   CALAMARI_FINETUNE_EPOCHS         default: 50
#   CALAMARI_FINETUNE_LR             default: 0.0001
#   CALAMARI_FINETUNE_ES_N_TO_GO     default: 10
#   CALAMARI_FINETUNE_N_AUGMENTATIONS default: 5
#   CALAMARI_FINETUNE_VAL_EVERY      default: 6  (every 6th line goes to validation)
#   CALAMARI_FINETUNE_PREPARE_ONLY   default: 0  (set to 1 to only build the pack)
#   CALAMARI_GPU                     default: 0  (set to "" for CPU)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_repo_root() {
  [[ -d "$1/ocr/calamari_ocr" && -d "$1/export 2" ]]
}

find_repo_root() {
  local base candidate resolved top
  for base in \
    "${CALAMARI_REPO:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD}" \
    "${SCRIPT_DIR}" \
    "${SCRIPT_DIR}/.." \
    "${SCRIPT_DIR}/../.." \
    "/home/math/gupta/work/greek_byzantine/greek-foundation"; do
    [[ -n "${base}" ]] || continue
    for candidate in "${base}" "${base}/.." "${base}/../.."; do
      [[ -d "${candidate}" ]] || continue
      resolved="$(cd "${candidate}" && pwd)"
      if top="$(git -C "${resolved}" rev-parse --show-toplevel 2>/dev/null)" && is_repo_root "${top}"; then
        echo "${top}"
        return 0
      fi
      if is_repo_root "${resolved}"; then
        echo "${resolved}"
        return 0
      fi
    done
  done
  return 1
}

REPO="$(find_repo_root || true)"
if [[ -z "${REPO}" ]]; then
  echo "Could not determine repository root. Set CALAMARI_REPO=/path/to/greek-foundation." >&2
  exit 1
fi
cd "${REPO}"

EXPORT_DIR="${CALAMARI_FINETUNE_EXPORT_DIR:-${REPO}/export 2}"
PACK="${CALAMARI_FINETUNE_PACK:-${REPO}/dataset/calamari_finetune_export2}"
OUT="${CALAMARI_FINETUNE_OUTPUT:-${REPO}/outputs/calamari-greek-export2-finetune}"
CHECKPOINT="${CALAMARI_FINETUNE_CHECKPOINT:-${REPO}/outputs/calamari-greek-bible/best.ckpt}"
EPOCHS="${CALAMARI_FINETUNE_EPOCHS:-50}"
LR="${CALAMARI_FINETUNE_LR:-0.0001}"
ES_N_TO_GO="${CALAMARI_FINETUNE_ES_N_TO_GO:-10}"
N_AUGMENTATIONS="${CALAMARI_FINETUNE_N_AUGMENTATIONS:-5}"
VAL_EVERY="${CALAMARI_FINETUNE_VAL_EVERY:-6}"
PREPARE_ONLY="${CALAMARI_FINETUNE_PREPARE_ONLY:-0}"
GPU="${CALAMARI_GPU:-0}"

LOG_DIR="${OUT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/finetune_${TIMESTAMP}.log"
METRICS_FILE="${LOG_DIR}/finetune_${TIMESTAMP}_metrics.csv"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Writing full run output to: ${LOG_FILE}"
echo "Writing metrics-only CSV to: ${METRICS_FILE}"
echo "Resolved repository root: ${REPO}"
echo "Using export directory:   ${EXPORT_DIR}"
echo "Using fine-tune pack:     ${PACK}"
echo "Using output directory:   ${OUT}"
echo "Using warmstart model:    ${CHECKPOINT}"

if [[ -x "${REPO}/.venv/bin/calamari-train" ]]; then
  CALAMARI_TRAIN="${REPO}/.venv/bin/calamari-train"
else
  CALAMARI_TRAIN="$(command -v calamari-train || true)"
fi

if [[ "${PREPARE_ONLY}" != "1" && -z "${CALAMARI_TRAIN}" ]]; then
  echo "Could not find calamari-train. Activate the venv or install calamari-ocr." >&2
  exit 1
fi

if [[ ! -d "${EXPORT_DIR}" ]]; then
  echo "Export directory not found: ${EXPORT_DIR}" >&2
  exit 1
fi

if ! [[ "${VAL_EVERY}" =~ ^[0-9]+$ ]] || [[ "${VAL_EVERY}" -lt 2 ]]; then
  echo "CALAMARI_FINETUNE_VAL_EVERY must be an integer >= 2." >&2
  exit 1
fi

if [[ "${PREPARE_ONLY}" != "1" && ! -e "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Expected the model directory, not only best.ckpt.json." >&2
  exit 1
fi
export EXPORT_DIR PACK VAL_EVERY

python - <<'PY'
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

export_dir = Path(os.environ["EXPORT_DIR"]).expanduser().resolve()
pack = Path(os.environ["PACK"]).expanduser().resolve()
val_every = int(os.environ["VAL_EVERY"])

image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_text_any_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def matching_image(label: Path) -> Path | None:
    for ext in image_exts:
        candidate = label.with_suffix(ext)
        if candidate.is_file():
            return candidate
        candidate = label.with_suffix(ext.upper())
        if candidate.is_file():
            return candidate
    return None


labels = sorted(
    p for p in export_dir.glob("*.txt")
    if not p.name.endswith(".gt.txt") and p.is_file()
)

pairs: list[tuple[Path, Path]] = []
for label in labels:
    image = matching_image(label)
    if image is None:
        print(f"Missing image for {label.name}; skipping", file=sys.stderr)
        continue
    text = read_text_any_encoding(label).strip()
    if not text:
        print(f"Empty label {label.name}; skipping", file=sys.stderr)
        continue
    pairs.append((image, label))

if len(pairs) < 2:
    raise SystemExit(
        f"Need at least 2 image/text pairs in {export_dir}; found {len(pairs)}."
    )

if pack.exists():
    shutil.rmtree(pack)
(pack / "train").mkdir(parents=True)
(pack / "val").mkdir(parents=True)

counts = {"train": 0, "val": 0}
for i, (image, label) in enumerate(pairs, start=1):
    split = "val" if i % val_every == 0 else "train"
    dest_dir = pack / split
    dest_img = dest_dir / image.name
    dest_gt = dest_dir / f"{image.stem}.gt.txt"

    dest_img.symlink_to(image)
    dest_gt.write_text(read_text_any_encoding(label).strip() + "\n", encoding="utf-8")
    counts[split] += 1

if counts["val"] == 0:
    # Keep a validation set even for tiny experiments.
    moved_img = next(
        p for p in (pack / "train").iterdir()
        if p.suffix.lower() in image_exts
    )
    moved_gt = pack / "train" / f"{moved_img.stem}.gt.txt"
    shutil.move(str(moved_img), pack / "val" / moved_img.name)
    shutil.move(str(moved_gt), pack / "val" / moved_gt.name)
    counts["train"] -= 1
    counts["val"] += 1

print(f"Prepared Calamari fine-tuning pack: {pack}")
print(f"  train lines: {counts['train']}")
print(f"  val lines:   {counts['val']}")
PY

mapfile -t TRAIN_IMGS < <(find "${PACK}/train" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \) | sort)
mapfile -t VAL_IMGS < <(find "${PACK}/val" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \) | sort)

if [[ ${#TRAIN_IMGS[@]} -eq 0 || ${#VAL_IMGS[@]} -eq 0 ]]; then
  echo "No train/val images found after preparing ${PACK}" >&2
  exit 1
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Prepared ${PACK}; skipping training because CALAMARI_FINETUNE_PREPARE_ONLY=1."
  echo "Log saved to: ${LOG_FILE}"
  exit 0
fi

{
  echo "========================================"
  echo "Calamari fine-tuning started: $(date)"
  echo "  Export dir:        ${EXPORT_DIR}"
  echo "  Pack:              ${PACK}"
  echo "  Warmstart model:   ${CHECKPOINT}"
  echo "  Output:            ${OUT}"
  echo "  Train images:      ${#TRAIN_IMGS[@]}"
  echo "  Val images:        ${#VAL_IMGS[@]}"
  echo "  Epochs:            ${EPOCHS}"
  echo "  Learning rate:     ${LR}"
  echo "  Augmentations:     ${N_AUGMENTATIONS}"
  echo "  Early stopping:    ${ES_N_TO_GO} epochs patience (-1 = disabled)"
  echo "  GPU:               ${GPU:-CPU}"
  echo "  Slurm job:         ${SLURM_JOB_ID:-not running under Slurm}"
  echo "  Slurm job GPUs:    ${SLURM_JOB_GPUS:-<unset>}"
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "========================================"
  nvidia-smi || true
  echo "========================================"
}

CMD=(
  "${CALAMARI_TRAIN}"
  --network def
  --warmstart.model "${CHECKPOINT}"
  --codec.keep_loaded true
  --n_augmentations "${N_AUGMENTATIONS}"
  --trainer.output_dir "${OUT}"
  --trainer.epochs "${EPOCHS}"
  --learning_rate.lr "${LR}"
  --early_stopping.n_to_go "${ES_N_TO_GO}"
  --early_stopping.frequency 1
  --train.gt_extension .gt.txt
  --val.gt_extension .gt.txt
  --train.images "${TRAIN_IMGS[@]}"
  --val.images "${VAL_IMGS[@]}"
)

if [[ -n "${GPU}" ]]; then
  CMD+=(--device.gpus "${GPU}")
fi

"${CMD[@]}"

echo ""
echo "Fine-tuning finished: $(date)"

export OUT METRICS_FILE
python - <<'PY'
from __future__ import annotations

import csv
import os
from pathlib import Path

out = Path(os.environ["OUT"]).expanduser().resolve()
metrics_file = Path(os.environ["METRICS_FILE"]).expanduser().resolve()


def load_scalars(directory: Path, tag: str) -> dict[int, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        print(f"Could not import TensorBoard event reader: {exc}")
        return {}

    if not directory.is_dir():
        return {}

    try:
        accumulator = EventAccumulator(str(directory))
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            return {}
        return {event.step: event.value for event in accumulator.Scalars(tag)}
    except Exception as exc:
        print(f"Could not read metrics from {directory}: {exc}")
        return {}


rows: list[dict[str, object]] = []
stage_dirs = [
    ("main", out),
    ("aug_data", out / "aug_data"),
    ("real_data", out / "real_data"),
]

for stage, stage_dir in stage_dirs:
    train_dir = stage_dir / "train"
    val_dir = stage_dir / "validation"
    train_cer = load_scalars(train_dir, "epoch_CER")
    train_loss = load_scalars(train_dir, "epoch_ctc-loss")
    val_cer = load_scalars(val_dir, "epoch_CER")
    val_loss = load_scalars(val_dir, "epoch_ctc-loss")

    steps = sorted(set(train_cer) | set(train_loss) | set(val_cer) | set(val_loss))
    for step in steps:
        rows.append(
            {
                "stage": stage,
                "epoch": step,
                "train_cer": train_cer.get(step, ""),
                "train_ctc_loss": train_loss.get(step, ""),
                "val_cer": val_cer.get(step, ""),
                "val_ctc_loss": val_loss.get(step, ""),
            }
        )

with metrics_file.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stage",
            "epoch",
            "train_cer",
            "train_ctc_loss",
            "val_cer",
            "val_ctc_loss",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Metrics-only CSV saved to: {metrics_file}")
if not rows:
    print("No TensorBoard scalar metrics found yet. The CSV contains only the header.")
print("Note: Calamari training logs CER and CTC loss. WER requires a separate calamari-eval run after prediction.")
PY

echo "Log saved to: ${LOG_FILE}"
echo "Metrics saved to: ${METRICS_FILE}"
echo "Fine-tuned checkpoints under: ${OUT}"
