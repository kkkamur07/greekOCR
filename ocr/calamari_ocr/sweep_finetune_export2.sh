#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --gres=gpu:1
# Run a small hyperparameter sweep for fine-tuning on "export 2".
#
# This script runs several full fine-tuning jobs serially. Each run writes to its
# own output folder under outputs/calamari-greek-export2-sweep/.
#
# Default sweep:
#   1. baseline_lr1e-4_aug5       LR 1e-4,  5 augmentations
#   2. low_lr5e-5_aug5            LR 5e-5,  5 augmentations
#   3. low_lr5e-5_aug10           LR 5e-5, 10 augmentations
#   4. very_low_lr2e-5_aug15      LR 2e-5, 15 augmentations
#
# Override common settings:
#   CALAMARI_SWEEP_EPOCHS=50
#   CALAMARI_SWEEP_ES_N_TO_GO=10
#   CALAMARI_SWEEP_ROOT=./outputs/calamari-greek-export2-sweep
#   CALAMARI_GPU=0
#
# Run:
#   ./ocr/calamari_ocr/sweep_finetune_export2.sh
#
# Run only the first two configs:
#   CALAMARI_SWEEP_LIMIT=2 ./ocr/calamari_ocr/sweep_finetune_export2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO}"

FINETUNE_SCRIPT="${REPO}/ocr/calamari_ocr/finetune_export2.sh"
if [[ ! -x "${FINETUNE_SCRIPT}" ]]; then
  echo "Fine-tuning script is not executable: ${FINETUNE_SCRIPT}" >&2
  echo "Run: chmod +x ${FINETUNE_SCRIPT}" >&2
  exit 1
fi

SWEEP_ROOT="${CALAMARI_SWEEP_ROOT:-${REPO}/outputs/calamari-greek-export2-sweep}"
EPOCHS="${CALAMARI_SWEEP_EPOCHS:-50}"
ES_N_TO_GO="${CALAMARI_SWEEP_ES_N_TO_GO:-10}"
LIMIT="${CALAMARI_SWEEP_LIMIT:-0}"

mkdir -p "${SWEEP_ROOT}"

# name|learning_rate|n_augmentations|val_every
# Keep the validation split fixed so runs are comparable.
CONFIGS=(
  "baseline_lr1e-4_aug5|0.0001|5|6"
  "low_lr5e-5_aug5|0.00005|5|6"
  "low_lr5e-5_aug10|0.00005|10|6"
  "very_low_lr2e-5_aug15|0.00002|15|6"
)

SUMMARY="${SWEEP_ROOT}/sweep_summary.csv"
echo "run,lr,n_augmentations,epochs,early_stopping_patience,best_epoch,best_val_cer,best_val_ctc_loss,metrics_file,checkpoint" > "${SUMMARY}"

echo "Hyperparameter sweep output root: ${SWEEP_ROOT}"
echo "Summary CSV: ${SUMMARY}"
echo "Epochs per run: ${EPOCHS}"
echo "Early stopping patience: ${ES_N_TO_GO}"
echo ""

run_count=0
for config in "${CONFIGS[@]}"; do
  IFS='|' read -r name lr n_aug val_every <<< "${config}"
  run_count=$((run_count + 1))

  if [[ "${LIMIT}" != "0" && "${run_count}" -gt "${LIMIT}" ]]; then
    echo "Stopping after ${LIMIT} configured run(s)."
    break
  fi

  out_dir="${SWEEP_ROOT}/${name}"
  echo "========================================"
  echo "Sweep run ${run_count}: ${name}"
  echo "  lr:              ${lr}"
  echo "  augmentations:   ${n_aug}"
  echo "  val_every:       ${val_every}"
  echo "  output:          ${out_dir}"
  echo "========================================"

  CALAMARI_FINETUNE_OUTPUT="${out_dir}" \
  CALAMARI_FINETUNE_EPOCHS="${EPOCHS}" \
  CALAMARI_FINETUNE_LR="${lr}" \
  CALAMARI_FINETUNE_N_AUGMENTATIONS="${n_aug}" \
  CALAMARI_FINETUNE_ES_N_TO_GO="${ES_N_TO_GO}" \
  CALAMARI_FINETUNE_VAL_EVERY="${val_every}" \
    "${FINETUNE_SCRIPT}"

  latest_metrics="$(ls -t "${out_dir}"/logs/*_metrics.csv 2>/dev/null | sed -n '1p' || true)"
  checkpoint="${out_dir}/best.ckpt"

  if [[ -z "${latest_metrics}" ]]; then
    echo "${name},${lr},${n_aug},${EPOCHS},${ES_N_TO_GO},,,,,${checkpoint}" >> "${SUMMARY}"
    echo "No metrics CSV found for ${name}; skipping summary extraction."
    continue
  fi

  python - <<'PY' "${SUMMARY}" "${name}" "${lr}" "${n_aug}" "${EPOCHS}" "${ES_N_TO_GO}" "${latest_metrics}" "${checkpoint}"
from __future__ import annotations

import csv
import sys
from pathlib import Path

summary, name, lr, n_aug, epochs, patience, metrics, checkpoint = sys.argv[1:]
metrics_path = Path(metrics)

best: dict[str, str] | None = None
with metrics_path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        val_cer = row.get("val_cer", "")
        if not val_cer:
            continue
        if best is None or float(val_cer) < float(best["val_cer"]):
            best = row

with Path(summary).open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            name,
            lr,
            n_aug,
            epochs,
            patience,
            "" if best is None else best.get("epoch", ""),
            "" if best is None else best.get("val_cer", ""),
            "" if best is None else best.get("val_ctc_loss", ""),
            metrics,
            checkpoint,
        ]
    )

if best is None:
    print(f"No validation CER found in {metrics_path}")
else:
    print(
        f"Best for {name}: epoch {best.get('epoch')} "
        f"val_CER={best.get('val_cer')} val_ctc_loss={best.get('val_ctc_loss')}"
    )
PY
done

echo ""
echo "Sweep finished."
echo "Summary CSV saved to: ${SUMMARY}"
