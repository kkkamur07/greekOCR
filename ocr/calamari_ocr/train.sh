#!/usr/bin/env bash
# Train Calamari with settings aligned to Kaddas et al. ICDAR 2023 (Section 4):
# - network "def" (best in their Table 1)
# - data augmentation ON (--n_augmentations > 0)
#
# Prerequisites:
#   pip install calamari-ocr
#   python scripts/prepare_calamari_pack.py --data-root "$BIBLE_ROOT" --out-dir ./data/calamari_pack
#
# Environment:
#   CALAMARI_PACK             — folder with train/ and val/ (default ./dataset/calamari)
#   CALAMARI_OUTPUT           — output dir for checkpoints and logs (default ./outputs/calamari-greek-bible)
#   CALAMARI_EPOCHS           — number of epochs (default 100)
#   CALAMARI_ES_N_TO_GO       — early stopping patience in epochs (default 20; -1 = disabled)
#   CALAMARI_GPU              — GPU index to use (default 0 = first GPU)
#
# Calamari writes automatically to output_dir:
#   train.log        — full training log with params and per-epoch metrics
#   tensorboard/     — TensorBoard event files (run: tensorboard --logdir <output_dir>)
#   best/            — continuously updated best model (by val loss)
#   checkpoint*/     — checkpoint saved every epoch (for resume)

set -euo pipefail

PACK="${CALAMARI_PACK:-./dataset/calamari}"
OUT="${CALAMARI_OUTPUT:-./outputs/calamari-greek-bible}"
EPOCHS="${CALAMARI_EPOCHS:-100}"
ES_N_TO_GO="${CALAMARI_ES_N_TO_GO:-20}"
GPU="${CALAMARI_GPU:-0}"

LOG_DIR="${OUT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

if [[ ! -d "${PACK}/train" || ! -d "${PACK}/val" ]]; then
  echo "Expected ${PACK}/train and ${PACK}/val with paired .png + .gt.txt"
  exit 1
fi

mapfile -t TRAIN_IMGS < <(find "${PACK}/train" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort)
mapfile -t VAL_IMGS < <(find "${PACK}/val" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort)

if [[ ${#TRAIN_IMGS[@]} -eq 0 || ${#VAL_IMGS[@]} -eq 0 ]]; then
  echo "No train/val images found under ${PACK}"
  exit 1
fi

{
  echo "========================================"
  echo "Calamari training started: $(date)"
  echo "  Pack:              ${PACK}"
  echo "  Output:            ${OUT}"
  echo "  Train images:      ${#TRAIN_IMGS[@]}"
  echo "  Val images:        ${#VAL_IMGS[@]}"
  echo "  Epochs:            ${EPOCHS}"
  echo "  Early stopping:    ${ES_N_TO_GO} epochs patience (-1 = disabled)"
  echo "  GPU:               ${GPU}"
  echo "========================================"
  nvidia-smi
  echo "========================================"
} | tee "${LOG_FILE}"

calamari-train \
  --network def \
  --n_augmentations 5 \
  --trainer.output_dir "${OUT}" \
  --trainer.epochs "${EPOCHS}" \
  --early_stopping.n_to_go "${ES_N_TO_GO}" \
  --early_stopping.frequency 1 \
  --device.gpus "${GPU}" \
  --train.gt_extension .gt.txt \
  --val.gt_extension .gt.txt \
  --train.images "${TRAIN_IMGS[@]}" \
  --val.images "${VAL_IMGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Training finished: $(date)" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
echo "Checkpoints under: ${OUT}"
