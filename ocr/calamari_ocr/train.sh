#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --gres=gpu:1
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


#! do not change the training configuration, this is recommended by the Calamari documentation, will experiment with it later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_repo_root() {
  [[ -d "$1/ocr/calamari_ocr" && -d "$1/dataset" ]]
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

PACK="${CALAMARI_PACK:-${REPO}/dataset/calamari}"
OUT="${CALAMARI_OUTPUT:-${REPO}/outputs/calamari-greek-bible}"
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
  echo "  Slurm job:         ${SLURM_JOB_ID:-not running under Slurm}"
  echo "  Slurm job GPUs:    ${SLURM_JOB_GPUS:-<unset>}"
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
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
