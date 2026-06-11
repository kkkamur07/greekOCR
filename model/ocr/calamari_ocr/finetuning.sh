#!/usr/bin/env bash
# Finetune a Calamari checkpoint on a prepared pack (train/ + val/ with *.gt.txt).
#
# Prerequisites:
#   pip install -e 'annote/backend[calamari]'
#   python model/ocr/calamari_ocr/prepare_calamari_pack.py --data-root ./data/greekData --out-dir ./data/calamari_pack_greek
#
# Usage (from repo root):
#   bash model/ocr/calamari_ocr/finetuning.sh
#
# Environment:
#   CALAMARI_PACK       — pack root with train/ and val/ (default: ./data/calamari_pack_greek)
#   CALAMARI_WARMSTART  — pretrained checkpoint path, without extra suffix (default: ./model/checkpoints/best.ckpt)
#   CALAMARI_OUTPUT     — run output dir (default: ./model/outputs/calamari-greek-finetune)
#   CALAMARI_EPOCHS         — max epochs (default: 100)
#   CALAMARI_LEARNING_RATE  — constant LR for --learning_rate.lr (default: 0.0001)
#   CALAMARI_ES_N_TO_GO     — early-stopping patience; -1 disables (default: 20)
#   CALAMARI_GPU            — GPU id, e.g. 0; empty string for CPU (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PACK="${CALAMARI_PACK:-${REPO_ROOT}/data/calamari_pack_greek}"
WARMSTART="${CALAMARI_WARMSTART:-${REPO_ROOT}/model/checkpoints/best.ckpt}"
OUT="${CALAMARI_OUTPUT:-${REPO_ROOT}/model/outputs/calamari-greek-finetune}"
EPOCHS="${CALAMARI_EPOCHS:-100}"
LEARNING_RATE="${CALAMARI_LEARNING_RATE:-0.0001}"
ES_N_TO_GO="${CALAMARI_ES_N_TO_GO:-20}"
GPU="${CALAMARI_GPU:-0}"

LOG_DIR="${OUT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"
CER_LOG_FILE="${LOG_FILE%.log}_cer.log"

if [[ ! -d "${PACK}/train" || ! -d "${PACK}/val" ]]; then
  echo "error: expected ${PACK}/train and ${PACK}/val" >&2
  exit 1
fi

if [[ ! -e "${WARMSTART}" || ! -e "${WARMSTART}.json" ]]; then
  echo "error: expected warmstart checkpoint and metadata:" >&2
  echo "  ${WARMSTART}" >&2
  echo "  ${WARMSTART}.json" >&2
  exit 1
fi

TRAIN_IMGS=()
while IFS= read -r img; do
  TRAIN_IMGS+=("${img}")
done < <(find "${PACK}/train" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort)

VAL_IMGS=()
while IFS= read -r img; do
  VAL_IMGS+=("${img}")
done < <(find "${PACK}/val" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort)

if [[ ${#TRAIN_IMGS[@]} -eq 0 || ${#VAL_IMGS[@]} -eq 0 ]]; then
  echo "error: no train/val images under ${PACK}" >&2
  exit 1
fi

GPU_ARGS=()
if [[ -n "${GPU}" ]]; then
  GPU_ARGS=(--device.gpus "${GPU}")
fi

{
  echo "Calamari finetune started: $(date)"
  echo "  Pack:       ${PACK} (${#TRAIN_IMGS[@]} train, ${#VAL_IMGS[@]} val)"
  echo "  Warmstart:  ${WARMSTART}"
  echo "  Output:     ${OUT}"
  echo "  Epochs:     ${EPOCHS}"
  echo "  LR:         ${LEARNING_RATE}"
  echo "  Patience:   ${ES_N_TO_GO}"
  echo "  GPU:        ${GPU:-CPU}"
  echo "  CER log:    ${CER_LOG_FILE}"
  command -v nvidia-smi >/dev/null && nvidia-smi || true
} | tee "${LOG_FILE}"

calamari-train \
  --network def \
  --n_augmentations 5 \
  --warmstart.model "${WARMSTART}" \
  --trainer.output_dir "${OUT}" \
  --trainer.epochs "${EPOCHS}" \
  --learning_rate.lr "${LEARNING_RATE}" \
  --early_stopping.n_to_go "${ES_N_TO_GO}" \
  --early_stopping.frequency 1 \
  "${GPU_ARGS[@]}" \
  --train.gt_extension .gt.txt \
  --val.gt_extension .gt.txt \
  --train.images "${TRAIN_IMGS[@]}" \
  --val.images "${VAL_IMGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}" >(awk '/(^|[^[:alnum:]_])(val_)?CER([^[:alnum:]_]|$)/ { print; fflush() }' >> "${CER_LOG_FILE}")

echo "Done: $(date)" | tee -a "${LOG_FILE}"
echo "Best model: ${OUT}/best.ckpt" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
echo "CER log: ${CER_LOG_FILE}"
