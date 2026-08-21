#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="var/runs/workstation-augmentation-smoke/${RUN_ID}"
LOG_ROOT="var/logs/workstation/augmentation-smoke/${RUN_ID}"
DATASETS=(combined greek armenian syriac)
START_FROM="${START_FROM:-}"
STARTED=false

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

command -v uv >/dev/null 2>&1 || {
    echo "Error: uv is not available on PATH." >&2
    exit 127
}
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "Error: nvidia-smi is not available; a CUDA workstation is required." >&2
    exit 127
}

for dataset in "${DATASETS[@]}"; do
    for split in train val; do
        manifest="data/processed/${dataset}/pretraining/gt_${split}.txt"
        [[ -s "${manifest}" ]] || {
            echo "Error: missing or empty manifest: ${manifest}" >&2
            exit 1
        }
    done
done

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

echo "Repository: ${REPO_ROOT}"
echo "Run ID: ${RUN_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

run_case() {
    local name="$1"
    shift
    if [[ "${STARTED}" == false ]]; then
        if [[ -n "${START_FROM}" && "${name}" != "${START_FROM}" ]]; then
            echo "=== Skipping ${name} ==="
            return
        fi
        STARTED=true
    fi
    local log_file="${LOG_ROOT}/${name}.log"
    echo
    echo "=== Starting ${name} ==="
    "$@" 2>&1 | tee "${log_file}"
    echo "=== Completed ${name} ==="
}

run_calamari() {
    local dataset="$1"
    local label="$2"
    local probability="$3"
    local variants="$4"
    local name="cal-$(dataset_code "${dataset}")-${label}"

    run_case "${name}" \
        bash scripts/workstation/calamari/train.sh \
        "data=pretraining/${dataset}" \
        training.epochs=1 \
        training.device=cuda \
        training.seed=1111 \
        "augmentation.probability=${probability}" \
        "augmentation.n_augmentations=${variants}" \
        "output.root=${RUN_ROOT}/calamari/${dataset}-${label}" \
        "wandb.name=${name}-${RUN_ID}"
}

trocr_tokenizer() {
    case "$1" in
        combined) echo "gpt_500" ;;
        greek) echo "gpt_greek_500" ;;
        armenian) echo "gpt_armenian_500" ;;
        syriac) echo "gpt_syriac_500" ;;
        *)
            echo "Unknown dataset: $1" >&2
            return 2
            ;;
    esac
}

dataset_code() {
    case "$1" in
        combined) echo "cmb" ;;
        greek) echo "gr" ;;
        armenian) echo "ar" ;;
        syriac) echo "sy" ;;
        *)
            echo "Unknown dataset: $1" >&2
            return 2
            ;;
    esac
}

run_trocr() {
    local dataset="$1"
    local label="$2"
    local probability="$3"
    local variants="$4"
    local name="trb-$(dataset_code "${dataset}")-${label}"

    run_case "${name}" \
        bash scripts/workstation/tr_ocr/train.sh \
        "data=pretraining/${dataset}" \
        "tokenizer=$(trocr_tokenizer "${dataset}")" \
        encoder=base \
        model.freeze_encoder=false \
        decoder.reinitialize=token_layers \
        lora_adaptors.enabled=false \
        training.epochs=1 \
        training.batch_size=32 \
        training.eval_batch_size=32 \
        "augmentation.probability=${probability}" \
        "augmentation.n_augmentations=${variants}" \
        "output.root=${RUN_ROOT}/trocr-base/${dataset}-${label}" \
        "wandb.name=${name}-${RUN_ID}"
}

for dataset in "${DATASETS[@]}"; do
    run_calamari "${dataset}" aug 1.0 3
    run_calamari "${dataset}" plain 0.0 0
done

for dataset in "${DATASETS[@]}"; do
    run_trocr "${dataset}" aug 1.0 3
    run_trocr "${dataset}" plain 0.0 0
done

if [[ -n "${START_FROM}" && "${STARTED}" == false ]]; then
    echo "Error: START_FROM case was not found: ${START_FROM}" >&2
    exit 2
fi

echo
echo "All workstation augmentation smoke tests completed."
