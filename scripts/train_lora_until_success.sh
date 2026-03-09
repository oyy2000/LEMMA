#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEMMA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${LEMMA_ROOT}/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128"}
export USE_BF16=${USE_BF16:-False}
export USE_FP16=${USE_FP16:-True}

export PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-1}
export CUTOFF_LEN=${CUTOFF_LEN:-2048}
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-examples/deepspeed/ds_z3_config.json}

LOG_DIR=${LOG_DIR:-${LEMMA_ROOT}/logs/sft_lora}
mkdir -p "$LOG_DIR"

N_EPOCH=${N_EPOCH:-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LR=${LR:-1e-5}

PRIMARY_MODEL_NAME=${PRIMARY_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
FALLBACK_MODEL_NAME=${FALLBACK_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
PRIMARY_MAX_ATTEMPTS=${PRIMARY_MAX_ATTEMPTS:-1}
RETRY_SLEEP_SECONDS=${RETRY_SLEEP_SECONDS:-30}
JOB_NAME=${JOB_NAME:-lora_retry}

DATA_PATHS=(
    lemma_sft_data
)

train_once() {
    local model_name="$1"
    local data_path="$2"

    START_TIME=$(date +%Y%m%d-%H:%M:%S)
    LOG_FILE=${LOG_DIR}/${START_TIME}_${JOB_NAME}.log

    echo "[$(date '+%F %T')] Start LoRA training"
    echo "Model: ${model_name}"
    echo "Dataset: ${data_path}"
    echo "Log: ${LOG_FILE}"

    bash "${LEMMA_ROOT}/LLaMA-Factory/sh/launch_sft_lora.sh" \
        "$model_name" "$data_path" "$LR" "$WARMUP_RATIO" "$GRADIENT_ACCUMULATION_STEPS" "$N_EPOCH" \
        >> "${LOG_FILE}" 2>&1
}

for DATA_PATH in "${DATA_PATHS[@]}"; do
    current_model="$PRIMARY_MODEL_NAME"
    attempt=0

    while true; do
        attempt=$((attempt + 1))
        echo "[$(date '+%F %T')] Attempt ${attempt} | model=${current_model} | data=${DATA_PATH}"

        if train_once "$current_model" "$DATA_PATH"; then
            echo "[$(date '+%F %T')] Success on attempt ${attempt} with ${current_model}"
            break
        fi

        echo "[$(date '+%F %T')] Failed on attempt ${attempt} with ${current_model}"

        if [[ "$current_model" == "$PRIMARY_MODEL_NAME" && "$attempt" -ge "$PRIMARY_MAX_ATTEMPTS" ]]; then
            echo "[$(date '+%F %T')] Switching model to fallback: ${FALLBACK_MODEL_NAME}"
            current_model="$FALLBACK_MODEL_NAME"
            attempt=0
        fi

        echo "[$(date '+%F %T')] Retry after ${RETRY_SLEEP_SECONDS}s ..."
        sleep "$RETRY_SLEEP_SECONDS"
    done
done
