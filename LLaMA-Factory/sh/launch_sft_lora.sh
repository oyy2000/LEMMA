#!/bin/bash

cd "$(dirname "$0")/.."

MODEL_NAME=$1
DATA_PATH=$2
LR=$3
WARMUP_RATIO=$4
GRADIENT_ACCUMULATION_STEPS=$5
N_EPOCH=$6

SAVE_LIMIT=1
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-1}
CUTOFF_LEN=${CUTOFF_LEN:-2048}
template="default"
base_model_name=${MODEL_NAME}

LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET=${LORA_TARGET:-all}
USE_BF16=${USE_BF16:-False}
USE_FP16=${USE_FP16:-True}

echo "Runing on: $(hostname)"
echo "SLURM ID: $SLURM_JOB_ID"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "DATA_PATH: ${DATA_PATH}"
echo "LR: ${LR}"
echo "WARMUP_RATIO: ${WARMUP_RATIO}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "N_EPOCH: ${N_EPOCH}"
echo "LORA_RANK: ${LORA_RANK}"
echo "LORA_ALPHA: ${LORA_ALPHA}"
echo "LORA_DROPOUT: ${LORA_DROPOUT}"
echo "LORA_TARGET: ${LORA_TARGET}"
echo "USE_BF16: ${USE_BF16}"
echo "USE_FP16: ${USE_FP16}"

export ds_config=${DEEPSPEED_CONFIG:-"examples/deepspeed/ds_z3_config.json"}
DEEPSPEED_ARGS=(--deepspeed "${ds_config}")

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path ${base_model_name} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target ${LORA_TARGET} \
    --template ${template} \
    --flash_attn fa2 \
    --dataset_dir data \
    --dataset ${DATA_PATH} \
    --cutoff_len ${CUTOFF_LEN} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --num_train_epochs ${N_EPOCH} \
    --max_samples 1000000 \
    --per_device_train_batch_size ${PER_GPU_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_steps -1 \
    --save_total_limit $SAVE_LIMIT \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir sft-models/${base_model_name}/${DATA_PATH}_lora_lr${LR}_warm${WARMUP_RATIO}_gas${GRADIENT_ACCUMULATION_STEPS}_epoch${N_EPOCH} \
    --bf16 ${USE_BF16} \
    --fp16 ${USE_FP16} \
    --gradient_checkpointing True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    "${DEEPSPEED_ARGS[@]}" \
    --save_only_model
