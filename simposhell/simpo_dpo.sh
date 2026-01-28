#!/bin/bash
# Fixed DPO training script - Using SimPO-style UltraFeedback on-policy dataset
# Based on SimPO Instruct setup

set -x

# Set environment variables - Optimize GPU memory usage
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,expandable_segments:True

# External parameter configuration - Using SimPO-style UltraFeedback on-policy dataset
SFT_MODEL="/home/ubuntu/basemodels/llama3/llama3-8b-instruct"  # Use local IT version model
DATASET_PATH="/home/ubuntu/rrhf/ultrafeedback_onpolicy_dpo.jsonl"  # SimPO-style UltraFeedback on-policy dataset

echo "=== SimPO-style DPO Training ==="
echo "SFT Model: $SFT_MODEL"
echo "Dataset: $DATASET_PATH"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file does not exist: $DATASET_PATH"
    echo "Please run first: bash /home/ubuntu/Open/rrhf/simpo_generate_data.sh"
    exit 1
fi

deepspeed --module openrlhf.cli.train_dpo \
    --save_path ./checkpoint/llama3-8b-simpo-instruct \
    --save_steps 1000 \
    --logging_steps 20 \
    --eval_steps 500 \
    --train_batch_size 32 \
    --micro_train_batch_size 2 \
    --pretrain $SFT_MODEL \
    --ref_pretrain $SFT_MODEL \
    --bf16 \
    --max_epochs 1 \
    --max_len 2048 \
    --zero_stage 3 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --dataset $DATASET_PATH \
    --chosen_key chosen \
    --rejected_key rejected \
    --prompt_key prompt \
    --gradient_checkpointing \
    --flash_attn \
    --use_wandb true \
    --wandb_project simpo_instruct_dpo \
    --wandb_run_name llama3-8b-simpo-instruct-ultrafeedback

echo "=== SimPO-style DPO Training Completed ==="
