#!/bin/bash
# DPO training script - Using Llama3-UltraFeedback-ArmoRM dataset + Mistral-7B-Instruct + OpenRLHF + DeepSpeed
# Usage: bash dpo_mistral.sh

set -x

# Set environment variables - Optimize GPU memory usage
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTORCH_CUDA_ALLOC_CONF=max_s

# External parameter configuration - Using Llama3-UltraFeedback-ArmoRM dataset and Mistral model
SFT_MODEL="<path-to-your-sft-model>"  # Path to your SFT model, e.g., mistralai/Mistral-7B-Instruct-v0.2
DATASET_PATH="<path-to-your-dataset>"  # Path to your DPO dataset in JSONL format

# Trainer selection (optional)
# Options: "simpo" (SimPOTrainer24), "ipo" (IPOTrainer25), "rdpo" (RDPOTrainer26), "orpo" (ORPOTrainer27), "dpo22" (DPOTrainer22), "dpo23" (DPOTrainer23)
# Leave empty to use standard DPOTrainer
TRAINER=""

deepspeed --module openrlhf.cli.train_dpo \
    --save_path ./checkpoint/<your-model-name>-dpo \
    --save_steps 1000 \
    --logging_steps 20 \
    --eval_steps 500 \
    --train_batch_size 32 \
    --micro_train_batch_size 2 \
    --pretrain $SFT_MODEL \
    --ref_pretrain $SFT_MODEL \
    --bf16 \
    --max_epochs 3 \
    --max_len 512 \
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
    --wandb_project <your-wandb-project-name> \
    --wandb_run_name <your-experiment-name> \
    ${TRAINER:+--formula $TRAINER}
