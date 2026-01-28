#!/bin/bash
# GRPO training script - Using Veribench-53K hardware verification dataset + Qwen2.5-Coder-7B-Instruct + OpenRLHF + DeepSpeed
# Usage: bash grpo_code.sh

set -x

# Set environment variables - Optimize GPU memory usage
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,2,3  # Use multiple GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export WANDB_MODE=online
export WANDB_PROJECT=<your-wandb-project-name>

# External parameter configuration - Using Veribench-53K hardware verification dataset and Qwen2.5-Coder-3B model
SFT_MODEL="<path-to-your-sft-model>"  # Path to your SFT model, e.g., Qwen/Qwen2.5-Coder-3B-Instruct
DATASET_PATH="<path-to-your-dataset>"  # Path to your dataset file

# Set working directory to project root (adjust path as needed)
# cd <path-to-your-project-root>
# export PYTHONPATH=<path-to-your-project-root>:$PYTHONPATH

# Check if DeepSpeed is available
if command -v deepspeed &> /dev/null; then
        deepspeed cli/train_dpo.py \
        --pretrain $SFT_MODEL \
        --ref_pretrain $SFT_MODEL \
        --dataset $DATASET_PATH \
        --prompt_key prompt \
        --chosen_key response \
        --rejected_key "" \
        --trainer_type multiturn_grpo \
        --save_path ./checkpoint/<your-model-name>-grpo \
        --max_epochs 2 \
        --micro_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-6 \
        --l2 0.01 \
        --max_norm 1.0 \
        --save_steps 500 \
        --logging_steps 10 \
        --eval_steps 250 \
        --bf16 \
        --gradient_checkpointing \
        --max_len 512 \
        --max_completion_length 256 \
        --num_refinements_per_draft 2 \
        --zero_stage 3 \
        --ref_offload \
        --gradient_checkpointing_use_reentrant \
        --overlap_comm \
        --zpg 8 \
        --max_train_turns 3 \
        --max_infer_turns 6 \
        --use_turn_aware_advantage \
        --gen_abs_weight 0.0 \
        --gen_imp_weight 1.0 \
        --ver_weight 1.0 \
        --stop_when_pass_1 \
        --old_model_sync_frequency 4 \
        --beta 0.01 \
        --temperature 0.7 \
        --top_p 0.9 \
        --do_sample \
        --scale_rewards group \
        --loss_type dr_grpo \
        --importance_sampling_level sequence \
        --draft_source ref \
        --use_refinement_advantage \
        --seed 42
else
    python cli/train_dpo.py \
        --pretrain $SFT_MODEL \
        --ref_pretrain $SFT_MODEL \
        --dataset $DATASET_PATH \
        --trainer_type multiturn_grpo \
        --save_path ./checkpoint/<your-model-name>-grpo \
        --max_epochs 2 \
        --micro_train_batch_size 1 \
        --micro_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 1e-6 \
        --l2 0.01 \
        --warmup_steps 200 \
        --max_norm 1.0 \
        --save_steps 500 \
        --logging_steps 10 \
        --eval_steps 250 \
        --dataloader_num_workers 4 \
        --report_to wandb \
        --wandb_project <your-wandb-project-name> \
        --wandb_run_name <your-experiment-name> \
        --bf16 \
        --gradient_checkpointing \
        --flash_attn \
        --max_len 512 \
        --max_completion_length 256 \
        --num_refinements_per_draft 2 \
        --max_train_turns 3 \
        --max_infer_turns 6 \
        --use_turn_aware_advantage \
        --gen_abs_weight 0.0 \
        --gen_imp_weight 1.0 \
        --ver_weight 1.0 \
        --stop_when_pass_1 \
        --old_model_sync_frequency 4 \
        --beta 0.01 \
        --temperature 0.7 \
        --top_p 0.9 \
        --do_sample \
        --scale_rewards group \
        --loss_type dr_grpo \
        --importance_sampling_level sequence \
        --draft_source ref \
        --use_refinement_advantage \
        --seed 42
fi