#!/bin/bash
# GRPO math reasoning training script - Simplified version

# Only show error messages, reduce redundant output
set -e

# =============================================================================
# Configurable parameter area - Please modify the following parameters as needed
# =============================================================================

# Training mode selection
MODE="grpo"  # Options: "grpo" (original GRPO) or "refine_grpo" (refined GRPO)

# Reward function mode selection
REWARD_MODE="math"  # Options: "math" (math reasoning) or "code" (code generation)

# Basic configuration
SFT_MODEL="<path-to-your-sft-model>"  # Path to your SFT model, e.g., Qwen/Qwen2.5-4B-Instruct
DATASET_PATH="<path-to-your-dataset>"  # Path to your dataset directory or file
CUDA_DEVICE="0,1,2,3"  # Specify GPU IDs to use for distributed training
# Force CUDA to use specified GPUs
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
# DeepSpeed GPU configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
WANDB_PROJECT="<your-wandb-project-name>"  # Your wandb project name
WANDB_ENTITY="<your-wandb-entity>"  # Your wandb entity/username
WANDB_TAGS="grpo,math,experiment"    # Tags, separated by commas
WANDB_NOTES="GRPO math reasoning training experiment"   # Experiment description
# Disable wandb to avoid permission issues
DISABLE_WANDB=true

# Common dataset path examples (uncomment and modify MODE and DATASET_PATH to use)
# GSM8K: DATASET_PATH="<path-to-gsm8k-dataset>"
# MATH:  DATASET_PATH="<path-to-math-dataset>"
# AQuA:  DATASET_PATH="<path-to-aqua-dataset>"
# SVAMP: DATASET_PATH="<path-to-svamp-dataset>"

# Training parameters - Normal training version
NUM_EPOCHS=3
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2  # Changed to 2 to ensure gradient accumulation steps > 0
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=100
MAX_GRAD_NORM=1.0

# Save and logging parameters - Normal training version
SAVE_STEPS=1000
LOGGING_STEPS=20
EVAL_STEPS=500
MAX_LEN=64

# GRPO specific parameters
BETA_DEFAULT=0.01
GAMMA=1.0
TEMPERATURE=1.0
TOP_P=0.9
REWARD_CLIP_MIN=-10.0
REWARD_CLIP_MAX=10.0

# RefineGRPO specific parameters - Normal training version
DRAFT_SOURCE="ref"
NUM_DRAFTS_PER_PROMPT=1
NUM_REFINEMENTS_PER_DRAFT=4
CLIP_RATIO_MAX=2.0
GRADER_TYPE="rlvr"  # Options: "rlvr" or "mlp"

# GRPO refinement training parameters
USE_SEQUENCE_LEVEL_IS=true
USE_REFINEMENT_ADVANTAGE=true
CLIP_RATIO=0.2
KL_COEF=0.01
SCALE_REWARDS="group"
IMPORTANCE_SAMPLING_LEVEL="sequence"
KL_STABLE=true
LOSS_TYPE="dr_grpo"
DO_SAMPLE=true
TEMPERATURE=1.0
TOP_P=0.9

# Dataset parameter configuration - Normal training version
NUM_COMPLETIONS=2
MAX_COMPLETION_LENGTH=64
BETA=0.1
LOSS_TYPE="dr_grpo"

# Recommended parameters for different datasets (adjust as needed)
# GSM8K: NUM_COMPLETIONS=2, MAX_COMPLETION_LENGTH=64, BETA=0.02, LOSS_TYPE="dr_grpo"
# MATH:  NUM_COMPLETIONS=8, MAX_COMPLETION_LENGTH=1024, BETA=0.03, LOSS_TYPE="dapo"
# AQuA:  NUM_COMPLETIONS=2, MAX_COMPLETION_LENGTH=64, BETA=0.02, LOSS_TYPE="dapo"

# =============================================================================
# Python path and environment variable settings
# =============================================================================

# Set Python path to ensure rlhf module can be found
export PYTHONPATH="<path-to-your-project-root>:$PYTHONPATH"

# Wandb environment variable settings
if [ "$DISABLE_WANDB" = "true" ]; then
    export WANDB_MODE=disabled
    echo "Wandb disabled to avoid permission issues"
else
    export WANDB_MODE=online
    export WANDB_PROJECT=$WANDB_PROJECT
    if [ -n "$WANDB_ENTITY" ]; then
        export WANDB_ENTITY=$WANDB_ENTITY
    fi
fi

# =============================================================================
# Environment variable settings
# =============================================================================

export CUDA_LAUNCH_BLOCKING=1  # Enable CUDA error debugging
export TORCH_USE_CUDA_DSA=1     # Enable device-side assertions
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
# DeepSpeed GPU configuration
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false


# =============================================================================
# Parameter validation - Simplified version
# =============================================================================

echo "=== Validating parameters ==="
echo "MODE: $MODE"
echo "REWARD_MODE: $REWARD_MODE"
echo "DATASET: $DATASET_PATH"
echo "MODEL: $SFT_MODEL"
echo "GPUS: $CUDA_DEVICE"
echo "GRPO configuration:"
echo "  - Draft Source: $DRAFT_SOURCE"
echo "  - Refinements per draft: $NUM_REFINEMENTS_PER_DRAFT"
echo "  - Loss Type: $LOSS_TYPE"
echo "  - Scale Rewards: $SCALE_REWARDS"
echo "  - Importance Sampling: $IMPORTANCE_SAMPLING_LEVEL"

# Quick validation of key parameters
[ "$MODE" = "sft" ] || [ "$MODE" = "grpo" ] || [ "$MODE" = "refine_grpo" ] || { echo "Error: MODE must be 'sft', 'grpo' or 'refine_grpo'"; exit 1; }
[ "$REWARD_MODE" = "math" ] || [ "$REWARD_MODE" = "code" ] || { echo "Error: REWARD_MODE must be 'math' or 'code'"; exit 1; }
[ -d "$DATASET_PATH" ] || [ -f "$DATASET_PATH" ] || { echo "Error: Dataset path does not exist: $DATASET_PATH"; exit 1; }
[ -d "$SFT_MODEL" ] || { echo "Error: Model path does not exist: $SFT_MODEL"; exit 1; }

# Validate GRPO parameters
[ "$DRAFT_SOURCE" = "ref" ] || [ "$DRAFT_SOURCE" = "old" ] || [ "$DRAFT_SOURCE" = "current" ] || { echo "Error: DRAFT_SOURCE must be 'ref', 'old' or 'current'"; exit 1; }
[ "$LOSS_TYPE" = "dr_grpo" ] || [ "$LOSS_TYPE" = "dapo" ] || [ "$LOSS_TYPE" = "bnpo" ] || { echo "Error: LOSS_TYPE must be 'dr_grpo', 'dapo' or 'bnpo'"; exit 1; }
[ "$SCALE_REWARDS" = "group" ] || [ "$SCALE_REWARDS" = "batch" ] || [ "$SCALE_REWARDS" = "none" ] || { echo "Error: SCALE_REWARDS must be 'group', 'batch' or 'none'"; exit 1; }
[ "$IMPORTANCE_SAMPLING_LEVEL" = "sequence" ] || [ "$IMPORTANCE_SAMPLING_LEVEL" = "token" ] || { echo "Error: IMPORTANCE_SAMPLING_LEVEL must be 'sequence' or 'token'"; exit 1; }

# Set working directory to project root
cd <path-to-your-project-root>

# =============================================================================
# Parameter settings
# =============================================================================

# Extract dataset name from dataset path
DATASET_NAME=$(basename "$DATASET_PATH")

# Generate wandb run name
if [ "$MODE" = "grpo" ]; then
    WANDB_RUN_NAME="<your-model-name>-grpo-$DATASET_NAME"
else
    WANDB_RUN_NAME="<your-model-name>-refine-grpo-$DATASET_NAME-$GRADER_TYPE"
fi

# =============================================================================
# Output directory and parameter settings
# =============================================================================

# Set output directory
if [ "$MODE" = "sft" ]; then
    OUTPUT_DIR="./checkpoint/<your-model-name>-sft-$DATASET_NAME"
elif [ "$MODE" = "grpo" ]; then
    OUTPUT_DIR="./checkpoint/<your-model-name>-grpo-$DATASET_NAME"
    IMPORTANCE_SAMPLING_LEVEL="token"
    REWARD_FUNCTION="format_check"
else
    OUTPUT_DIR="./checkpoint/<your-model-name>-refine-grpo-$DATASET_NAME-$GRADER_TYPE"
    IMPORTANCE_SAMPLING_LEVEL="sequence"
    REWARD_FUNCTION=""
fi

# =============================================================================
# Training execution - Simplified version
# =============================================================================

echo "=== Starting training ==="
echo "Output directory: $OUTPUT_DIR"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Clean up any residual processes
pkill -f openrlhf || true
sleep 2

# Simplified training command, reduced parameters, only core functionality
if [ "$MODE" = "sft" ]; then
    echo "Using SFT mode (Supervised Fine-tuning)"
    deepspeed --module openrlhf.cli.train_sft \
        --pretrain $SFT_MODEL \
        --dataset "$DATASET_PATH" \
        --prompt_key question \
        --chosen_key answer \
        --save_path $OUTPUT_DIR \
        --max_epochs $NUM_EPOCHS \
        --micro_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --train_batch_size $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 3)) \
        --learning_rate $LEARNING_RATE \
        --max_norm $MAX_GRAD_NORM \
        --save_steps $SAVE_STEPS \
        --logging_steps $LOGGING_STEPS \
        --eval_steps $EVAL_STEPS \
        --bf16 \
        --gradient_checkpointing \
        --attn_implementation flash_attention_2 \
        --max_len $MAX_LEN \
        --seed 42 \
        2>&1 | tee training.log | grep -E "(ERROR|Error|Exception|Traceback|Failed|KeyError|AttributeError|ImportError|ModuleNotFoundError|CUDA|GPU|memory|OOM|Success|Finished|Training|Epoch|Step|SFT|exits successfully)"
elif [ "$MODE" = "grpo" ]; then
    echo "Using original GRPO mode"
    deepspeed --module openrlhf.cli.train_dpo \
        --trainer_type grpo \
        --pretrain $SFT_MODEL \
        --ref_pretrain $SFT_MODEL \
        --dataset "$DATASET_PATH" \
        --prompt_key question \
        --chosen_key answer \
        --rejected_key answer \
        --save_path $OUTPUT_DIR \
        --max_epochs $NUM_EPOCHS \
        --micro_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --train_batch_size $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 3)) \
        --learning_rate $LEARNING_RATE \
        --max_norm $MAX_GRAD_NORM \
        --save_steps $SAVE_STEPS \
        --logging_steps $LOGGING_STEPS \
        --eval_steps $EVAL_STEPS \
        --bf16 \
        --gradient_checkpointing \
        --attn_implementation flash_attention_2 \
        --max_len $MAX_LEN \
        --beta $BETA \
        --seed 42 \
        --draft_source $DRAFT_SOURCE \
        --num_drafts_per_prompt $NUM_DRAFTS_PER_PROMPT \
        --num_refinements_per_draft $NUM_COMPLETIONS \
        --max_completion_length $MAX_LEN \
        --use_sequence_level_is \
        --use_refinement_advantage \
        --clip_ratio $CLIP_RATIO \
        --kl_coef $KL_COEF \
        --scale_rewards $SCALE_REWARDS \
        --importance_sampling_level $IMPORTANCE_SAMPLING_LEVEL \
        --kl_stable \
        --loss_type $LOSS_TYPE \
        --reward_function $REWARD_MODE \
        --do_sample \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        2>&1 | tee training.log | grep -E "(ERROR|Error|Exception|Traceback|Failed|KeyError|AttributeError|ImportError|ModuleNotFoundError|CUDA|GPU|memory|OOM|Success|完成|Finished|训练|Training|Epoch|Step|RefineGRPO|exits successfully)"
else
    echo "Using RefineGRPO mode (using GRPO functionality in train_dpo)"
    
    # Build base command
    WANDB_ARGS=""
    if [ "$DISABLE_WANDB" != "true" ]; then
        WANDB_ARGS="--use_wandb $WANDB_PROJECT --wandb_project $WANDB_PROJECT --wandb_run_name $WANDB_RUN_NAME --wandb_org \"$WANDB_ENTITY\""
    fi
    
    deepspeed --module openrlhf.cli.train_dpo \
        --trainer_type refine_grpo \
        --pretrain $SFT_MODEL \
        --ref_pretrain $SFT_MODEL \
        --dataset "$DATASET_PATH" \
        --prompt_key question \
        --chosen_key answer \
        --rejected_key answer \
        --save_path $OUTPUT_DIR \
        --max_epochs $NUM_EPOCHS \
        --micro_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --train_batch_size $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 3)) \
        --learning_rate $LEARNING_RATE \
        --max_norm $MAX_GRAD_NORM \
        --save_steps $SAVE_STEPS \
        --logging_steps $LOGGING_STEPS \
        --eval_steps $EVAL_STEPS \
        --bf16 \
        --gradient_checkpointing \
        --attn_implementation flash_attention_2 \
        --max_len $MAX_LEN \
        --beta $BETA \
        --seed 42 \
        --draft_source $DRAFT_SOURCE \
        --num_drafts_per_prompt $NUM_DRAFTS_PER_PROMPT \
        --num_refinements_per_draft $NUM_REFINEMENTS_PER_DRAFT \
        --max_completion_length $MAX_LEN \
        --use_sequence_level_is \
        --use_refinement_advantage \
        --clip_ratio $CLIP_RATIO \
        --kl_coef $KL_COEF \
        --scale_rewards $SCALE_REWARDS \
        --importance_sampling_level $IMPORTANCE_SAMPLING_LEVEL \
        --kl_stable \
        --loss_type $LOSS_TYPE \
        --reward_function $REWARD_MODE \
        --do_sample \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        $WANDB_ARGS \
        2>&1 | tee training.log | grep -E "(ERROR|Error|Exception|Traceback|Failed|KeyError|AttributeError|ImportError|ModuleNotFoundError|CUDA|GPU|memory|OOM|Success|完成|Finished|训练|Training|Epoch|Step|RefineGRPO|exits successfully)"
fi

# Check if training was successful
if [ -f "training.log" ] && grep -q "Training completed\|RefineGRPO training completed\|exits successfully" training.log 2>/dev/null; then
    echo "=== Training completed ==="
    echo "Training log saved to: training.log"
    echo "Checking training results:"
    ls -la $OUTPUT_DIR/ 2>/dev/null || echo "Output directory not created"
    
    echo ""
    echo "=== GRPO Training Summary ==="
    echo "Training mode: $MODE"
    echo "Refinement advantage: ΔR = R(x,y1,y2) - R(x,y1)"
    echo "Loss calculation: Only y2 segment (refinement segment)"
    echo "Importance sampling: $IMPORTANCE_SAMPLING_LEVEL level"
    echo "Reward scaling: $SCALE_REWARDS"
    echo "Loss type: $LOSS_TYPE"
    echo ""
    echo "To adjust parameters, please modify the configuration area at the top of the script"
else
    echo "=== Training ended abnormally ==="
    echo "Please check training.log file for detailed error information"
    echo "Output directory status:"
    ls -la $OUTPUT_DIR/ 2>/dev/null || echo "Output directory not created"
fi

