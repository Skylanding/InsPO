#!/bin/bash
# DPO model evaluation script - AlpacaEval, Arena-Hard, MT-Bench
# Usage: bash evaluate_dpo_model.sh [model_path]

set -x

# Configuration parameters
MODEL_PATH=${1:-"./checkpoint/llama3-8b-dpo-rrhf"}  # Default DPO model path
OUTPUT_DIR="./evaluation_results"
EVAL_DATASETS_DIR="/home/ubuntu/datasets/eval_dataset"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting DPO model evaluation: $MODEL_PATH"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please ensure the model has been trained or provide the correct model path"
    exit 1
fi

# Check if evaluation datasets exist
echo "=== Checking evaluation datasets ==="

if [ ! -f "$EVAL_DATASETS_DIR/alpacaeval_data.json" ]; then
    echo "Error: AlpacaEval dataset does not exist: $EVAL_DATASETS_DIR/alpacaeval_data.json"
    exit 1
fi

if [ ! -f "$EVAL_DATASETS_DIR/arena_hard_data.json" ]; then
    echo "Error: Arena-Hard dataset does not exist: $EVAL_DATASETS_DIR/arena_hard_data.json"
    exit 1
fi

if [ ! -f "$EVAL_DATASETS_DIR/mt_bench_data.json" ]; then
    echo "Error: MT-Bench dataset does not exist: $EVAL_DATASETS_DIR/mt_bench_data.json"
    exit 1
fi

echo "All evaluation datasets checked!"

# 2. Generate model outputs
echo "=== Generating model outputs ==="

# AlpacaEval output
echo "Generating AlpacaEval output..."
python -m openrlhf.cli.batch_inference \
    --pretrain "$MODEL_PATH" \
    --dataset "$EVAL_DATASETS_DIR/alpacaeval_data.json" \
    --input_key "instruction" \
    --output_path "$OUTPUT_DIR/alpacaeval_outputs.json" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --micro_batch_size 8 \
    --bf16 \
    --flash_attn

# Arena-Hard输出
echo "生成Arena-Hard输出..."
python -m openrlhf.cli.batch_inference \
    --pretrain "$MODEL_PATH" \
    --dataset "$EVAL_DATASETS_DIR/arena_hard_data.json" \
    --input_key "instruction" \
    --output_path "$OUTPUT_DIR/arena_hard_outputs.json" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --micro_batch_size 8 \
    --bf16 \
    --flash_attn

# MT-Bench输出
echo "生成MT-Bench输出..."
python -m openrlhf.cli.batch_inference \
    --pretrain "$MODEL_PATH" \
    --dataset "$EVAL_DATASETS_DIR/mt_bench_data.json" \
    --input_key "instruction" \
    --output_path "$OUTPUT_DIR/mt_bench_outputs.json" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --micro_batch_size 8 \
    --bf16 \
    --flash_attn

echo "模型输出生成完成！"

# 3. 运行评估
echo "=== 运行评估 ==="

# 检查是否设置了OpenAI API密钥
if [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: 未设置OPENAI_API_KEY环境变量"
    echo "请设置: export OPENAI_API_KEY=your_api_key"
    echo "或者修改脚本中的API密钥"
    echo "跳过需要API的评估..."
else
    # AlpacaEval评估
    echo "运行AlpacaEval评估..."
    alpaca_eval \
        --model_outputs "$OUTPUT_DIR/alpacaeval_outputs.json" \
        --annotators_config "weighted_alpaca_eval_gpt4_turbo" \
        --reference_outputs "$EVAL_DATASETS_DIR/alpacaeval_data.json" \
        --output_path "$OUTPUT_DIR/alpacaeval_results"
    
    # Arena-Hard评估
    echo "运行Arena-Hard评估..."
    python -m arena_hard.evaluate \
        --model_outputs "$OUTPUT_DIR/arena_hard_outputs.json" \
        --reference_outputs "$EVAL_DATASETS_DIR/arena_hard_data.json" \
        --output_path "$OUTPUT_DIR/arena_hard_results"
    
    # MT-Bench评估
    echo "运行MT-Bench评估..."
    python -m mt_bench.evaluate \
        --model_outputs "$OUTPUT_DIR/mt_bench_outputs.json" \
        --reference_outputs "$EVAL_DATASETS_DIR/mt_bench_data.json" \
        --output_path "$OUTPUT_DIR/mt_bench_results"
fi

echo ""
echo "=== 评估完成 ==="
echo "所有结果保存在: $OUTPUT_DIR"
echo ""
echo "结果文件："
echo "- AlpacaEval: $OUTPUT_DIR/alpacaeval_results"
echo "- Arena-Hard: $OUTPUT_DIR/arena_hard_results"
echo "- MT-Bench: $OUTPUT_DIR/mt_bench_results"
echo ""
echo "模型输出文件："
echo "- AlpacaEval: $OUTPUT_DIR/alpacaeval_outputs.json"
echo "- Arena-Hard: $OUTPUT_DIR/arena_hard_outputs.json"
echo "- MT-Bench: $OUTPUT_DIR/mt_bench_outputs.json"
