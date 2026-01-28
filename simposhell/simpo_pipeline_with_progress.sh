#!/bin/bash
# SimPO Instruct Setup 带详细进度显示的完整流程
# 严格按照论文描述实现，包含详细的进度信息

set -x

echo "=========================================="
echo "🚀 SimPO Instruct Setup 完整流程开始"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 参数配置
SFT_MODEL="/home/ubuntu/basemodels/llama3/llama3-8b-instruct"
OUTPUT_DIR="/home/ubuntu/rrhf/ultrafeedback_onpolicy"
REWARD_MODEL="llm-blender/PairRM"
DATASET_DIR="HuggingFaceH4/ultrafeedback_binarized"

# SimPO使用的5个seeds（对应5个不同的响应）
SEEDS=(13 21 42 79 100)

echo "📋 配置信息："
echo "   SFT模型: $SFT_MODEL"
echo "   输出目录: $OUTPUT_DIR"
echo "   奖励模型: $REWARD_MODEL"
echo "   数据集: $DATASET_DIR"
echo "   Seeds: ${SEEDS[@]}"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "📝 步骤1: 使用SFT模型生成5个不同响应"
echo "=========================================="
echo "📌 说明: 对每个prompt生成5个响应（使用5个不同seeds）"
echo "📌 参数: temperature=0.8, max_tokens=4096"
echo ""

# 为每个seed生成响应
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    step_num=$((i + 1))
    echo "🔄 进度: $step_num/5 - 生成seed=$seed的响应..."
    
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_decode_final.py \
        --model $SFT_MODEL \
        --temperature 0.8 \
        --top_p 0.95 \
        --max_tokens 4096 \
        --seed $seed \
        --output_dir $OUTPUT_DIR \
        --max_prompts 1000
    
    if [ $? -eq 0 ]; then
        echo "✅ seed=$seed 响应生成完成"
    else
        echo "❌ seed=$seed 响应生成失败"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "📝 步骤2: 后处理生成结果"
echo "=========================================="
echo "📌 说明: 合并5个响应文件，过滤相同响应"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_post_process.py \
    --generation_file_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ 后处理完成"
    if [ -f "$OUTPUT_DIR/all_outputs.json" ]; then
        echo "📊 后处理结果: $(wc -l < $OUTPUT_DIR/all_outputs.json) 个样本"
    fi
else
    echo "❌ 后处理失败"
    exit 1
fi
echo ""

echo "=========================================="
echo "📝 步骤3: 使用PairRM奖励模型进行偏好标注"
echo "=========================================="
echo "📌 说明: 使用PairRM对5个响应评分，选择最高分和最低分"
echo "📌 奖励模型: $REWARD_MODEL"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_reward_annotate_no_datasets.py \
    --generation_file $OUTPUT_DIR/all_outputs.json \
    --reward_model $REWARD_MODEL \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ 奖励模型标注完成"
    if [ -f "$OUTPUT_DIR/all_outputs_bin.json" ]; then
        echo "📊 标注结果: $(jq length $OUTPUT_DIR/all_outputs_bin.json) 个偏好对"
    fi
else
    echo "❌ 奖励模型标注失败"
    exit 1
fi
echo ""

echo "=========================================="
echo "📝 步骤4: 转换为DPO训练格式"
echo "=========================================="
echo "📌 说明: 转换为DPO训练所需的JSONL格式"
echo ""

echo "步骤4: 转换为SimPO和DPO格式..."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/convert_to_simpo_format.py \
    --input_file $OUTPUT_DIR/all_outputs_bin.json \
    --simpo_output $OUTPUT_DIR/ultrafeedback_onpolicy_simpo.json \
    --dpo_output $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl

if [ $? -eq 0 ]; then
    echo "✅ DPO格式转换完成"
    if [ -f "$OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl" ]; then
        echo "📊 DPO格式数据: $(wc -l < $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl) 个偏好对"
    fi
else
    echo "❌ DPO格式转换失败"
    exit 1
fi
echo ""

echo "=========================================="
echo "🎉 SimPO Instruct Setup 完成！"
echo "=========================================="
echo "📁 输出文件："
echo "   SimPO格式数据: $OUTPUT_DIR/ultrafeedback_onpolicy_simpo.json"
echo "   DPO格式数据: $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl"
echo "   HuggingFace格式: $OUTPUT_DIR/"
echo ""

# 显示结果统计
if [ -f "$OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl" ]; then
    echo "📊 最终统计："
    echo "   DPO偏好对数量: $(wc -l < $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl)"
    echo ""
    echo "📄 数据样本预览："
    head -n 2 $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl
    echo ""
fi

echo "✅ 所有步骤完成！现在可以使用生成的DPO数据进行训练。"
