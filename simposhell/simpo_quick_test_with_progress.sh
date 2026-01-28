#!/bin/bash
# SimPO Instruct Setup 快速测试版本 - 带详细进度显示
# 使用少量prompts验证完整流程

set -x

echo "=========================================="
echo "🧪 SimPO Instruct Setup 快速测试"
echo "=========================================="

# 设置参数
SFT_MODEL="/home/ubuntu/basemodels/llama3/llama3-8b-instruct"
OUTPUT_DIR="/home/ubuntu/rrhf/ultrafeedback_onpolicy_test"
REWARD_MODEL="llm-blender/PairRM"

# 使用5个seeds（对应5个响应）
SEEDS=(13 21 42 79 100)

echo "📋 测试配置："
echo "   SFT模型: $SFT_MODEL"
echo "   输出目录: $OUTPUT_DIR"
echo "   奖励模型: $REWARD_MODEL"
echo "   Seeds: ${SEEDS[@]}"
echo "   Prompts数量: 5 (测试用)"
echo ""

# 创建测试目录
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "📝 步骤1: 生成5个不同响应"
echo "=========================================="
echo "📌 说明: 为5个prompts生成5个不同响应"
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
        --max_tokens 512 \
        --seed $seed \
        --output_dir $OUTPUT_DIR \
        --max_prompts 5
    
    if [ $? -eq 0 ]; then
        echo "✅ seed=$seed 响应生成完成"
        if [ -f "$OUTPUT_DIR/output_$seed.json" ]; then
            echo "📊 生成了 $(jq length $OUTPUT_DIR/output_$seed.json) 个响应"
        fi
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
        echo "📊 后处理结果: $(jq length $OUTPUT_DIR/all_outputs.json) 个样本"
        echo "📄 后处理结果预览："
        head -n 5 $OUTPUT_DIR/all_outputs.json
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
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_reward_annotate_no_datasets.py \
    --generation_file $OUTPUT_DIR/all_outputs.json \
    --reward_model $REWARD_MODEL \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ 奖励模型标注完成"
    if [ -f "$OUTPUT_DIR/all_outputs_bin.json" ]; then
        echo "📊 标注结果: $(jq length $OUTPUT_DIR/all_outputs_bin.json) 个偏好对"
        echo "📄 标注结果预览："
        head -n 5 $OUTPUT_DIR/all_outputs_bin.json
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

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/convert_to_dpo_format.py \
    --input_file $OUTPUT_DIR/all_outputs_bin.json \
    --output_file $OUTPUT_DIR/test_dpo.jsonl

if [ $? -eq 0 ]; then
    echo "✅ DPO格式转换完成"
    if [ -f "$OUTPUT_DIR/test_dpo.jsonl" ]; then
        echo "📊 DPO格式数据: $(wc -l < $OUTPUT_DIR/test_dpo.jsonl) 个偏好对"
        echo "📄 DPO格式数据样本："
        head -n 2 $OUTPUT_DIR/test_dpo.jsonl
    fi
else
    echo "❌ DPO格式转换失败"
    exit 1
fi
echo ""

echo "=========================================="
echo "🎉 SimPO快速测试完成！"
echo "=========================================="
echo "📁 测试结果保存在: $OUTPUT_DIR"
echo "📊 最终文件列表："
ls -la $OUTPUT_DIR/
echo ""
echo "✅ 快速测试成功！可以运行完整流程了。"
