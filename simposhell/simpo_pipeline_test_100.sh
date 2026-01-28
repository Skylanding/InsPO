#!/bin/bash
# SimPO Instruct Setup - 快速测试版本 (使用100个prompts)
# 验证整个流程是否正常工作

set -x

echo "=========================================="
echo "🚀 SimPO Instruct Setup - 快速测试版本"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 参数配置
SFT_MODEL="/home/ubuntu/basemodels/llama3/llama3-8b-instruct"
OUTPUT_DIR="/home/ubuntu/rrhf/ultrafeedback_test_100"
REWARD_MODEL="llm-blender/PairRM"

# SimPO使用的5个seeds（对应5个不同的响应）
SEEDS=(13 21 42 79 100)

echo "📋 配置信息："
echo "   SFT模型: $SFT_MODEL"
echo "   输出目录: $OUTPUT_DIR"
echo "   奖励模型: $REWARD_MODEL"
echo "   测试规模: 100个prompts"
echo "   Seeds: ${SEEDS[@]}"
echo "   预计生成: ~500 个偏好对"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "📝 步骤1: 使用SFT模型生成5个不同响应"
echo "=========================================="
echo "📌 说明: 对100个prompts生成5个响应"
echo "📌 预计时间: 每个seed约10-15分钟"
echo ""

# 为每个seed生成响应
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    step_num=$((i + 1))
    echo "🔄 进度: $step_num/5 - 生成seed=$seed的响应..."
    
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_decode_hf_hub.py \
        --model $SFT_MODEL \
        --temperature 0.8 \
        --top_p 0.95 \
        --max_tokens 4096 \
        --seed $seed \
        --output_dir $OUTPUT_DIR \
        --max_prompts 100 \
        --gpu_ids "0" \
        --batch_size 1
    
    if [ $? -eq 0 ]; then
        echo "✅ seed=$seed 响应生成完成"
        if [ -f "$OUTPUT_DIR/output_$seed.json" ]; then
            echo "📊 生成响应数量: $(jq length $OUTPUT_DIR/output_$seed.json)"
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
echo "📝 步骤4: 转换为SimPO和DPO格式"
echo "=========================================="
echo "📌 说明: 转换为SimPO和DPO训练所需的格式"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/convert_to_simpo_format.py \
    --input_file $OUTPUT_DIR/all_outputs_bin.json \
    --simpo_output $OUTPUT_DIR/ultrafeedback_test_simpo.json \
    --dpo_output $OUTPUT_DIR/ultrafeedback_test_dpo.jsonl

if [ $? -eq 0 ]; then
    echo "✅ 格式转换完成"
    if [ -f "$OUTPUT_DIR/ultrafeedback_test_dpo.jsonl" ]; then
        echo "📊 DPO格式数据: $(wc -l < $OUTPUT_DIR/ultrafeedback_test_dpo.jsonl) 个偏好对"
    fi
else
    echo "❌ 格式转换失败"
    exit 1
fi
echo ""

echo "=========================================="
echo "🎉 SimPO 快速测试完成！"
echo "=========================================="
echo "📁 输出文件："
echo "   SimPO格式数据: $OUTPUT_DIR/ultrafeedback_test_simpo.json"
echo "   DPO格式数据: $OUTPUT_DIR/ultrafeedback_test_dpo.jsonl"
echo ""

# 显示结果统计
if [ -f "$OUTPUT_DIR/ultrafeedback_test_dpo.jsonl" ]; then
    echo "📊 最终统计："
    echo "   DPO偏好对数量: $(wc -l < $OUTPUT_DIR/ultrafeedback_test_dpo.jsonl)"
    echo ""
    echo "📄 数据样本预览："
    head -n 2 $OUTPUT_DIR/ultrafeedback_test_dpo.jsonl
    echo ""
fi

echo "✅ 快速测试完成！如果结果正常，可以运行完整版本。"
