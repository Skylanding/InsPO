#!/usr/bin/env python3
"""
Llama3-UltraFeedback-ArmoRM数据集预处理脚本
将Llama3-UltraFeedback-ArmoRM数据集转换为OpenRLHF DPO训练所需的格式
"""

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm


def convert_llama3_ultrafeedback_armorm_to_dpo_format(input_dataset, output_file, max_samples=None):
    """
    将Llama3-UltraFeedback-ArmoRM数据集转换为DPO格式
    
    Args:
        input_dataset: Llama3-UltraFeedback-ArmoRM数据集
        output_file: 输出文件路径
        max_samples: 最大样本数量
    """
    dpo_data = []
    
    # 处理数据集
    for i, sample in enumerate(tqdm(input_dataset, desc="Converting Llama3-UltraFeedback-ArmoRM")):
        if max_samples and i >= max_samples:
            break
            
        # Llama3-UltraFeedback-ArmoRM数据集的格式：
        # - prompt: 用户输入
        # - chosen: 更好的回答（对话格式）
        # - rejected: 较差的回答（对话格式）
        # - all_generated_responses: 所有生成的回答
        # - all_rm_scores: 所有回答的RM分数
        
        # 检查数据格式
        if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
            # 提取prompt
            prompt = sample['prompt']
            
            # 提取chosen和rejected的assistant回复
            chosen_response = ""
            rejected_response = ""
            
            # 处理chosen回复 - 对话格式
            if isinstance(sample['chosen'], list):
                for msg in sample['chosen']:
                    if msg.get('role') == 'assistant':
                        chosen_response = msg.get('content', '')
                        break
            else:
                chosen_response = str(sample['chosen'])
            
            # 处理rejected回复 - 对话格式
            if isinstance(sample['rejected'], list):
                for msg in sample['rejected']:
                    if msg.get('role') == 'assistant':
                        rejected_response = msg.get('content', '')
                        break
            else:
                rejected_response = str(sample['rejected'])
            
            # 验证数据有效性
            if chosen_response and rejected_response and chosen_response != rejected_response:
                dpo_sample = {
                    'prompt': prompt,
                    'chosen': chosen_response,
                    'rejected': rejected_response
                }
                dpo_data.append(dpo_sample)
            else:
                print(f"Warning: Skipping sample {i} due to invalid chosen/rejected responses")
                continue
        else:
            print(f"Warning: Skipping sample {i} due to unexpected format: {sample.keys()}")
            continue
    
    # 保存为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dpo_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(dpo_data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert Llama3-UltraFeedback-ArmoRM dataset to DPO format")
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/llama3-ultrafeedback-armorm",
                       help="Llama3-UltraFeedback-ArmoRM dataset name")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--output_file", type=str, default="./data/Princeton_llama3_ultrafeedback_armorm_dpo.jsonl",
                       help="Output file path")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to convert")
    
    args = parser.parse_args()
    
    # 加载数据集
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    # 转换格式
    convert_llama3_ultrafeedback_armorm_to_dpo_format(dataset, args.output_file, args.max_samples)
    
    print("Conversion completed!")


if __name__ == "__main__":
    main()
