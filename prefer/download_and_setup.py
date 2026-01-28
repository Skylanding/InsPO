#!/usr/bin/env python3
"""
SFT模型和数据集下载配置脚本
自动下载RTO的SFT模型和UltraFeedback数据集，并转换为OpenRLHF格式
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_sft_model(model_name="OpenRLHF/Llama-3-8b-sft-mixture", save_dir="./models"):
    """
    下载SFT模型
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
    """
    print(f"正在下载SFT模型: {model_name}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    # 使用Princeton前缀避免覆盖原有模型
    model_path = os.path.join(save_dir, f"Princeton_{model_name.split('/')[-1]}")
    
    if os.path.exists(model_path):
        print(f"✅ 模型已存在: {model_path}")
        return model_path
    
    try:
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        
        # 下载模型（仅下载配置，不下载权重以节省时间）
        print("下载模型配置...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.save_pretrained(model_path)
        
        print(f"✅ SFT模型下载完成: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ SFT模型下载失败: {e}")
        return None


def download_ultrafeedback_dataset(dataset_name="princeton-nlp/llama3-ultrafeedback-armorm", save_dir="./data"):
    """
    下载Llama3-UltraFeedback-ArmoRM数据集
    
    Args:
        dataset_name: 数据集名称
        save_dir: 保存目录
    """
    print(f"正在下载Llama3-UltraFeedback-ArmoRM数据集: {dataset_name}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 下载数据集 - 使用train split
        dataset = load_dataset(dataset_name, split="train")
        print(f"✅ 数据集下载完成，包含 {len(dataset)} 个样本")
        
        # 保存到本地 - 使用Princeton前缀避免覆盖原有数据
        dataset_path = os.path.join(save_dir, "Princeton_llama3_ultrafeedback_armorm")
        dataset.save_to_disk(dataset_path)
        print(f"✅ 数据集保存到: {dataset_path}")
        
        return dataset_path
        
    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        return None


def convert_to_openrlhf_format(dataset_path, output_file, max_samples=None):
    """
    转换为OpenRLHF DPO格式
    
    Args:
        dataset_path: 数据集路径
        output_file: 输出文件
        max_samples: 最大样本数
    """
    print("转换为OpenRLHF DPO格式...")
    
    try:
        # 直接从磁盘加载数据集
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        
        dpo_data = []
        sample_count = 0
        
        for sample in dataset:
            if max_samples and sample_count >= max_samples:
                break
                
            # Llama3-UltraFeedback-ArmoRM数据集的格式：
            # - prompt: 用户输入
            # - chosen: 更好的回答（对话格式）
            # - rejected: 较差的回答（对话格式）
            
            if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
                # 提取prompt（用户消息）
                prompt = sample['prompt']
                
                # 提取chosen和rejected的assistant回复
                chosen_response = ""
                rejected_response = ""
                
                # 处理chosen回复 - 新数据集使用对话格式
                if isinstance(sample['chosen'], list):
                    for msg in sample['chosen']:
                        if msg.get('role') == 'assistant':
                            chosen_response = msg.get('content', '')
                            break
                else:
                    chosen_response = str(sample['chosen'])
                
                # 处理rejected回复 - 新数据集使用对话格式
                if isinstance(sample['rejected'], list):
                    for msg in sample['rejected']:
                        if msg.get('role') == 'assistant':
                            rejected_response = msg.get('content', '')
                            break
                else:
                    rejected_response = str(sample['rejected'])
                
                # 验证数据有效性
                if chosen_response and rejected_response and chosen_response != rejected_response:
                    # 创建DPO格式样本
                    dpo_sample = {
                        'prompt': prompt,
                        'chosen': chosen_response,
                        'rejected': rejected_response
                    }
                    dpo_data.append(dpo_sample)
                    sample_count += 1
                else:
                    print(f"跳过无效样本 {sample_count}: chosen和rejected相同或为空")
                    continue
                
            else:
                print(f"跳过格式不匹配的样本: {sample.keys()}")
                continue
        
        # 保存为JSONL格式
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dpo_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✅ 转换完成，共 {len(dpo_data)} 个样本保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 格式转换失败: {e}")
        return False


def create_config_file(model_path, dataset_path, output_dir="./config"):
    """
    创建OpenRLHF配置文件
    
    Args:
        model_path: 模型路径
        dataset_path: 数据集路径
        output_dir: 输出目录
    """
    print("创建OpenRLHF配置文件...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "model": {
            "pretrain": model_path,
            "ref_pretrain": model_path
        },
        "dataset": {
            "path": dataset_path,
            "split": "train",
            "max_samples": 100000
        },
        "training": {
            "train_batch_size": 128,
            "micro_train_batch_size": 4,
            "max_epochs": 3,
            "max_len": 1024,
            "learning_rate": 5e-7,
            "beta": 0.1
        },
        "optimization": {
            "zero_stage": 2,
            "adam_offload": True,
            "flash_attn": True,
            "gradient_checkpointing": True
        },
        "logging": {
            "save_path": "./checkpoint/Princeton_llama3-8b-dpo-ultrafeedback-armorm",
            "save_steps": 500,
            "logging_steps": 10,
            "eval_steps": 200,
            "use_wandb": True,
            "wandb_project": "princeton_dpo",
            "wandb_run_name": "Princeton_llama3-8b-dpo-ultrafeedback-armorm"
        },
        "gpu": {
            "cuda_visible_devices": "0,1,2,3"
        }
    }
    
    config_file = os.path.join(output_dir, "Princeton_openrlhf_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置文件保存到: {config_file}")
    return config_file


def main():
    parser = argparse.ArgumentParser(description="下载和配置SFT模型和数据集")
    parser.add_argument("--model_name", type=str, default="OpenRLHF/Llama-3-8b-sft-mixture",
                       help="SFT模型名称")
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/llama3-ultrafeedback-armorm",
                       help="数据集名称")
    parser.add_argument("--model_dir", type=str, default="./models",
                       help="模型保存目录")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="数据保存目录")
    parser.add_argument("--config_dir", type=str, default="./config",
                       help="配置保存目录")
    parser.add_argument("--max_samples", type=int, default=100000,
                       help="最大样本数量")
    parser.add_argument("--skip_model", action="store_true",
                       help="跳过模型下载")
    parser.add_argument("--skip_dataset", action="store_true",
                       help="跳过数据集下载")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SFT模型和数据集下载配置脚本")
    print("=" * 60)
    
    # 下载SFT模型
    model_path = None
    if not args.skip_model:
        model_path = download_sft_model(args.model_name, args.model_dir)
        if not model_path:
            print("❌ SFT模型下载失败，退出")
            return
    
    # 下载数据集
    dataset_path = None
    if not args.skip_dataset:
        dataset_path = download_ultrafeedback_dataset(args.dataset_name, args.data_dir)
        if not dataset_path:
            print("❌ 数据集下载失败，退出")
            return
    
    # 转换数据集格式
    if dataset_path:
        output_file = os.path.join(args.data_dir, "Princeton_llama3_ultrafeedback_armorm_dpo.jsonl")
        if not convert_to_openrlhf_format(dataset_path, output_file, args.max_samples):
            print("❌ 数据集格式转换失败，退出")
            return
    
    # 创建配置文件
    if model_path and dataset_path:
        config_file = create_config_file(model_path, output_file, args.config_dir)
        print(f"✅ 配置完成！配置文件: {config_file}")
    
    print("=" * 60)
    print("🎉 所有配置完成！")
    print("现在可以运行DPO训练:")
    print("bash train_dpo_rto_sft_improved.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
