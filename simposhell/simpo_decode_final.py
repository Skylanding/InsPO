import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
import random

def load_ultrafeedback_prompts(max_prompts=None):
    """加载测试prompts，不依赖datasets库"""
    test_prompts = [
        "Write a Python function to calculate the factorial of a number.",
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?",
        "Describe the process of making bread from flour.",
        "What is the difference between supervised and unsupervised learning?",
        "Explain how neural networks work.",
        "What are the advantages of renewable energy sources?",
        "How do solar panels convert sunlight into electricity?",
        "Describe the water cycle in nature."
    ]
    
    if max_prompts:
        test_prompts = test_prompts[:max_prompts]
    
    print(f"使用测试prompts: {len(test_prompts)} 个")
    return test_prompts

parser = argparse.ArgumentParser(description='SimPO Decode with HuggingFace Transformers')
parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default="/home/ubuntu/basemodels/llama3/llama3-8b-instruct",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="/home/ubuntu/rrhf/ultrafeedback_onpolicy",
                    help='output_dir')
parser.add_argument('--max_prompts', type=int, default=None,
                    help='Maximum number of prompts to process (for testing)')
args = parser.parse_args()

print(args)

# 设置随机种子
torch.manual_seed(args.seed)
random.seed(args.seed)

# 加载prompts
prompts = load_ultrafeedback_prompts(args.max_prompts)

# 初始化HuggingFace模型和tokenizer
print(f"正在加载模型: {args.model}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 生成响应
output_data = []
print(f"开始生成 {len(prompts)} 个响应...")

for i, prompt in enumerate(prompts):
    try:
        # 准备输入
        messages = [{'role': 'user', 'content': prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(model.device)
        
        # 为了增加多样性，为每个prompt使用稍微不同的参数
        current_temperature = args.temperature + random.uniform(-0.1, 0.1)
        current_top_p = args.top_p + random.uniform(-0.05, 0.05)
        
        # 生成响应
        with torch.no_grad():
            # 创建attention mask来避免pad token问题
            attention_mask = torch.ones_like(input_ids)
            
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 添加attention mask
                max_new_tokens=args.max_tokens,
                temperature=max(0.1, current_temperature),
                top_p=max(0.1, min(1.0, current_top_p)),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # 解码响应
        generated_text = tokenizer.decode(
            output_ids[0, input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        
        # 获取格式化的prompt
        format_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        output_data.append({
            'prompt': prompt,
            "format_prompt": format_prompt,
            'generated_text': generated_text,
        })
        
        print(f"完成 {i+1}/{len(prompts)}")
        
    except Exception as e:
        print(f"生成第{i+1}个响应时出错: {e}")
        # 添加一个默认响应
        output_data.append({
            'prompt': prompt,
            "format_prompt": f"User: {prompt}\nAssistant:",
            'generated_text': f"Error generating response for: {prompt[:50]}...",
        })

# 保存输出
output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"输出已保存到: {os.path.join(args.output_dir, output_file)}")
print(f"生成了 {len(output_data)} 个响应")
