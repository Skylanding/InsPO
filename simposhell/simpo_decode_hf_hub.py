import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
import random
import requests
from huggingface_hub import hf_hub_download
import pandas as pd
import time
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading

def generate_batch_responses(args_tuple):
    """Generate a batch of responses on a single GPU"""
    prompts, model_path, temperature, top_p, max_tokens, seed, gpu_id = args_tuple
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        print(f"GPU {gpu_id}: Starting to load model...")
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"GPU {gpu_id}: Model loaded, starting to generate {len(prompts)} responses...")
        
        output_data = []
        
        # Process prompts in batch
        for i, prompt in enumerate(prompts):
            try:
                # Prepare input
                messages = [{'role': 'user', 'content': prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                ).to(model.device)
                
                # Use slightly different parameters to increase diversity
                current_temperature = temperature + random.uniform(-0.1, 0.1)
                current_top_p = top_p + random.uniform(-0.05, 0.05)
                
                # Generate response
                with torch.no_grad():
                    attention_mask = torch.ones_like(input_ids)
                    
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        temperature=max(0.1, current_temperature),
                        top_p=max(0.1, min(1.0, current_top_p)),
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # Decode response
                generated_text = tokenizer.decode(
                    output_ids[0, input_ids.shape[-1]:], 
                    skip_special_tokens=True
                )
                
                # Get formatted prompt
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
                
                # Show progress every 100 prompts
                if (i + 1) % 100 == 0:
                    print(f"GPU {gpu_id}: Completed {i+1}/{len(prompts)} responses")
                
            except Exception as e:
                print(f"GPU {gpu_id}: Error generating response {i+1}: {e}")
                output_data.append({
                    'prompt': prompt,
                    "format_prompt": f"User: {prompt}\nAssistant:",
                    'generated_text': f"Error generating response for: {prompt[:50]}...",
                })
        
        print(f"GPU {gpu_id}: Completed all {len(prompts)} responses")
        return output_data
        
    except Exception as e:
        print(f"GPU {gpu_id}: Batch processing failed: {e}")
        return [{
            'prompt': prompt,
            "format_prompt": f"User: {prompt}\nAssistant:",
            'generated_text': f"Error generating response for: {prompt[:50]}...",
        } for prompt in prompts]
    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_responses_parallel(prompts, model_path, temperature, top_p, max_tokens, seed, gpu_ids, batch_size):
    """Generate responses in parallel using multiple GPUs"""
    if len(gpu_ids) == 1:
        # Single GPU mode, use original serial method
        return generate_responses_serial(prompts, model_path, temperature, top_p, max_tokens, seed, gpu_ids[0])
    
    print(f"Using multi-GPU parallel generation: {len(gpu_ids)} GPUs")
    print(f"GPU list: {gpu_ids}")
    
    # Distribute prompts in batches to different GPUs
    prompts_per_gpu = len(prompts) // len(gpu_ids)
    remainder = len(prompts) % len(gpu_ids)
    
    args_list = []
    start_idx = 0
    
    for i, gpu_id in enumerate(gpu_ids):
        # Calculate number of prompts for this GPU
        current_batch_size = prompts_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        
        if current_batch_size > 0:
            gpu_prompts = prompts[start_idx:end_idx]
            args_tuple = (gpu_prompts, model_path, temperature, top_p, max_tokens, seed, gpu_id)
            args_list.append(args_tuple)
            print(f"GPU {gpu_id}: Assigned {len(gpu_prompts)} prompts")
        
        start_idx = end_idx
    
    output_data = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        # Submit all tasks
        future_to_gpu = {executor.submit(generate_batch_responses, args): args[5] for args in args_list}
        
        # Collect results
        for future in as_completed(future_to_gpu):
            gpu_id = future_to_gpu[future]
            try:
                result = future.result()
                output_data.extend(result)
                print(f"GPU {gpu_id}: Returned {len(result)} responses")
            except Exception as e:
                print(f"GPU {gpu_id}: Processing failed: {e}")
    
    return output_data

def generate_responses_serial(prompts, model_path, temperature, top_p, max_tokens, seed, gpu_id):
    """Generate responses serially on single GPU (original method)"""
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    output_data = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Serial generation", unit="prompt")):
        try:
            # Prepare input
            messages = [{'role': 'user', 'content': prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                add_generation_prompt=True
            ).to(model.device)
            
            # Use slightly different parameters to increase diversity
            current_temperature = temperature + random.uniform(-0.1, 0.1)
            current_top_p = top_p + random.uniform(-0.05, 0.05)
            
            # Generate response
            with torch.no_grad():
                attention_mask = torch.ones_like(input_ids)
                
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=max(0.1, current_temperature),
                    top_p=max(0.1, min(1.0, current_top_p)),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            generated_text = tokenizer.decode(
                output_ids[0, input_ids.shape[-1]:], 
                skip_special_tokens=True
            )
            
            # Get formatted prompt
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
            
        except Exception as e:
            print(f"Error generating response {i+1}: {e}")
            output_data.append({
                'prompt': prompt,
                "format_prompt": f"User: {prompt}\nAssistant:",
                'generated_text': f"Error generating response for: {prompt[:50]}...",
            })
    
    return output_data

def load_ultrafeedback_prompts(max_prompts=None):
    """Download UltraFeedback dataset directly from HuggingFace Hub"""
    try:
        print("Downloading UltraFeedback dataset from HuggingFace Hub...")
        
        # Directly download parquet file
        parquet_file = hf_hub_download(
            repo_id="HuggingFaceH4/ultrafeedback_binarized",
            filename="data/train_prefs-00000-of-00001.parquet",
            repo_type="dataset"
        )
        
        # Use pandas to read parquet file
        df = pd.read_parquet(parquet_file)
        prompts = sorted(list(set(df['prompt'].tolist())))
        
        print(f"Successfully loaded UltraFeedback dataset: {len(prompts)} unique prompts")
        
        if max_prompts and len(prompts) > max_prompts:
            prompts = prompts[:max_prompts]
            print(f"Limited to first {max_prompts} prompts")
        
        return prompts
        
    except Exception as e:
        print(f"Failed to load UltraFeedback dataset: {e}")
        print("Falling back to test prompts...")
        
        # Fall back to test prompts
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
        
        print(f"Using test prompts: {len(test_prompts)} prompts")
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
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for generation (for multi-GPU)')
parser.add_argument('--gpu_ids', type=str, default=None,
                    help='Comma-separated GPU IDs to use (e.g., "0,1,2,3")')
args = parser.parse_args()

print(args)

# Set random seed
torch.manual_seed(args.seed)
random.seed(args.seed)

# Set GPU
if args.gpu_ids:
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    print(f"Using GPUs: {gpu_ids}")
else:
    gpu_ids = [0]  # Default to GPU 0
    print("Using default GPU: 0")

# Load prompts
prompts = load_ultrafeedback_prompts(args.max_prompts)

print(f"Starting to generate {len(prompts)} responses...")
print(f"Number of GPUs: {len(gpu_ids)}")
print(f"Batch size: {args.batch_size}")
print(f"Estimated completion time: {len(prompts) * 2 / len(gpu_ids) / 60:.1f} minutes (estimate)")

# Record start time
start_time = time.time()

# Use parallel generation
output_data = generate_responses_parallel(
    prompts, 
    args.model, 
    args.temperature, 
    args.top_p, 
    args.max_tokens, 
    args.seed, 
    gpu_ids, 
    args.batch_size
)

# Calculate total time
total_time = time.time() - start_time
print(f"\n🎉 Generation completed!")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Average speed: {total_time/len(prompts):.2f} seconds/prompt")
print(f"   Generated count: {len(output_data)} responses")
print(f"   Parallel efficiency: {len(gpu_ids)}x GPU acceleration")

# Save output
output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Output saved to: {os.path.join(args.output_dir, output_file)}")
print(f"Generated {len(output_data)} responses")
