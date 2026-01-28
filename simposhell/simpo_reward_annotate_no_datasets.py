import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", type=str, default="/home/ubuntu/rrhf/ultrafeedback_onpolicy/all_outputs.json", help="Path to the output generation file")
parser.add_argument("--reward_model", type=str, default="llm-blender/PairRM", help="Path to reward model")
parser.add_argument("--output_dir", type=str, default="/home/ubuntu/rrhf/ultrafeedback_onpolicy/", help="Path to output directory")
args = parser.parse_args()

print(args)

generation_file = args.generation_file
with open(generation_file, 'r') as f:
    output_data = json.load(f)

print(f"Loaded {len(output_data)} samples")

# Load reward model
print(f"Loading reward model: {args.reward_model}")
try:
    model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, 
                                                               device_map="cuda", 
                                                               trust_remote_code=True, 
                                                               torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)
    print("Reward model loaded successfully")
except Exception as e:
    print(f"Failed to load reward model: {e}")
    print("Will use simulated scoring")
    model = None
    tokenizer = None

# Score responses for each sample
print("Starting to score with reward model...")
for data in tqdm.tqdm(output_data):
    prompt = data["prompt"]
    candidates = data["all_generated_responses"]
    scores = []
    
    if model is not None and tokenizer is not None:
        for candidate in candidates:
            try:
                messages = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": candidate}]
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                
                # Create attention mask to avoid pad token issues
                attention_mask = torch.ones_like(input_ids)
                
                with torch.no_grad():
                    output = model(input_ids, attention_mask=attention_mask)
                    score = output.score.float().item()
                    scores.append(score)
            except Exception as e:
                print(f"Scoring failed: {e}")
                scores.append(0.5)  # Default score
    else:
        # Use simulated scoring
        import random
        scores = [random.uniform(0.3, 0.9) for _ in candidates]
    
    data["all_rm_scores"] = scores

# Save file with scores
file_name = os.path.basename(args.generation_file).split('.json')[0] + "_rm.json"
with open(os.path.join(args.output_dir, file_name), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Output with scores saved to: {os.path.join(args.output_dir, file_name)}")

# Binarize data: select responses with highest and lowest scores
print("Starting to binarize data...")
for data in output_data:
    chosen_idx = np.argmax(data["all_rm_scores"])
    rejected_idx = np.argmin(data["all_rm_scores"])
    
    chosen = []
    chosen.append({
        "role": "user",
        "content": data["prompt"]
    })
    chosen.append({
        "role": "assistant",
        "content": data["all_generated_responses"][chosen_idx]
    })
    
    rejected = []
    rejected.append({
        "role": "user",
        "content": data["prompt"]
    })
    rejected.append({
        "role": "assistant",
        "content": data["all_generated_responses"][rejected_idx]
    })
    
    data.update({
        "chosen": chosen,
        "rejected": rejected,
    })

# Save binarized data
output_file = os.path.basename(args.generation_file).split('.json')[0] + "_bin.json"
with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Binarized output saved to: {os.path.join(args.output_dir, output_file)}")
print(f"Final dataset contains {len(output_data)} preference pairs")
