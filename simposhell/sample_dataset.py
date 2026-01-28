#!/usr/bin/env python3
"""
Dataset sampling script - Sample 1/10 of data from full dataset for quick training tests
"""

import json
import random
import argparse
from pathlib import Path


def sample_dataset(input_file, output_file, sample_ratio=0.1, seed=42):
    """
    Sample specified ratio of data from input dataset
    
    Args:
        input_file: Input dataset file path
        output_file: Output dataset file path
        sample_ratio: Sampling ratio (0.1 means 1/10 of data)
        seed: Random seed
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all data
    print(f"Reading dataset from {input_file}...")
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line.strip()))
    
    total_samples = len(all_data)
    sample_size = int(total_samples * sample_ratio)
    
    print(f"Total samples: {total_samples}")
    print(f"Sample ratio: {sample_ratio}")
    print(f"Sample size: {sample_size}")
    
    # Random sampling
    sampled_data = random.sample(all_data, sample_size)
    
    # Save sampled data
    print(f"Writing sampled dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in sampled_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Sampling completed! Saved {len(sampled_data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Sample dataset for quick training")
    parser.add_argument("--input_file", type=str, 
                       default="/home/ubuntu/Open/prefer/data/ultrafeedback_dpo.jsonl",
                       help="Input dataset file path")
    parser.add_argument("--output_file", type=str,
                       default="/home/ubuntu/Open/prefer/data/ultrafeedback_dpo_sample.jsonl",
                       help="Output sampled dataset file path")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                       help="Sample ratio (0.1 means 1/10 of data)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist!")
        return
    
    # Create output directory
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Execute sampling
    sample_dataset(args.input_file, args.output_file, args.sample_ratio, args.seed)


if __name__ == "__main__":
    main()
