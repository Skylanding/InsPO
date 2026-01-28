#!/usr/bin/env python3
"""
Convert SimPO preference data to DPO training format
"""

import json
import argparse
import os

def convert_to_dpo_format(input_file, output_file):
    """Convert SimPO format to DPO training format"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    dpo_data = []
    
    for item in data:
        # Extract prompt from chosen
        prompt = item['chosen'][0]['content']
        
        # Extract chosen and rejected responses
        chosen_response = item['chosen'][1]['content']
        rejected_response = item['rejected'][1]['content']
        
        dpo_item = {
            'prompt': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response
        }
        
        dpo_data.append(dpo_item)
    
    # Save as JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed: {len(dpo_data)} preference pairs")
    print(f"DPO format data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert SimPO data to DPO format')
    parser.add_argument('--input_file', type=str, 
                       default='/home/ubuntu/rrhf/ultrafeedback_onpolicy/all_outputs_bin.json',
                       help='Input SimPO format file')
    parser.add_argument('--output_file', type=str,
                       default='/home/ubuntu/rrhf/ultrafeedback_onpolicy_dpo.jsonl',
                       help='Output DPO format file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Input file does not exist: {args.input_file}")
        return
    
    convert_to_dpo_format(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
