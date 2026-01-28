import json
import argparse
import os

def convert_to_simpo_format(input_file, output_file):
    """Convert to SimPO expected format"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    simpo_data = []
    for item in data:
        # Build prompt messages
        prompt_messages = [
            {"role": "user", "content": item['chosen'][0]['content']}
        ]
        
        # chosen and rejected are already in correct OpenAI format
        chosen_messages = item['chosen']
        rejected_messages = item['rejected']
        
        simpo_item = {
            'prompt': prompt_messages,
            'chosen': chosen_messages,
            'rejected': rejected_messages
        }
        simpo_data.append(simpo_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simpo_data, f, indent=4, ensure_ascii=False)
    
    print(f"Conversion completed: {len(simpo_data)} preference pairs")
    print(f"SimPO format data saved to: {output_file}")

def convert_to_dpo_format(input_file, output_file):
    """Convert to DPO training format (string format)"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    dpo_data = []
    for item in data:
        prompt = item['chosen'][0]['content']
        chosen_response = item['chosen'][1]['content']
        rejected_response = item['rejected'][1]['content']
        
        dpo_item = {
            'prompt': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response
        }
        dpo_data.append(dpo_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed: {len(dpo_data)} preference pairs")
    print(f"DPO format data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert SimPO data to different formats')
    parser.add_argument('--input_file', type=str, 
                       default='/home/ubuntu/rrhf/ultrafeedback_onpolicy/all_outputs_bin.json',
                       help='Input SimPO format file')
    parser.add_argument('--simpo_output', type=str,
                       default='/home/ubuntu/rrhf/ultrafeedback_onpolicy_simpo.json',
                       help='Output SimPO format file')
    parser.add_argument('--dpo_output', type=str,
                       default='/home/ubuntu/rrhf/ultrafeedback_onpolicy_dpo.jsonl',
                       help='Output DPO format file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Input file does not exist: {args.input_file}")
        return
    
    # Convert to SimPO format
    convert_to_simpo_format(args.input_file, args.simpo_output)
    
    # Convert to DPO format
    convert_to_dpo_format(args.input_file, args.dpo_output)

if __name__ == "__main__":
    main()
