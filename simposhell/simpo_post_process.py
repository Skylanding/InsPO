import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file_dir", type=str, help="Directory containing the generation files", default="/home/ubuntu/rrhf/ultrafeedback_onpolicy")
args = parser.parse_args()

print(args)

all_data = []
for file_name in os.listdir(args.generation_file_dir):
    if file_name.startswith("output") and file_name.endswith(".json"):
        generation_file = os.path.join(args.generation_file_dir, file_name)
        print(f"Loading {generation_file}")
        with open(generation_file, 'r') as f:
            output_data = json.load(f)
            all_data.append(output_data)

if not all_data:
    print("No generation files found!")
    exit(1)

num_samples = len(all_data[0])
all_res = []
num_identical = 0
for i in range(num_samples):
    prompt = all_data[0][i]["prompt"]
    gen_text = []
    for data in all_data:
        gen_text.append(data[i]["generated_text"])

    if len(set(gen_text)) == 1:
        # filter out samples where all generated responses are identical
        num_identical += 1
        continue

    all_res.append(
        {
            "prompt": prompt,
            "all_generated_responses": gen_text,
        }
    )

print(f"Filtered out {num_identical} samples with identical generated responses")

with open(os.path.join(args.generation_file_dir, 'all_outputs.json'), 'w') as f:
    json.dump(all_res, f, indent=4)

print(f"Processed outputs saved to {os.path.join(args.generation_file_dir, 'all_outputs.json')}")