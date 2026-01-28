#!/bin/bash
# Dataset sampling script - Quickly create sampled datasets with different ratios

set -x

# Default parameters
SAMPLE_RATIO=${1:-0.1}  # Default sample 1/10, can be passed as parameter
INPUT_FILE="/home/ubuntu/Open/prefer/data/ultrafeedback_dpo.jsonl"
OUTPUT_FILE="/home/ubuntu/Open/prefer/data/ultrafeedback_dpo_sample_${SAMPLE_RATIO}.jsonl"

echo "Creating sample dataset with ratio: ${SAMPLE_RATIO}"
echo "Input file: ${INPUT_FILE}"
echo "Output file: ${OUTPUT_FILE}"

# Run sampling script
python sample_dataset.py \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --sample_ratio "${SAMPLE_RATIO}" \
    --seed 42

echo "Sample dataset created successfully!"
echo "You can now update your training script to use: ${OUTPUT_FILE}"
