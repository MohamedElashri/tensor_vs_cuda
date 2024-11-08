#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define GPU mapping for sneezy (A100 GPUs on IDs 0, 1, 2, and 3)
declare -A gpu_mapping_sneezy=(
    [A100_0]="0"
    [A100_1]="1"
    [A100_2]="2"
    [A100_3]="3"
)

# Iterate over each GPU and repeat factor to run the binaries
for gpu_label in "${!gpu_mapping_sneezy[@]}"; do
    gpu_id="${gpu_mapping_sneezy[$gpu_label]}"
    
    for repeat_factor in 1 10 30 50 100; do
        echo "Running on GPU: $gpu_label with repeat factor: $repeat_factor"

        # Run inference_cuda and save log
        CUDA_VISIBLE_DEVICES=$gpu_id ./inference_cuda 0 $repeat_factor > "logs/sneezy_${gpu_label}_cuda_rf${repeat_factor}.log"
        
        # Run inference_tensor and save log
        CUDA_VISIBLE_DEVICES=$gpu_id ./inference_tensor 0 $repeat_factor > "logs/sneezy_${gpu_label}_tensor_rf${repeat_factor}.log"
        
        echo "Completed GPU: $gpu_label with repeat factor: $repeat_factor"
    done
done

echo "All runs completed on sneezy."
