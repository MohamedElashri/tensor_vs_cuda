#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define GPU mapping for sleepy (RTX 2080 Ti GPUs on IDs 0, 1 and RTX 3090 on ID 2)
declare -A gpu_mapping_sleepy=(
    [RTX2080TI_0]="0"
    [RTX2080TI_1]="1"
    [RTX3090]="2"
)

# Iterate over each GPU and repeat factor to run the binaries
for gpu_label in "${!gpu_mapping_sleepy[@]}"; do
    gpu_id="${gpu_mapping_sleepy[$gpu_label]}"
    
    for repeat_factor in 1 10 30 50 100; do
        echo "Running on GPU: $gpu_label with repeat factor: $repeat_factor"

        # Run inference_cuda and save log
        CUDA_VISIBLE_DEVICES=$gpu_id ./inference_cuda 0 $repeat_factor > "logs/sleepy_${gpu_label}_cuda_rf${repeat_factor}.log"
        
        # Run inference_tensor and save log
        CUDA_VISIBLE_DEVICES=$gpu_id ./inference_tensor 0 $repeat_factor > "logs/sleepy_${gpu_label}_tensor_rf${repeat_factor}.log"
        
        echo "Completed GPU: $gpu_label with repeat factor: $repeat_factor"
    done
done

echo "All runs completed on sleepy."
