#!/bin/bash
# Define the repository path (project root)
REPO_PATH="../.."

# Define the binary path (inference/build)
BIN_PATH="../inference/build_full"

# Ensure the data and weights paths are correctly set
DATA_PATH="$REPO_PATH/data/validation"
WEIGHTS_PATH="$REPO_PATH/data/weights"

# Choose the machine ('sleepy' or 'sneezy')
MACHINE="sneezy"  # or "sleepy"

# Loop over FP precision and batch sizes
for FP in 16 32; do
  for BATCH_SIZE in 256 512; do
    ./run_inference.sh --machine $MACHINE --fp $FP --repo-path $REPO_PATH --bin-path $BIN_PATH --batch-size $BATCH_SIZE --profile yes
  done
done
