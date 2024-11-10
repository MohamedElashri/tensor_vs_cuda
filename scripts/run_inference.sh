#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Color definitions
green="\033[0;32m"
yellow="\033[1;33m"
red="\033[0;31m"
nc="\033[0m" # No Color

# Logging functions
log_info() {
    echo -e "${green}[INFO]${nc} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${yellow}[WARNING]${nc} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${red}[ERROR]${nc} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Command execution with logging
print_and_execute() {
    echo -e "${green}+ $@${nc}" >&2
    eval "$@"
}

# Usage instructions
usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Options:
    --machine    Machine to run on (sleepy/sneezy)
    --fp         Floating point precision (16/32, default: 32)
    --repo-path  Path to the repository root directory (required)
    --bin-path   Path to the directory containing inference binaries (required)
    --help       Show this help message and exit

Example:
    $(basename "$0") --machine sleepy --fp 32 --repo-path /path/to/repo --bin-path /path/to/binaries
    $(basename "$0") --machine sneezy --fp 16 --repo-path /path/to/repo --bin-path /path/to/binaries
EOF
}

# Function to verify directory exists
verify_directory() {
    local dir=$1
    local dir_type=$2
    if [ ! -d "$dir" ]; then
        log_error "$dir_type directory does not exist: $dir"
        exit 1
    fi
}

# Function to verify binary exists
verify_binary() {
    local binary=$1
    if [ ! -x "$binary" ]; then
        log_error "Binary not found or not executable: $binary"
        exit 1
    fi
}

# Function to switch git branch based on FP precision
switch_branch() {
    local repo_path=$1
    local fp=$2
    
    # Change to repo directory
    cd "$repo_path"
    
    if [ "$fp" == "16" ]; then
        print_and_execute git checkout fp16
    else
        print_and_execute git checkout master
    fi
}

# Function to run inference on specified machine
run_inference() {
    local machine=$1
    local gpu_label=$2
    local gpu_id=$3
    local repeat_factor=$4
    local bin_path=$5
    local logs_dir=$6
    
    log_info "Running on GPU: $gpu_label with repeat factor: $repeat_factor"
    
    # Store current directory
    local current_dir=$(pwd)
    
    # Change to build directory
    cd "$bin_path"
    
    # Run inference_cuda and save log
    print_and_execute "CUDA_VISIBLE_DEVICES=$gpu_id ./inference_cuda 0 $repeat_factor > '${logs_dir}/${machine}_${gpu_label}_cuda_rf${repeat_factor}.log'"
    
    # Run inference_tensor and save log
    print_and_execute "CUDA_VISIBLE_DEVICES=$gpu_id ./inference_tensor 0 $repeat_factor > '${logs_dir}/${machine}_${gpu_label}_tensor_rf${repeat_factor}.log'"
    
    # Return to original directory
    cd "$current_dir"
    
    log_info "Completed GPU: $gpu_label with repeat factor: $repeat_factor"
}


# Parse command line arguments
machine=""
fp="32"  # Default to FP32
repo_path=""
bin_path=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --machine)
            machine="$2"
            shift 2
            ;;
        --fp)
            fp="$2"
            if [[ ! "$fp" =~ ^(16|32)$ ]]; then
                log_error "Invalid FP precision. Must be either 16 or 32"
                exit 1
            fi
            shift 2
            ;;
        --repo-path)
            repo_path="$2"
            shift 2
            ;;
        --bin-path)
            bin_path="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$machine" ] || [ -z "$repo_path" ] || [ -z "$bin_path" ]; then
    log_error "Missing required arguments"
    usage
    exit 1
fi

# Validate machine argument
if [[ ! "$machine" =~ ^(sleepy|sneezy)$ ]]; then
    log_error "Invalid machine specified. Must be either 'sleepy' or 'sneezy'"
    usage
    exit 1
fi

# Verify directories exist
verify_directory "$repo_path" "Repository"
verify_directory "$bin_path" "Binary"

# Verify binaries exist and are executable
verify_binary "${bin_path}/inference_cuda"
verify_binary "${bin_path}/inference_tensor"

# Create logs directory with precision and machine name
logs_dir="$(pwd)/logs_fp${fp}_${machine}"
print_and_execute mkdir -p "$logs_dir"

# Switch to appropriate branch based on FP precision
switch_branch "$repo_path" "$fp"

# Define GPU mappings
declare -A gpu_mapping_sleepy=(
    [RTX2080TI_0]="0"
    [RTX2080TI_1]="1"
    [RTX3090]="2"
)

declare -A gpu_mapping_sneezy=(
    [A100_0]="0"
    [A100_1]="1"
    [A100_2]="2"
    [A100_3]="3"
)

# Select appropriate GPU mapping based on machine
if [ "$machine" == "sleepy" ]; then
    declare -n gpu_mapping=gpu_mapping_sleepy
else
    declare -n gpu_mapping=gpu_mapping_sneezy
fi

# Run inference for each GPU and repeat factor
for gpu_label in "${!gpu_mapping[@]}"; do
    gpu_id="${gpu_mapping[$gpu_label]}"
    for repeat_factor in 1 10 30 50 100; do
        run_inference "$machine" "$gpu_label" "$gpu_id" "$repeat_factor" "$bin_path" "$logs_dir"
    done
done

log_info "All runs completed on $machine with FP$fp precision."
log_info "Logs saved in: $logs_dir"