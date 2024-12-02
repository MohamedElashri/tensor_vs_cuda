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
    --machine      Machine to run on (sleepy/sneezy)
    --fp           Floating point precision (16/32, default: 32)
    --repo-path    Path to the repository root directory (required)
    --bin-path     Path to the directory containing inference binaries (required)
    --batch-size   Batch size (256/512, default: 256)
    --help         Show this help message and exit

Example:
    $(basename "$0") --machine sneezy --fp 32 --repo-path . --bin-path build --batch-size 256
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

# Function to run inference on specified machine
run_inference() {
    local machine=$1
    local gpu_label=$2
    local gpu_id=$3
    local repeat_factor=$4
    local bin_path=$5
    local logs_dir=$6
    local precision=$7
    local batch_size=$8

    # Determine binary names based on precision and batch size
    local cuda_binary="standard_inference_fp${precision}_bs${batch_size}"
    local tensor_binary="tensor_inference_fp${precision}_bs${batch_size}"

    # Verify binaries exist
    verify_binary "${bin_path}/${cuda_binary}"
    verify_binary "${bin_path}/${tensor_binary}"

    log_info "Running on GPU: $gpu_label with repeat factor: $repeat_factor, batch size: $batch_size, and FP precision: $precision"

    # Save current directory
    local current_dir=$(pwd)

    # Change directory to bin_path
    cd "${bin_path}"

    # Set CUDA_VISIBLE_DEVICES and remap the GPU ID to 0
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    local remapped_gpu_id=0

    # Run standard_inference and save log
    binary_name=${cuda_binary}

    if [ "$profile" == "yes" ]; then
        cmd="nsys profile \
            -o ${logs_dir}/${machine}_${gpu_label}_${binary_name}_fp${precision}_rf${repeat_factor}_bs${batch_size} \
            --export=sqlite \
            --force-overwrite true \
            --env CUDA_VISIBLE_DEVICES=$remapped_gpu_id \
            ./${binary_name} $remapped_gpu_id $repeat_factor $batch_size $data_path $weights_path"
    else
        cmd="CUDA_VISIBLE_DEVICES=$remapped_gpu_id ./${binary_name} $remapped_gpu_id $repeat_factor $batch_size $data_path $weights_path"
    fi

    print_and_execute "$cmd > '${logs_dir}/${machine}_${gpu_label}_standard_fp${precision}_rf${repeat_factor}_bs${batch_size}.log'"

    # Run tensor_inference and save log
    binary_name=${tensor_binary}

    if [ "$profile" == "yes" ]; then
        cmd="nsys profile \
            -o ${logs_dir}/${machine}_${gpu_label}_${binary_name}_fp${precision}_rf${repeat_factor}_bs${batch_size} \
            --export=sqlite \
            --force-overwrite true \
            --env CUDA_VISIBLE_DEVICES=$remapped_gpu_id \
            ./${binary_name} $remapped_gpu_id $repeat_factor $batch_size $data_path $weights_path"
    else
        cmd="CUDA_VISIBLE_DEVICES=$remapped_gpu_id ./${binary_name} $remapped_gpu_id $repeat_factor $batch_size $data_path $weights_path"
    fi

    print_and_execute "$cmd > '${logs_dir}/${machine}_${gpu_label}_tensor_fp${precision}_rf${repeat_factor}_bs${batch_size}.log'"

    # Change back to original directory
    cd "${current_dir}"

    log_info "Completed GPU: $gpu_label with repeat factor: $repeat_factor, batch size: $batch_size, and FP precision: $precision"
}


# Parse command line arguments
machine=""
fp="32"  # Default to FP32
repo_path=""
bin_path=""
batch_size="256"  # Default batch size
profile="no"  # Options: yes, no

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
        --batch-size)
            batch_size="$2"
            if [[ ! "$batch_size" =~ ^(256|512)$ ]]; then
                log_error "Invalid batch size. Must be either 256 or 512"
                exit 1
            fi
            shift 2
            ;;
        --profile)
            profile="$2"
            if [[ ! "$profile" =~ ^(yes|no)$ ]]; then
                log_error "Invalid profile option. Must be 'yes' or 'no'"
                exit 1
            fi
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

# Paths for data and weights
data_path="${repo_path}/data/validation"
weights_path="${repo_path}/data/weights"

# Verify directories exist
verify_directory "$repo_path" "Repository"
verify_directory "$bin_path" "Binary"

# Create logs directory with precision, machine name, and batch size
logs_dir="$(pwd)/logs_fp${fp}_${machine}_bs${batch_size}"
print_and_execute mkdir -p "$logs_dir"

# Select appropriate GPU mapping based on machine
gpu_labels=()
gpu_ids=()

if [ "$machine" == "sleepy" ]; then
    gpu_labels=("RTX2080TI_0" "RTX2080TI_1" "RTX3090")
    gpu_ids=("0" "1" "2")
else
    gpu_labels=("A100_0" "A100_1" "A100_2" "A100_3")
    gpu_ids=("0" "1" "2" "3")
fi

# Run inference for each GPU and repeat factor
for i in "${!gpu_labels[@]}"; do
    gpu_label="${gpu_labels[$i]}"
    gpu_id="${gpu_ids[$i]}"
    for repeat_factor in 1 10 30 50 100; do
        run_inference "$machine" "$gpu_label" "$gpu_id" "$repeat_factor" "$bin_path" "$logs_dir" "$fp" "$batch_size"
    done
done

log_info "All runs completed on $machine with FP$fp precision and batch size $batch_size."
log_info "Logs saved in: $logs_dir"
