# Inference Runner Script

A script to automate running inference tests across different GPU configurations with support for FP16 and FP32 precision.

## Overview

This script automates the process of running inference tests on different machines (sleepy/sneezy) with different GPU configurations. It supports both FP16 and FP32 precision modes and automatically handles:

- Git branch switching between FP16/FP32 versions
- GPU device selection
- Log file organization
- Multiple repeat factors for thorough testing

## Usage

```bash
./run_inference.sh [options]
```

### Required Arguments

- `--machine`: Target machine to run tests on (sleepy/sneezy)
- `--repo-path`: Path to the repository root directory
- `--bin-path`: Path to the directory containing inference binaries

### Optional Arguments

- `--fp`: Floating point precision (16/32, default: 32)
- `--help`: Show help message and exit

### GPU Configurations

#### Sleepy Machine (RTX GPUs)
- RTX2080TI_0 (GPU ID: 0)
- RTX2080TI_1 (GPU ID: 1)
- RTX3090 (GPU ID: 2)

#### Sneezy Machine (A100 GPUs)
- A100_0 (GPU ID: 0)
- A100_1 (GPU ID: 1)
- A100_2 (GPU ID: 2)
- A100_3 (GPU ID: 3)

## Examples

### Running FP32 Tests on Sleepy

```bash
./run_inference.sh \
    --machine sleepy \
    --repo-path /path/to/repo \
    --bin-path /path/to/repo/inference/unified/build
```

### Running FP16 Tests on Sneezy

```bash
./run_inference.sh \
    --machine sneezy \
    --fp 16 \
    --repo-path /path/to/repo \
    --bin-path /path/to/repo/inference/unified/build
```

### Using Relative Paths

```bash
# From scripts directory
./run_inference.sh \
    --machine sneezy \
    --fp 16 \
    --repo-path . \
    --bin-path ../inference/unified/build
```

### Getting Help

```bash
./run_inference.sh --help
```

## Output Structure

The script creates a logs directory with the following naming convention:
```
logs_fp{16|32}_{machine}/
├── {machine}_GPU_cuda_rf{repeat_factor}.log
└── {machine}_GPU_tensor_rf{repeat_factor}.log
```

Example:
```
logs_fp16_sneezy/
├── sneezy_A100_0_cuda_rf1.log
├── sneezy_A100_0_tensor_rf1.log
├── sneezy_A100_0_cuda_rf10.log
├── sneezy_A100_0_tensor_rf10.log
...
```

## Features

- **Automatic Branch Switching**: Switches between fp16 branch and master based on precision selection
- **Colored Output**: Uses color-coded logging for better visibility
- **Error Handling**: Comprehensive error checking for paths and executables
- **Multiple Repeat Factors**: Tests with repeat factors: 1, 10, 30, 50, 100
- **Organized Logging**: Creates structured log directories with precision and machine information

## Notes

- Script must be run with proper permissions to execute binaries and write logs
- Binaries are executed from their build directory to maintain correct relative paths
- All logs are saved in the current working directory where the script is run from

