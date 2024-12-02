cmake_minimum_required(VERSION 3.10)
project(InferenceEngine CUDA CXX)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Machine-specific compiler settings
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "sneezy")
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "sleepy")
    set(CMAKE_C_COMPILER "/data/apps/gcc-10.1.0/bin/gcc")
    set(CMAKE_CXX_COMPILER "/data/apps/gcc-10.1.0/bin/g++")
endif()

# CUDA and cuDNN paths
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1" CACHE PATH "Path to CUDA toolkit")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDNN_INCLUDE_DIR "/usr/include" CACHE PATH "Path to cuDNN include")
set(CUDNN_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "Path to cuDNN libraries")

# Include and library directories
include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDNN_LIBRARY_DIR})

# CUDA architecture setting
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to build for")

# Source files with corrected paths
set(TENSOR_FP16_SOURCES tensor/inference_fp16.cu)
set(TENSOR_FP32_SOURCES tensor/inference_fp32.cu)
set(STANDARD_FP16_SOURCES cuda/inference_fp16.cu)
set(STANDARD_FP32_SOURCES cuda/inference_fp32.cu)

# Common CUDA compile options
set(CUDA_COMPILE_OPTIONS
    --use_fast_math
    -lineinfo
    --expt-relaxed-constexpr
    -gencode arch=compute_80,code=sm_80
)

# Batch sizes to build
set(BATCH_SIZES 256 512)

# Function to add inference target
function(add_inference_target TARGET_NAME SOURCE_FILE BATCH_SIZE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE cudart cublas cudnn)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)
    target_compile_definitions(${TARGET_NAME} PRIVATE BATCH_SIZE=${BATCH_SIZE})
endfunction()

# Build targets for each batch size
foreach(BATCH_SIZE ${BATCH_SIZES})
    # Build tensor_inference_fp16
    add_inference_target(tensor_inference_fp16_bs${BATCH_SIZE} "${TENSOR_FP16_SOURCES}" ${BATCH_SIZE})
    # Build tensor_inference_fp32
    add_inference_target(tensor_inference_fp32_bs${BATCH_SIZE} "${TENSOR_FP32_SOURCES}" ${BATCH_SIZE})
    # Build standard_inference_fp16
    add_inference_target(standard_inference_fp16_bs${BATCH_SIZE} "${STANDARD_FP16_SOURCES}" ${BATCH_SIZE})
    # Build standard_inference_fp32
    add_inference_target(standard_inference_fp32_bs${BATCH_SIZE} "${STANDARD_FP32_SOURCES}" ${BATCH_SIZE})
endforeach()

# Set build type
set(CMAKE_BUILD_TYPE Release)

# Print configuration for verification
message(STATUS "Using GCC: ${CMAKE_C_COMPILER}")
message(STATUS "Using G++: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuDNN Include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
