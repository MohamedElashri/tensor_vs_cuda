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
set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures to build for")

# Source groups (updated paths)
set(CUDA_FP16_SOURCES cuda/inference_fp16.cu)
set(CUDA_FP32_SOURCES cuda/inference_fp32.cu)
set(TENSOR_FP16_SOURCES tensor/inference_fp16.cu)
set(TENSOR_FP32_SOURCES tensor/inference_fp32.cu)

# Common CUDA compile options
set(CUDA_COMPILE_OPTIONS
    --use_fast_math
    -lineinfo
    --expt-relaxed-constexpr
)

# Target configuration helper
function(add_inference_target TARGET SOURCES)
    add_executable(${TARGET} ${SOURCES})
    target_include_directories(${TARGET} PRIVATE ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE cudart cublas cudnn cudnn_adv_infer cudnn_cnn_infer cudnn_ops_infer)
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)
endfunction()

# Options for selective builds
option(BUILD_FP16_ONLY "Build only FP16 versions" OFF)
option(BUILD_FP32_ONLY "Build only FP32 versions" OFF)

# Build selection logic
if (BUILD_FP16_ONLY)
    add_inference_target(inference_cuda_fp16 "${CUDA_FP16_SOURCES}")
    add_inference_target(inference_tensor_fp16 "${TENSOR_FP16_SOURCES}")
    message(STATUS "Building only FP16 versions")

elseif (BUILD_FP32_ONLY)
    add_inference_target(inference_cuda_fp32 "${CUDA_FP32_SOURCES}")
    add_inference_target(inference_tensor_fp32 "${TENSOR_FP32_SOURCES}")
    message(STATUS "Building only FP32 versions")

else()
    add_inference_target(inference_cuda_fp16 "${CUDA_FP16_SOURCES}")
    add_inference_target(inference_cuda_fp32 "${CUDA_FP32_SOURCES}")
    add_inference_target(inference_tensor_fp16 "${TENSOR_FP16_SOURCES}")
    add_inference_target(inference_tensor_fp32 "${TENSOR_FP32_SOURCES}")
    message(STATUS "Building both FP16 and FP32 versions")
endif()

# Set build type
set(CMAKE_BUILD_TYPE Release)

# Print configuration for verification
message(STATUS "Using GCC: ${CMAKE_C_COMPILER}")
message(STATUS "Using G++: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuDNN Include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
