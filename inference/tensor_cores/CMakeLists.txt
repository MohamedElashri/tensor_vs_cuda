cmake_minimum_required(VERSION 3.10)
project(InferenceEngine CUDA CXX)

# Set CUDA architecture and standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Optionally set compilers (can be overridden by user)
if (NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
endif()

if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

# CUDA and cuDNN paths
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1" CACHE PATH "Path to CUDA toolkit")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDNN_INCLUDE_DIR "/usr/include" CACHE PATH "Path to cuDNN include")
set(CUDNN_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "Path to cuDNN libraries")

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

# Library directories
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_LIBRARY_DIR}
)

# Source file (inference.cu)
set(SOURCES inference.cu)

# Common CUDA properties
set(CUDA_COMPILE_OPTIONS
    --use_fast_math
    -lineinfo
    --expt-relaxed-constexpr
    -arch=sm_80
)

# Set properties for all CUDA files
foreach(CUDA_SOURCE ${SOURCES})
    set_source_files_properties(
        ${CUDA_SOURCE}
        PROPERTIES
        LANGUAGE CUDA
    )
endforeach()

# Add executable
add_executable(inference_engine ${SOURCES})

target_include_directories(inference_engine
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

target_link_libraries(inference_engine
    PRIVATE
    cudart
    cublas
    cudnn
    cudnn_adv_infer
    cudnn_cnn_infer
    cudnn_ops_infer
)

target_compile_options(inference_engine
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
)

set(CMAKE_BUILD_TYPE Release)

# Print configuration
message(STATUS "Using system GCC: ${CMAKE_C_COMPILER}")
message(STATUS "Using system G++: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuDNN Include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
