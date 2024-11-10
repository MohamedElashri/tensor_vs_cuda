// CUDA Cores + Tensor Cores Inference Engine

#include </usr/include/cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cuda_fp16.h>

// Utility function to convert between FP32 and FP16
__global__ void convertFP32ToFP16(float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void convertFP16ToFP32(half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

// Helper function to launch conversion kernels
void convertToFP16(float* input, half* output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    convertFP32ToFP16<<<numBlocks, blockSize>>>(input, output, size);
}

void convertToFP32(half* input, float* output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    convertFP16ToFP32<<<numBlocks, blockSize>>>(input, output, size);
}

template <typename T>
std::vector<T> loadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    // Get file size in bytes
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Calculate number of elements
    size_t num_elements = file_size / sizeof(T);
    
    std::cout << "Loading " << filename << " - File size: " << file_size 
              << " bytes, Elements: " << num_elements << std::endl;
    
    std::vector<T> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    file.close();
    
    return buffer;
}

// Add specialized version for half
template <>
std::vector<half> loadBinaryFile<half>(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(half);
    file.seekg(0, std::ios::beg);
    std::vector<half> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size * sizeof(half));
    file.close();
    return buffer;
}

// Helper function to handle CuDNN errors
#define CUDNN_CHECK(call) {                                                        \
    cudnnStatus_t err = call;                                                     \
    if (err != CUDNN_STATUS_SUCCESS) {                                           \
        std::cerr << "CuDNN Error at " << __FILE__ << ":" << __LINE__ << ": "   \
                  << cudnnGetErrorString(err) << std::endl;                      \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
}

// Helper function to handle CUDA errors
#define CUDA_CHECK(call) {                                                        \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "    \
                  << cudaGetErrorString(err) << std::endl;                       \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
}

class TensorCNNInference {
public:
    TensorCNNInference();
    ~ TensorCNNInference();
    void loadWeights();
    void initializeLayers();
    void checkTensorCoreUsage();
    void infer(const std::vector<float>& input_data);
    std::vector<float> getOutput();
    void evaluate(const std::vector<std::vector<float>>& images, 
                 const std::vector<int>& labels);
private:
    cudnnHandle_t cudnn;
    
    // Layer descriptors
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t conv1_output_desc;
    cudnnTensorDescriptor_t pool1_output_desc;
    cudnnTensorDescriptor_t conv2_output_desc;
    cudnnTensorDescriptor_t pool2_output_desc;
    cudnnTensorDescriptor_t pool2_flat_desc;
    cudnnTensorDescriptor_t fc1_output_desc;
    cudnnTensorDescriptor_t fc2_output_desc;

    // Filter descriptors
    cudnnFilterDescriptor_t conv1_filter_desc;
    cudnnFilterDescriptor_t conv2_filter_desc;
    cudnnFilterDescriptor_t fc1_filter_desc;
    cudnnFilterDescriptor_t fc2_filter_desc;
    
    // Bias descriptors
    cudnnTensorDescriptor_t conv1_bias_desc;
    cudnnTensorDescriptor_t conv2_bias_desc;
    cudnnTensorDescriptor_t fc1_bias_desc;
    cudnnTensorDescriptor_t fc2_bias_desc;
    
    // Convolution descriptors
    cudnnConvolutionDescriptor_t conv1_desc;
    cudnnConvolutionDescriptor_t conv2_desc;
    cudnnConvolutionDescriptor_t fc1_desc;
    cudnnConvolutionDescriptor_t fc2_desc;
    
    // Activation and pooling descriptors
    cudnnActivationDescriptor_t relu_activation;
    cudnnPoolingDescriptor_t pooling_desc;

    // Convolution algorithms
    cudnnConvolutionFwdAlgo_t conv1_algo;
    cudnnConvolutionFwdAlgo_t conv2_algo;
    cudnnConvolutionFwdAlgo_t fc1_algo;
    cudnnConvolutionFwdAlgo_t fc2_algo;

    // Device memory pointers
    float *d_input;
    half *d_conv1_weight, *d_conv1_bias, *d_conv1_output;
    half *d_pool1_output;
    half *d_conv2_weight, *d_conv2_bias, *d_conv2_output;
    half *d_pool2_output;
    half *d_fc1_weight, *d_fc1_bias, *d_fc1_output;
    half *d_fc2_weight, *d_fc2_bias, *d_fc2_output;
    
    // Workspace for convolutions
    size_t workspace_size;
    void *d_workspace;

    // Dimensions
    int batch_size;
    struct LayerDims {
        int n, c, h, w;
    };
    LayerDims input_dims, conv1_dims, pool1_dims, conv2_dims, pool2_dims, 
              fc1_dims, fc2_dims;

    // Find best convolution algorithm that uses Tensor Cores
    cudnnConvolutionFwdAlgo_t findBestConvAlgorithm(
        cudnnTensorDescriptor_t input_desc,
        cudnnFilterDescriptor_t filter_desc,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnTensorDescriptor_t output_desc,
        size_t* workspace_size);

};

TensorCNNInference::TensorCNNInference() : batch_size(1) {
    std::cout << "Initializing TensorCore CNN..." << std::endl;
    
    // Create cuDNN handle first
    CUDNN_CHECK(cudnnCreate(&cudnn));
    
    // Create descriptors
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv1_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool1_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv2_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool2_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool2_flat_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc1_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc2_output_desc));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&conv1_filter_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&conv2_filter_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&fc1_filter_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&fc2_filter_desc));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv1_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv2_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc1_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc2_bias_desc));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv1_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv2_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&fc1_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&fc2_desc));

    CUDNN_CHECK(cudnnCreateActivationDescriptor(&relu_activation));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc));

    // Now load weights and initialize layers
    loadWeights();
    initializeLayers();
    checkTensorCoreUsage();
}

cudnnConvolutionFwdAlgo_t TensorCNNInference::findBestConvAlgorithm(
    cudnnTensorDescriptor_t input_desc,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t output_desc,
    size_t* workspace_size) {
    
    const int requestedAlgoCount = 8;
    int returnedAlgoCount;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(requestedAlgoCount);
    
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResults.data()));

    // Find the fastest algorithm that uses Tensor Cores
    cudnnConvolutionFwdAlgo_t bestAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    float bestTime = std::numeric_limits<float>::max();
    
    for (int i = 0; i < returnedAlgoCount; i++) {
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS &&
            perfResults[i].mathType == CUDNN_TENSOR_OP_MATH &&  // Require Tensor Core operations
            perfResults[i].time < bestTime) {
            bestTime = perfResults[i].time;
            bestAlgo = perfResults[i].algo;
            *workspace_size = std::max(*workspace_size, perfResults[i].memory);
        }
    }

    // If no Tensor Core algorithm was found, try again with any algorithm
    if (bestTime == std::numeric_limits<float>::max()) {
        for (int i = 0; i < returnedAlgoCount; i++) {
            if (perfResults[i].status == CUDNN_STATUS_SUCCESS &&
                perfResults[i].time < bestTime) {
                bestTime = perfResults[i].time;
                bestAlgo = perfResults[i].algo;
                *workspace_size = std::max(*workspace_size, perfResults[i].memory);
            }
        }
    }

    return bestAlgo;
}


void TensorCNNInference::initializeLayers() {
    // Input dimensions configuration
    input_dims = {batch_size, 3, 32, 32};
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, 
        CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_HALF,  // Changed to HALF
        input_dims.n, input_dims.c, input_dims.h, input_dims.w));

    // Conv1 layer configuration
    conv1_dims = {batch_size, 32, 32, 32};
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv1_filter_desc, 
        CUDNN_DATA_HALF, 
        CUDNN_TENSOR_NCHW, 
        32, 3, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_HALF));  // Changed to HALF
    
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv1_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_output_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        conv1_dims.n, conv1_dims.c, conv1_dims.h, conv1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_bias_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        1, conv1_dims.c, 1, 1));

    // Pooling configuration
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc, 
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        2, 2,    // window size
        0, 0,    // padding
        2, 2));  // stride

    // Pool1 layer configuration
    pool1_dims = {conv1_dims.n, conv1_dims.c, conv1_dims.h/2, conv1_dims.w/2};
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1_output_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        pool1_dims.n, pool1_dims.c, pool1_dims.h, pool1_dims.w));

    // Conv2 layer configuration
    conv2_dims = {pool1_dims.n, 64, pool1_dims.h, pool1_dims.w};
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv2_filter_desc, 
        CUDNN_DATA_HALF, 
        CUDNN_TENSOR_NCHW, 
        64, 32, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_HALF));  // Changed to HALF
    
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv2_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_output_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        conv2_dims.n, conv2_dims.c, conv2_dims.h, conv2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_bias_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        1, conv2_dims.c, 1, 1));

    // Pool2 layer configuration
    pool2_dims = {conv2_dims.n, conv2_dims.c, conv2_dims.h/2, conv2_dims.w/2};
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_output_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        pool2_dims.n, pool2_dims.c, pool2_dims.h, pool2_dims.w));

    // Flatten pool2 output for FC layers
    int fc_input_size = pool2_dims.c * pool2_dims.h * pool2_dims.w;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_flat_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        batch_size, fc_input_size, 1, 1));

    // FC1 layer configuration
    fc1_dims = {batch_size, 128, 1, 1};
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc1_filter_desc, 
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW, 
        128, fc_input_size, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc1_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF));  // Changed to HALF
    
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc1_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_output_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        fc1_dims.n, fc1_dims.c, fc1_dims.h, fc1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_bias_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        1, fc1_dims.c, 1, 1));

    // FC2 layer configuration
    fc2_dims = {batch_size, 10, 1, 1};
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc2_filter_desc, 
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW, 
        10, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc2_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF));  // Changed to HALF
    
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc2_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_output_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        fc2_dims.n, fc2_dims.c, fc2_dims.h, fc2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_bias_desc, 
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,  // Changed to HALF
        1, fc2_dims.c, 1, 1));

    // ReLU activation configuration
    CUDNN_CHECK(cudnnSetActivationDescriptor(relu_activation,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0));

    // Find best algorithms for convolution operations
    workspace_size = 0;
    conv1_algo = findBestConvAlgorithm(input_desc, conv1_filter_desc, conv1_desc, 
                                     conv1_output_desc, &workspace_size);
    conv2_algo = findBestConvAlgorithm(pool1_output_desc, conv2_filter_desc, conv2_desc, 
                                     conv2_output_desc, &workspace_size);
    fc1_algo = findBestConvAlgorithm(pool2_flat_desc, fc1_filter_desc, fc1_desc, 
                                   fc1_output_desc, &workspace_size);
    fc2_algo = findBestConvAlgorithm(fc1_output_desc, fc2_filter_desc, fc2_desc, 
                                   fc2_output_desc, &workspace_size);

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    // Allocate device memory for intermediate results
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_output, batch_size * conv1_dims.c * conv1_dims.h * conv1_dims.w * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pool1_output, batch_size * pool1_dims.c * pool1_dims.h * pool1_dims.w * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_conv2_output, batch_size * conv2_dims.c * conv2_dims.h * conv2_dims.w * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pool2_output, batch_size * pool2_dims.c * pool2_dims.h * pool2_dims.w * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_fc1_output, batch_size * fc1_dims.c * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_fc2_output, batch_size * fc2_dims.c * sizeof(half)));

    std::cout << "Layer initialization complete." << std::endl;
    std::cout << "Workspace size: " << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;
}

void TensorCNNInference::checkTensorCoreUsage() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\nGPU Configuration:" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Tensor Cores are available on:
    // - Volta (7.0) and above for FP16
    // - Ampere (8.0) and above for TF32
    bool hasTensorCores = false;
    bool supportsTF32 = false;
    
    if (prop.major >= 7) {
        hasTensorCores = true;
        if (prop.major >= 8) {
            supportsTF32 = true;
        }
    }
    
    std::cout << "Tensor Cores Available: " << (hasTensorCores ? "Yes" : "No") << std::endl;
    if (hasTensorCores) {
        std::cout << "TF32 Support: " << (supportsTF32 ? "Yes" : "No") << std::endl;
    }
    
    // Check math type configuration for each convolution
    cudnnMathType_t mathType;
    CUDNN_CHECK(cudnnGetConvolutionMathType(conv1_desc, &mathType));
    std::cout << "\nConvolution Layer Math Types:" << std::endl;
    std::cout << "Conv1: " << 
        (mathType == CUDNN_TENSOR_OP_MATH ? "Tensor Core" : "Standard") << std::endl;
    
    CUDNN_CHECK(cudnnGetConvolutionMathType(conv2_desc, &mathType));
    std::cout << "Conv2: " << 
        (mathType == CUDNN_TENSOR_OP_MATH ? "Tensor Core" : "Standard") << std::endl;
    
    CUDNN_CHECK(cudnnGetConvolutionMathType(fc1_desc, &mathType));
    std::cout << "FC1: " << 
        (mathType == CUDNN_TENSOR_OP_MATH ? "Tensor Core" : "Standard") << std::endl;
    
    CUDNN_CHECK(cudnnGetConvolutionMathType(fc2_desc, &mathType));
    std::cout << "FC2: " << 
        (mathType == CUDNN_TENSOR_OP_MATH ? "Tensor Core" : "Standard") << std::endl;
    
    std::cout << "\nWorkspace Size: " << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;
}

void TensorCNNInference::infer(const std::vector<float>& input_data) {
    // Use float values for alpha/beta
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    const void* alpha_ptr = &alpha_f;
    const void* beta_ptr = &beta_f;

    // Copy input FP32 data to device
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), 
                         input_data.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));

    // Convert input from FP32 to FP16
    half* d_input_half;
    CUDA_CHECK(cudaMalloc(&d_input_half, input_data.size() * sizeof(half)));
    convertToFP16((float*)d_input, d_input_half, input_data.size());

    // Conv1 + ReLU
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
        alpha_ptr,
        input_desc, d_input_half,
        conv1_filter_desc, d_conv1_weight,
        conv1_desc, conv1_algo,
        d_workspace, workspace_size,
        beta_ptr,
        conv1_output_desc, d_conv1_output));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        alpha_ptr,
        conv1_bias_desc, d_conv1_bias,
        alpha_ptr,
        conv1_output_desc, d_conv1_output));

    // ReLU activation
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        alpha_ptr,
        conv1_output_desc, d_conv1_output,
        beta_ptr,
        conv1_output_desc, d_conv1_output));

    // MaxPool1
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        alpha_ptr,
        conv1_output_desc, d_conv1_output,
        beta_ptr,
        pool1_output_desc, d_pool1_output));

    // Conv2 + ReLU
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
        alpha_ptr,
        pool1_output_desc, d_pool1_output,
        conv2_filter_desc, d_conv2_weight,
        conv2_desc, conv2_algo,
        d_workspace, workspace_size,
        beta_ptr,
        conv2_output_desc, d_conv2_output));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        alpha_ptr,
        conv2_bias_desc, d_conv2_bias,
        alpha_ptr,
        conv2_output_desc, d_conv2_output));

    // ReLU activation
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        alpha_ptr,
        conv2_output_desc, d_conv2_output,
        beta_ptr,
        conv2_output_desc, d_conv2_output));

    // MaxPool2
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        alpha_ptr,
        conv2_output_desc, d_conv2_output,
        beta_ptr,
        pool2_output_desc, d_pool2_output));

    // FC1 + ReLU
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
        alpha_ptr,
        pool2_flat_desc, d_pool2_output,
        fc1_filter_desc, d_fc1_weight,
        fc1_desc, fc1_algo,
        d_workspace, workspace_size,
        beta_ptr,
        fc1_output_desc, d_fc1_output));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        alpha_ptr,
        fc1_bias_desc, d_fc1_bias,
        alpha_ptr,
        fc1_output_desc, d_fc1_output));

    // ReLU activation
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        alpha_ptr,
        fc1_output_desc, d_fc1_output,
        beta_ptr,
        fc1_output_desc, d_fc1_output));

    // FC2 (final layer)
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
        alpha_ptr,
        fc1_output_desc, d_fc1_output,
        fc2_filter_desc, d_fc2_weight,
        fc2_desc, fc2_algo,
        d_workspace, workspace_size,
        beta_ptr,
        fc2_output_desc, d_fc2_output));

    // Add final bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        alpha_ptr,
        fc2_bias_desc, d_fc2_bias,
        alpha_ptr,
        fc2_output_desc, d_fc2_output));

    // Clean up temporary FP16 input buffer
    CUDA_CHECK(cudaFree(d_input_half));

    // Check for any CUDA errors
    CUDA_CHECK(cudaGetLastError());
}

std::vector<float> TensorCNNInference::getOutput() {
    std::vector<float> output(10);
    
    // Convert FP16 output to FP32
    float* d_output_float;
    CUDA_CHECK(cudaMalloc(&d_output_float, 10 * sizeof(float)));
    convertToFP32((half*)d_fc2_output, d_output_float, 10);
    
    // Copy the FP32 output from device to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_output_float, 
                         output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_output_float);
    
    // Apply softmax normalization
    float max_val = *std::max_element(output.begin(), output.end());
    float sum = 0.0f;
    
    for (float& val : output) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    for (float& val : output) {
        val /= sum;
    }
    
    return output;
}


TensorCNNInference::~TensorCNNInference() {
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_conv1_weight);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv1_output);
    cudaFree(d_pool1_output);
    cudaFree(d_conv2_weight);
    cudaFree(d_conv2_bias);
    cudaFree(d_conv2_output);
    cudaFree(d_pool2_output);
    cudaFree(d_fc1_weight);
    cudaFree(d_fc1_bias);
    cudaFree(d_fc1_output);
    cudaFree(d_fc2_weight);
    cudaFree(d_fc2_bias);
    cudaFree(d_fc2_output);
    cudaFree(d_workspace);

    // Destroy descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(conv1_output_desc);
    cudnnDestroyTensorDescriptor(pool1_output_desc);
    cudnnDestroyTensorDescriptor(conv2_output_desc);
    cudnnDestroyTensorDescriptor(pool2_output_desc);
    cudnnDestroyTensorDescriptor(pool2_flat_desc);
    cudnnDestroyTensorDescriptor(fc1_output_desc);
    cudnnDestroyTensorDescriptor(fc2_output_desc);
    
    cudnnDestroyFilterDescriptor(conv1_filter_desc);
    cudnnDestroyFilterDescriptor(conv2_filter_desc);
    cudnnDestroyFilterDescriptor(fc1_filter_desc);
    cudnnDestroyFilterDescriptor(fc2_filter_desc);
    
    cudnnDestroyTensorDescriptor(conv1_bias_desc);
    cudnnDestroyTensorDescriptor(conv2_bias_desc);
    cudnnDestroyTensorDescriptor(fc1_bias_desc);
    cudnnDestroyTensorDescriptor(fc2_bias_desc);
    
    cudnnDestroyConvolutionDescriptor(conv1_desc);
    cudnnDestroyConvolutionDescriptor(conv2_desc);
    cudnnDestroyConvolutionDescriptor(fc1_desc);
    cudnnDestroyConvolutionDescriptor(fc2_desc);
    
    cudnnDestroyActivationDescriptor(relu_activation);
    cudnnDestroyPoolingDescriptor(pooling_desc);
    
    cudnnDestroy(cudnn);
}

void TensorCNNInference::loadWeights() {
    std::cout << "Loading FP16 model weights..." << std::endl;
    
    // Load the FP16 weights from binary files
    auto conv1_weights = loadBinaryFile<half>("../../../data/weights/conv1.weight_fp16.bin");
    auto conv1_biases = loadBinaryFile<half>("../../../data/weights/conv1.bias_fp16.bin");
    auto conv2_weights = loadBinaryFile<half>("../../../data/weights/conv2.weight_fp16.bin");
    auto conv2_biases = loadBinaryFile<half>("../../../data/weights/conv2.bias_fp16.bin");
    auto fc1_weights = loadBinaryFile<half>("../../../data/weights/fc1.weight_fp16.bin");
    auto fc1_biases = loadBinaryFile<half>("../../../data/weights/fc1.bias_fp16.bin");
    auto fc2_weights = loadBinaryFile<half>("../../../data/weights/fc2.weight_fp16.bin");
    auto fc2_biases = loadBinaryFile<half>("../../../data/weights/fc2.bias_fp16.bin");
    
    // Verify sizes based on PyTorch model shapes
    const size_t conv1_weights_size = 32 * 3 * 3 * 3;      // (32, 3, 3, 3)
    const size_t conv1_bias_size = 32;                     // (32,)
    const size_t conv2_weights_size = 64 * 32 * 3 * 3;     // (64, 32, 3, 3)
    const size_t conv2_bias_size = 64;                     // (64,)
    const size_t fc1_weights_size = 128 * (64 * 8 * 8);    // (128, 4096)
    const size_t fc1_bias_size = 128;                      // (128,)
    const size_t fc2_weights_size = 10 * 128;              // (10, 128)
    const size_t fc2_bias_size = 10;                       // (10,)

    // Debug print
    std::cout << "\nWeight sizes comparison:" << std::endl;
    std::cout << "Conv1 weights - Expected: " << conv1_weights_size << ", Actual: " << conv1_weights.size() 
              << " (Shape: 32x3x3x3)" << std::endl;
    std::cout << "Conv1 bias - Expected: " << conv1_bias_size << ", Actual: " << conv1_biases.size() 
              << " (Shape: 32)" << std::endl;
    std::cout << "Conv2 weights - Expected: " << conv2_weights_size << ", Actual: " << conv2_weights.size() 
              << " (Shape: 64x32x3x3)" << std::endl;
    std::cout << "Conv2 bias - Expected: " << conv2_bias_size << ", Actual: " << conv2_biases.size() 
              << " (Shape: 64)" << std::endl;
    std::cout << "FC1 weights - Expected: " << fc1_weights_size << ", Actual: " << fc1_weights.size() 
              << " (Shape: 128x4096)" << std::endl;
    std::cout << "FC1 bias - Expected: " << fc1_bias_size << ", Actual: " << fc1_biases.size() 
              << " (Shape: 128)" << std::endl;
    std::cout << "FC2 weights - Expected: " << fc2_weights_size << ", Actual: " << fc2_weights.size() 
              << " (Shape: 10x128)" << std::endl;
    std::cout << "FC2 bias - Expected: " << fc2_bias_size << ", Actual: " << fc2_biases.size() 
              << " (Shape: 10)" << std::endl;

    // Verify sizes
    if (conv1_weights.size() != conv1_weights_size ||
        conv1_biases.size() != conv1_bias_size ||
        conv2_weights.size() != conv2_weights_size ||
        conv2_biases.size() != conv2_bias_size ||
        fc1_weights.size() != fc1_weights_size ||
        fc1_biases.size() != fc1_bias_size ||
        fc2_weights.size() != fc2_weights_size ||
        fc2_biases.size() != fc2_bias_size) {
        
        std::stringstream error_msg;
        error_msg << "Weight file sizes do not match expected dimensions:\n";
        error_msg << "Conv1 weights: expected " << conv1_weights_size << ", got " << conv1_weights.size() << "\n";
        error_msg << "Conv1 bias: expected " << conv1_bias_size << ", got " << conv1_biases.size() << "\n";
        // Add similar lines for other weights...
        throw std::runtime_error(error_msg.str());
    }    
    // Allocate and copy weights to device
    CUDA_CHECK(cudaMalloc(&d_conv1_weight, conv1_weights_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_conv1_bias, conv1_bias_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_conv1_weight, conv1_weights.data(), 
                         conv1_weights_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_bias, conv1_biases.data(), 
                         conv1_bias_size * sizeof(half), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&d_conv2_weight, conv2_weights_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias, conv2_bias_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_conv2_weight, conv2_weights.data(),
                         conv2_weights_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_bias, conv2_biases.data(),
                         conv2_bias_size * sizeof(half), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fc1_weight, fc1_weights_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_fc1_bias, fc1_bias_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_fc1_weight, fc1_weights.data(),
                         fc1_weights_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc1_bias, fc1_biases.data(),
                         fc1_bias_size * sizeof(half), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fc2_weight, fc2_weights_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_fc2_bias, fc2_bias_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_fc2_weight, fc2_weights.data(),
                         fc2_weights_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc2_bias, fc2_biases.data(),
                         fc2_bias_size * sizeof(half), cudaMemcpyHostToDevice));

    // Verify weights were loaded successfully
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error while loading weights: ") +
                               cudaGetErrorString(error));
    }

    std::cout << "Successfully loaded all FP16 weights to GPU." << std::endl;
}

void parseArguments(int argc, char** argv, int& gpu_id, int& repeat_factor) {
    if (argc >= 3) {
        gpu_id = std::atoi(argv[1]);
        repeat_factor = std::atoi(argv[2]);
    } else {
        std::cerr << "Usage: " << argv[0] << " <gpu_id> <repeat_factor>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 0 10" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}




int main(int argc, char** argv) {
    int gpu_id = 0;
    int repeat_factor = 1;

    // Parse GPU ID and repeat factor from arguments
    parseArguments(argc, argv, gpu_id, repeat_factor);

    // Set the GPU device at runtime
    CUDA_CHECK(cudaSetDevice(gpu_id));

    std::cout << "Running on GPU: " << gpu_id << std::endl;
    std::cout << "Repeat factor: " << repeat_factor << std::endl;

    // Load validation data, set up the inference model, and evaluate
    try {
        std::cout << "Loading validation data..." << std::endl;
        auto validation_images = loadBinaryFile<float>("../../../data/validation/validation_images.bin");
        auto validation_labels = loadBinaryFile<int>("../../../data/validation/validation_labels.bin");

        // Original image size for CIFAR-10 (3 channels, 32x32 resolution)
        size_t image_size = 3 * 32 * 32;

        // Organize the original data into individual images
        std::vector<std::vector<float>> images;
        for (size_t i = 0; i < validation_images.size(); i += image_size) {
            images.push_back(std::vector<float>(validation_images.begin() + i, 
                                              validation_images.begin() + i + image_size));
        }

        // Repeat the dataset
        std::vector<std::vector<float>> repeated_images;
        std::vector<int> repeated_labels;

        for (int i = 0; i < repeat_factor; ++i) {
            repeated_images.insert(repeated_images.end(), images.begin(), images.end());
            repeated_labels.insert(repeated_labels.end(), validation_labels.begin(), validation_labels.end());
        }
        
        int total_images = repeated_images.size();

        std::cout << "Total images after repeating: " << total_images << std::endl;

        std::cout << "Creating Tensor Core inference engine..." << std::endl;
        TensorCNNInference cnn;
        
        // Print Tensor Core capabilities and configuration
        cnn.checkTensorCoreUsage();

        std::cout << "\n=== Starting Evaluation ===" << std::endl;
        std::cout << "Model type: Tensor Core" << std::endl;

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        size_t correct_count = 0;
        float total_time = 0.0f;

        // Warmup run
        std::cout << "Performing warmup runs..." << std::endl;
        for (int i = 0; i < 10; i++) {
            cnn.infer(repeated_images[0]);
        }

        // Main evaluation loop
        std::cout << "Starting main evaluation..." << std::endl;
        for (size_t i = 0; i < total_images; ++i) {
            try {
                CUDA_CHECK(cudaEventRecord(start));
                
                cnn.infer(repeated_images[i]);
                std::vector<float> output = cnn.getOutput();
                
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                
                float milliseconds = 0;
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                total_time += milliseconds;

                int predicted_label = std::distance(output.begin(), 
                                                    std::max_element(output.begin(), output.end()));
                
                if (predicted_label == repeated_labels[i]) {
                    ++correct_count;
                }

                // if (i % 100 == 0) {
                //     float running_accuracy = (static_cast<float>(correct_count) / (i + 1)) * 100.0f;

                    // Print intermediate statistics (debugging)
                    // std::cout << "\nProcessed " << i + 1 << "/" << total_images << " images" << std::endl;
                    // std::cout << "Running accuracy: " << std::fixed << std::setprecision(2) 
                    //           << running_accuracy << "%" << std::endl;
                    // std::cout << "Current inference time: " << std::fixed << std::setprecision(3) 
                    //           << milliseconds << " ms" << std::endl;
                    
                    // Print top 5 predictions for current image (debugging)
                    // std::vector<std::pair<int, float>> scores;
                    // for (size_t j = 0; j < output.size(); ++j) {
                    //     scores.emplace_back(j, output[j]);
                    // }
                    // std::sort(scores.begin(), scores.end(),
                    //           [](const auto& a, const auto& b) { return a.second > b.second; });
                    
                    // std::cout << "Top 5 predictions for current image:" << std::endl;
                    // for (int k = 0; k < std::min(5, static_cast<int>(scores.size())); ++k) {
                    //     std::cout << "  Class " << std::setw(2) << scores[k].first 
                    //               << ": " << std::fixed << std::setprecision(4) 
                    //               << (scores[k].second * 100.0f) << "%" << std::endl;
                    // }
                    // std::cout << "True label: " << repeated_labels[i] << std::endl;
                // }
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing image " << i << ": " << e.what() << std::endl;
                continue;
            }
        }

        // Print final statistics
        float accuracy = static_cast<float>(correct_count) / total_images * 100.0f;
        float avg_time = total_time / total_images;
        float throughput = 1000.0f / avg_time;

        std::cout << "\n=== Final Results ===" << std::endl;
        std::cout << "Model type: Tensor Core" << std::endl;
        std::cout << "Total images: " << total_images << std::endl;
        std::cout << "Correct predictions: " << correct_count << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "Average inference time: " << std::fixed << std::setprecision(3) 
                  << avg_time << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(1) 
                  << throughput << " images/second" << std::endl;
        std::cout << "Total evaluation time: " << std::fixed << std::setprecision(2) 
                  << total_time / 1000.0f << " seconds" << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}