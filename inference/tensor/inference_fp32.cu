// tensor_inference_fp32.cu

#include </usr/include/cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>

// Helper macros for error checking
#define CUDNN_CHECK(call)                                                         \
    {                                                                             \
        cudnnStatus_t err = call;                                                 \
        if (err != CUDNN_STATUS_SUCCESS) {                                        \
            std::cerr << "CuDNN Error: " << cudnnGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    }

#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    }

// Template function to load binary data from file
template <typename T>
std::vector<T> loadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(T);
    file.seekg(0, std::ios::beg);
    std::vector<T> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size * sizeof(T));
    file.close();
    return buffer;
}

class TensorCNNInference {
public:
    TensorCNNInference(int batch_size, const std::string& weights_path);
    ~TensorCNNInference();
    void loadWeights(const std::string& weights_path);
    void initializeLayers();
    void checkTensorCoreUsage();
    void infer(const std::vector<float>& input_data);
    std::vector<float> getOutput();
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

    cudnnFilterDescriptor_t conv1_filter_desc;
    cudnnFilterDescriptor_t conv2_filter_desc;
    cudnnFilterDescriptor_t fc1_filter_desc;
    cudnnFilterDescriptor_t fc2_filter_desc;
    cudnnTensorDescriptor_t conv1_bias_desc;
    cudnnTensorDescriptor_t conv2_bias_desc;
    cudnnTensorDescriptor_t fc1_bias_desc;
    cudnnTensorDescriptor_t fc2_bias_desc;
    cudnnConvolutionDescriptor_t conv1_desc;
    cudnnConvolutionDescriptor_t conv2_desc;
    cudnnConvolutionDescriptor_t fc1_desc;
    cudnnConvolutionDescriptor_t fc2_desc;

    cudnnActivationDescriptor_t relu_activation;
    cudnnPoolingDescriptor_t pooling_desc;

    cudnnConvolutionFwdAlgo_t conv1_algo;
    cudnnConvolutionFwdAlgo_t conv2_algo;
    cudnnConvolutionFwdAlgo_t fc1_algo;
    cudnnConvolutionFwdAlgo_t fc2_algo;

    int batch_size;

    // Device memory pointers
    float *d_input;
    float *d_conv1_weight, *d_conv1_bias, *d_conv1_output;
    float *d_pool1_output;
    float *d_conv2_weight, *d_conv2_bias, *d_conv2_output;
    float *d_pool2_output;
    float *d_fc1_weight, *d_fc1_bias, *d_fc1_output;
    float *d_fc2_weight, *d_fc2_bias, *d_fc2_output;

    // Workspace for convolutions
    size_t workspace_size;
    void *d_workspace;

    // Output dimensions
    struct LayerDims {
        int n, c, h, w;
    };
    LayerDims input_dims, conv1_dims, pool1_dims, conv2_dims, pool2_dims, fc1_dims, fc2_dims;

    cudnnConvolutionFwdAlgo_t findBestConvAlgorithm(
        cudnnTensorDescriptor_t input_desc,
        cudnnFilterDescriptor_t filter_desc,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnTensorDescriptor_t output_desc,
        size_t* workspace_size);
};

TensorCNNInference::TensorCNNInference(int batch_size_, const std::string& weights_path)
    : batch_size(batch_size_) {
    std::cout << "Initializing CuDNN..." << std::endl;
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

    loadWeights(weights_path);
    initializeLayers();
    checkTensorCoreUsage();
}

TensorCNNInference::~TensorCNNInference() {
    // Free device memory for layer outputs
    cudaFree(d_input);
    cudaFree(d_conv1_output);
    cudaFree(d_pool1_output);
    cudaFree(d_conv2_output);
    cudaFree(d_pool2_output);
    cudaFree(d_fc1_output);
    cudaFree(d_fc2_output);

    // Free device memory for weights and biases
    cudaFree(d_conv1_weight);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv2_weight);
    cudaFree(d_conv2_bias);
    cudaFree(d_fc1_weight);
    cudaFree(d_fc1_bias);
    cudaFree(d_fc2_weight);
    cudaFree(d_fc2_bias);

    // Free workspace memory
    if (d_workspace) {
        cudaFree(d_workspace);
    }

    // Destroy tensor descriptors
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

void TensorCNNInference::loadWeights(const std::string& weights_path) {
    std::cout << "Loading model weights..." << std::endl;

    // Build full paths to weight files
    std::string conv1_weight_path = weights_path + "/conv1.weight_fp32.bin";
    std::string conv1_bias_path = weights_path + "/conv1.bias_fp32.bin";
    std::string conv2_weight_path = weights_path + "/conv2.weight_fp32.bin";
    std::string conv2_bias_path = weights_path + "/conv2.bias_fp32.bin";
    std::string fc1_weight_path = weights_path + "/fc1.weight_fp32.bin";
    std::string fc1_bias_path = weights_path + "/fc1.bias_fp32.bin";
    std::string fc2_weight_path = weights_path + "/fc2.weight_fp32.bin";
    std::string fc2_bias_path = weights_path + "/fc2.bias_fp32.bin";

    // Conv1 weights
    auto conv1_weights = loadBinaryFile<float>(conv1_weight_path);
    auto conv1_biases = loadBinaryFile<float>(conv1_bias_path);

    const size_t conv1_weights_size = 32 * 3 * 3 * 3; // (32, 3, 3, 3)
    const size_t conv1_bias_size = 32; // (32,)

    // Conv2 weights
    auto conv2_weights = loadBinaryFile<float>(conv2_weight_path);
    auto conv2_biases = loadBinaryFile<float>(conv2_bias_path);

    const size_t conv2_weights_size = 64 * 32 * 3 * 3; // (64, 32, 3, 3)
    const size_t conv2_bias_size = 64; // (64,)

    // FC1 weights
    auto fc1_weights = loadBinaryFile<float>(fc1_weight_path);
    auto fc1_biases = loadBinaryFile<float>(fc1_bias_path);

    const size_t fc1_weights_size = 128 * (64 * 8 * 8); // (128, 4096)
    const size_t fc1_bias_size = 128; // (128,)

    // FC2 weights
    auto fc2_weights = loadBinaryFile<float>(fc2_weight_path);
    auto fc2_biases = loadBinaryFile<float>(fc2_bias_path);

    const size_t fc2_weights_size = 10 * 128; // (10, 128)
    const size_t fc2_bias_size = 10; // (10,)

    // Verify sizes
    if (conv1_weights.size() != conv1_weights_size ||
        conv1_biases.size() != conv1_bias_size ||
        conv2_weights.size() != conv2_weights_size ||
        conv2_biases.size() != conv2_bias_size ||
        fc1_weights.size() != fc1_weights_size ||
        fc1_biases.size() != fc1_bias_size ||
        fc2_weights.size() != fc2_weights_size ||
        fc2_biases.size() != fc2_bias_size) {

        std::cerr << "Error: Weight file sizes do not match expected dimensions." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Allocate and copy all weights
    CUDA_CHECK(cudaMalloc(&d_conv1_weight, conv1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_bias, conv1_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv1_weight, conv1_weights.data(), conv1_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_bias, conv1_biases.data(), conv1_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_conv2_weight, conv2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias, conv2_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv2_weight, conv2_weights.data(), conv2_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_bias, conv2_biases.data(), conv2_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fc1_weight, fc1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_bias, fc1_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc1_weight, fc1_weights.data(), fc1_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc1_bias, fc1_biases.data(), fc1_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fc2_weight, fc2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_bias, fc2_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc2_weight, fc2_weights.data(), fc2_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc2_bias, fc2_biases.data(), fc2_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error while loading weights: " << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Successfully loaded all weights to GPU." << std::endl;
}

cudnnConvolutionFwdAlgo_t TensorCNNInference::findBestConvAlgorithm(
    cudnnTensorDescriptor_t input_desc,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t output_desc,
    size_t* workspace_size) {

    const int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
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
            (perfResults[i].mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION ||
             perfResults[i].mathType == CUDNN_TENSOR_OP_MATH)) {
            if (perfResults[i].time < bestTime) {
                bestTime = perfResults[i].time;
                bestAlgo = perfResults[i].algo;
                *workspace_size = std::max(*workspace_size, perfResults[i].memory);
            }
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
        CUDNN_DATA_FLOAT,
        input_dims.n, input_dims.c, input_dims.h, input_dims.w));

    // Conv1 layer configuration
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv1_filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        32, 3, 3, 3));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1_desc,
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetConvolutionMathType(conv1_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // Get Conv1 output dimensions
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv1_desc, input_desc,
        conv1_filter_desc, &conv1_dims.n, &conv1_dims.c, &conv1_dims.h, &conv1_dims.w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        conv1_dims.n, conv1_dims.c, conv1_dims.h, conv1_dims.w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, conv1_dims.c, 1, 1));

    // Pooling configuration
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        2, 2,    // window size
        0, 0,    // padding
        2, 2));  // stride

    // Get Pool1 dimensions
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pooling_desc,
        conv1_output_desc,
        &pool1_dims.n, &pool1_dims.c, &pool1_dims.h, &pool1_dims.w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1_output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        pool1_dims.n, pool1_dims.c, pool1_dims.h, pool1_dims.w));

    // Conv2 layer configuration
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv2_filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        64, 32, 3, 3));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2_desc,
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetConvolutionMathType(conv2_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // Get Conv2 output dimensions
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv2_desc, pool1_output_desc,
        conv2_filter_desc, &conv2_dims.n, &conv2_dims.c, &conv2_dims.h, &conv2_dims.w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        conv2_dims.n, conv2_dims.c, conv2_dims.h, conv2_dims.w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, conv2_dims.c, 1, 1));

    // Pool2 layer configuration
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pooling_desc,
        conv2_output_desc,
        &pool2_dims.n, &pool2_dims.c, &pool2_dims.h, &pool2_dims.w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        pool2_dims.n, pool2_dims.c, pool2_dims.h, pool2_dims.w));

    // Flatten pool2 output for FC layers
    int fc_input_size = pool2_dims.c * pool2_dims.h * pool2_dims.w;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_flat_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, fc_input_size, 1, 1));

    // FC1 layer configuration
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc1_filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        128, fc_input_size, 1, 1));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc1_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetConvolutionMathType(fc1_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, 128, 1, 1));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, 128, 1, 1));

    // FC2 layer configuration
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc2_filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        10, 128, 1, 1));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc2_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetConvolutionMathType(fc2_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, 10, 1, 1));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, 10, 1, 1));

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

    // Allocate memory for layer outputs
    size_t input_bytes = batch_size * 3 * 32 * 32 * sizeof(float);
    size_t conv1_output_bytes = batch_size * conv1_dims.c * conv1_dims.h * conv1_dims.w * sizeof(float);
    size_t pool1_output_bytes = batch_size * pool1_dims.c * pool1_dims.h * pool1_dims.w * sizeof(float);
    size_t conv2_output_bytes = batch_size * conv2_dims.c * conv2_dims.h * conv2_dims.w * sizeof(float);
    size_t pool2_output_bytes = batch_size * pool2_dims.c * pool2_dims.h * pool2_dims.w * sizeof(float);
    size_t fc1_output_bytes = batch_size * 128 * sizeof(float);
    size_t fc2_output_bytes = batch_size * 10 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_conv1_output, conv1_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_pool1_output, pool1_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_conv2_output, conv2_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_pool2_output, pool2_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_fc1_output, fc1_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_fc2_output, fc2_output_bytes));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error after memory allocation: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA memory allocation failed");
    }

    std::cout << "Layer initialization complete." << std::endl;
    std::cout << "Workspace size: " << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;
}

void TensorCNNInference::checkTensorCoreUsage() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\nGPU Configuration:" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

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
        ((mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION || mathType == CUDNN_TENSOR_OP_MATH) ? "Tensor Core" : "Standard") << std::endl;

    CUDNN_CHECK(cudnnGetConvolutionMathType(conv2_desc, &mathType));
    std::cout << "Conv2: " <<
        ((mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION || mathType == CUDNN_TENSOR_OP_MATH) ? "Tensor Core" : "Standard") << std::endl;

    CUDNN_CHECK(cudnnGetConvolutionMathType(fc1_desc, &mathType));
    std::cout << "FC1: " <<
        ((mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION || mathType == CUDNN_TENSOR_OP_MATH) ? "Tensor Core" : "Standard") << std::endl;

    CUDNN_CHECK(cudnnGetConvolutionMathType(fc2_desc, &mathType));
    std::cout << "FC2: " <<
        ((mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION || mathType == CUDNN_TENSOR_OP_MATH) ? "Tensor Core" : "Standard") << std::endl;

    std::cout << "\nWorkspace Size: " << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;
}

void TensorCNNInference::infer(const std::vector<float>& input_data) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Verify input data size and copy to device
    size_t expected_input_size = batch_size * 3 * 32 * 32;
    if (input_data.size() != expected_input_size) {
        throw std::runtime_error("Input data size mismatch");
    }

    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Conv1 layer
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        input_desc, d_input,
        conv1_filter_desc, d_conv1_weight,
        conv1_desc, conv1_algo,
        d_workspace, workspace_size,
        &beta, conv1_output_desc, d_conv1_output));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        &alpha,
        conv1_bias_desc, d_conv1_bias,
        &alpha,
        conv1_output_desc, d_conv1_output));

    // ReLU
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, conv1_output_desc, d_conv1_output,
        &beta, conv1_output_desc, d_conv1_output));

    // MaxPool1
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        &alpha, conv1_output_desc, d_conv1_output,
        &beta, pool1_output_desc, d_pool1_output));

    // Conv2 + ReLU
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        pool1_output_desc, d_pool1_output,
        conv2_filter_desc, d_conv2_weight,
        conv2_desc, conv2_algo,
        d_workspace, workspace_size,
        &beta, conv2_output_desc, d_conv2_output));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        &alpha,
        conv2_bias_desc, d_conv2_bias,
        &alpha,
        conv2_output_desc, d_conv2_output));

    // ReLU
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, conv2_output_desc, d_conv2_output,
        &beta, conv2_output_desc, d_conv2_output));

    // MaxPool2
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        &alpha, conv2_output_desc, d_conv2_output,
        &beta, pool2_output_desc, d_pool2_output));

    // FC1 + ReLU
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        pool2_flat_desc, d_pool2_output,
        fc1_filter_desc, d_fc1_weight,
        fc1_desc, fc1_algo,
        d_workspace, workspace_size,
        &beta, fc1_output_desc, d_fc1_output));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        &alpha,
        fc1_bias_desc, d_fc1_bias,
        &alpha,
        fc1_output_desc, d_fc1_output));

    // ReLU
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, fc1_output_desc, d_fc1_output,
        &beta, fc1_output_desc, d_fc1_output));

    // FC2 (final layer)
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        fc1_output_desc, d_fc1_output,
        fc2_filter_desc, d_fc2_weight,
        fc2_desc, fc2_algo,
        d_workspace, workspace_size,
        &beta, fc2_output_desc, d_fc2_output));

    // Add final bias
    CUDNN_CHECK(cudnnAddTensor(cudnn,
        &alpha,
        fc2_bias_desc, d_fc2_bias,
        &alpha,
        fc2_output_desc, d_fc2_output));

    // Check for any CUDA errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error during inference: ") +
                               cudaGetErrorString(cuda_status));
    }
}

std::vector<float> TensorCNNInference::getOutput() {
    size_t output_size = batch_size * 10;
    std::vector<float> output(output_size);

    // Copy the output from device to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_fc2_output,
                         output_size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Apply softmax normalization for each sample
    for (int i = 0; i < batch_size; ++i) {
        float max_val = *std::max_element(output.begin() + i * 10, output.begin() + (i + 1) * 10);
        float sum = 0.0f;

        for (int j = 0; j < 10; ++j) {
            output[i * 10 + j] = std::exp(output[i * 10 + j] - max_val);
            sum += output[i * 10 + j];
        }

        for (int j = 0; j < 10; ++j) {
            output[i * 10 + j] /= sum;
        }
    }

    return output;
}

void parseArguments(int argc, char** argv, int& gpu_id, int& repeat_factor, int& batch_size, std::string& data_path, std::string& weights_path) {
    if (argc >= 6) {
        gpu_id = std::atoi(argv[1]);
        repeat_factor = std::atoi(argv[2]);
        batch_size = std::atoi(argv[3]);
        data_path = argv[4];
        weights_path = argv[5];
    } else {
        std::cerr << "Usage: " << argv[0] << " <gpu_id> <repeat_factor> <batch_size> <data_path> <weights_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 0 10 256 /path/to/data/validation /path/to/data/weights" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int gpu_id = 0;
    int repeat_factor = 1;
    int batch_size = 256;
    std::string data_path;
    std::string weights_path;

    parseArguments(argc, argv, gpu_id, repeat_factor, batch_size, data_path, weights_path);
    CUDA_CHECK(cudaSetDevice(gpu_id));

    std::cout << "Running on GPU: " << gpu_id << std::endl;
    std::cout << "Repeat factor: " << repeat_factor << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    try {
        auto validation_images_path = data_path + "/validation_images.bin";
        auto validation_labels_path = data_path + "/validation_labels.bin";

        auto validation_images = loadBinaryFile<float>(validation_images_path);
        auto validation_labels = loadBinaryFile<int>(validation_labels_path);

        size_t image_size = 3 * 32 * 32;
        std::vector<std::vector<float>> images;
        for (size_t i = 0; i < validation_images.size(); i += image_size) {
            images.push_back(std::vector<float>(validation_images.begin() + i,
                                              validation_images.begin() + i + image_size));
        }

        std::vector<std::vector<float>> repeated_images;
        std::vector<int> repeated_labels;

        for (int i = 0; i < repeat_factor; ++i) {
            repeated_images.insert(repeated_images.end(), images.begin(), images.end());
            repeated_labels.insert(repeated_labels.end(), validation_labels.begin(), validation_labels.end());
        }

        size_t total_images = repeated_images.size();
        std::cout << "Total images after repeating: " << total_images << std::endl;

        TensorCNNInference cnn(batch_size, weights_path);
        cnn.checkTensorCoreUsage();

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        size_t correct_count = 0;
        float total_time = 0.0f;

        // Warmup
        std::vector<float> warmup_batch;
        warmup_batch.reserve(batch_size * image_size);
        for (int i = 0; i < batch_size && i < total_images; ++i) {
            warmup_batch.insert(warmup_batch.end(), repeated_images[i].begin(), repeated_images[i].end());
        }
        for (int i = 0; i < 10; i++) {
            cnn.infer(warmup_batch);
        }

        // Main evaluation loop
        size_t total_batches = (total_images + batch_size - 1) / batch_size;
        std::cout << "Starting main evaluation..." << std::endl;

        for (size_t batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
            size_t batch_start = batch_idx * batch_size;
            size_t batch_end = std::min(batch_start + batch_size, total_images);
            size_t current_batch_size = batch_end - batch_start;

            std::vector<float> batch_input;
            batch_input.reserve(batch_size * image_size);

            // Load actual images
            for (size_t i = batch_start; i < batch_end; ++i) {
                batch_input.insert(batch_input.end(), repeated_images[i].begin(), repeated_images[i].end());
            }

            // Pad the batch if necessary
            if (current_batch_size < batch_size) {
                // Duplicate the last image to fill the batch
                const auto& last_image = repeated_images[batch_end - 1];
                for (size_t i = current_batch_size; i < batch_size; ++i) {
                    batch_input.insert(batch_input.end(), last_image.begin(), last_image.end());
                }
            }

            CUDA_CHECK(cudaEventRecord(start));
            cnn.infer(batch_input);
            std::vector<float> output = cnn.getOutput();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            total_time += milliseconds;

            for (size_t i = 0; i < current_batch_size; ++i) {
                int predicted_label = std::distance(
                    output.begin() + i * 10,
                    std::max_element(output.begin() + i * 10, output.begin() + (i + 1) * 10)
                );
                if (predicted_label == repeated_labels[batch_start + i]) {
                    ++correct_count;
                }
            }

            if (batch_idx % 10 == 0) {
                float running_accuracy = (static_cast<float>(correct_count) / ((batch_idx + 1) * batch_size)) * 100.0f;
                std::cout << "Processed " << (batch_idx + 1) * batch_size << "/" << total_images
                         << " images. Accuracy: " << std::fixed
                         << std::setprecision(2) << running_accuracy << "%" << std::endl;
            }
        }

        float accuracy = static_cast<float>(correct_count) / total_images * 100.0f;
        float avg_time = total_time / total_batches;
        float throughput = (batch_size * 1000.0f) / avg_time;

        std::cout << "\n=== Final Results ===" << std::endl;
        std::cout << "Model type: Tensor Core FP32" << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Total images: " << total_images << std::endl;
        std::cout << "Correct predictions: " << correct_count << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "Average batch time: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " images/sec" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time / 1000.0f << " seconds" << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
