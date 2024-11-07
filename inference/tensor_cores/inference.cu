// Tensor Cores Inference Engine


#include </usr/include/cudnn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>


__global__ void convertAndStabilizeOutput(const __half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(input[idx]);
        val = fmaxf(fminf(val, 88.0f), -88.0f);  // Clamp extreme values to prevent overflow
        output[idx] = val;
    }
}

__global__ void convertToFP16Output(const float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// Load binary data from file with FP16 conversion
template <typename T>
std::vector<T> loadBinaryFile(const std::string& filename, bool convert_to_fp16 = false) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(float);  // Original data is float
    file.seekg(0, std::ios::beg);
    
    if (convert_to_fp16) {
        // Load as float first
        std::vector<float> float_buffer(size);
        file.read(reinterpret_cast<char*>(float_buffer.data()), size * sizeof(float));
        file.close();
        
        // Convert to half precision
        std::vector<__half> half_buffer(size);
        for (size_t i = 0; i < size; ++i) {
            half_buffer[i] = __float2half(float_buffer[i]);
        }
        return std::vector<T>(half_buffer.begin(), half_buffer.end());
    } else {
        std::vector<T> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size * sizeof(T));
        file.close();
        return buffer;
    }
}

// Helper function to handle CuDNN errors
#define CUDNN_CHECK(call)                                                         \
    {                                                                             \
        cudnnStatus_t err = call;                                                 \
        if (err != CUDNN_STATUS_SUCCESS) {                                        \
            std::cerr << "CuDNN Error: " << cudnnGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    }

class TensorCoreCNNInference {
public:
    TensorCoreCNNInference();
    ~TensorCoreCNNInference();
    void loadWeights();
    void initializeLayers();
    void infer(const std::vector<float>& input_data);
    std::vector<float> getOutput();
    void evaluate(const std::vector<std::vector<float>>& images, const std::vector<int>& labels);

private:
    cudnnHandle_t cudnn;
    
    // Layer descriptors
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t conv1_output_desc;
    cudnnTensorDescriptor_t pool1_output_desc;
    cudnnTensorDescriptor_t conv2_output_desc;
    cudnnTensorDescriptor_t pool2_output_desc;
    cudnnTensorDescriptor_t pool2_flat_desc;
    cudnnTensorDescriptor_t fc1_input_desc;
    cudnnTensorDescriptor_t fc1_output_desc;
    cudnnTensorDescriptor_t fc2_input_desc;
    cudnnTensorDescriptor_t fc2_output_desc;
    cudnnFilterDescriptor_t fc1_weight_desc;
    cudnnFilterDescriptor_t fc2_weight_desc;
    
    // Bias descriptors
    cudnnTensorDescriptor_t conv1_bias_desc;
    cudnnTensorDescriptor_t conv2_bias_desc;
    cudnnTensorDescriptor_t fc1_bias_desc;
    cudnnTensorDescriptor_t fc2_bias_desc;
    
    // Convolution descriptors
    cudnnFilterDescriptor_t conv1_filter_desc;
    cudnnFilterDescriptor_t conv2_filter_desc;
    cudnnConvolutionDescriptor_t conv1_desc;
    cudnnConvolutionDescriptor_t conv2_desc;
    cudnnConvolutionDescriptor_t fc1_desc;
    cudnnConvolutionDescriptor_t fc2_desc;
    
    // Activation and pooling descriptors
    cudnnActivationDescriptor_t relu_activation;
    cudnnPoolingDescriptor_t pooling_desc;

    // Tensor Core specific parameters
    cudnnMathType_t math_type;
    cudnnDataType_t compute_type;
    cudnnDataType_t data_type;
    
    // Convolution algorithms for Tensor Cores
    cudnnConvolutionFwdAlgo_t conv1_algo;
    cudnnConvolutionFwdAlgo_t conv2_algo;
    cudnnConvolutionFwdAlgo_t fc1_algo;
    cudnnConvolutionFwdAlgo_t fc2_algo;

    int fc1_input_size;
    int batch_size = 1;

    // Device memory pointers (now using __half type)
    __half *d_input;
    __half *d_conv1_weight, *d_conv1_bias, *d_conv1_output;
    __half *d_pool1_output;
    __half *d_conv2_weight, *d_conv2_bias, *d_conv2_output;
    __half *d_pool2_output;
    __half *d_fc1_weight, *d_fc1_bias, *d_fc1_output;
    __half *d_fc2_weight, *d_fc2_bias, *d_fc2_output;

    // Workspace for convolutions
    size_t workspace_size;
    void *d_workspace;

    struct LayerDims {
        int n, c, h, w;
    };
    LayerDims conv1_dims, pool1_dims, conv2_dims, pool2_dims, fc1_dims, fc2_dims;
};

TensorCoreCNNInference::TensorCoreCNNInference() {
    std::cout << "Initializing CuDNN with Tensor Core support..." << std::endl;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Set up Tensor Core specific parameters
    math_type = CUDNN_TENSOR_OP_MATH;
    compute_type = CUDNN_DATA_FLOAT;  // Changed from HALF to FLOAT
    data_type = CUDNN_DATA_HALF;      // Keep input/output as HALF

    // Create all descriptors
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv1_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool1_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv2_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool2_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool2_flat_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc1_input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc1_output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc2_input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc2_output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&fc1_weight_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&fc2_weight_desc));

    // Create bias descriptors
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv1_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv2_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc1_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc2_bias_desc));
    
    // Create filter and convolution descriptors
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&conv1_filter_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&conv2_filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv1_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv2_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&fc1_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&fc2_desc));
    
    // Set math type for convolution operations
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv1_desc, math_type));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv2_desc, math_type));
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc1_desc, math_type));
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc2_desc, math_type));
    
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&relu_activation));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc));

    loadWeights();
    initializeLayers();
}

TensorCoreCNNInference::~TensorCoreCNNInference() {
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

    // Destroy all descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(conv1_output_desc);
    cudnnDestroyTensorDescriptor(pool1_output_desc);
    cudnnDestroyTensorDescriptor(conv2_output_desc);
    cudnnDestroyTensorDescriptor(pool2_output_desc);
    cudnnDestroyTensorDescriptor(pool2_flat_desc);
    cudnnDestroyTensorDescriptor(fc1_input_desc);
    cudnnDestroyTensorDescriptor(fc1_output_desc);
    cudnnDestroyTensorDescriptor(fc2_input_desc);
    cudnnDestroyTensorDescriptor(fc2_output_desc);
    cudnnDestroyFilterDescriptor(fc1_weight_desc);
    cudnnDestroyFilterDescriptor(fc2_weight_desc);
    cudnnDestroyTensorDescriptor(conv1_bias_desc);
    cudnnDestroyTensorDescriptor(conv2_bias_desc);
    cudnnDestroyTensorDescriptor(fc1_bias_desc);
    cudnnDestroyTensorDescriptor(fc2_bias_desc);
    cudnnDestroyFilterDescriptor(conv1_filter_desc);
    cudnnDestroyFilterDescriptor(conv2_filter_desc);
    cudnnDestroyConvolutionDescriptor(conv1_desc);
    cudnnDestroyConvolutionDescriptor(conv2_desc);
    cudnnDestroyConvolutionDescriptor(fc1_desc);
    cudnnDestroyConvolutionDescriptor(fc2_desc);
    cudnnDestroyActivationDescriptor(relu_activation);
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroy(cudnn);
}

void TensorCoreCNNInference::loadWeights() {
    std::cout << "Loading model weights with FP16 conversion and scaling..." << std::endl;
    
    // Load weights and biases for Conv1
    auto conv1_weights = loadBinaryFile<float>("../../../data/weights/conv1.weight.bin");
    auto conv1_biases = loadBinaryFile<float>("../../../data/weights/conv1.bias.bin");
    
    // Calculate scaling factors for different layers
    // For convolution layers: 1/sqrt(kernel_size * in_channels)
    // For FC layers: 1/sqrt(in_features)
    const float conv1_scale = 1.0f / std::sqrt(3.0f * 3.0f * 3.0f);  // kernel=3x3, in_channels=3
    const float conv2_scale = 1.0f / std::sqrt(3.0f * 3.0f * 32.0f); // kernel=3x3, in_channels=32
    const float fc1_scale = 1.0f / std::sqrt(64.0f * 8.0f * 8.0f);   // input size = 64*8*8
    const float fc2_scale = 1.0f / std::sqrt(128.0f);                 // input size = 128

    // Verify Conv1 dimensions
    const size_t conv1_weights_size = 32 * 3 * 3 * 3;  // out_channels * in_channels * kernel_size^2
    const size_t conv1_bias_size = 32;                 // out_channels
    
    if (conv1_weights.size() != conv1_weights_size || conv1_biases.size() != conv1_bias_size) {
        std::cerr << "Error: Conv1 weight/bias size mismatch! Expected weights: " 
                  << conv1_weights_size << ", got: " << conv1_weights.size() 
                  << ". Expected biases: " << conv1_bias_size 
                  << ", got: " << conv1_biases.size() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Convert and scale Conv1 weights to FP16
    std::vector<__half> conv1_weights_fp16(conv1_weights_size);
    for (size_t i = 0; i < conv1_weights_size; ++i) {
        conv1_weights_fp16[i] = __float2half(conv1_weights[i] * conv1_scale);
    }


    std::vector<__half> conv1_biases_fp16(conv1_bias_size);
    for (size_t i = 0; i < conv1_bias_size; ++i) {
        conv1_biases_fp16[i] = __float2half(conv1_biases[i]);
    }

    // Conv2 weights and biases
    auto conv2_weights = loadBinaryFile<float>("../../../data/weights/conv2.weight.bin");
    auto conv2_biases = loadBinaryFile<float>("../../../data/weights/conv2.bias.bin");
    
    const size_t conv2_weights_size = 64 * 32 * 3 * 3;
    const size_t conv2_bias_size = 64;
    
    if (conv2_weights.size() != conv2_weights_size || conv2_biases.size() != conv2_bias_size) {
        std::cerr << "Error: Conv2 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Convert and scale Conv2 weights
    std::vector<__half> conv2_weights_fp16(conv2_weights_size);
    for (size_t i = 0; i < conv2_weights_size; ++i) {
        conv2_weights_fp16[i] = __float2half(conv2_weights[i] * conv2_scale);
    }

    std::vector<__half> conv2_biases_fp16(conv2_bias_size);
    for (size_t i = 0; i < conv2_bias_size; ++i) {
        conv2_biases_fp16[i] = __float2half(conv2_biases[i]);
    }

    // FC1 weights and biases
    auto fc1_weights = loadBinaryFile<float>("../../../data/weights/fc1.weight.bin");
    auto fc1_biases = loadBinaryFile<float>("../../../data/weights/fc1.bias.bin");
    
    const size_t fc1_weights_size = 128 * (64 * 8 * 8);
    const size_t fc1_bias_size = 128;
    
    if (fc1_weights.size() != fc1_weights_size || fc1_biases.size() != fc1_bias_size) {
        std::cerr << "Error: FC1 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Convert and scale FC1 weights
    std::vector<__half> fc1_weights_fp16(fc1_weights_size);
    for (size_t i = 0; i < fc1_weights_size; ++i) {
        fc1_weights_fp16[i] = __float2half(fc1_weights[i] * fc1_scale);
    }

    std::vector<__half> fc1_biases_fp16(fc1_bias_size);
    for (size_t i = 0; i < fc1_bias_size; ++i) {
        fc1_biases_fp16[i] = __float2half(fc1_biases[i]);
    }

    // FC2 weights and biases
    auto fc2_weights = loadBinaryFile<float>("../../../data/weights/fc2.weight.bin");
    auto fc2_biases = loadBinaryFile<float>("../../../data/weights/fc2.bias.bin");
    
    const size_t fc2_weights_size = 10 * 128;
    const size_t fc2_bias_size = 10;
    
    if (fc2_weights.size() != fc2_weights_size || fc2_biases.size() != fc2_bias_size) {
        std::cerr << "Error: FC2 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Convert and scale FC2 weights
    std::vector<__half> fc2_weights_fp16(fc2_weights_size);
    for (size_t i = 0; i < fc2_weights_size; ++i) {
        fc2_weights_fp16[i] = __float2half(fc2_weights[i] * fc2_scale);
    }

    std::vector<__half> fc2_biases_fp16(fc2_bias_size);
    for (size_t i = 0; i < fc2_bias_size; ++i) {
        fc2_biases_fp16[i] = __float2half(fc2_biases[i]);
    }

    // Allocate and copy weights to GPU in order
    cudaMalloc(&d_conv1_weight, conv1_weights_size * sizeof(__half));
    cudaMalloc(&d_conv1_bias, conv1_bias_size * sizeof(__half));
    cudaMemcpy(d_conv1_weight, conv1_weights_fp16.data(), conv1_weights_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_bias, conv1_biases_fp16.data(), conv1_bias_size * sizeof(__half), cudaMemcpyHostToDevice);

    cudaMalloc(&d_conv2_weight, conv2_weights_size * sizeof(__half));
    cudaMalloc(&d_conv2_bias, conv2_bias_size * sizeof(__half));
    cudaMemcpy(d_conv2_weight, conv2_weights_fp16.data(), conv2_weights_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_bias, conv2_biases_fp16.data(), conv2_bias_size * sizeof(__half), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fc1_weight, fc1_weights_size * sizeof(__half));
    cudaMalloc(&d_fc1_bias, fc1_bias_size * sizeof(__half));
    cudaMemcpy(d_fc1_weight, fc1_weights_fp16.data(), fc1_weights_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, fc1_biases_fp16.data(), fc1_bias_size * sizeof(__half), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fc2_weight, fc2_weights_size * sizeof(__half));
    cudaMalloc(&d_fc2_bias, fc2_bias_size * sizeof(__half));
    cudaMemcpy(d_fc2_weight, fc2_weights_fp16.data(), fc2_weights_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_bias, fc2_biases_fp16.data(), fc2_bias_size * sizeof(__half), cudaMemcpyHostToDevice);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error while loading weights: " << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Add debug printing of weight statistics
    std::cout << "\nWeight Statistics:" << std::endl;
    std::cout << "Conv1 scale factor: " << conv1_scale << std::endl;
    std::cout << "Conv2 scale factor: " << conv2_scale << std::endl;
    std::cout << "FC1 scale factor: " << fc1_scale << std::endl;
    std::cout << "FC2 scale factor: " << fc2_scale << std::endl;

    // Print first few weights for verification
    std::cout << "\nSample weights after conversion:" << std::endl;
    std::cout << "Conv1 first weight: " << __half2float(conv1_weights_fp16[0]) << std::endl;
    std::cout << "Conv2 first weight: " << __half2float(conv2_weights_fp16[0]) << std::endl;
    std::cout << "FC1 first weight: " << __half2float(fc1_weights_fp16[0]) << std::endl;
    std::cout << "FC2 first weight: " << __half2float(fc2_weights_fp16[0]) << std::endl;
    
    std::cout << "Successfully loaded all weights to GPU with FP16 precision." << std::endl;
}

void TensorCoreCNNInference::initializeLayers() {
    // Input: 3x32x32 with FP16
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, 
        data_type, batch_size, 3, 32, 32));

    // Conv1: 3 -> 32 channels, 3x3 kernel
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv1_filter_desc, data_type, 
        CUDNN_TENSOR_NCHW, 32, 3, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_FLOAT));  // Changed from compute_type to CUDNN_DATA_FLOAT
    
    // Get Conv1 output dimensions
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv1_desc, input_desc, 
        conv1_filter_desc, &conv1_dims.n, &conv1_dims.c, &conv1_dims.h, &conv1_dims.w));
    
    std::cout << "Conv1 output dimensions: " << conv1_dims.n << "x" << conv1_dims.c 
              << "x" << conv1_dims.h << "x" << conv1_dims.w << std::endl;
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_output_desc, CUDNN_TENSOR_NCHW,
        data_type, conv1_dims.n, conv1_dims.c, conv1_dims.h, conv1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_bias_desc, CUDNN_TENSOR_NCHW,
        data_type, 1, conv1_dims.c, 1, 1));

    // Find algorithm for Conv1
    int requestedAlgoCount = 1;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        input_desc,
        conv1_filter_desc,
        conv1_desc,
        conv1_output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &perfResults));
    
    conv1_algo = perfResults.algo;
    
    // Pooling setup
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc, 
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        2, 2,    // window
        0, 0,    // padding
        2, 2));  // stride

    // Get Pool1 dimensions
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pooling_desc,
        conv1_output_desc,
        &pool1_dims.n, &pool1_dims.c, &pool1_dims.h, &pool1_dims.w));
    
    std::cout << "Pool1 dimensions: " << pool1_dims.n << "x" << pool1_dims.c 
              << "x" << pool1_dims.h << "x" << pool1_dims.w << std::endl;
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1_output_desc, CUDNN_TENSOR_NCHW,
        data_type, pool1_dims.n, pool1_dims.c, pool1_dims.h, pool1_dims.w));

    // Conv2 setup
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv2_filter_desc, data_type, 
        CUDNN_TENSOR_NCHW, 64, 32, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_FLOAT));  // Use FLOAT for compute type
    
    // Get Conv2 output dimensions
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv2_desc, pool1_output_desc, 
        conv2_filter_desc, &conv2_dims.n, &conv2_dims.c, &conv2_dims.h, &conv2_dims.w));
    
    std::cout << "Conv2 output dimensions: " << conv2_dims.n << "x" << conv2_dims.c 
              << "x" << conv2_dims.h << "x" << conv2_dims.w << std::endl;
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_output_desc, CUDNN_TENSOR_NCHW,
        data_type, conv2_dims.n, conv2_dims.c, conv2_dims.h, conv2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_bias_desc, CUDNN_TENSOR_NCHW,
        data_type, 1, conv2_dims.c, 1, 1));

    // Find algorithm for Conv2
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        pool1_output_desc,
        conv2_filter_desc,
        conv2_desc,
        conv2_output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &perfResults));
    
    conv2_algo = perfResults.algo;

    // Pool2 setup
    pool2_dims = {conv2_dims.n, conv2_dims.c, conv2_dims.h/2, conv2_dims.w/2};
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_output_desc, CUDNN_TENSOR_NCHW,
        data_type, pool2_dims.n, pool2_dims.c, pool2_dims.h, pool2_dims.w));

    std::cout << "Pool2 dimensions: " << pool2_dims.n << "x" << pool2_dims.c 
              << "x" << pool2_dims.h << "x" << pool2_dims.w << std::endl;

    // FC layers setup
    fc1_input_size = pool2_dims.c * pool2_dims.h * pool2_dims.w;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_flat_desc, CUDNN_TENSOR_NCHW,
        data_type, batch_size, fc1_input_size, 1, 1));

    // FC1 setup
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc1_weight_desc, data_type,
        CUDNN_TENSOR_NCHW, 128, fc1_input_size, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc1_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));  // Use FLOAT for compute type

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_output_desc, CUDNN_TENSOR_NCHW,
        data_type, batch_size, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_bias_desc, CUDNN_TENSOR_NCHW,
        data_type, 1, 128, 1, 1));

    // Find algorithm for FC1
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        pool2_flat_desc,
        fc1_weight_desc,
        fc1_desc,
        fc1_output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &perfResults));
    
    fc1_algo = perfResults.algo;

    // FC2 setup
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc2_weight_desc, data_type,
        CUDNN_TENSOR_NCHW, 10, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc2_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));  // Use FLOAT for compute type

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_output_desc, CUDNN_TENSOR_NCHW,
        data_type, batch_size, 10, 1, 1));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_bias_desc, CUDNN_TENSOR_NCHW,
        data_type, 1, 10, 1, 1));

    // Find algorithm for FC2
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        fc1_output_desc,
        fc2_weight_desc,
        fc2_desc,
        fc2_output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &perfResults));
    
    fc2_algo = perfResults.algo;

    // ReLU activation setup
    CUDNN_CHECK(cudnnSetActivationDescriptor(relu_activation,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0));

    // Calculate workspace size for all operations
    size_t workspace_sizes[4];
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_desc, conv1_filter_desc, conv1_desc, conv1_output_desc,
        conv1_algo, &workspace_sizes[0]));
    
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        pool1_output_desc, conv2_filter_desc, conv2_desc, conv2_output_desc,
        conv2_algo, &workspace_sizes[1]));
    
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        pool2_flat_desc, fc1_weight_desc, fc1_desc, fc1_output_desc,
        fc1_algo, &workspace_sizes[2]));
    
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        fc1_output_desc, fc2_weight_desc, fc2_desc, fc2_output_desc,
        fc2_algo, &workspace_sizes[3]));

    workspace_size = *std::max_element(workspace_sizes, workspace_sizes + 4);
    cudaMalloc(&d_workspace, workspace_size);

    // Allocate memory for layer outputs
    cudaMalloc(&d_input, batch_size * 3 * 32 * 32 * sizeof(__half));
    cudaMalloc(&d_conv1_output, batch_size * conv1_dims.c * conv1_dims.h * conv1_dims.w * sizeof(__half));
    cudaMalloc(&d_pool1_output, batch_size * pool1_dims.c * pool1_dims.h * pool1_dims.w * sizeof(__half));
    cudaMalloc(&d_conv2_output, batch_size * conv2_dims.c * conv2_dims.h * conv2_dims.w * sizeof(__half));
    cudaMalloc(&d_pool2_output, batch_size * pool2_dims.c * pool2_dims.h * pool2_dims.w * sizeof(__half));
    cudaMalloc(&d_fc1_output, batch_size * 128 * sizeof(__half));
    cudaMalloc(&d_fc2_output, batch_size * 10 * sizeof(__half));

    std::cout << "Layer initialization complete." << std::endl;
    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;
}

void TensorCoreCNNInference::infer(const std::vector<float>& input_data) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Convert input to FP16 with proper scaling
    std::vector<__half> input_fp16(input_data.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        // Scale inputs to [-1, 1]
        float normalized_input = (input_data[i] / 255.0f - 0.5f) / 0.5f;
        input_fp16[i] = __float2half(normalized_input);

        // Print the normalized inputs for verification (optional)
        // if (i < 5) {  
        //     std::cout << "Input[" << i << "] after adjusted normalization: " 
        //             << normalized_input << std::endl;
        // }
    }

    cudaMemcpy(d_input, input_fp16.data(), input_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice);

    // Conv1 + ReLU
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, 
        input_desc, d_input,
        conv1_filter_desc, d_conv1_weight,
        conv1_desc, conv1_algo,
        d_workspace, workspace_size,
        &beta, conv1_output_desc, d_conv1_output));
    
    CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, 
        conv1_bias_desc, d_conv1_bias,
        &alpha, conv1_output_desc, d_conv1_output));
    
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, conv1_output_desc, d_conv1_output,
        &beta, conv1_output_desc, d_conv1_output));

    // Pool1
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
    
    CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha,
        conv2_bias_desc, d_conv2_bias,
        &alpha, conv2_output_desc, d_conv2_output));
        
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, conv2_output_desc, d_conv2_output,
        &beta, conv2_output_desc, d_conv2_output));

    // Pool2
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        &alpha, conv2_output_desc, d_conv2_output,
        &beta, pool2_output_desc, d_pool2_output));

    // FC1 + ReLU
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        pool2_flat_desc, d_pool2_output,
        fc1_weight_desc, d_fc1_weight,
        fc1_desc, fc1_algo,
        d_workspace, workspace_size,
        &beta, fc1_output_desc, d_fc1_output));

    CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha,
        fc1_bias_desc, d_fc1_bias,
        &alpha, fc1_output_desc, d_fc1_output));
    
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, fc1_output_desc, d_fc1_output,
        &beta, fc1_output_desc, d_fc1_output));

    // FC2 (final layer)
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        fc1_output_desc, d_fc1_output,
        fc2_weight_desc, d_fc2_weight,
        fc2_desc, fc2_algo,
        d_workspace, workspace_size,
        &beta, fc2_output_desc, d_fc2_output));

    CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha,
        fc2_bias_desc, d_fc2_bias,
        &alpha, fc2_output_desc, d_fc2_output));

    // Additional numerical stability check after final layer
    cudnnTensorDescriptor_t temp_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&temp_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(temp_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, batch_size, 10, 1, 1));

    // Convert final output to FP32 temporarily for stable computation
    float *d_final_fp32;
    cudaMalloc(&d_final_fp32, batch_size * 10 * sizeof(float));
    
    // Convert and stabilize outputs
    const int BLOCK_SIZE = 256;
    const int grid_size = (10 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch convertAndStabilizeOutput and convertToFP16Output kernels
    convertAndStabilizeOutput<<<grid_size, BLOCK_SIZE>>>(d_fc2_output, d_final_fp32, 10);
    convertToFP16Output<<<grid_size, BLOCK_SIZE>>>(d_final_fp32, d_fc2_output, 10);    

    // Cleanup
    cudaFree(d_final_fp32);
    cudnnDestroyTensorDescriptor(temp_desc);

    // Check for any CUDA errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error during inference: ") + 
                               cudaGetErrorString(cuda_status));
    }
}
std::vector<float> TensorCoreCNNInference::getOutput() {
    std::vector<float> output_fp32(10);
    
    // Copy the FP16 output from device to host and convert to FP32
    cudaError_t status = cudaMemcpy(output_fp32.data(), d_fc2_output, 
                                    output_fp32.size() * sizeof(__half), 
                                    cudaMemcpyDeviceToHost);
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy output from device: " 
                  << cudaGetErrorString(status) << std::endl;
        throw std::runtime_error("CUDA memcpy failed");
    }

    // Softmax computation in FP32
    float max_val = *std::max_element(output_fp32.begin(), output_fp32.end());
    float sum = 0.0f;
    
    // Subtract max for numerical stability and compute exp
    for (float& val : output_fp32) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    // Normalize
    for (float& val : output_fp32) {
        val /= sum;
    }
    
    return output_fp32;
}


void TensorCoreCNNInference::evaluate(const std::vector<std::vector<float>>& images, 
                                    const std::vector<int>& labels) {
    size_t correct_count = 0;
    auto start = std::chrono::high_resolution_clock::now();

    
    for (size_t i = 0; i < images.size(); ++i) {
        try {
            infer(images[i]);
            std::vector<float> output = getOutput();
            int predicted_label = std::distance(output.begin(), 
                                             std::max_element(output.begin(), output.end()));
            
            // Print predictions for every 100th image
            if (i % 100 == 0) {
                std::cout << "\nImage " << i << " predictions:" << std::endl;
                std::vector<std::pair<int, float>> scores;
                for (int j = 0; j < output.size(); ++j) {
                    scores.emplace_back(j, output[j]);
                }
                std::sort(scores.begin(), scores.end(), 
                         [](const auto& a, const auto& b) { return a.second > b.second; });
                
                for (int k = 0; k < std::min(5, static_cast<int>(scores.size())); ++k) {
                    std::cout << "Class " << scores[k].first 
                             << " - Score: " << scores[k].second << std::endl;
                }
                std::cout << "True label: " << labels[i] << std::endl;
            }

            if (predicted_label == labels[i]) {
                ++correct_count;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing image " << i << ": " << e.what() << std::endl;
            continue;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    float accuracy = static_cast<float>(correct_count) / images.size() * 100.0f;
    double throughput = images.size() / elapsed.count();
    
    std::cout << "\nEvaluation Results:" << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Throughput: " << throughput << " images/second" << std::endl;
    std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;
}

int main() {
    try {
        std::cout << "Loading validation data..." << std::endl;
        auto validation_images = loadBinaryFile<float>("../../../data/validation/validation_images.bin");
        auto validation_labels = loadBinaryFile<int>("../../../data/validation/validation_labels.bin");

        std::vector<std::vector<float>> images;
        size_t image_size = 3 * 32 * 32;
        for (size_t i = 0; i < validation_images.size(); i += image_size) {
            images.push_back(std::vector<float>(
                validation_images.begin() + i,
                validation_images.begin() + i + image_size
            ));
        }

        std::cout << "Creating Tensor Core inference engine..." << std::endl;
        TensorCoreCNNInference cnn_inference;
        
        std::cout << "Running evaluation on validation data..." << std::endl;
        cnn_inference.evaluate(images, validation_labels);
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}        