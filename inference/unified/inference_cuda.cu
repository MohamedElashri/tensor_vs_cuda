// CUDA Cores Inference Engine


#include </usr/include/cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Load binary data from file
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

// Helper function to handle CuDNN errors
#define CUDNN_CHECK(call)                                                         \
    {                                                                             \
        cudnnStatus_t err = call;                                                 \
        if (err != CUDNN_STATUS_SUCCESS) {                                        \
            std::cerr << "CuDNN Error: " << cudnnGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    }

// Helper function to handle CUDA errors

#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    }

class CUDACNNInference {
public:
    CUDACNNInference();
    ~CUDACNNInference();
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
    
    // FC layer convolution descriptors
    cudnnConvolutionDescriptor_t fc1_desc;
    cudnnConvolutionDescriptor_t fc2_desc;
    
    // Activation and pooling descriptors
    cudnnActivationDescriptor_t relu_activation;
    cudnnPoolingDescriptor_t pooling_desc;

    int fc1_input_size;


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
    int batch_size = 1;
    struct LayerDims {
        int n, c, h, w;
    };
    LayerDims conv1_dims, pool1_dims, conv2_dims, pool2_dims, fc1_dims, fc2_dims;
};

CUDACNNInference::CUDACNNInference() {
    std::cout << "Initializing CuDNN..." << std::endl;
    CUDNN_CHECK(cudnnCreate(&cudnn));

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
    
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&relu_activation));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc));

    loadWeights();
    initializeLayers();
}

CUDACNNInference::~CUDACNNInference() {
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
    cudnnDestroyTensorDescriptor(fc1_input_desc);
    cudnnDestroyTensorDescriptor(fc1_output_desc);
    cudnnDestroyTensorDescriptor(fc2_input_desc);
    cudnnDestroyTensorDescriptor(fc2_output_desc);
    cudnnDestroyFilterDescriptor(fc1_weight_desc);
    cudnnDestroyFilterDescriptor(fc2_weight_desc);

    // Destroy bias descriptors
    cudnnDestroyTensorDescriptor(conv1_bias_desc);
    cudnnDestroyTensorDescriptor(conv2_bias_desc);
    cudnnDestroyTensorDescriptor(fc1_bias_desc);
    cudnnDestroyTensorDescriptor(fc2_bias_desc);

    // Destroy filter and convolution descriptors
    cudnnDestroyFilterDescriptor(conv1_filter_desc);
    cudnnDestroyFilterDescriptor(conv2_filter_desc);
    cudnnDestroyConvolutionDescriptor(conv1_desc);
    cudnnDestroyConvolutionDescriptor(conv2_desc);
    cudnnDestroyConvolutionDescriptor(fc1_desc);
    cudnnDestroyConvolutionDescriptor(fc2_desc);

    // Destroy activation and pooling descriptors
    cudnnDestroyActivationDescriptor(relu_activation);
    cudnnDestroyPoolingDescriptor(pooling_desc);

    // Destroy cuDNN handle
    cudnnDestroy(cudnn);
}

void CUDACNNInference::loadWeights() {
    std::cout << "Loading model weights..." << std::endl;
    
    // Conv1 weights
    auto conv1_weights = loadBinaryFile<float>("../../../data/weights/conv1.weight_fp32.bin");
    auto conv1_biases = loadBinaryFile<float>("../../../data/weights/conv1.bias_fp32.bin");
    
    //  print first few weights for verification (debugging)
    // std::cout << "Conv1 weights first values: ";
    // for(int i = 0; i < 5; i++) {
    //     std::cout << conv1_weights[i] << " ";
    // }
    // std::cout << std::endl;
    
    const size_t conv1_weights_size = 32 * 3 * 3 * 3;
    const size_t conv1_bias_size = 32;
    
    if (conv1_weights.size() != conv1_weights_size || conv1_biases.size() != conv1_bias_size) {
        std::cerr << "Error: Conv1 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Conv2 weights
    auto conv2_weights = loadBinaryFile<float>("../../../data/weights/conv2.weight_fp32.bin");
    auto conv2_biases = loadBinaryFile<float>("../../../data/weights/conv2.bias_fp32.bin");
    
    const size_t conv2_weights_size = 64 * 32 * 3 * 3;
    const size_t conv2_bias_size = 64;
    
    // FC1 weights
    auto fc1_weights = loadBinaryFile<float>("../../../data/weights/fc1.weight_fp32.bin");
    auto fc1_biases = loadBinaryFile<float>("../../../data/weights/fc1.bias_fp32.bin");
    
    const size_t fc1_weights_size = 128 * (64 * 8 * 8);
    const size_t fc1_bias_size = 128;  // Added definition
    
    // FC2 weights
    auto fc2_weights = loadBinaryFile<float>("../../../data/weights/fc2.weight_fp32.bin");
    auto fc2_biases = loadBinaryFile<float>("../../../data/weights/fc2.bias_fp32.bin");
    
    const size_t fc2_weights_size = 10 * 128;
    const size_t fc2_bias_size = 10;  // Added definition

    // Size verification
    if (conv2_weights.size() != conv2_weights_size || conv2_biases.size() != conv2_bias_size) {
        std::cerr << "Error: Conv2 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (fc1_weights.size() != fc1_weights_size || fc1_biases.size() != fc1_bias_size) {
        std::cerr << "Error: FC1 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (fc2_weights.size() != fc2_weights_size || fc2_biases.size() != fc2_bias_size) {
        std::cerr << "Error: FC2 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Allocate and copy all weights in order
    cudaMalloc(&d_conv1_weight, conv1_weights_size * sizeof(float));
    cudaMalloc(&d_conv1_bias, conv1_bias_size * sizeof(float));
    cudaMemcpy(d_conv1_weight, conv1_weights.data(), conv1_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_bias, conv1_biases.data(), conv1_bias_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_conv2_weight, conv2_weights_size * sizeof(float));
    cudaMalloc(&d_conv2_bias, conv2_bias_size * sizeof(float));
    cudaMemcpy(d_conv2_weight, conv2_weights.data(), conv2_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_bias, conv2_biases.data(), conv2_bias_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fc1_weight, fc1_weights_size * sizeof(float));
    cudaMalloc(&d_fc1_bias, fc1_bias_size * sizeof(float));
    cudaMemcpy(d_fc1_weight, fc1_weights.data(), fc1_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, fc1_biases.data(), fc1_bias_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fc2_weight, fc2_weights_size * sizeof(float));
    cudaMalloc(&d_fc2_bias, fc2_bias_size * sizeof(float));
    cudaMemcpy(d_fc2_weight, fc2_weights.data(), fc2_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_bias, fc2_biases.data(), fc2_bias_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error while loading weights: " << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    std::cout << "Successfully loaded all weights to GPU." << std::endl;
}


void CUDACNNInference::initializeLayers() {
    // Input: 3x32x32
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, batch_size, 3, 32, 32));

    // Conv1: 3 -> 32 channels, 3x3 kernel
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv1_filter_desc, CUDNN_DATA_FLOAT, 
        CUDNN_TENSOR_NCHW, 32, 3, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_FLOAT));

    // Get Conv1 output dimensions
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv1_desc, input_desc, 
        conv1_filter_desc, &conv1_dims.n, &conv1_dims.c, &conv1_dims.h, &conv1_dims.w));
    
    std::cout << "Conv1 output dimensions: " << conv1_dims.n << "x" << conv1_dims.c 
              << "x" << conv1_dims.h << "x" << conv1_dims.w << std::endl;
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, conv1_dims.n, conv1_dims.c, conv1_dims.h, conv1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, conv1_dims.c, 1, 1));

    // Find best algorithm for Conv1
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
    
    // Pooling setup
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc, 
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        2, 2,    // window height, width
        0, 0,    // padding height, width
        2, 2));  // stride height, width

    // Get Pool1 dimensions
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pooling_desc,
        conv1_output_desc,
        &pool1_dims.n, &pool1_dims.c, &pool1_dims.h, &pool1_dims.w));
    
    std::cout << "Pool1 dimensions: " << pool1_dims.n << "x" << pool1_dims.c 
              << "x" << pool1_dims.h << "x" << pool1_dims.w << std::endl;
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, pool1_dims.n, pool1_dims.c, pool1_dims.h, pool1_dims.w));

    // Conv2 setup
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv2_filter_desc, CUDNN_DATA_FLOAT, 
        CUDNN_TENSOR_NCHW, 64, 32, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_FLOAT));
    
    // Get Conv2 output dimensions
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv2_desc, pool1_output_desc, 
        conv2_filter_desc, &conv2_dims.n, &conv2_dims.c, &conv2_dims.h, &conv2_dims.w));
    
    std::cout << "Conv2 output dimensions: " << conv2_dims.n << "x" << conv2_dims.c 
              << "x" << conv2_dims.h << "x" << conv2_dims.w << std::endl;
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, conv2_dims.n, conv2_dims.c, conv2_dims.h, conv2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, conv2_dims.c, 1, 1));

    // Find best algorithm for Conv2
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        pool1_output_desc,
        conv2_filter_desc,
        conv2_desc,
        conv2_output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &perfResults));

    // Pool2 setup
    pool2_dims = {conv2_dims.n, conv2_dims.c, conv2_dims.h/2, conv2_dims.w/2};
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, pool2_dims.n, pool2_dims.c, pool2_dims.h, pool2_dims.w));

    std::cout << "Pool2 dimensions: " << pool2_dims.n << "x" << pool2_dims.c 
              << "x" << pool2_dims.h << "x" << pool2_dims.w << std::endl;

    // Calculate flattened size for FC1 input
    fc1_input_size = pool2_dims.c * pool2_dims.h * pool2_dims.w;

    // Set up the flattened pool2 descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_flat_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, batch_size, fc1_input_size, 1, 1));

    std::cout << "Pool2 flattened dimensions: " << batch_size << "x" 
              << fc1_input_size << "x1x1" << std::endl;

    // FC1 setup
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc1_weight_desc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, 128, fc1_input_size, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc1_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, batch_size, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, 128, 1, 1));

    // Find best algorithm for FC1
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        pool2_flat_desc,
        fc1_weight_desc,
        fc1_desc,
        fc1_output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &perfResults));

    // FC2 setup
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc2_weight_desc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, 10, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc2_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, batch_size, 10, 1, 1));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, 10, 1, 1));

    // Find best algorithm for FC2
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        fc1_output_desc,
        fc2_weight_desc,
        fc2_desc,
        fc2_output_desc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &perfResults));

    // ReLU activation setup
    CUDNN_CHECK(cudnnSetActivationDescriptor(relu_activation,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0));

    // Calculate workspace sizes for all operations
    size_t workspace_sizes[4];
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_desc, conv1_filter_desc, conv1_desc, conv1_output_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &workspace_sizes[0]));
    
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        pool1_output_desc, conv2_filter_desc, conv2_desc, conv2_output_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &workspace_sizes[1]));
    
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        pool2_flat_desc, fc1_weight_desc, fc1_desc, fc1_output_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_sizes[2]));
    
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        fc1_output_desc, fc2_weight_desc, fc2_desc, fc2_output_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_sizes[3]));

    // Find maximum workspace size needed
    workspace_size = *std::max_element(workspace_sizes, workspace_sizes + 4);
    cudaMalloc(&d_workspace, workspace_size);

    // Allocate memory for layer outputs
    size_t input_bytes = batch_size * 3 * 32 * 32 * sizeof(float);
    size_t conv1_output_bytes = batch_size * conv1_dims.c * conv1_dims.h * conv1_dims.w * sizeof(float);
    size_t pool1_output_bytes = batch_size * pool1_dims.c * pool1_dims.h * pool1_dims.w * sizeof(float);
    size_t conv2_output_bytes = batch_size * conv2_dims.c * conv2_dims.h * conv2_dims.w * sizeof(float);
    size_t pool2_output_bytes = batch_size * pool2_dims.c * pool2_dims.h * pool2_dims.w * sizeof(float);
    size_t fc1_output_bytes = batch_size * 128 * sizeof(float);
    size_t fc2_output_bytes = batch_size * 10 * sizeof(float);

    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_conv1_output, conv1_output_bytes);
    cudaMalloc(&d_pool1_output, pool1_output_bytes);
    cudaMalloc(&d_conv2_output, conv2_output_bytes);
    cudaMalloc(&d_pool2_output, pool2_output_bytes);
    cudaMalloc(&d_fc1_output, fc1_output_bytes);
    cudaMalloc(&d_fc2_output, fc2_output_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error after memory allocation: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA memory allocation failed");
    }

    std::cout << "Layer initialization complete." << std::endl;
    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;
}

void checkTensorDimensions(cudnnTensorDescriptor_t desc, const char* name) {
    int n, c, h, w, nStride, cStride, hStride, wStride;
    cudnnDataType_t dtype;
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(desc, &dtype, &n, &c, &h, &w,
                              &nStride, &cStride, &hStride, &wStride));
    std::cout << name << " tensor dimensions: " << n << "x" << c << "x" << h << "x" << w << std::endl;
}

void checkFilterDimensions(cudnnFilterDescriptor_t desc, const char* name) {
    int k, c, h, w;
    cudnnDataType_t dtype;
    cudnnTensorFormat_t format;
    CUDNN_CHECK(cudnnGetFilter4dDescriptor(desc, &dtype, &format, &k, &c, &h, &w));
    std::cout << name << " filter dimensions: " << k << "x" << c << "x" << h << "x" << w << std::endl;
}

void CUDACNNInference::infer(const std::vector<float>& input_data) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Verify input data size and copy to device
    size_t expected_input_size = batch_size * 3 * 32 * 32;
    if (input_data.size() != expected_input_size) {
        throw std::runtime_error("Input data size mismatch");
    }

    cudaMemcpy(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Conv1 layer
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, 
        input_desc, d_input,
        conv1_filter_desc, d_conv1_weight,
        conv1_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        d_workspace, workspace_size,
        &beta, conv1_output_desc, d_conv1_output));
    
    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        &alpha,
        conv1_bias_desc, d_conv1_bias,
        &alpha,  // Important: using alpha here, not beta
        conv1_output_desc, d_conv1_output));
    
    // ReLU
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, conv1_output_desc, d_conv1_output,
        &beta, conv1_output_desc, d_conv1_output));

    // MaxPool1
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        &alpha, conv1_output_desc, d_conv1_output,
        &beta, pool1_output_desc, d_pool1_output));

    // Conv2 layer
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        pool1_output_desc, d_pool1_output,
        conv2_filter_desc, d_conv2_weight,
        conv2_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        d_workspace, workspace_size,
        &beta, conv2_output_desc, d_conv2_output));
    
    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        &alpha,
        conv2_bias_desc, d_conv2_bias,
        &alpha,  // Important: using alpha here, not beta
        conv2_output_desc, d_conv2_output));
    
    // ReLU
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, conv2_output_desc, d_conv2_output,
        &beta, conv2_output_desc, d_conv2_output));

    // MaxPool2
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        &alpha, conv2_output_desc, d_conv2_output,
        &beta, pool2_output_desc, d_pool2_output));

    // FC1 layer
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        pool2_flat_desc, d_pool2_output,
        fc1_weight_desc, d_fc1_weight,
        fc1_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        d_workspace, workspace_size,
        &beta, fc1_output_desc, d_fc1_output));
    
    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        &alpha,
        fc1_bias_desc, d_fc1_bias,
        &alpha,  // Important: using alpha here, not beta
        fc1_output_desc, d_fc1_output));
    
    // ReLU
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        &alpha, fc1_output_desc, d_fc1_output,
        &beta, fc1_output_desc, d_fc1_output));

    // FC2 layer (final layer)
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
        fc1_output_desc, d_fc1_output,
        fc2_weight_desc, d_fc2_weight,
        fc2_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        d_workspace, workspace_size,
        &beta, fc2_output_desc, d_fc2_output));
    
    // Add final bias
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        &alpha,
        fc2_bias_desc, d_fc2_bias,
        &alpha,  // Important: using alpha here, not beta
        fc2_output_desc, d_fc2_output));

    // Check for any CUDA errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error during inference: ") + 
                               cudaGetErrorString(cuda_status));
    }
}


std::vector<float> CUDACNNInference::getOutput() {
    std::vector<float> output(10);
    
    // Copy the output from device to host
    cudaError_t status = cudaMemcpy(output.data(), d_fc2_output, 
                                   output.size() * sizeof(float), 
                                   cudaMemcpyDeviceToHost);
    
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to copy output from device: ") + 
                               cudaGetErrorString(status));
    }
    
    // Apply softmax normalization
    float max_val = *std::max_element(output.begin(), output.end());
    float sum = 0.0f;
    
    // Subtract max for numerical stability and compute exp
    for (float& val : output) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    // Normalize
    for (float& val : output) {
        val /= sum;
    }
    
    return output;
}

int main() {
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

        // Repeat the dataset to increase the total number of images
        int repeat_factor = 10; // Adjust this factor as needed to increase the dataset size
        std::vector<std::vector<float>> repeated_images;
        std::vector<int> repeated_labels;

        for (int i = 0; i < repeat_factor; ++i) {
            repeated_images.insert(repeated_images.end(), images.begin(), images.end());
            repeated_labels.insert(repeated_labels.end(), validation_labels.begin(), validation_labels.end());
        }

        int total_images = repeated_images.size();

        std::cout << "Total images after repeating: " << total_images << std::endl;

        std::cout << "Creating CUDA inference engine..." << std::endl;
        CUDACNNInference cnn;

        std::cout << "\n=== Starting Evaluation ===" << std::endl;
        std::cout << "Model type: CUDA Core" << std::endl;

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

                //     // Print running statistics (debugging)
                //     // float running_accuracy = (static_cast<float>(correct_count) / (i + 1)) * 100.0f;
                //     // std::cout << "\nProcessed " << i + 1 << "/" << total_images << " images" << std::endl;
                //     // std::cout << "Running accuracy: " << std::fixed << std::setprecision(2) 
                //     //           << running_accuracy << "%" << std::endl;
                //     // std::cout << "Current inference time: " << std::fixed << std::setprecision(3) 
                //     //           << milliseconds << " ms" << std::endl;
                    
                //     // Print top 5 predictions for current image (debugging)
                //     // std::vector<std::pair<int, float>> scores;
                //     // for (size_t j = 0; j < output.size(); ++j) {
                //     //     scores.emplace_back(j, output[j]);
                //     // }
                //     // std::sort(scores.begin(), scores.end(),
                //     //           [](const auto& a, const auto& b) { return a.second > b.second; });
                    
                //     // std::cout << "Top 5 predictions for current image:" << std::endl;
                //     // for (int k = 0; k < std::min(5, static_cast<int>(scores.size())); ++k) {
                //     //     std::cout << "  Class " << std::setw(2) << scores[k].first 
                //     //               << ": " << std::fixed << std::setprecision(4) 
                //     //               << (scores[k].second * 100.0f) << "%" << std::endl;
                //     // }
                //     // std::cout << "True label: " << repeated_labels[i] << std::endl;
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
        std::cout << "Model type: CUDA Core" << std::endl;
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