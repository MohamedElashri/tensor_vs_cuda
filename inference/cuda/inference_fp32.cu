#include </usr/include/cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Error checking macros
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file,
           const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

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

class CUDACNNInference {
public:
    CUDACNNInference(int batch_size, const std::string& weights_path);
    ~CUDACNNInference();
    void loadWeights(const std::string& weights_path);
    void initializeLayers();
    void infer(const std::vector<float>& input_data);
    std::vector<float> getOutput();

private:
    cudnnHandle_t cudnn;
    
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
    
    cudnnTensorDescriptor_t conv1_bias_desc;
    cudnnTensorDescriptor_t conv2_bias_desc;
    cudnnTensorDescriptor_t fc1_bias_desc;
    cudnnTensorDescriptor_t fc2_bias_desc;
    
    cudnnFilterDescriptor_t conv1_filter_desc;
    cudnnFilterDescriptor_t conv2_filter_desc;
    cudnnConvolutionDescriptor_t conv1_desc;
    cudnnConvolutionDescriptor_t conv2_desc;
    cudnnConvolutionDescriptor_t fc1_desc;
    cudnnConvolutionDescriptor_t fc2_desc;
    
    cudnnActivationDescriptor_t relu_activation;
    cudnnPoolingDescriptor_t pooling_desc;

    int fc1_input_size;

    float *d_input;
    float *d_conv1_weight, *d_conv1_bias, *d_conv1_output;
    float *d_pool1_output;
    float *d_conv2_weight, *d_conv2_bias, *d_conv2_output;
    float *d_pool2_output;
    float *d_fc1_weight, *d_fc1_bias, *d_fc1_output;
    float *d_fc2_weight, *d_fc2_bias, *d_fc2_output;

    size_t workspace_size;
    void *d_workspace;

    int batch_size;

    struct LayerDims {
        int n, c, h, w;
    };
    LayerDims conv1_dims, pool1_dims, conv2_dims, pool2_dims, fc1_dims, fc2_dims;
};

CUDACNNInference::CUDACNNInference(int batch_size_, const std::string& weights_path)
    : batch_size(batch_size_) {
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

    loadWeights(weights_path);
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

void CUDACNNInference::loadWeights(const std::string& weights_path) {
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

    const size_t conv1_weights_size = 32 * 3 * 3 * 3;
    const size_t conv1_bias_size = 32;

    if (conv1_weights.size() != conv1_weights_size || conv1_biases.size() != conv1_bias_size) {
        std::cerr << "Error: Conv1 weight/bias size mismatch!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Conv2 weights
    auto conv2_weights = loadBinaryFile<float>(conv2_weight_path);
    auto conv2_biases = loadBinaryFile<float>(conv2_bias_path);

    const size_t conv2_weights_size = 64 * 32 * 3 * 3;
    const size_t conv2_bias_size = 64;

    // FC1 weights
    auto fc1_weights = loadBinaryFile<float>(fc1_weight_path);
    auto fc1_biases = loadBinaryFile<float>(fc1_bias_path);

    const size_t fc1_weights_size = 128 * (64 * 8 * 8);
    const size_t fc1_bias_size = 128;

    // FC2 weights
    auto fc2_weights = loadBinaryFile<float>(fc2_weight_path);
    auto fc2_biases = loadBinaryFile<float>(fc2_bias_path);

    const size_t fc2_weights_size = 10 * 128;
    const size_t fc2_bias_size = 10;

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
    size_t output_size = batch_size * 10;
    std::vector<float> output(output_size);
    
    // Copy the output from device to host
    cudaError_t status = cudaMemcpy(output.data(), d_fc2_output, 
                                   output.size() * sizeof(float), 
                                   cudaMemcpyDeviceToHost);
    
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to copy output from device: ") + 
                               cudaGetErrorString(status));
    }
    
    // Apply softmax normalization for each sample
    for (int i = 0; i < batch_size; ++i) {
        float max_val = *std::max_element(output.begin() + i * 10, output.begin() + (i + 1) * 10);
        float sum = 0.0f;

        // Subtract max for numerical stability and compute exp
        for (int j = 0; j < 10; ++j) {
            output[i * 10 + j] = std::exp(output[i * 10 + j] - max_val);
            sum += output[i * 10 + j];
        }

        // Normalize
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
    int batch_size = 256;  // Default value
    std::string data_path;
    std::string weights_path;

    parseArguments(argc, argv, gpu_id, repeat_factor, batch_size, data_path, weights_path);
    CUDA_CHECK(cudaSetDevice(gpu_id));

    std::cout << "Running on GPU: " << gpu_id << std::endl;
    std::cout << "Repeat factor: " << repeat_factor << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    try {
        std::cout << "Loading validation data..." << std::endl;
        std::string validation_images_path = data_path + "/validation_images.bin";
        std::string validation_labels_path = data_path + "/validation_labels.bin";

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

        CUDACNNInference cnn(batch_size, weights_path);

        std::cout << "\n=== Starting Evaluation ===" << std::endl;
        std::cout << "Model type: CUDA FP32" << std::endl;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        size_t correct_count = 0;
        float total_time = 0.0f;

        // Warmup run with first batch
        std::cout << "Performing warmup runs..." << std::endl;
        std::vector<float> warmup_batch;
        warmup_batch.reserve(batch_size * image_size);
        for (int i = 0; i < batch_size && i < total_images; ++i) {
            warmup_batch.insert(warmup_batch.end(), repeated_images[i].begin(), repeated_images[i].end());
        }
        for (int i = 0; i < 10; i++) {
            cnn.infer(warmup_batch);
        }

        // Main evaluation loop with batching
        std::cout << "Starting main evaluation..." << std::endl;
        size_t total_batches = (total_images + batch_size - 1) / batch_size;
        
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
                         << " images. Running accuracy: " << std::fixed 
                         << std::setprecision(2) << running_accuracy << "%" << std::endl;
            }
        }

        float accuracy = static_cast<float>(correct_count) / total_images * 100.0f;
        float avg_time = total_time / total_batches;  // Average time per batch
        float throughput = (batch_size * 1000.0f) / avg_time;  // Images per second

        std::cout << "\n=== Final Results ===" << std::endl;
        std::cout << "Model type: CUDA FP32" << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Total images: " << total_images << std::endl;
        std::cout << "Correct predictions: " << correct_count << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "Average batch inference time: " << std::fixed << std::setprecision(3) 
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
