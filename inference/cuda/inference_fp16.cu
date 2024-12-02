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
#include <sstream>
#include <algorithm>

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
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_elements = file_size / sizeof(T);
    
    std::cout << "Loading " << filename << " - File size: " << file_size 
              << " bytes, Elements: " << num_elements << std::endl;
    
    std::vector<T> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
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

__global__ void floatToHalf(float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void halfToFloat(half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
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
    half *d_conv1_weight, *d_conv1_bias, *d_conv1_output;
    half *d_pool1_output;
    half *d_conv2_weight, *d_conv2_bias, *d_conv2_output;
    half *d_pool2_output;
    half *d_fc1_weight, *d_fc1_bias, *d_fc1_output;
    half *d_fc2_weight, *d_fc2_bias, *d_fc2_output;

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

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv1_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv2_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc1_bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&fc2_bias_desc));
    
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

void CUDACNNInference::loadWeights(const std::string& weights_path) {
    std::cout << "Loading FP16 model weights..." << std::endl;
    
    // Build full paths to the weight files
    std::string conv1_weight_path = weights_path + "/conv1.weight_fp16.bin";
    std::string conv1_bias_path = weights_path + "/conv1.bias_fp16.bin";
    std::string conv2_weight_path = weights_path + "/conv2.weight_fp16.bin";
    std::string conv2_bias_path = weights_path + "/conv2.bias_fp16.bin";
    std::string fc1_weight_path = weights_path + "/fc1.weight_fp16.bin";
    std::string fc1_bias_path = weights_path + "/fc1.bias_fp16.bin";
    std::string fc2_weight_path = weights_path + "/fc2.weight_fp16.bin";
    std::string fc2_bias_path = weights_path + "/fc2.bias_fp16.bin";

    // Load weights and biases
    auto conv1_weights = loadBinaryFile<half>(conv1_weight_path);
    auto conv1_biases = loadBinaryFile<half>(conv1_bias_path);
    auto conv2_weights = loadBinaryFile<half>(conv2_weight_path);
    auto conv2_biases = loadBinaryFile<half>(conv2_bias_path);
    auto fc1_weights = loadBinaryFile<half>(fc1_weight_path);
    auto fc1_biases = loadBinaryFile<half>(fc1_bias_path);
    auto fc2_weights = loadBinaryFile<half>(fc2_weight_path);
    auto fc2_biases = loadBinaryFile<half>(fc2_bias_path);

    // Verify sizes based on model architecture
    const size_t conv1_weights_size = 32 * 3 * 3 * 3;
    const size_t conv1_bias_size = 32;
    const size_t conv2_weights_size = 64 * 32 * 3 * 3;
    const size_t conv2_bias_size = 64;
    const size_t fc1_weights_size = 128 * (64 * 8 * 8);
    const size_t fc1_bias_size = 128;
    const size_t fc2_weights_size = 10 * 128;
    const size_t fc2_bias_size = 10;
    
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

    std::cout << "Successfully loaded all weights to GPU." << std::endl;
}

void CUDACNNInference::initializeLayers() {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_HALF, batch_size, 3, 32, 32));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv1_filter_desc, CUDNN_DATA_HALF, 
        CUDNN_TENSOR_NCHW, 32, 3, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_HALF));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv1_desc, CUDNN_DEFAULT_MATH));

    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv1_desc, input_desc, 
        conv1_filter_desc, &conv1_dims.n, &conv1_dims.c, &conv1_dims.h, &conv1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, conv1_dims.n, conv1_dims.c, conv1_dims.h, conv1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, 1, conv1_dims.c, 1, 1));

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc, 
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        2, 2,    // window size
        0, 0,    // padding
        2, 2));  // stride

    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pooling_desc,
        conv1_output_desc,
        &pool1_dims.n, &pool1_dims.c, &pool1_dims.h, &pool1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, pool1_dims.n, pool1_dims.c, pool1_dims.h, pool1_dims.w));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv2_filter_desc, CUDNN_DATA_HALF, 
        CUDNN_TENSOR_NCHW, 64, 32, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_HALF));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv2_desc, CUDNN_DEFAULT_MATH));
    
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv2_desc, pool1_output_desc, 
        conv2_filter_desc, &conv2_dims.n, &conv2_dims.c, &conv2_dims.h, &conv2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, conv2_dims.n, conv2_dims.c, conv2_dims.h, conv2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, 1, conv2_dims.c, 1, 1));

    pool2_dims = {conv2_dims.n, conv2_dims.c, conv2_dims.h/2, conv2_dims.w/2};
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, pool2_dims.n, pool2_dims.c, pool2_dims.h, pool2_dims.w));

    fc1_input_size = pool2_dims.c * pool2_dims.h * pool2_dims.w;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_flat_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, batch_size, fc1_input_size, 1, 1));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc1_weight_desc, CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW, 128, fc1_input_size, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc1_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF));
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc1_desc, CUDNN_DEFAULT_MATH));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, batch_size, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, 1, 128, 1, 1));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc2_weight_desc, CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW, 10, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc2_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF));
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc2_desc, CUDNN_DEFAULT_MATH));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, batch_size, 10, 1, 1));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF, 1, 10, 1, 1));

    CUDNN_CHECK(cudnnSetActivationDescriptor(relu_activation,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0));

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

    workspace_size = *std::max_element(workspace_sizes, workspace_sizes + 4);
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    size_t input_bytes = batch_size * 3 * 32 * 32 * sizeof(float);
    size_t conv1_output_bytes = batch_size * conv1_dims.c * conv1_dims.h * conv1_dims.w * sizeof(half);
    size_t pool1_output_bytes = batch_size * pool1_dims.c * pool1_dims.h * pool1_dims.w * sizeof(half);
    size_t conv2_output_bytes = batch_size * conv2_dims.c * conv2_dims.h * conv2_dims.w * sizeof(half);
    size_t pool2_output_bytes = batch_size * pool2_dims.c * pool2_dims.h * pool2_dims.w * sizeof(half);
    size_t fc1_output_bytes = batch_size * 128 * sizeof(half);
    size_t fc2_output_bytes = batch_size * 10 * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_conv1_output, conv1_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_pool1_output, pool1_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_conv2_output, conv2_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_pool2_output, pool2_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_fc1_output, fc1_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_fc2_output, fc2_output_bytes));

    std::cout << "Layer initialization complete." << std::endl;
    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;
}

void CUDACNNInference::infer(const std::vector<float>& input_data) {
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    const void* alpha_ptr = &alpha_f;
    const void* beta_ptr = &beta_f;
    
    size_t expected_input_size = batch_size * 3 * 32 * 32;
    if (input_data.size() != expected_input_size) {
        throw std::runtime_error("Input data size mismatch");
    }

    float* d_input_float;
    CUDA_CHECK(cudaMalloc(&d_input_float, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input_float, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (input_data.size() + blockSize - 1) / blockSize;
    floatToHalf<<<numBlocks, blockSize>>>(d_input_float, (half*)d_input, input_data.size());
    CHECK_CUDA_ERROR(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_input_float));

    CUDNN_CHECK(cudnnConvolutionForward(cudnn, alpha_ptr, 
        input_desc, d_input,
        conv1_filter_desc, d_conv1_weight,
        conv1_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        d_workspace, workspace_size,
        beta_ptr, conv1_output_desc, d_conv1_output));
    
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        alpha_ptr,
        conv1_bias_desc, d_conv1_bias,
        alpha_ptr,
        conv1_output_desc, d_conv1_output));
    
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        alpha_ptr, conv1_output_desc, d_conv1_output,
        beta_ptr, conv1_output_desc, d_conv1_output));

    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        alpha_ptr, conv1_output_desc, d_conv1_output,
        beta_ptr, pool1_output_desc, d_pool1_output));

    CUDNN_CHECK(cudnnConvolutionForward(cudnn, alpha_ptr,
        pool1_output_desc, d_pool1_output,
        conv2_filter_desc, d_conv2_weight,
        conv2_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        d_workspace, workspace_size,
        beta_ptr, conv2_output_desc, d_conv2_output));
    
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        alpha_ptr,
        conv2_bias_desc, d_conv2_bias,
        alpha_ptr,
        conv2_output_desc, d_conv2_output));
    
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        alpha_ptr, conv2_output_desc, d_conv2_output,
        beta_ptr, conv2_output_desc, d_conv2_output));

    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc,
        alpha_ptr, conv2_output_desc, d_conv2_output,
        beta_ptr, pool2_output_desc, d_pool2_output));

    CUDNN_CHECK(cudnnConvolutionForward(cudnn, alpha_ptr,
        pool2_flat_desc, d_pool2_output,
        fc1_weight_desc, d_fc1_weight,
        fc1_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        d_workspace, workspace_size,
        beta_ptr, fc1_output_desc, d_fc1_output));
    
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        alpha_ptr,
        fc1_bias_desc, d_fc1_bias,
        alpha_ptr,
        fc1_output_desc, d_fc1_output));
    
    CUDNN_CHECK(cudnnActivationForward(cudnn, relu_activation,
        alpha_ptr, fc1_output_desc, d_fc1_output,
        beta_ptr, fc1_output_desc, d_fc1_output));

    CUDNN_CHECK(cudnnConvolutionForward(cudnn, alpha_ptr,
        fc1_output_desc, d_fc1_output,
        fc2_weight_desc, d_fc2_weight,
        fc2_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        d_workspace, workspace_size,
        beta_ptr, fc2_output_desc, d_fc2_output));
    
    CUDNN_CHECK(cudnnAddTensor(cudnn, 
        alpha_ptr,
        fc2_bias_desc, d_fc2_bias,
        alpha_ptr,
        fc2_output_desc, d_fc2_output));

    CHECK_CUDA_ERROR(cudaGetLastError());
}

std::vector<float> CUDACNNInference::getOutput() {
    size_t output_size = batch_size * 10;
    std::vector<float> output(output_size);
    
    float* d_output_float;
    CUDA_CHECK(cudaMalloc(&d_output_float, output_size * sizeof(float)));
    
    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;
    halfToFloat<<<numBlocks, blockSize>>>((half*)d_fc2_output, d_output_float, output_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(output.data(), d_output_float, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output_float));
    
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

CUDACNNInference::~CUDACNNInference() {
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

void parseArguments(int argc, char** argv, int& gpu_id, int& repeat_factor, size_t& batch_size, std::string& data_path, std::string& weights_path) {
    if (argc >= 6) {
        gpu_id = std::atoi(argv[1]);
        repeat_factor = std::atoi(argv[2]);
        batch_size = std::stoul(argv[3]);  // Use std::stoul for size_t
        data_path = argv[4];
        weights_path = argv[5];
    } else {
        std::cerr << "Usage: " << argv[0] << " <gpu_id> <repeat_factor> <batch_size> <data_path> <weights_path>" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


int main(int argc, char** argv) {
    int gpu_id = 0;
    int repeat_factor = 1;
    size_t batch_size = 256;// Default value
    std::string data_path;
    std::string weights_path;

    parseArguments(argc, argv, gpu_id, repeat_factor, batch_size, data_path, weights_path);
    CUDA_CHECK(cudaSetDevice(gpu_id));

    std::cout << "Running on GPU: " << gpu_id << std::endl;
    std::cout << "Repeat factor: " << repeat_factor << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    try {
        std::cout << "Loading validation data..." << std::endl;

        // Build paths to validation data files
        std::string validation_images_path = data_path + "/validation_images.bin";
        std::string validation_labels_path = data_path + "/validation_labels.bin";

        auto validation_images = loadBinaryFile<float>(validation_images_path);
        auto validation_labels = loadBinaryFile<int>(validation_labels_path);

        size_t image_size = 3 * 32 * 32;
        std::vector<std::vector<float>> images;
        for (size_t i = 0; i < validation_images.size(); i += image_size) {
            images.emplace_back(validation_images.begin() + i,
                                validation_images.begin() + i + image_size);
        }

        // Repeat images and labels
        std::vector<std::vector<float>> repeated_images;
        std::vector<int> repeated_labels;

        for (int i = 0; i < repeat_factor; ++i) {
            repeated_images.insert(repeated_images.end(), images.begin(), images.end());
            repeated_labels.insert(repeated_labels.end(), validation_labels.begin(), validation_labels.end());
        }

        size_t total_images = repeated_images.size();
        std::cout << "Total images after repeating: " << total_images << std::endl;

        // Create CNN object with batch_size and weights_path
        CUDACNNInference cnn(batch_size, weights_path);

        std::cout << "\n=== Starting Evaluation ===" << std::endl;
        std::cout << "Model type: CUDA FP16" << std::endl;

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
        std::cout << "Model type: CUDA FP16" << std::endl;
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
