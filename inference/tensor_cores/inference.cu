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

class TensorCoreCNNInference {
public:
    TensorCoreCNNInference();
    ~TensorCoreCNNInference();
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
    float *d_conv1_weight, *d_conv1_bias, *d_conv1_output;
    float *d_pool1_output;
    float *d_conv2_weight, *d_conv2_bias, *d_conv2_output;
    float *d_pool2_output;
    float *d_fc1_weight, *d_fc1_bias, *d_fc1_output;
    float *d_fc2_weight, *d_fc2_bias, *d_fc2_output;

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

TensorCoreCNNInference::TensorCoreCNNInference() : batch_size(1) {
    std::cout << "Initializing TensorCore CNN..." << std::endl;
    
    // Create cuDNN handle
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

    loadWeights();
    initializeLayers();
    checkTensorCoreUsage();
}

cudnnConvolutionFwdAlgo_t TensorCoreCNNInference::findBestConvAlgorithm(
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
            perfResults[i].mathType == CUDNN_TENSOR_OP_MATH &&
            perfResults[i].time < bestTime) {
            bestTime = perfResults[i].time;
            bestAlgo = perfResults[i].algo;
            *workspace_size = std::max(*workspace_size, perfResults[i].memory);
        }
    }

    return bestAlgo;
}



void TensorCoreCNNInference::initializeLayers() {
    // Input layer: 3x32x32
    input_dims = {batch_size, 3, 32, 32};
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, input_dims.n, input_dims.c, input_dims.h, input_dims.w));

    // Conv1 layer setup
    conv1_dims = {batch_size, 32, 32, 32};  // Output size after padding
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv1_filter_desc, CUDNN_DATA_FLOAT, 
        CUDNN_TENSOR_NCHW, 32, 3, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_FLOAT));
    
    // Enable Tensor Core operation for conv1
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv1_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, conv1_dims.n, conv1_dims.c, conv1_dims.h, conv1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, conv1_dims.c, 1, 1));

    // Pooling setup
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc, 
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        2, 2,    // window height, width
        0, 0,    // padding height, width
        2, 2));  // stride height, width

    // Get Pool1 dimensions
    pool1_dims = {conv1_dims.n, conv1_dims.c, conv1_dims.h/2, conv1_dims.w/2};
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, pool1_dims.n, pool1_dims.c, pool1_dims.h, pool1_dims.w));

    // Conv2 setup
    conv2_dims = {pool1_dims.n, 64, pool1_dims.h, pool1_dims.w};
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv2_filter_desc, CUDNN_DATA_FLOAT, 
        CUDNN_TENSOR_NCHW, 64, 32, 3, 3));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2_desc, 
        1, 1,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION, 
        CUDNN_DATA_FLOAT));
    
    // Enable Tensor Core operation for conv2
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv2_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, conv2_dims.n, conv2_dims.c, conv2_dims.h, conv2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, conv2_dims.c, 1, 1));

    // Pool2 setup
    pool2_dims = {conv2_dims.n, conv2_dims.c, conv2_dims.h/2, conv2_dims.w/2};
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, pool2_dims.n, pool2_dims.c, pool2_dims.h, pool2_dims.w));

    // Setup flattened pool2 for FC layers
    int fc_input_size = pool2_dims.c * pool2_dims.h * pool2_dims.w;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2_flat_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, batch_size, fc_input_size, 1, 1));

    // FC1 setup
    fc1_dims = {batch_size, 128, 1, 1};
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc1_filter_desc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, 128, fc_input_size, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc1_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    
    // Enable Tensor Core operation for FC1
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc1_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, fc1_dims.n, fc1_dims.c, fc1_dims.h, fc1_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc1_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, fc1_dims.c, 1, 1));

    // FC2 setup
    fc2_dims = {batch_size, 10, 1, 1};
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(fc2_filter_desc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, 10, 128, 1, 1));
    
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(fc2_desc,
        0, 0,    // padding
        1, 1,    // stride
        1, 1,    // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    
    // Enable Tensor Core operation for FC2
    CUDNN_CHECK(cudnnSetConvolutionMathType(fc2_desc, CUDNN_TENSOR_OP_MATH));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, fc2_dims.n, fc2_dims.c, fc2_dims.h, fc2_dims.w));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(fc2_bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, fc2_dims.c, 1, 1));

    // ReLU activation setup
    CUDNN_CHECK(cudnnSetActivationDescriptor(relu_activation,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0));

    // Find best algorithms and workspace size for all convolutions
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
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_output, batch_size * conv1_dims.c * conv1_dims.h * conv1_dims.w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_output, batch_size * pool1_dims.c * pool1_dims.h * pool1_dims.w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_output, batch_size * conv2_dims.c * conv2_dims.h * conv2_dims.w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_output, batch_size * pool2_dims.c * pool2_dims.h * pool2_dims.w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_output, batch_size * fc1_dims.c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_output, batch_size * fc2_dims.c * sizeof(float)));

    std::cout << "Layer initialization complete." << std::endl;
    std::cout << "Workspace size: " << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;
}


void TensorCoreCNNInference::checkTensorCoreUsage() {
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

void TensorCoreCNNInference::infer(const std::vector<float>& input_data) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), 
                         input_data.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));

    // Conv1 + ReLU
// Conv1 + ReLU
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
    
    // ReLU activation
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
    
    // ReLU activation
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
    
    // ReLU activation
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
    CUDA_CHECK(cudaGetLastError());
}

std::vector<float> TensorCoreCNNInference::getOutput() {
    std::vector<float> output(10);
    
    // Copy the output from device to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_fc2_output, 
                         output.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
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

void TensorCoreCNNInference::evaluate(const std::vector<std::vector<float>>& images, 
                                    const std::vector<int>& labels) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    size_t correct_count = 0;
    float total_time = 0.0f;
    
    std::cout << "\nStarting evaluation..." << std::endl;
    
    for (size_t i = 0; i < images.size(); ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        
        try {
            infer(images[i]);
            std::vector<float> output = getOutput();
            
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            total_time += milliseconds;
            
            int predicted_label = std::distance(output.begin(), 
                                              std::max_element(output.begin(), output.end()));
            
            if (i % 100 == 0) {
                std::cout << "Processing image " << i << "..." << std::endl;
                std::cout << "Top 5 predictions:" << std::endl;
                
                std::vector<std::pair<int, float>> scores;
                for (size_t j = 0; j < output.size(); ++j) {
                    scores.emplace_back(j, output[j]);
                }
                
                std::sort(scores.begin(), scores.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
                
                for (int k = 0; k < std::min(5, static_cast<int>(scores.size())); ++k) {
                    std::cout << "  Class " << scores[k].first 
                             << ": " << std::fixed << std::setprecision(4) 
                             << scores[k].second * 100.0f << "%" << std::endl;
                }
                
                std::cout << "True label: " << labels[i] << std::endl;
                std::cout << "Inference time: " << milliseconds << " ms" << std::endl;
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

    float accuracy = static_cast<float>(correct_count) / images.size() * 100.0f;
    float avg_time = total_time / images.size();
    float throughput = 1000.0f / avg_time;  // images per second
    
    std::cout << "\nEvaluation Results:" << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "Average inference time: " << std::fixed << std::setprecision(3) 
              << avg_time << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) 
              << throughput << " images/second" << std::endl;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

TensorCoreCNNInference::~TensorCoreCNNInference() {
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

void TensorCoreCNNInference::loadWeights() {
    std::cout << "Loading model weights..." << std::endl;
    
    // Load weights from binary files
    auto conv1_weights = loadBinaryFile<float>("../../../data/weights/conv1.weight_fp32.bin");
    auto conv1_biases = loadBinaryFile<float>("../../../data/weights/conv1.bias_fp32.bin");
    auto conv2_weights = loadBinaryFile<float>("../../../data/weights/conv2.weight_fp32.bin");
    auto conv2_biases = loadBinaryFile<float>("../../../data/weights/conv2.bias_fp32.bin");
    auto fc1_weights = loadBinaryFile<float>("../../../data/weights/fc1.weight_fp32.bin");
    auto fc1_biases = loadBinaryFile<float>("../../../data/weights/fc1.bias_fp32.bin");
    auto fc2_weights = loadBinaryFile<float>("../../../data/weights/fc2.weight_fp32.bin");
    auto fc2_biases = loadBinaryFile<float>("../../../data/weights/fc2.bias_fp32.bin");
    
    // Verify sizes
    const size_t conv1_weights_size = 32 * 3 * 3 * 3;
    const size_t conv1_bias_size = 32;
    const size_t conv2_weights_size = 64 * 32 * 3 * 3;
    const size_t conv2_bias_size = 64;
    const size_t fc1_weights_size = 128 * (64 * 8 * 8);
    const size_t fc1_bias_size = 128;
    const size_t fc2_weights_size = 10 * 128;
    const size_t fc2_bias_size = 10;
    
    // Verify sizes match expected dimensions
    if (conv1_weights.size() != conv1_weights_size ||
        conv1_biases.size() != conv1_bias_size ||
        conv2_weights.size() != conv2_weights_size ||
        conv2_biases.size() != conv2_bias_size ||
        fc1_weights.size() != fc1_weights_size ||
        fc1_biases.size() != fc1_bias_size ||
        fc2_weights.size() != fc2_weights_size ||
        fc2_biases.size() != fc2_bias_size) {
        throw std::runtime_error("Weight file sizes do not match expected dimensions");
    }
    
    // Allocate and copy weights to device
    CUDA_CHECK(cudaMalloc(&d_conv1_weight, conv1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_bias, conv1_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv1_weight, conv1_weights.data(), 
                         conv1_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_bias, conv1_biases.data(), 
                         conv1_bias_size * sizeof(float), cudaMemcpyHostToDevice));
    
CUDA_CHECK(cudaMalloc(&d_conv2_weight, conv2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias, conv2_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv2_weight, conv2_weights.data(),
                         conv2_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_bias, conv2_biases.data(),
                         conv2_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fc1_weight, fc1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_bias, fc1_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc1_weight, fc1_weights.data(),
                         fc1_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc1_bias, fc1_biases.data(),
                         fc1_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fc2_weight, fc2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_bias, fc2_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc2_weight, fc2_weights.data(),
                         fc2_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc2_bias, fc2_biases.data(),
                         fc2_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    // Verify weights were loaded successfully
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error while loading weights: ") +
                               cudaGetErrorString(error));
    }

    std::cout << "Successfully loaded all weights to GPU." << std::endl;
    
    // Print first few weights for verification
    std::cout << "Conv1 weights first values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::fixed << std::setprecision(6) << conv1_weights[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        std::cout << "Loading validation data..." << std::endl;
        auto validation_images = loadBinaryFile<float>("../../../data/validation/validation_images.bin");
        auto validation_labels = loadBinaryFile<int>("../../../data/validation/validation_labels.bin");

        std::vector<std::vector<float>> images;
        const size_t image_size = 3 * 32 * 32;
        for (size_t i = 0; i < validation_images.size(); i += image_size) {
            images.push_back(std::vector<float>(
                validation_images.begin() + i,
                validation_images.begin() + i + image_size));
        }

        std::cout << "Creating TensorCore inference engine..." << std::endl;
        TensorCoreCNNInference cnn_inference;
        
        std::cout << "Running evaluation on validation data..." << std::endl;
        cnn_inference.evaluate(images, validation_labels);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}