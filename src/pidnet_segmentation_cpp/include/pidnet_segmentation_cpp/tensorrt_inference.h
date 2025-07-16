#ifndef TENSORRT_INFERENCE_H
#define TENSORRT_INFERENCE_H

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <npp.h> 

// --- Forward declare your CUDA kernels ---
// Wrap in extern "C" to prevent C++ name mangling
#ifdef __cplusplus
extern "C" {
#endif

void preprocess_kernel(const uint8_t* src, float* dst, int width, int height, const float3 mean, const float3 std);
void argmax_kernel(const float* src, uint8_t* dst, int width, int height, int num_classes);
void argmax_with_confidence_kernel(const float* src, uchar2* dst, int width, int height, int num_classes);

#ifdef __cplusplus
}
#endif

class TensorRTInference {
public:
    TensorRTInference();
    ~TensorRTInference();
    
    bool loadEngine(const std::string& engine_path);
    // CPU Path
    cv::Mat preprocess(const cv::Mat& frame);
    cv::Mat inference(const cv::Mat& preprocessed_frame);
    cv::Mat postprocess(const cv::Mat& raw_output, const cv::Size& original_size);
    
    // GPU Path
    void inference_gpu();
    cv::Mat preprocess_gpu(const cv::Mat& frame);
    cv::Mat postprocess_gpu(const cv::Mat& raw_output, const cv::Size& original_size);
    cv::Mat postprocess_gpu_with_confidence(const cv::Mat& raw_output, const cv::Size& original_size);

    // Common
    cv::Mat applyColormap(const cv::Mat& segmentation_map);

private:
    // Reordena los miembros para que coincidan con el orden de inicialización
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    void* input_memory_gpu_;
    void* output_memory_gpu_;
    float* output_buffer_cpu_;
    uint8_t* gpu_input_buffer_;
    uint8_t* gpu_map_low_res_;
    uint8_t* gpu_map_high_res_;
    uint8_t* cpu_final_map_;
    cudaStream_t stream_;
    int input_height_;
    int input_width_;
    int output_height_;
    int output_width_;
    int num_classes_;
    const float mean_[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3] = {0.229f, 0.224f, 0.225f};
    static const uint8_t colormap_[20][3];
    // --- Buffer para logits reescalados en alta resolución ---
    float* gpu_logits_high_res_;
    uchar2* gpu_combined_map_;
    uchar2* cpu_combined_map_;
};

#endif // TENSORRT_INFERENCE_H