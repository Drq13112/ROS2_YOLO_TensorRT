#include "tensorrt_inference.h"
#include <iostream>
#include <fstream>
#include <NvOnnxParser.h>
#include "opencv2/opencv.hpp"

// Simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// Colormap definition fort cityscapes (19 classes)
const uint8_t TensorRTInference::colormap_[20][3] = {
    {128, 64, 128},   // road
    {244, 35, 232},   // sidewalk
    {70, 70, 70},     // building
    {102, 102, 156},  // wall
    {190, 153, 153},  // fence
    {153, 153, 153},  // pole
    {250, 170, 30},   // traffic light
    {220, 220, 0},    // traffic sign
    {107, 142, 35},   // vegetation
    {152, 251, 152},  // terrain
    {70, 130, 180},   // sky
    {220, 20, 60},    // person
    {255, 0, 0},      // rider
    {0, 0, 142},      // car
    {0, 0, 70},       // truck
    {0, 60, 100},     // bus
    {0, 80, 100},     // train
    {0, 0, 230},      // motorcycle
    {119, 11, 32},    // bicycle
    {0, 0, 0}         // background
};

// Colormap definition for Mapillary Vistas (66 classes)
// const uint8_t TensorRTInference::colormap_[66][3] = {
//     {128, 64, 128},   // 0: road
//     {244, 35, 232},   // 1: sidewalk
//     {70, 70, 70},     // 2: building
//     {102, 102, 156},  // 3: wall
//     {190, 153, 153},  // 4: fence
//     {153, 153, 153},  // 5: pole
//     {250, 170, 30},   // 6: traffic light
//     {220, 220, 0},    // 7: traffic sign
//     {107, 142, 35},   // 8: vegetation
//     {152, 251, 152},  // 9: terrain
//     {70, 130, 180},   // 10: sky
//     {220, 20, 60},    // 11: person
//     {255, 0, 0},      // 12: rider
//     {0, 0, 142},      // 13: car
//     {0, 0, 70},       // 14: truck
//     {0, 60, 100},     // 15: bus
//     {0, 80, 100},     // 16: train
//     {0, 0, 230},      // 17: motorcycle
//     {119, 11, 32},    // 18: bicycle
//     {110, 190, 160},  // 19: lane-marking-general
//     {170, 170, 170},  // 20: manhole
//     {81, 0, 81},      // 21: curb
//     {230, 150, 140},  // 22: traffic-cone
//     {180, 165, 180},  // 23: barrier
//     {150, 100, 100},  // 24: crosswalk-zebra
//     {150, 120, 90},   // 25: box
//     {250, 170, 160},  // 26: billboard
//     {255, 255, 255},  // 27: catch-basin
//     {200, 150, 150},  // 28: streetlight
//     {250, 128, 114},  // 29: junction-box
//     {140, 60, 60},    // 30: fire-hydrant
//     {128, 128, 0},    // 31: bike-rack
//     {220, 190, 150},  // 32: phone-booth
//     {128, 0, 128},    // 33: pothole
//     {150, 150, 150},  // 34: parking-meter
//     {128, 128, 128},  // 35: bench
//     {0, 255, 0},      // 36: lane-marking-dashed
//     {0, 0, 255},      // 37: lane-marking-solid
//     {255, 255, 0},    // 38: crosswalk-other
//     {255, 0, 255},    // 39: traffic-sign-front
//     {0, 255, 255},    // 40: traffic-sign-back
//     {192, 192, 192},  // 41: traffic-signal-single
//     {64, 64, 128},    // 42: traffic-signal-frame
//     {128, 0, 0},      // 43: traffic-signal-box
//     {0, 128, 0},      // 44: traffic-signal-light
//     {0, 0, 128},      // 45: traffic-signal-pedestrian
//     {128, 128, 64},   // 46: traffic-sign-warning
//     {128, 64, 0},     // 47: traffic-sign-info
//     {64, 0, 128},     // 48: traffic-sign-other
//     {192, 0, 0},      // 49: traffic-sign-main
//     {0, 192, 0},      // 50: traffic-sign-top
//     {0, 0, 192},      // 51: traffic-sign-direction
//     {192, 192, 0},    // 52: traffic-sign-distance
//     {192, 0, 192},    // 53: traffic-sign-temporary
//     {0, 192, 192},    // 54: traffic-sign-shape
//     {64, 64, 0},      // 55: traffic-sign-text
//     {64, 0, 64},      // 56: traffic-sign-arrow
//     {0, 64, 64},      // 57: traffic-sign-symbol
//     {64, 128, 128},   // 58: traffic-sign-map
//     {128, 64, 128},   // 59: traffic-sign-speed
//     {128, 128, 192},  // 60: traffic-sign-no
//     {192, 64, 64},    // 61: traffic-sign-pedestrian
//     {64, 192, 64},    // 62: traffic-sign-animal
//     {64, 64, 192},    // 63: traffic-sign-bicycle
//     {192, 128, 128},  // 64: traffic-sign-parking
//     {0, 0, 0}         // 65: unlabeled
// };


TensorRTInference::TensorRTInference()
    : input_memory_gpu_(nullptr), output_memory_gpu_(nullptr), output_buffer_cpu_(nullptr), stream_(nullptr),
      gpu_input_buffer_(nullptr), gpu_map_low_res_(nullptr), gpu_map_high_res_(nullptr), cpu_final_map_(nullptr),
      input_height_(1200), input_width_(1920), num_classes_(20) {
    
    output_height_ = input_height_ / 8;
    output_width_ = input_width_ / 8;
    
    cudaStreamCreate(&stream_);

   // Asignar memoria para los nuevos buffers
    cudaMalloc(reinterpret_cast<void**>(&gpu_input_buffer_), input_width_ * input_height_ * 3 * sizeof(uint8_t));
    cudaMalloc(reinterpret_cast<void**>(&gpu_map_low_res_), output_width_ * output_height_ * sizeof(uint8_t));
    cudaMalloc(reinterpret_cast<void**>(&gpu_map_high_res_), input_width_ * input_height_ * sizeof(uint8_t));
    cudaMallocHost(reinterpret_cast<void**>(&cpu_final_map_), input_width_ * input_height_ * sizeof(uint8_t));

    // --- Asignar memoria para los logits de alta resolución ---
    cudaMalloc(reinterpret_cast<void**>(&gpu_logits_high_res_), input_width_ * input_height_ * num_classes_ * sizeof(float));

    // --- Asignar memoria para el mapa combinado (clase + confianza) ---
    // uchar2 contiene dos uint8_t, perfecto para nuestro caso.
    size_t combined_map_size = input_width_ * input_height_ * sizeof(uchar2);
    cudaMalloc(reinterpret_cast<void**>(&gpu_combined_map_), combined_map_size);
    cudaMallocHost(reinterpret_cast<void**>(&cpu_combined_map_), combined_map_size);
}

TensorRTInference::~TensorRTInference() {
    if (stream_) cudaStreamDestroy(stream_);
    if (input_memory_gpu_) cudaFree(input_memory_gpu_);
    if (output_memory_gpu_) cudaFree(output_memory_gpu_);
    if (output_buffer_cpu_) cudaFreeHost(output_buffer_cpu_);
    if (gpu_input_buffer_) cudaFree(gpu_input_buffer_);
    if (gpu_map_low_res_) cudaFree(gpu_map_low_res_);
    if (gpu_map_high_res_) cudaFree(gpu_map_high_res_);
    if (cpu_final_map_) cudaFreeHost(cpu_final_map_);
    if (gpu_logits_high_res_) cudaFree(gpu_logits_high_res_);
    if (gpu_combined_map_) cudaFree(gpu_combined_map_);
    if (cpu_combined_map_) cudaFreeHost(cpu_combined_map_);
    // Smart pointers se liberan solos
}

bool TensorRTInference::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: could not open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> trt_model_stream(size);
    file.read(trt_model_stream.data(), size);
    file.close();

    Logger logger;
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    engine_.reset(runtime_->deserializeCudaEngine(trt_model_stream.data(), size));
    if (!engine_) {
        std::cerr << "Error: failed to deserialize engine." << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Error: failed to create execution context." << std::endl;
        return false;
    }

    // Asignar memoria
    auto input_dims = nvinfer1::Dims4{1, 3, input_height_, input_width_};
    context_->setInputShape(engine_->getIOTensorName(0), input_dims);

    size_t input_size = 1 * 3 * input_height_ * input_width_ * sizeof(float);
    size_t output_size = 1 * num_classes_ * output_height_ * output_width_ * sizeof(float);

    cudaMalloc(&input_memory_gpu_, input_size);
    cudaMalloc(&output_memory_gpu_, output_size);
    cudaMallocHost(reinterpret_cast<void**>(&output_buffer_cpu_), output_size);

    context_->setTensorAddress(engine_->getIOTensorName(0), input_memory_gpu_);
    context_->setTensorAddress(engine_->getIOTensorName(1), output_memory_gpu_);

    return true;
}

cv::Mat TensorRTInference::preprocess(const cv::Mat& frame) {
    cv::Mat resized_frame, rgb_frame, float_frame;
    
    // 1. Redimensionar
    cv::resize(frame, resized_frame, cv::Size(input_width_, input_height_));
    
    // 2. BGR a RGB
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
    
    // 3. Convertir a float y normalizar
    rgb_frame.convertTo(float_frame, CV_32FC3, 1.0 / 255.0);
    cv::subtract(float_frame, cv::Scalar(mean_[0], mean_[1], mean_[2]), float_frame);
    cv::divide(float_frame, cv::Scalar(std_[0], std_[1], std_[2]), float_frame);

    // 4. Transponer de HWC a CHW manualmente
    // Crear un buffer lineal para CHW (Canal, Alto, Ancho)
    cv::Mat chw_frame(1, 3 * input_height_ * input_width_, CV_32F);
    float* chw_data = chw_frame.ptr<float>();
    
    // Separar canales
    std::vector<cv::Mat> channels(3);
    cv::split(float_frame, channels);
    
    // Copiar cada canal completo secuencialmente: R, G, B
    size_t channel_size = input_height_ * input_width_;
    
    // Canal R (índice 0)
    std::memcpy(chw_data, channels[0].ptr<float>(), channel_size * sizeof(float));
    
    // Canal G (índice 1)
    std::memcpy(chw_data + channel_size, channels[1].ptr<float>(), channel_size * sizeof(float));
    
    // Canal B (índice 2)
    std::memcpy(chw_data + 2 * channel_size, channels[2].ptr<float>(), channel_size * sizeof(float));

    return chw_frame;
}

cv::Mat TensorRTInference::inference(const cv::Mat& preprocessed_frame) {
    // preprocessed_frame ya está en formato CHW
    size_t input_size = preprocessed_frame.total() * preprocessed_frame.elemSize();
    cudaMemcpyAsync(input_memory_gpu_, preprocessed_frame.data, input_size, cudaMemcpyHostToDevice, stream_);
    
    context_->enqueueV3(stream_);
    
    size_t output_size = 1 * num_classes_ * output_height_ * output_width_ * sizeof(float);
    cudaMemcpyAsync(output_buffer_cpu_, output_memory_gpu_, output_size, cudaMemcpyDeviceToHost, stream_);
    
    cudaStreamSynchronize(stream_);
    
    // Devolver los logits como un Mat 4D para una interpretación correcta
    // return cv::Mat(std::vector<int>{1, num_classes_, output_height_, output_width_}, CV_32F, output_buffer_cpu_);
    return cv::Mat(std::vector<int>{1, num_classes_, output_height_, output_width_}, CV_32F, output_buffer_cpu_);
}

void TensorRTInference::inference_gpu() {
    // Los datos ya están en la GPU, solo ejecutar el contexto.
    context_->enqueueV3(stream_);
}

cv::Mat TensorRTInference::postprocess(const cv::Mat& raw_output, const cv::Size& original_size) {
    // Método rápido: argmax primero, luego upsampling
    
    // 1. Crear matriz para el argmax en resolución baja
    cv::Mat segmentation_map_low(output_height_, output_width_, CV_8UC1);
    
    // 2. Realizar argmax en resolución baja (mucho más rápido)
    float* data = (float*)raw_output.data;
    
    #pragma omp parallel for
    for (int y = 0; y < output_height_; ++y) {
        for (int x = 0; x < output_width_; ++x) {
            float max_val = -std::numeric_limits<float>::infinity();
            uint8_t max_idx = 0;
            
            // Buscar el máximo entre todas las clases para este píxel
            for (int c = 0; c < num_classes_; ++c) {
                size_t offset = c * output_height_ * output_width_ + y * output_width_ + x;
                float val = data[offset];
                if (val > max_val) {
                    max_val = val;
                    max_idx = static_cast<uint8_t>(c);
                }
            }
            segmentation_map_low.at<uint8_t>(y, x) = max_idx;
        }
    }
    
    // 3. Hacer upsampling del mapa de segmentación (una sola operación)
    cv::Mat segmentation_map;
    cv::resize(segmentation_map_low, segmentation_map, original_size, 0, 0, cv::INTER_NEAREST);
    
    return segmentation_map;
}

cv::Mat TensorRTInference::applyColormap(const cv::Mat& segmentation_map) {
    cv::Mat colored_map(segmentation_map.size(), CV_8UC3);
    #pragma omp parallel for
    for (int y = 0; y < segmentation_map.rows; ++y) {
        for (int x = 0; x < segmentation_map.cols; ++x) {
            uint8_t class_id = segmentation_map.at<uint8_t>(y, x);
            if (class_id < 20) {
                // El colormap está en RGB. OpenCV espera BGR.
                // Por tanto, asignamos (B, G, R)
                colored_map.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    colormap_[class_id][2], // Blue
                    colormap_[class_id][1], // Green
                    colormap_[class_id][0]  // Red
                );
            }
        }
    }
    return colored_map;
}

cv::Mat TensorRTInference::preprocess_gpu(const cv::Mat& frame) {
    // Copiar frame de CPU a GPU usando cudaMemcpy2DAsync para manejar el stride de cv::Mat
    // frame.step es el número de bytes por fila en la imagen de origen (CPU)
    // input_width_ * 3 * sizeof(uint8_t) es el número de bytes por fila en el destino (GPU)
    cudaMemcpy2DAsync(gpu_input_buffer_, input_width_ * 3 * sizeof(uint8_t), 
                      frame.data, frame.step,
                      input_width_ * 3 * sizeof(uint8_t), input_height_,
                      cudaMemcpyHostToDevice, stream_);

    // Lanzar kernel de preprocesado
    const float3 mean = {mean_[0], mean_[1], mean_[2]};
    const float3 std = {std_[0], std_[1], std_[2]};
    preprocess_kernel(gpu_input_buffer_, (float*)input_memory_gpu_, input_width_, input_height_, mean, std);
    
    // No se devuelve Mat, la entrada para inferencia ya está en input_memory_gpu_
    return cv::Mat(); 
}

cv::Mat TensorRTInference::postprocess_gpu(const cv::Mat& /*raw_output*/, const cv::Size& original_size) {
    // --- MÉTODO DE ALTA CALIDAD: Reescalar logits primero, luego argmax ---

    // 1. Reescalar los logits de cada clase de baja a alta resolución usando interpolación bilineal.
    NppiSize oSrcSize = {output_width_, output_height_};
    NppiRect oSrcRect = {0, 0, output_width_, output_height_};
    NppiSize oDstSize = {original_size.width, original_size.height};
    NppiRect oDstRect = {0, 0, original_size.width, original_size.height};

    size_t low_res_channel_size = output_width_ * output_height_;
    size_t high_res_channel_size = original_size.width * original_size.height;

    // Iterar sobre cada canal de clase y reescalarlo
    for (int c = 0; c < num_classes_; ++c) {
        const float* src_ptr = (const float*)output_memory_gpu_ + c * low_res_channel_size;
        float* dst_ptr = gpu_logits_high_res_ + c * high_res_channel_size;

        // nppiResize_32f_C1R reescala una imagen de 1 canal y 32-bit float
        nppiResize_32f_C1R(src_ptr,                // Puntero al canal de origen (baja res)
                           output_width_ * sizeof(float), // Stride de origen en bytes
                           oSrcSize,
                           oSrcRect,
                           dst_ptr,                // Puntero al canal de destino (alta res)
                           original_size.width * sizeof(float), // Stride de destino en bytes
                           oDstSize,
                           oDstRect,
                           NPPI_INTER_LINEAR);     // Usar interpolación LINEAR (bilineal) para suavizar logits
    }

    // 2. Ejecutar Argmax en los logits de ALTA resolución para obtener el mapa de clases final.
    argmax_kernel(gpu_logits_high_res_, gpu_map_high_res_, original_size.width, original_size.height, num_classes_);

    // 3. Copiar el mapa final de alta resolución de la GPU a la CPU.
    cudaMemcpy(cpu_final_map_, gpu_map_high_res_, original_size.width * original_size.height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 4. Envolver el buffer de la CPU en un cv::Mat para devolverlo.
    return cv::Mat(original_size.height, original_size.width, CV_8UC1, cpu_final_map_);
}

cv::Mat TensorRTInference::postprocess_gpu_with_confidence(const cv::Mat& /*raw_output*/, const cv::Size& original_size) {

    // 1. Reescalar los logits de cada clase de baja a alta resolución usando interpolación bilineal.
    NppiSize oSrcSize = {output_width_, output_height_};
    NppiRect oSrcRect = {0, 0, output_width_, output_height_};
    NppiSize oDstSize = {original_size.width, original_size.height};
    NppiRect oDstRect = {0, 0, original_size.width, original_size.height};

    size_t low_res_channel_size = output_width_ * output_height_;
    size_t high_res_channel_size = original_size.width * original_size.height;

    // Iterar sobre cada canal de clase y reescalarlo
    for (int c = 0; c < num_classes_; ++c) {
        const float* src_ptr = (const float*)output_memory_gpu_ + c * low_res_channel_size;
        float* dst_ptr = gpu_logits_high_res_ + c * high_res_channel_size;

        // nppiResize_32f_C1R reescala una imagen de 1 canal y 32-bit float
        nppiResize_32f_C1R(src_ptr,                // Puntero al canal de origen (baja res)
                           output_width_ * sizeof(float), // Stride de origen en bytes
                           oSrcSize,
                           oSrcRect,
                           dst_ptr,                // Puntero al canal de destino (alta res)
                           original_size.width * sizeof(float), // Stride de destino en bytes
                           oDstSize,
                           oDstRect,
                           NPPI_INTER_LINEAR);     // Usar interpolación LINEAR (bilineal) para suavizar logits
    }


    // 2. Ejecutar Argmax en los logits de ALTA resolución para obtener el mapa de clases y el mapa de confianza.
    argmax_with_confidence_kernel(gpu_logits_high_res_, (uchar2*)gpu_combined_map_, original_size.width, original_size.height, num_classes_);

    // 3. Copiar el mapa combinado final de la GPU a la CPU.
    size_t combined_map_size_bytes = original_size.width * original_size.height * sizeof(uchar2);
    cudaMemcpy(cpu_combined_map_, gpu_combined_map_, combined_map_size_bytes, cudaMemcpyDeviceToHost);

    // 4. Envolver el buffer de la CPU en un cv::Mat de 2 canales (CV_8UC2) y devolverlo.
    return cv::Mat(original_size.height, original_size.width, CV_8UC2, cpu_combined_map_);
}