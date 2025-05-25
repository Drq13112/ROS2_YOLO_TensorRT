/**
 * @file model.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 模型实现
 * @date 2025-01-16
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

#include "deploy/model.hpp"
#include "deploy/result.hpp"
#include <cmath>

namespace deploy {

template <typename ResultType>
std::unique_ptr<BaseModel<ResultType>> BaseModel<ResultType>::clone() const {
    auto clone_model              = std::make_unique<BaseModel<ResultType>>();
    clone_model->backend_         = backend_->clone();  // < 克隆 TrtBackend
    clone_model->infer_gpu_trace_ = std::make_unique<GpuTimer>(clone_model->backend_->stream);
    clone_model->infer_cpu_trace_ = std::make_unique<CpuTimer>();
    return clone_model;
}

template <typename ResultType>
std::vector<ResultType> BaseModel<ResultType>::predict(const std::vector<Image>& images) {


    std::cout << "Preprocessing " << images.size() << " images..." << std::endl;
    if (backend_->option.enable_performance_report) {
        total_request_ += (backend_->dynamic ? images.size() : backend_->max_shape.x);
        infer_cpu_trace_->start();
        infer_gpu_trace_->start();
    }

    std::cout << "Predicting " << images.size() << " images..." << std::endl;
    backend_->infer(images);  // 调用推理方法
    std::cout << "Inference completed." << std::endl;

    // 预分配结果空间
    std::vector<ResultType> results(images.size());
    for (auto idx = 0u; idx < images.size(); ++idx) {
        results[idx] = postProcess(idx);
    }

    if (backend_->option.enable_performance_report) {
        infer_gpu_trace_->stop();
        infer_cpu_trace_->stop();
    }

    return results;
}

template <typename ResultType>
ResultType BaseModel<ResultType>::predict(const Image& image) {
    std::cout << "INside deploy" << std::endl;
    return predict(std::vector<Image>{image}).front();
}

template <typename ResultType>
int BaseModel<ResultType>::batch_size() const {
    return backend_->max_shape.x;
}

template <typename ResultType>
std::tuple<std::string, std::string, std::string> BaseModel<ResultType>::performanceReport() {
    if (backend_->option.enable_performance_report) {
        float const       throughput = total_request_ / infer_cpu_trace_->totalMilliseconds() * 1000;
        std::stringstream ss;

        // 构建吞吐量字符串
        ss << "Throughput: " << throughput << " qps";
        std::string throughputStr = ss.str();
        ss.str("");  // 清空 stringstream

        auto percentiles = std::vector<float>{90, 95, 99};

        auto getLatencyStr = [&](const auto& trace, const std::string& device) {
            auto result = getPerformanceResult(trace->milliseconds(), {0.90, 0.95, 0.99});
            ss << device << " Latency: min = " << result.min << " ms, max = " << result.max << " ms, mean = " << result.mean << " ms, median = " << result.median << " ms";
            for (int32_t i = 0, n = percentiles.size(); i < n; ++i) {
                ss << ", percentile(" << percentiles[i] << "%) = " << result.percentiles[i] << " ms";
            }
            std::string output = ss.str();
            ss.str("");  // 清空 stringstream
            return output;
        };

        std::string cpuLatencyStr = getLatencyStr(infer_cpu_trace_, "CPU");
        std::string gpuLatencyStr = getLatencyStr(infer_gpu_trace_, "GPU");

        total_request_ = 0;
        infer_cpu_trace_->reset();
        infer_gpu_trace_->reset();

        return std::make_tuple(throughputStr, cpuLatencyStr, gpuLatencyStr);
    } else {
        // 性能报告未启用时返回空字符串
        return std::make_tuple("", "", "");
    }
}

// ClassifyModel 的后处理方法实现
template <>
ClassifyRes BaseModel<ClassifyRes>::postProcess(int idx) {
    auto&  tensor_info = backend_->tensor_infos[1];
    float* topk        = static_cast<float*>(tensor_info.buffer->host()) + idx * tensor_info.shape.d[1] * tensor_info.shape.d[2];

    ClassifyRes result;
    result.num = tensor_info.shape.d[1];
    result.scores.reserve(result.num);
    result.classes.reserve(result.num);

    for (int i = 0; i < result.num; ++i) {
        result.scores.push_back(topk[i * tensor_info.shape.d[2]]);
        result.classes.push_back(topk[i * tensor_info.shape.d[2] + 1]);
    }

    return result;
}

// DetectModel 的后处理方法实现
template <>
DetectRes BaseModel<DetectRes>::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];

    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

    DetectRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);
    }

    return result;
}

// OBBModel 的后处理方法实现
template <>
OBBRes BaseModel<OBBRes>::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];

    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

    OBBRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];
        float theta = boxes[base_index + 4];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(RotatedBox{left, top, right, bottom, theta});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);
    }

    return result;
}

// // SegmentModel 的后处理方法实现
template <>
SegmentRes BaseModel<SegmentRes>::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];
    auto& mask_tensor  = backend_->tensor_infos[5];
    int   mask_height  = mask_tensor.shape.d[2];
    int   mask_width   = mask_tensor.shape.d[3];

    int      num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float*   boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float*   scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*     classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    uint8_t* masks   = static_cast<uint8_t*>(mask_tensor.buffer->host()) + idx * mask_tensor.shape.d[1] * mask_height * mask_width;

    SegmentRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);
    result.masks.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);

        Mask mask(mask_width - 2 * affine_transform.dst_offset_x, mask_height - 2 * affine_transform.dst_offset_y);

        // Crop the mask's edge area, applying offset to adjust the position
        int start_idx = i * mask_height * mask_width;
        int src_idx   = start_idx + affine_transform.dst_offset_y * mask_width + affine_transform.dst_offset_x;
        for (int y = 0; y < mask.height; ++y) {
            std::memcpy(&mask.data[y * mask.width], masks + src_idx, mask.width);
            src_idx += mask_width;
        }

        result.masks.emplace_back(std::move(mask));
    }

    return result;
}

// Versión modificada de SegmentModel::postProcess para el output de Ultralytics YOLO-seg
// template <>
// SegmentRes BaseModel<SegmentRes>::postProcess(int idx) {
//     std::cout << "[DEBUG] postProcess called" << std::endl;
//     // Ahora asumimos que:
//     // - backend_->tensor_infos[0]: tensor de detección, shape (B, 116, 143640)
//     // - backend_->tensor_infos[1]: tensor de máscara prototype, shape (B, 32, 304, 1440)
//     auto& det_tensor = backend_->tensor_infos[0];
//     auto& mask_tensor = backend_->tensor_infos[1];

//     // Dimensiones del tensor de detección
//     int vector_length = det_tensor.shape.d[1]; // 116
//     int total_candidates = det_tensor.shape.d[2]; // 143640

//     // Dimensiones del proto de máscara
//     int mask_channels = mask_tensor.shape.d[1]; // 32
//     int mask_height = mask_tensor.shape.d[2];     // 304
//     int mask_width = mask_tensor.shape.d[3];        // 1440

//     // Punteros a los datos (suponemos que el tensor de detección ya es float)
//     float* det_data = static_cast<float*>(det_tensor.buffer->host()) + idx * vector_length * total_candidates;
//     // Usaremos el tensor de máscara como float (a pesar de que originalmente se manejaba como uint8_t)
//     float* proto = reinterpret_cast<float*>(mask_tensor.buffer->host()) + idx * (mask_channels * mask_height * mask_width);

//     SegmentRes result;
//     // Fijamos un umbral para filtrar detecciones bajas en score
//     float score_thresh = 0.3f;
//     // Número de coeficientes por detección (se extraen del vector de detección: índices 6..37)
//     const int num_coeff = 32;

//         // (Opcional) Debug: imprimir algunos parámetros
//     std::cout << "[DEBUG] vector_length: " << vector_length 
//               << ", total_candidates: " << total_candidates << std::endl;
//     std::cout << "[DEBUG] mask_channels: " << mask_channels 
//               << ", mask_height: " << mask_height 
//               << ", mask_width: " << mask_width << std::endl;

//     // Iteramos sobre todos los candidatos
//     for (int i = 0; i < total_candidates; ++i) {

//         // Calcular el offset para este candidato en el tensor de detección
//         int base_offset = i * vector_length;

//         // Comprobación (defensiva): asegurarse de que tenemos espacio para leer todos los datos
//         if (base_offset + 6 + num_coeff > total_candidates * vector_length) {
//             std::cerr << "[WARN] Límite de detección excedido para candidato " << i << std::endl;
//             break;
//         }

//         float score = det_data[i * vector_length + 4];
//         if (score < score_thresh)
//             continue; // descarta candidatos con score bajo

//         // Extraer datos de la caja: se asumen (x, y, w, h)
//         float x = det_data[i * vector_length + 0];
//         float y = det_data[i * vector_length + 1];
//         float w = det_data[i * vector_length + 2];
//         float h = det_data[i * vector_length + 3];
//         float left = x;
//         float top = y;
//         float right = x + w;
//         float bottom = y + h;
//         int cls = static_cast<int>(det_data[i * vector_length + 5]);

//         // Obtener los coeficientes de máscara (suponiendo 32 coeficientes)
//         float* coeffs = det_data + i * vector_length + 6; // índices de 6 a 37

//         // Se obtiene la transformación afín
//         auto& affine_transform = backend_->option.input_shape.has_value()
//                                      ? backend_->affine_transforms.front()
//                                      : backend_->affine_transforms[idx];
//         affine_transform.applyTransform(left, top, &left, &top);
//         affine_transform.applyTransform(right, bottom, &right, &bottom);

//         result.boxes.emplace_back(Box{left, top, right, bottom});
//         result.scores.push_back(score);
//         result.classes.push_back(cls);

//         // Calcular la máscara del candidato aplicando la fórmula:
//         // instance_mask = sigmoid( sum_{c=0}^{31} ( proto[c, y, x] * coeff[c] ) )
//         // Crearemos una máscara de tamaño (mask_width, mask_height)
//         Mask inst_mask(mask_width, mask_height);
//         for (int y_pix = 0; y_pix < mask_height; ++y_pix) {
//             for (int x_pix = 0; x_pix < mask_width; ++x_pix) {
//                 float sum = 0.0f;
//                 for (int c = 0; c < num_coeff; ++c) {
//                     int proto_index = c * mask_height * mask_width + y_pix * mask_width + x_pix;
//                     sum += proto[proto_index] * coeffs[c];
//                 }
//                 // Sigmoid
//                 float prob = 1.0f / (1.0f + std::exp(-sum));
//                 // Umbralizamos la máscara (por ejemplo, 0.5)
//                 inst_mask.data[y_pix * mask_width + x_pix] = (prob > 0.5f ? 255 : 0);
//             }
//         }
//         result.masks.emplace_back(std::move(inst_mask));

//         // Se incrementa el contador de detecciones (el formato original lo guardaba en result.num)
//         result.num++;
//     }
//     return result;
// }

// PoseModel 的后处理方法实现
template <>
PoseRes BaseModel<PoseRes>::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];
    auto& kpt_tensor   = backend_->tensor_infos[5];
    int   nkpt         = kpt_tensor.shape.d[2];
    int   ndim         = kpt_tensor.shape.d[3];

    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    float* kpts    = static_cast<float*>(kpt_tensor.buffer->host()) + idx * kpt_tensor.shape.d[1] * nkpt * ndim;

    PoseRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);
    result.kpts.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);

        std::vector<KeyPoint> keypoints;
        for (int j = 0; j < nkpt; ++j) {
            float x = kpts[i * nkpt * ndim + j * ndim];
            float y = kpts[i * nkpt * ndim + j * ndim + 1];
            affine_transform.applyTransform(x, y, &x, &y);
            keypoints.emplace_back((ndim == 2) ? KeyPoint(x, y) : KeyPoint(x, y, kpts[i * nkpt * ndim + j * ndim + 2]));
        }
        result.kpts.emplace_back(std::move(keypoints));
    }

    return result;
}

}  // namespace deploy
