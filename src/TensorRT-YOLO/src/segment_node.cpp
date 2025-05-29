#include <chrono>
#include <memory>
#include <mutex>
#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <time.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

// Incluir las cabeceras de la librería deploy
#include "deploy/model.hpp"      // Para deploy::SegmentModel y deploy::Image (si está allí)
#include "deploy/option.hpp"     // Para deploy::InferOption
#include "deploy/result.hpp"     // Para deploy::SegmentRes
#include <cuda_runtime_api.h>

// Helper macro para simplificar la revisión de errores CUDA en el constructor/destructor
#define LOG_CUDA_ERROR(err, msg_prefix, logger) \
    if (err != cudaSuccess) { \
        RCLCPP_ERROR(logger, "%s: %s", msg_prefix, cudaGetErrorString(err)); \
    }

namespace fs = std::filesystem;
using namespace std::chrono_literals;

class YoloBatchNode : public rclcpp::Node
{
public:
    YoloBatchNode() : Node("yolo_batch_node")
    {
        // Parámetros del nodo
        this->declare_parameter<std::string>("engine_path", "yolo11m-seg.engine");
        this->declare_parameter<double>("rescale_factor", 1.0);
        // Tamaño de imagen de salida del pre-procesamiento (ancho x alto):
        // Se espera que el batch sea de 3 imágenes de 416x640 (por ejemplo)
        this->declare_parameter<int>("input_width", 640);
        this->declare_parameter<int>("input_height", 416);
        this->declare_parameter<std::string>("image_topic_1", "/left/image_raw");
        this->declare_parameter<std::string>("image_topic_2", "/front/image_raw");
        this->declare_parameter<std::string>("image_topic_3", "/right/image_raw");
        this->declare_parameter<bool>("use_pinned_input_memory", true);

        std::array<std_msgs::msg::Header, 3> image_headers_;

        auto engine_path = this->get_parameter("engine_path").get_value<std::string>();
        rescale_factor_ = this->get_parameter("rescale_factor").get_value<double>();
        input_width_ = this->get_parameter("input_width").get_value<int>();
        input_height_ = this->get_parameter("input_height").get_value<int>();
        topic_names_[0] = this->get_parameter("image_topic_1").get_value<std::string>();
        topic_names_[1] = this->get_parameter("image_topic_2").get_value<std::string>();
        topic_names_[2] = this->get_parameter("image_topic_3").get_value<std::string>();
        use_pinned_input_memory_ = this->get_parameter("use_pinned_input_memory").get_value<bool>();

        RCLCPP_INFO(this->get_logger(), "Engine path: %s", engine_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Input resized image size: %dx%d", input_width_, input_height_);
        RCLCPP_INFO(this->get_logger(), "Subscripciones a: [%s] , [%s] , [%s]",
        topic_names_[0].c_str(), topic_names_[1].c_str(), topic_names_[2].c_str());

        if (use_pinned_input_memory_) {
            // Asumimos imágenes BGR de 8 bits (3 canales)
            input_channels_ = 3;
            single_image_pinned_bytes_ = static_cast<size_t>(input_width_ * input_height_ * input_channels_);
            total_pinned_input_bytes_ = 3 * single_image_pinned_bytes_;
            
            cudaError_t err = cudaHostAlloc(&h_pinned_input_buffer_, total_pinned_input_bytes_, cudaHostAllocDefault);
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate pinned host memory for input: %s. Disabling pinned memory.", cudaGetErrorString(err));
                h_pinned_input_buffer_ = nullptr; // Asegura que no se use
                use_pinned_input_memory_ = false; // Desactiva la funcionalidad
            } else {
                RCLCPP_INFO(this->get_logger(), "Allocated %zu bytes of pinned host memory for input.", total_pinned_input_bytes_);
            }
        }else {
            RCLCPP_INFO(this->get_logger(), "Pinned input memory is disabled.");
            h_pinned_input_buffer_ = nullptr; // No se usa memoria pinneada
        }


        // Inicializar inferencia (configurar opciones según corresponda)
        deploy::InferOption option;
        option.enableSwapRB();

        try {
            model_ = std::make_unique<deploy::SegmentModel>(engine_path, option);
            if (!model_) {
                throw std::runtime_error("No se pudo cargar el engine");
            }
            RCLCPP_INFO(this->get_logger(), "Modelo cargado exitosamente.");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error al cargar el modelo: %s", e.what());
            rclcpp::shutdown();
        }

        // Crear suscriptores para tres tópicos
        for (size_t i = 0; i < 3; i++) {
            image_subs_[i] = this->create_subscription<sensor_msgs::msg::Image>(
                topic_names_[i],
                10,
                [this, i](const sensor_msgs::msg::Image::SharedPtr msg) { this->imageCallback(msg, i); });
        }

        // Crear publicadores para cada imagen del batch
        for (size_t i = 0; i < 3; i++) {
            seg_pubs_[i] = this->create_publisher<sensor_msgs::msg::Image>(
                "/segmentation/mask_" + std::to_string(i + 1), 20);
            scores_pubs_[i] = this->create_publisher<std_msgs::msg::Float32MultiArray>(
                "/segmentation/scores_" + std::to_string(i + 1), 20);
            classes_pubs_[i] = this->create_publisher<std_msgs::msg::Int32MultiArray>(
                "/segmentation/classes_" + std::to_string(i + 1), 20);
        }
    }

private:
    // Parámetros
    double rescale_factor_;
    int input_width_;
    int input_height_;
    std::array<std::string, 3> topic_names_;
    bool use_pinned_input_memory_ = false;
    unsigned char* h_pinned_input_buffer_ = nullptr;
    size_t single_image_pinned_bytes_ = 0;
    size_t total_pinned_input_bytes_ = 0;
    int input_channels_ = 3;
    // static std::chrono::steady_clock::time_point last_publish_time = clock::now(); // REMOVE THIS
    static std::chrono::steady_clock::time_point last_publish_time; // DECLARE only

    // Subscriptores y publicadores
    std::array<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr, 3> image_subs_;
    std::array<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr, 3> seg_pubs_;
    std::array<rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr, 3> scores_pubs_;
    std::array<rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr, 3> classes_pubs_;

    // Buffers para guardar imágenes recibidas y sus headers
    std::array<cv::Mat, 3> image_buffers_;
    std::array<std_msgs::msg::Header, 3> image_headers_;
    std::array<bool, 3> received_{false, false, false};
    std::mutex buffer_mutex_;

    // Modelo de segmentación
    std::unique_ptr<deploy::SegmentModel> model_;

    // Callback de cada imagen
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg, size_t index)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            image_buffers_[index] = cv_ptr->image;
            image_headers_[index] = msg->header;
            received_[index] = true;
        }
        // Si se tiene un batch completo, se procesa
        if (received_[0] && received_[1] && received_[2]) {
            processBatch();
            // Resetear flags
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            received_ = {false, false, false};
        }
    }

    // Función para escalar polígonos
    std::vector<std::vector<cv::Point>> scalePolygons(const std::vector<std::vector<cv::Point>>& polys, double scale)
    {
        std::vector<std::vector<cv::Point>> scaled;
        for (const auto& poly : polys) {
            std::vector<cv::Point> poly_scaled;
            for (const auto& pt : poly) {
                poly_scaled.push_back(cv::Point(static_cast<int>(pt.x * scale), static_cast<int>(pt.y * scale)));
            }
            scaled.push_back(poly_scaled);
        }
        return scaled;
    }

    // Postprocesamiento: dado el resultado y tamaño original, genera una máscara de segmentación simple.
    cv::Mat generateSegMask(const deploy::SegmentRes &result, const cv::Size &orig_size, double scale_up_factor)
    {
        (void)scale_up_factor; // Mark as unused to silence warning for now
        cv::Mat seg_mask = cv::Mat::zeros(orig_size, CV_8UC1);

        size_t num_detections = static_cast<size_t>(result.num);
        // Ensure we don't access out of bounds if vector sizes don't match num
        size_t num_items_to_process = std::min({num_detections, result.masks.size(), result.classes.size()});

        if (num_detections > 0 && (num_detections != result.masks.size() || num_detections != result.classes.size())) {
            fprintf(stderr, "[WARN] generateSegMask: Mismatch between result.num (%d) and sizes of masks (%zu) or classes (%zu).\\n",
                    result.num, result.masks.size(), result.classes.size());
        }

        for (size_t mask_idx = 0; mask_idx < num_items_to_process; ++mask_idx) {
            if (result.masks[mask_idx].data.empty())
                continue;
            
            // Se usa const_cast para convertir el puntero constante a void*
            cv::Mat raw_mask(result.masks[mask_idx].height, result.masks[mask_idx].width, CV_8UC1,
                            const_cast<void*>(static_cast<const void*>(result.masks[mask_idx].data.data())));
            if (raw_mask.empty())
                continue;
            
            cv::Mat mask_resized;

            cv::resize(raw_mask, mask_resized, orig_size, 0, 0, cv::INTER_NEAREST);

            int class_id = result.classes[mask_idx];
            int class_val = class_id + 1; 

            if (class_val < 1) class_val = 1;   // Smallest positive value
            if (class_val > 255) class_val = 255; // Max value for CV_8UC1

            // Apply the mask: set pixels in seg_mask to class_val where mask_resized is non-zero
            seg_mask.setTo(static_cast<unsigned char>(class_val), mask_resized);
        }
        return seg_mask;
    }

    // Función para procesar las 3 imágenes en batch
    void processBatch()
    {
        using clock = std::chrono::steady_clock;
        auto t_total_start = clock::now();

        std::array<cv::Mat, 3> originals;
        std::array<cv::Size, 3> orig_sizes;
        std::vector<deploy::Image> img_batch;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            for (size_t i = 0; i < 3; ++i) {
                originals[i] = image_buffers_[i].clone(); // Clonar para evitar problemas de concurrencia
                orig_sizes[i] = originals[i].size();
            }
        }
        
        // Preprocesamiento
        auto t_pre_start = clock::now();
        cv::Size target_size(input_width_, input_height_);
        unsigned char* current_pinned_ptr = h_pinned_input_buffer_;

        for (size_t i = 0; i < 3; ++i) {
            cv::Mat resized_img; // Temporal para la operación de resize
            cv::resize(originals[i], resized_img, target_size, 0, 0, cv::INTER_LINEAR);
            
            if (use_pinned_input_memory_ && h_pinned_input_buffer_) {
                // Asegurarse de que la imagen redimensionada es continua y del tamaño esperado
                if (!resized_img.isContinuous()) {
                    resized_img = resized_img.clone(); // Hacerla continua
                }
                if (resized_img.total() * resized_img.elemSize() == single_image_pinned_bytes_) {
                    std::memcpy(current_pinned_ptr, resized_img.data, single_image_pinned_bytes_);
                    img_batch.emplace_back(current_pinned_ptr, resized_img.cols, resized_img.rows); 
                    current_pinned_ptr += single_image_pinned_bytes_;
                } else {
                    //  RCLCPP_WARN(this->get_logger(), "Resized image %zu size mismatch for pinned memory. Using its own data.", i);
                    img_batch.emplace_back(resized_img.data, resized_img.cols, resized_img.rows); // Fallback
                }
            } else {
                // resized_imgs[i] = resized_img; // Si no se usa pinned, se podría guardar aquí
                img_batch.emplace_back(resized_img.data, resized_img.cols, resized_img.rows);
            }
        }

        auto t_pre_end = clock::now();

        // Inferencia en batch
        auto t_inf_start = clock::now();
        std::vector<deploy::SegmentRes> results = model_->predict(img_batch);
        auto t_inf_end = clock::now();

        // Postprocesamiento y publicación
        auto t_post_start = clock::now();
        // Se define scale_up_factor para redimensionar resultados a tamaño original.
        double scale_up_factor = 1.0; // En este ejemplo no hay re-escalado adicional; ajustar si se aplica rescale.
        

        for (size_t i = 0; i < results.size() && i < 3; ++i) {
            // RCLCPP_INFO(this->get_logger(), "Procesando resultado para imagen %zu:", i);
            // RCLCPP_INFO(this->get_logger(), "  Numero de detecciones (results[%zu].num): %d", i, results[i].num);
            // RCLCPP_INFO(this->get_logger(), "  Available data sizes: scores=%zu, classes=%zu, boxes=%zu, masks=%zu",
            //             results[i].scores.size(), results[i].classes.size(), results[i].boxes.size(), results[i].masks.size());

            // Para este ejemplo se genera una máscara de segmentación a partir del resultado.
            cv::Mat seg_mask = generateSegMask(results[i], orig_sizes[i], scale_up_factor);

            // Convertir a mensaje ROS usando cv_bridge
            cv_bridge::CvImage cv_img;
            cv_img.header = image_headers_[i];
            cv_img.encoding = "mono8";
            cv_img.image = seg_mask;
            auto mask_msg = cv_img.toImageMsg();
            seg_pubs_[i]->publish(*mask_msg);

            // Publicar mensajes de scores y clases
            std_msgs::msg::Float32MultiArray scores_msg;
            std_msgs::msg::Int32MultiArray classes_msg;
            
            if (results[i].num > 0) {
                size_t num_to_iterate = static_cast<size_t>(results[i].num);
                // Safety check: ensure all relevant vectors are at least `num` long
                if (results[i].scores.size() < num_to_iterate ||
                    results[i].classes.size() < num_to_iterate ||
                    results[i].boxes.size() < num_to_iterate) {
                    
                    // RCLCPP_WARN(this->get_logger(),
                    //             "  Image %zu: Mismatch between results.num (%d) and vector sizes (scores: %zu, classes: %zu, boxes: %zu). Clamping iteration count.",
                    //             i, results[i].num, results[i].scores.size(), results[i].classes.size(), results[i].boxes.size());
                    // num_to_iterate = std::min({num_to_iterate, results[i].scores.size(), results[i].classes.size(), results[i].boxes.size()});
                }

                // RCLCPP_INFO(this->get_logger(), "  Iterating over %zu items for image %zu.", num_to_iterate, i);

                for (size_t j = 0; j < num_to_iterate; ++j) {
                    scores_msg.data.push_back(results[i].scores[j]);
                    classes_msg.data.push_back(results[i].classes[j]);
                    
                    const auto& box = results[i].boxes[j]; // For logging box info
                    // RCLCPP_INFO(this->get_logger(), "    Item %zu: Score=%.2f, ClassID=%d, Box(L:%.1f, T:%.1f, R:%.1f, B:%.1f)",
                    //             j, results[i].scores[j], results[i].classes[j], box.left, box.top, box.right, box.bottom);
                }
            }
            
            if (!scores_msg.data.empty()) {
                std::stringstream ss_scores;
                for(size_t k=0; k < scores_msg.data.size(); ++k) {
                    ss_scores << scores_msg.data[k] << (k == scores_msg.data.size() - 1 ? "" : ", ");
                }
                // RCLCPP_INFO(this->get_logger(), "  Publicando Scores para imagen %zu: [%s]", i, ss_scores.str().c_str());
            }

            if (!classes_msg.data.empty()) {
                std::stringstream ss_classes;
                for(size_t k=0; k < classes_msg.data.size(); ++k) {
                    ss_classes << classes_msg.data[k] << (k == classes_msg.data.size() - 1 ? "" : ", ");
                }
                // RCLCPP_INFO(this->get_logger(), "  Publicando Classes para imagen %zu: [%s]", i, ss_classes.str().c_str());
            }

            scores_pubs_[i]->publish(scores_msg);
            classes_pubs_[i]->publish(classes_msg);
        }
        // RCLCPP_INFO(this->get_logger(), "--- Fin Postprocesamiento Batch ---");
        auto t_post_end = clock::now();

        // Calcular tiempos
        auto dt_pre = std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_end - t_pre_start).count();
        auto dt_inf = std::chrono::duration_cast<std::chrono::milliseconds>(t_inf_end - t_inf_start).count();
        auto dt_post = std::chrono::duration_cast<std::chrono::milliseconds>(t_post_end - t_post_start).count();
        auto dt_total = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t_total_start).count();

        RCLCPP_INFO(this->get_logger(),
                    "Batch procesado: pre=%ld ms | inf=%ld ms | post=%ld ms | total=%ld ms",
                    dt_pre, dt_inf, dt_post, dt_total);

        // Para calcular la frecuencia de publicación de los batches:
        auto current_publish_time = clock::now();
        // REMOVE: static std::chrono::steady_clock::time_point last_publish_time;
        auto time_since_last_publish = std::chrono::duration_cast<std::chrono::duration<double>>(current_publish_time - YoloBatchNode::last_publish_time);
        if (time_since_last_publish.count() > 0) { // Evitar división por cero al inicio
            double frequency = 1.0 / time_since_last_publish.count();
            RCLCPP_INFO(this->get_logger(), "Frecuencia de publicación de batch: %.2f Hz", frequency);
        }
        YoloBatchNode::last_publish_time = current_publish_time;
    }
};

// Define and initialize the static member AFTER the class definition
std::chrono::steady_clock::time_point YoloBatchNode::last_publish_time = std::chrono::steady_clock::now();

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloBatchNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}