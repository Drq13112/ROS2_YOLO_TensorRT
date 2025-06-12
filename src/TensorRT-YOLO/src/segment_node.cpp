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
#include <thread>
#include <condition_variable>
#include <atomic>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include <opencv2/videoio.hpp> 
#include "deploy/model.hpp"      
#include "deploy/option.hpp"     
#include "deploy/result.hpp"     
#include <cuda_runtime_api.h>
#include <image_transport/image_transport.hpp>
#include <rclcpp/callback_group.hpp> 
#include "yolo_custom_interfaces/msg/instance_segmentation_info.hpp"

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
        this->declare_parameter<int>("input_width", 640);
        this->declare_parameter<int>("input_height", 416);
        this->declare_parameter<std::string>("image_topic_1", "/left/image_raw");
        this->declare_parameter<std::string>("image_topic_2", "/front/image_raw");
        this->declare_parameter<std::string>("image_topic_3", "/right/image_raw");
        this->declare_parameter<bool>("use_pinned_input_memory", true);

        // Parámetros para la escritura de vídeo
        this->declare_parameter<bool>("enable_inferred_video_writing", false);
        this->declare_parameter<bool>("enable_mask_video_writing", false);
        this->declare_parameter<std::string>("video_output_path", "/home/david/ros_videos");
        this->declare_parameter<std::string>("inferred_video_filename", "inferred_output.avi");
        this->declare_parameter<std::string>("mask_video_filename", "mask_output.avi");
        this->declare_parameter<double>("video_fps", 10.0);
        this->declare_parameter<int>("video_frame_width", 1920 * 3); // 5760
        this->declare_parameter<int>("video_frame_height", 1200);


        auto engine_path = this->get_parameter("engine_path").get_value<std::string>();
        input_width_ = this->get_parameter("input_width").get_value<int>();
        input_height_ = this->get_parameter("input_height").get_value<int>();
        topic_names_[0] = this->get_parameter("image_topic_1").get_value<std::string>();
        topic_names_[1] = this->get_parameter("image_topic_2").get_value<std::string>();
        topic_names_[2] = this->get_parameter("image_topic_3").get_value<std::string>();
        use_pinned_input_memory_ = this->get_parameter("use_pinned_input_memory").get_value<bool>();


        enable_inferred_video_writing_ = this->get_parameter("enable_inferred_video_writing").get_value<bool>();
        enable_mask_video_writing_ = this->get_parameter("enable_mask_video_writing").get_value<bool>();
        video_output_path_str_ = this->get_parameter("video_output_path").get_value<std::string>();
        inferred_video_filename_ = this->get_parameter("inferred_video_filename").get_value<std::string>();
        mask_video_filename_ = this->get_parameter("mask_video_filename").get_value<std::string>();
        video_fps_ = this->get_parameter("video_fps").get_value<double>();
        video_frame_size_ = cv::Size(this->get_parameter("video_frame_width").get_value<int>(), this->get_parameter("video_frame_height").get_value<int>());


        RCLCPP_INFO(this->get_logger(), "Engine path: %s", engine_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Input resized image size: %dx%d", input_width_, input_height_);
        RCLCPP_INFO(this->get_logger(), "Subscripciones a: [%s] , [%s] , [%s]",
        topic_names_[0].c_str(), topic_names_[1].c_str(), topic_names_[2].c_str());

        this->declare_parameter<std::string>("_image_transport", "raw");
        std::string transport_type;
        if(this->get_parameter("_image_transport", transport_type)){
            RCLCPP_INFO(this->get_logger(), "Using image_transport: %s", transport_type.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "No _image_transport parameter set. Using default (raw).");
        }

        // Crear QoS profile con política best effort
        rclcpp::QoS qos_sensors(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
        qos_sensors.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        // qos_sensors.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        qos_sensors.keep_last(1); // depth = 1


        // Inicializar paleta de colores para clases
        // BGR format
        class_colors_.push_back(cv::Scalar(255, 0, 0));     // Blue
        class_colors_.push_back(cv::Scalar(0, 255, 0));     // Green
        class_colors_.push_back(cv::Scalar(0, 0, 255));     // Red
        class_colors_.push_back(cv::Scalar(255, 255, 0));   // Yellow
        class_colors_.push_back(cv::Scalar(0, 255, 255));   // Cyan
        class_colors_.push_back(cv::Scalar(255, 0, 255));   // Magenta
        class_colors_.push_back(cv::Scalar(192, 192, 192)); // Silver
        class_colors_.push_back(cv::Scalar(128, 128, 128)); // Gray
        class_colors_.push_back(cv::Scalar(128, 0, 0));     // Maroon
        class_colors_.push_back(cv::Scalar(128, 128, 0));   // Olive
        class_colors_.push_back(cv::Scalar(0, 128, 0));     // Dark Green
        class_colors_.push_back(cv::Scalar(128, 0, 128));   // Purple
        class_colors_.push_back(cv::Scalar(0, 128, 128));   // Teal
        class_colors_.push_back(cv::Scalar(0, 0, 128));     // Navy
        class_colors_.push_back(cv::Scalar(255,165,0));     // Orange
        class_colors_.push_back(cv::Scalar(255,192,203));   // Pink 

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


        image_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        
        for (size_t i = 0; i < 3; i++) {
            instance_info_pubs_[i] = this->create_publisher<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
                "/segmentation/instance_info_" + std::to_string(i + 1), qos_sensors);
        }

        // Iniciar el hilo de procesamiento del modelo
        processing_thread_ = std::thread(&YoloBatchNode::modelProcessingLoop, this);
    }

    void initSubscriptions() {
        image_transport::ImageTransport it(shared_from_this());
        image_transport::TransportHints hints(this, "raw", "raw"); // o "compressed"

        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = image_callback_group_;

        for (size_t i = 0; i < 3; i++) {
            image_subs_[i] = it.subscribe( // El error de asignación se corregirá cambiando el tipo de image_subs_
                topic_names_[i],
                1, // queue_size
                // La lambda captura msg como ConstSharedPtr, imageCallback debe aceptarlo así
                [this, i](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                    this->imageCallback(msg, i);
                },
                shared_from_this(), 
                &hints,            
                sub_options        
            );
        }
        RCLCPP_INFO(this->get_logger(), "Image subscriptions initialized with CallbackGroup.");
    }

    ~YoloBatchNode() {
        RCLCPP_INFO(this->get_logger(), "Shutting down YoloBatchNode...");
        stop_processing_thread_ = true;
        cv_batch_ready_.notify_one(); // Wake up processing thread if it's waiting
        if (processing_thread_.joinable()) {
            processing_thread_.join();
            RCLCPP_INFO(this->get_logger(), "Processing thread joined.");
        } else {
            RCLCPP_WARN(this->get_logger(), "Processing thread was not joinable.");
        }

        if (h_pinned_input_buffer_) {
            cudaError_t err = cudaFreeHost(h_pinned_input_buffer_);
            LOG_CUDA_ERROR(err, "Destructor: Failed to free pinned host memory", this->get_logger());
            h_pinned_input_buffer_ = nullptr;
        }
        RCLCPP_INFO(this->get_logger(), "YoloBatchNode shutdown complete.");
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
    std::string mask_encoding_; 
    static std::chrono::steady_clock::time_point last_publish_time;


    // Variables para escritura de vídeo
    cv::VideoWriter video_writer_inferred_;
    cv::VideoWriter video_writer_instance_mask_;
    std::string video_output_path_str_;
    std::string inferred_video_filename_;
    std::string mask_video_filename_;
    double video_fps_;
    cv::Size video_frame_size_; // e.g., 1920*3 x 1216
    bool enable_inferred_video_writing_ = false;
    bool enable_mask_video_writing_ = false;
    bool video_writers_initialized_ = false;
    std::vector<cv::Scalar> class_colors_;

    // Subscriptores y publicadores
    std::array<image_transport::Subscriber, 3> image_subs_; 
    std::array<rclcpp::Publisher<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr, 3> instance_info_pubs_;
    rclcpp::CallbackGroup::SharedPtr image_callback_group_; // CallbackGroup para suscriptores de imagen

    // Buffers para guardar imágenes recibidas y sus headers
    std::array<cv::Mat, 3> image_buffers_;
    std::array<std_msgs::msg::Header, 3> image_headers_;
    std::array<timespec, 3> image_monotonic_entry_times_; // para T2_mono
    std::array<bool, 3> received_{false, false, false};
    std::mutex buffer_mutex_;   // Mutex para proteger el acceso a buffers y flags

    // Modelo de segmentación
    std::unique_ptr<deploy::SegmentModel> model_;
    
    // Para el hilo de procesamiento dedicado
    std::thread processing_thread_;
    std::condition_variable cv_batch_ready_;
    std::atomic<bool> stop_processing_thread_{false};

    
    void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg, size_t index)
    {
        RCLCPP_INFO(this->get_logger(), "Imagen recibida en cámara %zu, encoding: %s", index, msg->encoding.c_str());
        auto callback_entry_time = std::chrono::steady_clock::now(); // Tiempo de entrada al callback
        
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception for topic %s: %s", topic_names_[index].c_str(), e.what());
            return;
        }
        auto cv_bridge_end_time = std::chrono::steady_clock::now();

        if (!cv_ptr || cv_ptr->image.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty image on topic %s", topic_names_[index].c_str());
            return;
        }
        
        auto t_before_lock = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            auto t_after_lock = std::chrono::steady_clock::now();
            
            //image_buffers_[index] = cv_ptr->image.clone(); // Asegurar una copia propia
            image_buffers_[index] = cv_ptr->image; // Usar la imagen directamente, asumiendo que no se modificará fuera de este contexto
            image_headers_[index] = msg->header;
            received_[index] = true;
            
            auto buffer_write_end_time = std::chrono::steady_clock::now();

            auto cv_bridge_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(cv_bridge_end_time - callback_entry_time).count();
            auto lock_wait_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_lock - t_before_lock).count();
            auto buffer_write_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(buffer_write_end_time - t_after_lock).count();
            auto total_callback_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(buffer_write_end_time - callback_entry_time).count();

            RCLCPP_INFO(this->get_logger(), 
                        "ImageCallback[%zu](%s): TS=%d.%09u | CVBridge=%ldus | LockWait=%ldus | BufWrite=%ldus | TotalCB=%ldus | Status=[C1:%d,C2:%d,C3:%d]",
                        index, topic_names_[index].c_str(),
                        msg->header.stamp.sec, msg->header.stamp.nanosec,
                        cv_bridge_duration_us,
                        lock_wait_duration_us,
                        buffer_write_duration_us,
                        total_callback_duration_us,
                        received_[0], received_[1], received_[2]);

            if (received_[0] && received_[1] && received_[2]) {
                cv_batch_ready_.notify_one();
                RCLCPP_INFO(this->get_logger(),"Notified processing thread. All images for batch received. Last to arrive: %s (index %zu)", topic_names_[index].c_str(), index);
            }
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


    // Funcion que funciona bien
    //     cv::Mat generateInstanceIdMaskROI(const deploy::SegmentRes &result,
    //                              const cv::Size &orig_size,
    //                              const cv::Size &net_input_size)
    // {
    //     cv::Mat instance_id_mask;
    //     // Usar this->mask_encoding_
    //     if (this->mask_encoding_ == "mono16") {
    //         instance_id_mask = cv::Mat::zeros(orig_size, CV_16UC1);
    //     } else { 
    //         instance_id_mask = cv::Mat::zeros(orig_size, CV_8UC1);
    //     }

    //     size_t num_detections = static_cast<size_t>(result.num);
    //     size_t num_items_to_process = std::min({num_detections, result.masks.size(), result.classes.size(), result.boxes.size()});

    //     if (num_detections > 0 && (num_detections != result.masks.size() ||
    //                                num_detections != result.classes.size() ||
    //                                num_detections != result.boxes.size())) {
    //         RCLCPP_WARN(this->get_logger(), "[generateInstanceIdMaskROI] Mismatch between result.num (%d) and vector sizes (masks: %zu, classes: %zu, boxes: %zu). Processing %zu items.",
    //                     result.num, result.masks.size(), result.classes.size(), result.boxes.size(), num_items_to_process);
    //     }

    //     double scale_x = static_cast<double>(orig_size.width) / net_input_size.width;
    //     double scale_y = static_cast<double>(orig_size.height) / net_input_size.height;

    //     for (size_t item_idx = 0; item_idx < num_items_to_process; ++item_idx) {
    //         if (result.masks[item_idx].data.empty() || result.masks[item_idx].width <= 0 || result.masks[item_idx].height <= 0) {
    //             RCLCPP_WARN(this->get_logger(), "[generateInstanceIdMaskROI] Item %zu has empty or invalid mask data.", item_idx);
    //             continue;
    //         }

    //         const deploy::Box& net_box = result.boxes[item_idx];
    //         cv::Rect orig_roi_rect(
    //             static_cast<int>(net_box.left * scale_x),
    //             static_cast<int>(net_box.top * scale_y),
    //             static_cast<int>((net_box.right - net_box.left) * scale_x),
    //             static_cast<int>((net_box.bottom - net_box.top) * scale_y)
    //         );
    //         orig_roi_rect &= cv::Rect(0, 0, orig_size.width, orig_size.height); // Clip to image bounds

    //         if (orig_roi_rect.width <= 0 || orig_roi_rect.height <= 0) {
    //             RCLCPP_WARN(this->get_logger(), "[generateInstanceIdMaskROI] Item %zu has zero or negative ROI width/height after scaling.", item_idx);
    //             continue;
    //         }

    //         cv::Mat raw_instance_mask_from_lib(result.masks[item_idx].height, result.masks[item_idx].width, CV_8UC1,
    //                                   const_cast<void*>(static_cast<const void*>(result.masks[item_idx].data.data())));
    //         if (raw_instance_mask_from_lib.empty()) {
    //             RCLCPP_WARN(this->get_logger(), "[generateInstanceIdMaskROI] raw_instance_mask_from_lib for item %zu is empty.", item_idx);
    //             continue;
    //         }

    //         // 1. Resize the raw instance mask to the FULL original image size
    //         cv::Mat instance_mask_at_orig_res;
    //         cv::resize(raw_instance_mask_from_lib, instance_mask_at_orig_res, orig_size, 0, 0, cv::INTER_NEAREST);

    //         // 2. Get the ROI from this full-sized mask that corresponds to the bounding box
    //         // This part of the mask is already binary (0 or non-zero) due to INTER_NEAREST and source mask characteristics.
    //         cv::Mat relevant_part_of_full_mask = instance_mask_at_orig_res(orig_roi_rect);

    //         uint16_t instance_pixel_value = static_cast<uint16_t>(item_idx + 1); // Instance IDs start from 1

    //         cv::Mat final_mask_roi = instance_id_mask(orig_roi_rect);
            
    //         // 3. Apply this correctly scaled and cropped part to the output instance_id_mask's ROI
    //         if (this->mask_encoding_ == "mono16") {
    //             final_mask_roi.setTo(instance_pixel_value, relevant_part_of_full_mask);
    //         } else { // mono8
    //             if (instance_pixel_value > 255) {
    //                 RCLCPP_WARN_ONCE(this->get_logger(), "[generateInstanceIdMaskROI] Instance ID %u exceeds 255 for mono8 mask. Clamping. Consider using mono16.", instance_pixel_value);
    //                 final_mask_roi.setTo(static_cast<unsigned char>(255), relevant_part_of_full_mask);
    //             } else {
    //                 final_mask_roi.setTo(static_cast<unsigned char>(instance_pixel_value), relevant_part_of_full_mask);
    //             }
    //         }
    //     }
    //     return instance_id_mask;
    // }

    cv::Mat generateInstanceIdMaskROI(const deploy::SegmentRes &result,
                                    const cv::Size &orig_size,
                                    const cv::Size &net_input_size)
    {
        cv::Mat instance_id_mask;
        if (this->mask_encoding_ == "mono16") {
            instance_id_mask = cv::Mat::zeros(orig_size, CV_16UC1);
        } else {
            instance_id_mask = cv::Mat::zeros(orig_size, CV_8UC1);
        }

        size_t num_detections = static_cast<size_t>(result.num);
        size_t num_items = std::min({num_detections, result.masks.size(),
                                    result.classes.size(), result.boxes.size()});

        if (num_detections > 0 && (num_detections != result.masks.size() ||
                                num_detections != result.classes.size() ||
                                num_detections != result.boxes.size())) {
            RCLCPP_WARN(this->get_logger(),
                "[generateInstanceIdMaskROI] Mismatch between result.num (%d) and vector sizes (masks: %zu, classes: %zu, boxes: %zu). Processing %zu items.",
                        result.num, result.masks.size(), result.classes.size(), result.boxes.size(), num_items);
        }
        
        // Estos factores se usan para pasar de net_input_size a imagen original.
        double scale_x = static_cast<double>(orig_size.width) / net_input_size.width;
        double scale_y = static_cast<double>(orig_size.height) / net_input_size.height;
        // Estos factores se usan para pasar de net_input_size a la resolución de la máscara de salida.
        // Se asume que la máscara (raw) se produce en dimensiones relacionadas a net_input_size.
        // (Si no fuera así, se deberá ajustar este cálculo.)
        for (size_t item_idx = 0; item_idx < num_items; ++item_idx) {
            if (result.masks[item_idx].data.empty() ||
                result.masks[item_idx].width <= 0 ||
                result.masks[item_idx].height <= 0) {
                RCLCPP_WARN(this->get_logger(),
                    "[generateInstanceIdMaskROI] Item %zu has empty or invalid mask data.", item_idx);
                continue;
            }
            const deploy::Box& net_box = result.boxes[item_idx];
            // Bounding box en la imagen original
            cv::Rect orig_roi_rect(
                static_cast<int>(net_box.left * scale_x),
                static_cast<int>(net_box.top * scale_y),
                static_cast<int>((net_box.right - net_box.left) * scale_x),
                static_cast<int>((net_box.bottom - net_box.top) * scale_y)
            );
            orig_roi_rect &= cv::Rect(0, 0, orig_size.width, orig_size.height);
            if (orig_roi_rect.width <= 0 || orig_roi_rect.height <= 0) {
                RCLCPP_WARN(this->get_logger(),
                    "[generateInstanceIdMaskROI] Item %zu has zero or negative ROI size after scaling.", item_idx);
                continue;
            }
            // Cargar la máscara cruda (raw) del objeto
            cv::Mat raw_instance_mask(result.masks[item_idx].height,
                                    result.masks[item_idx].width, CV_8UC1,
                                    const_cast<void*>(static_cast<const void*>(result.masks[item_idx].data.data())));
            if (raw_instance_mask.empty()) {
                RCLCPP_WARN(this->get_logger(),
                    "[generateInstanceIdMaskROI] raw_instance_mask for item %zu is empty.", item_idx);
                continue;
            }
            // Calcular el ROI en la raw mask.
            double raw_scale_x = static_cast<double>(raw_instance_mask.cols) / net_input_size.width;
            double raw_scale_y = static_cast<double>(raw_instance_mask.rows) / net_input_size.height;
            cv::Rect raw_roi(
                static_cast<int>(net_box.left * raw_scale_x),
                static_cast<int>(net_box.top * raw_scale_y),
                static_cast<int>((net_box.right - net_box.left) * raw_scale_x),
                static_cast<int>((net_box.bottom - net_box.top) * raw_scale_y)
            );
            raw_roi &= cv::Rect(0, 0, raw_instance_mask.cols, raw_instance_mask.rows);
            if (raw_roi.width <= 0 || raw_roi.height <= 0) {
                RCLCPP_WARN(this->get_logger(),
                    "[generateInstanceIdMaskROI] Item %zu has zero or negative ROI in raw mask.", item_idx);
                continue;
            }
            // Redimensionar solo la submáscara correspondiente al ROI, en lugar de redimensionar la máscara completa.
            cv::Mat submask = raw_instance_mask(raw_roi);
            cv::Mat resized_mask;
            cv::resize(submask, resized_mask, orig_roi_rect.size(), 0, 0, cv::INTER_NEAREST);
            // Opcional: Si se requiere asegurar que el resultado sea binario.
            cv::threshold(resized_mask, resized_mask, 0, 255, cv::THRESH_BINARY);
            
            uint16_t instance_pixel_value = static_cast<uint16_t>(item_idx + 1);
            cv::Mat roi = instance_id_mask(orig_roi_rect);
            if (this->mask_encoding_ == "mono16") {
                roi.setTo(instance_pixel_value, resized_mask);
            } else { // mono8
                if (instance_pixel_value > 255) {
                    RCLCPP_WARN_ONCE(this->get_logger(),
                        "[generateInstanceIdMaskROI] Instance ID %u exceeds 255 in mono8. Clamping.", instance_pixel_value);
                    roi.setTo(static_cast<unsigned char>(255), resized_mask);
                } else {
                    roi.setTo(static_cast<unsigned char>(instance_pixel_value), resized_mask);
                }
            }
        }
        return instance_id_mask;
    }

    void modelProcessingLoop() {
        RCLCPP_INFO(this->get_logger(), "Model processing thread started.");
        while (rclcpp::ok() && !stop_processing_thread_.load()) {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            cv_batch_ready_.wait(lock, [this] {
                // Esperar hasta que todas las imágenes sean recibidas O se indique parar
                return (received_[0] && received_[1] && received_[2]) || stop_processing_thread_.load();
            });

            if (stop_processing_thread_.load()) {
                RCLCPP_INFO(this->get_logger(), "Model processing thread stopping...");
                break; // Salir del bucle si se indica parar
            }


            if (!video_writers_initialized_ && (enable_inferred_video_writing_ || enable_mask_video_writing_)) {
                RCLCPP_INFO(this->get_logger(), "Initializing video writers...");
                // Inicializar los escritores de vídeo
                initializeVideoWriters();
            }

            processBatchSharedMemory();

            std::fill(received_.begin(), received_.end(), false);
            
        }
        RCLCPP_INFO(this->get_logger(), "Model processing thread finished.");
    }


    void processBatchSharedMemory()
    {
        using clock = std::chrono::steady_clock;
        auto t_batch_entry_time = clock::now(); 
        if (YoloBatchNode::last_publish_time.time_since_epoch().count() != 0 && 
            YoloBatchNode::last_publish_time < t_batch_entry_time) { 
            auto inter_batch_wait_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_batch_entry_time - YoloBatchNode::last_publish_time);
            RCLCPP_INFO(this->get_logger(), "Inter-batch wait time (SharedMem): %ld ms", inter_batch_wait_duration.count());
        }
        auto t_total_start = clock::now();
        auto t_buffer_copy_start = clock::now(); // Declare t_buffer_copy_start here

        std::array<cv::Mat, 3> originals;
        std::array<cv::Size, 3> orig_sizes;
        std::array<std_msgs::msg::Header, 3> current_batch_headers;
        std::array<cv::Mat, 3> instance_id_masks_for_batch; 
        std::vector<deploy::Image> img_batch; 
        std::vector<cv::Mat> temp_resized_images_for_batch_data;
        
        
        for (size_t i = 0; i < 3; ++i) {
            if (image_buffers_[i].empty()) { 
                RCLCPP_ERROR(this->get_logger(), "Image buffer %zu is empty in processBatchSharedMemory. Skipping batch.", i);
                return; 
            }
            originals[i] = image_buffers_[i].clone(); 
            orig_sizes[i] = originals[i].size();
            current_batch_headers[i] = image_headers_[i]; // Copiar el header para este lote
        }
        auto t_buffer_copy_end = clock::now();
        auto dt_buffer_copy_us = std::chrono::duration_cast<std::chrono::microseconds>(t_buffer_copy_end - t_buffer_copy_start).count();

        RCLCPP_INFO(this->get_logger(), "Processing batch with image timestamps: Cam1=%d.%09u, Cam2=%d.%09u, Cam3=%d.%09u",
                    current_batch_headers[0].stamp.sec, current_batch_headers[0].stamp.nanosec,
                    current_batch_headers[1].stamp.sec, current_batch_headers[1].stamp.nanosec,
                    current_batch_headers[2].stamp.sec, current_batch_headers[2].stamp.nanosec);
        
        // Calcular la diferencia máxima de timestamps en el lote actual
        if (current_batch_headers[0].stamp.sec > 0 && current_batch_headers[1].stamp.sec > 0 && current_batch_headers[2].stamp.sec > 0) {
            auto to_nanos = [](const builtin_interfaces::msg::Time& t) {
                return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
            };
            int64_t ts0_ns = to_nanos(current_batch_headers[0].stamp);
            int64_t ts1_ns = to_nanos(current_batch_headers[1].stamp);
            int64_t ts2_ns = to_nanos(current_batch_headers[2].stamp);

            int64_t min_ts = std::min({ts0_ns, ts1_ns, ts2_ns});
            int64_t max_ts = std::max({ts0_ns, ts1_ns, ts2_ns});
            double diff_ms = static_cast<double>(max_ts - min_ts) / 1.0e6;
            RCLCPP_INFO(this->get_logger(), "Max timestamp difference in current batch: %.3f ms", diff_ms);
        }
        
        auto t_pre_start = clock::now();
        // Removed redeclaration of img_batch and temp_resized_images_for_batch_data
        temp_resized_images_for_batch_data.reserve(3);
        cv::Size network_input_target_size(input_width_, input_height_);
        unsigned char* current_pinned_ptr = h_pinned_input_buffer_;

        for (size_t i = 0; i < 3; ++i) {
            cv::Mat resized_img; 
            if (originals[i].empty()) {
                 RCLCPP_ERROR(this->get_logger(), "Original image %zu is empty before resize in processBatchSharedMemory. Skipping batch.", i);
                 return;
            }
            cv::resize(originals[i], resized_img, network_input_target_size, 0, 0, cv::INTER_LINEAR);
            
            if (resized_img.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Resized image %zu is empty in processBatchSharedMemory. Skipping batch.", i);
                return;
            }

            if (use_pinned_input_memory_ && h_pinned_input_buffer_) {
                if (!resized_img.isContinuous()) {
                    resized_img = resized_img.clone(); 
                }
                size_t current_image_bytes = static_cast<size_t>(resized_img.cols * resized_img.rows * resized_img.channels());
                if (current_image_bytes == single_image_pinned_bytes_ && resized_img.channels() == input_channels_) {
                    std::memcpy(current_pinned_ptr, resized_img.data, single_image_pinned_bytes_);
                    img_batch.emplace_back(current_pinned_ptr, resized_img.cols, resized_img.rows); 
                    current_pinned_ptr += single_image_pinned_bytes_;
                } else {
                    RCLCPP_WARN(this->get_logger(), "Resized image %zu (dims: %dx%d, channels: %d, expected_bytes: %zu, actual_bytes: %zu, expected_channels: %d) size/channel mismatch for pinned memory. Using its own data.",
                                i, resized_img.cols, resized_img.rows, resized_img.channels(), single_image_pinned_bytes_, current_image_bytes, input_channels_);
                    temp_resized_images_for_batch_data.push_back(resized_img.clone());
                    img_batch.emplace_back(temp_resized_images_for_batch_data.back().data, temp_resized_images_for_batch_data.back().cols, temp_resized_images_for_batch_data.back().rows);
                }
            } else {
                temp_resized_images_for_batch_data.push_back(resized_img.clone());
                img_batch.emplace_back(temp_resized_images_for_batch_data.back().data, temp_resized_images_for_batch_data.back().cols, temp_resized_images_for_batch_data.back().rows);
            }
        }
        auto t_pre_end = clock::now();

        if (img_batch.size() != 3) {
            RCLCPP_ERROR(this->get_logger(), "Incorrect batch size before predict in processBatchSharedMemory: %zu. Expected 3. Skipping inference.", img_batch.size());
            return;
        }

        auto t_inf_start = clock::now();
        std::vector<deploy::SegmentRes> results;
        try {
            results = model_->predict(img_batch);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during model_->predict() in processBatchSharedMemory: %s. Batch processing aborted.", e.what());
            return;
        }
        auto t_inf_end = clock::now();
        
        auto t_post_loop_start = clock::now(); // Corrected uto to auto
        std::array<long, 3> mask_gen_duration_us{};
        std::array<long, 3> msg_creation_duration_us{};
        std::array<long, 3> publish_duration_us{};
        
        // for (size_t i = 0; i < results.size() && i < 3; ++i) {
        //     if (orig_sizes[i].width <= 0 || orig_sizes[i].height <= 0) {
        //         RCLCPP_ERROR(this->get_logger(), "Original size for image %zu is invalid. Skipping post-processing.", i);
        //         continue;
        //     }
            
        //     auto t_mask_gen_start = clock::now();
        //     instance_id_mask_cv = generateInstanceIdMaskROI(results[i], orig_sizes[i], network_input_target_size);
        //     auto t_mask_gen_end = clock::now();
        //     mask_gen_duration_us[i] = std::chrono::duration_cast<std::chrono::microseconds>(t_mask_gen_end - t_mask_gen_start).count();
        
        //     auto t_msg_create_start = clock::now();
        //     auto instance_info_msg = std::make_unique<yolo_custom_interfaces::msg::InstanceSegmentationInfo>();
        //     instance_info_msg->header = current_batch_headers[i]; 

        //     cv_bridge::CvImage cv_img_mask_instance;
        //     cv_img_mask_instance.header = instance_info_msg->header; 
        //     cv_img_mask_instance.encoding = this->mask_encoding_; 
        //     cv_img_mask_instance.image = instance_id_mask_cv;
        //     instance_info_msg->mask = *cv_img_mask_instance.toImageMsg();

        //     size_t num_detected_instances = static_cast<size_t>(results[i].num);
        //     num_detected_instances = std::min({num_detected_instances, results[i].scores.size(), results[i].classes.size()});

        //     instance_info_msg->scores.reserve(num_detected_instances);
        //     instance_info_msg->classes.reserve(num_detected_instances);

        //     for (size_t j = 0; j < num_detected_instances; ++j) {
        //         instance_info_msg->scores.push_back(results[i].scores[j]);
        //         instance_info_msg->classes.push_back(results[i].classes[j]);
        //     }
        //     auto t_msg_create_end = clock::now();
        //     msg_creation_duration_us[i] = std::chrono::duration_cast<std::chrono::microseconds>(t_msg_create_end - t_msg_create_start).count();
            
        //     auto t_publish_start = clock::now();
        //     instance_info_pubs_[i]->publish(std::move(instance_info_msg));
        //     auto t_publish_end = clock::now();
        //     publish_duration_us[i] = std::chrono::duration_cast<std::chrono::microseconds>(t_publish_end - t_publish_start).count();
        // }

        // Variable para la máscara de la imagen actual en el bucle (usada para el mensaje ROS)
        cv::Mat instance_id_mask_cv; // Esta se genera para cada imagen

        for (size_t i = 0; i < 3; ++i) { // Asegurarse de iterar 3 veces si hay resultados para las 3
            if (i >= results.size()) {
                RCLCPP_WARN(this->get_logger(), "No result for image index %zu (results.size() = %zu). Creating empty mask.", i, results.size());
                // Crear una máscara vacía/negra para instance_id_masks_for_batch[i] si no hay resultado
                instance_id_masks_for_batch[i] = cv::Mat::zeros(orig_sizes[i].empty() ? cv::Size(10,10) : orig_sizes[i], (mask_encoding_ == "mono16" ? CV_16UC1 : CV_8UC1));
                if (orig_sizes[i].width <= 0 || orig_sizes[i].height <= 0) { // Si no hay resultado, y el tamaño original es inválido, saltar el mensaje
                    continue;
                }
                // Para el mensaje ROS, también usar una máscara vacía si no hay resultado
                instance_id_mask_cv = instance_id_masks_for_batch[i].clone();
            } else {
                 if (orig_sizes[i].width <= 0 || orig_sizes[i].height <= 0) {
                    RCLCPP_ERROR(this->get_logger(), "Original size for image %zu is invalid. Skipping post-processing for this image.", i);
                    instance_id_masks_for_batch[i] = cv::Mat::zeros(network_input_target_size, (mask_encoding_ == "mono16" ? CV_16UC1 : CV_8UC1)); // Placeholder
                    continue;
                }
                auto t_mask_gen_start = clock::now();
                instance_id_mask_cv = generateInstanceIdMaskROI(results[i], orig_sizes[i], network_input_target_size);
                auto t_mask_gen_end = clock::now();
                mask_gen_duration_us[i] = std::chrono::duration_cast<std::chrono::microseconds>(t_mask_gen_end - t_mask_gen_start).count();
            
                // Llenar el array que se pasará a las funciones de vídeo
                if (instance_id_mask_cv.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Generated instance_id_mask_cv for image %zu is empty. Using black mask for video.", i);
                    instance_id_masks_for_batch[i] = cv::Mat::zeros(orig_sizes[i], (mask_encoding_ == "mono16" ? CV_16UC1 : CV_8UC1));
                } else {
                    instance_id_masks_for_batch[i] = instance_id_mask_cv.clone();
                }
            }
        
            auto t_msg_create_start = clock::now();
            auto instance_info_msg = std::make_unique<yolo_custom_interfaces::msg::InstanceSegmentationInfo>();
            instance_info_msg->header = current_batch_headers[i]; 

            cv_bridge::CvImage cv_img_mask_instance;
            cv_img_mask_instance.header = instance_info_msg->header; 
            cv_img_mask_instance.encoding = this->mask_encoding_; 
            
            if (instance_id_mask_cv.empty()) { // Usar la máscara generada para el mensaje
                 RCLCPP_ERROR(this->get_logger(), "instance_id_mask_cv for image %zu (ROS msg) is EMPTY. Using black mask.", i);
                 cv_img_mask_instance.image = cv::Mat::zeros(orig_sizes[i].empty() ? cv::Size(10,10) : orig_sizes[i], (mask_encoding_ == "mono16" ? CV_16UC1 : CV_8UC1));
            } else {
                cv_img_mask_instance.image = instance_id_mask_cv;
            }
            instance_info_msg->mask = *cv_img_mask_instance.toImageMsg();

            if (i < results.size()){ // Solo llenar scores y classes si hay resultado
                size_t num_detected_instances = static_cast<size_t>(results[i].num);
                num_detected_instances = std::min({num_detected_instances, results[i].scores.size(), results[i].classes.size()});

                instance_info_msg->scores.reserve(num_detected_instances);
                instance_info_msg->classes.reserve(num_detected_instances);

                for (size_t j = 0; j < num_detected_instances; ++j) {
                    instance_info_msg->scores.push_back(results[i].scores[j]);
                    instance_info_msg->classes.push_back(results[i].classes[j]);
                }
            }
            auto t_msg_create_end = clock::now();
            msg_creation_duration_us[i] = std::chrono::duration_cast<std::chrono::microseconds>(t_msg_create_end - t_msg_create_start).count();
            
            auto t_publish_start = clock::now();
            instance_info_pubs_[i]->publish(std::move(instance_info_msg));
            auto t_publish_end = clock::now();
            publish_duration_us[i] = std::chrono::duration_cast<std::chrono::microseconds>(t_publish_end - t_publish_start).count();
        }
        auto t_post_loop_end = clock::now();

        // Escritura de vídeo después de que todos los mensajes ROS han sido preparados/publicados para el lote
        if (video_writers_initialized_) {
            if (enable_inferred_video_writing_ && video_writer_inferred_.isOpened()) {

                RCLCPP_DEBUG(this->get_logger(), "Calling writeInferredImagesToVideo. originals[0] empty: %d, results empty: %d, instance_id_masks_for_batch[0] empty: %d",
                            originals[0].empty(), results.empty(), (instance_id_masks_for_batch.empty() ? -1 : instance_id_masks_for_batch[0].empty()) );
                writeInferredImagesToVideo(originals, results, instance_id_masks_for_batch, orig_sizes, network_input_target_size);
            }
            if (enable_mask_video_writing_ && video_writer_instance_mask_.isOpened()) {
                RCLCPP_DEBUG(this->get_logger(), "Calling writeInstanceMasksToVideo. results empty: %d, instance_id_masks_for_batch[0] empty: %d",
                             results.empty(), (instance_id_masks_for_batch.empty() ? -1 : instance_id_masks_for_batch[0].empty()) );
                writeInstanceMasksToVideo(instance_id_masks_for_batch, results, orig_sizes);
            }
        }

        auto dt_pre_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_end - t_pre_start).count();
        auto dt_inf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_inf_end - t_inf_start).count();
        auto dt_post_loop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_post_loop_end - t_post_loop_start).count();
        auto dt_total_from_buffer_copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_post_loop_end - t_buffer_copy_start).count();

        RCLCPP_INFO(this->get_logger(), "--- Batch Processing Summary ---");
        RCLCPP_INFO(this->get_logger(), "  Buffer Copy   : %ld us", dt_buffer_copy_us);
        RCLCPP_INFO(this->get_logger(), "  Preprocessing : %ld ms", dt_pre_ms);
        RCLCPP_INFO(this->get_logger(), "  Inference     : %ld ms", dt_inf_ms);
        RCLCPP_INFO(this->get_logger(), "  Postproc. Loop: %ld ms (Total for 3 images' maskgen, msgcreate, publish)", dt_post_loop_ms);
        for(size_t i=0; i<3; ++i) {
            if (orig_sizes[i].width > 0 && orig_sizes[i].height > 0 && i < results.size()) {
                 RCLCPP_INFO(this->get_logger(), "    Img[%zu] Times: MaskGen=%ldus, MsgCreate=%ldus, PublishCall=%ldus | SumItemPost=%ldus",
                    i, mask_gen_duration_us[i], msg_creation_duration_us[i], publish_duration_us[i],
                    (mask_gen_duration_us[i] + msg_creation_duration_us[i] + publish_duration_us[i]));
            }
        }
        RCLCPP_INFO(this->get_logger(), "  Total Batch Time (BufCopy+Pre+Inf+PostLoop): %ld ms", dt_total_from_buffer_copy_ms);
        
        auto current_batch_processed_time = clock::now();
        if (YoloBatchNode::last_publish_time.time_since_epoch().count() != 0) {
            auto time_since_last_batch_processed = std::chrono::duration_cast<std::chrono::duration<double>>(current_batch_processed_time - YoloBatchNode::last_publish_time);
             if (time_since_last_batch_processed.count() > 1e-9) { 
                double frequency = 1.0 / time_since_last_batch_processed.count();
                RCLCPP_INFO(this->get_logger(), "  Batch Processing Frequency: %.2f Hz", frequency);
            }
        }
        YoloBatchNode::last_publish_time = current_batch_processed_time; 


    // --- Calcular la diferencia máxima de timestamps en el lote actual ---
    double max_diff_ms = 0.0;
    if (current_batch_headers[0].stamp.sec > 0 && current_batch_headers[1].stamp.sec > 0 && current_batch_headers[2].stamp.sec > 0) {
        auto to_nanos = [](const builtin_interfaces::msg::Time& t) {
            return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
        };
        int64_t ts0_ns = to_nanos(current_batch_headers[0].stamp);
        int64_t ts1_ns = to_nanos(current_batch_headers[1].stamp);
        int64_t ts2_ns = to_nanos(current_batch_headers[2].stamp);
        int64_t min_ts = std::min({ts0_ns, ts1_ns, ts2_ns});
        int64_t max_ts = std::max({ts0_ns, ts1_ns, ts2_ns});
        max_diff_ms = static_cast<double>(max_ts - min_ts) / 1.0e6;
        RCLCPP_INFO(this->get_logger(), "Max timestamp difference in current batch: %.3f ms", max_diff_ms);
    }

    // --- Agregar la media (promedio) de los tiempos importantes entre imágenes ---
    long total_mask_gen = 0, total_msg_create = 0, total_publish = 0;
    int valid_count = 0;
    for (size_t i = 0; i < 3; ++i) {
        // Se considera válida la medición si el tamaño original es válido y hay resultado para esa imagen
        if (orig_sizes[i].width > 0 && orig_sizes[i].height > 0 && i < results.size()) {
            total_mask_gen += mask_gen_duration_us[i];
            total_msg_create += msg_creation_duration_us[i];
            total_publish += publish_duration_us[i];
            valid_count++;
        }
    }
    long avg_mask_gen = (valid_count > 0) ? total_mask_gen / valid_count : 0;
    long avg_msg_create = (valid_count > 0) ? total_msg_create / valid_count : 0;
    long avg_publish = (valid_count > 0) ? total_publish / valid_count : 0;

    // --- Escritura de tiempos al archivo ---
    std::ofstream timing_file("frame_times.txt", std::ios_base::app);
    if (timing_file.is_open()) {
        // Se escribe la hora actual (timestamp en segundos), el max timestamp diff, y todos los tiempos medidos separados por comas (CSV)
        auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        timing_file << now_time << ", " 
                    << dt_buffer_copy_us << ", " 
                    << dt_pre_ms << ", " 
                    << dt_inf_ms << ", " 
                    << dt_post_loop_ms << ", " 
                    << dt_total_from_buffer_copy_ms << ", " 
                    << max_diff_ms << ", " 
                    << avg_mask_gen << ", " 
                    << avg_msg_create << ", " 
                    << avg_publish;
        timing_file << std::endl;
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open frame_times.txt for writing.");
    }
}


    void initializeVideoWriters() {
        if (video_writers_initialized_) return;

        fs::path output_dir(video_output_path_str_);

        if (enable_inferred_video_writing_) {
            fs::path inferred_video_full_path = output_dir / inferred_video_filename_;
            RCLCPP_INFO(this->get_logger(), "Initializing inferred video writer: %s", inferred_video_full_path.string().c_str());
            if (!video_writer_inferred_.open(inferred_video_full_path.string(), cv::VideoWriter::fourcc('M','J','P','G'), video_fps_, video_frame_size_, true)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open video writer for inferred images. Disabling.");
                enable_inferred_video_writing_ = false;
            }
        }

        if (enable_mask_video_writing_) {
            fs::path mask_video_full_path = output_dir / mask_video_filename_;
            RCLCPP_INFO(this->get_logger(), "Initializing mask video writer: %s", mask_video_full_path.string().c_str());
            if (!video_writer_instance_mask_.open(mask_video_full_path.string(), cv::VideoWriter::fourcc('M','J','P','G'), video_fps_, video_frame_size_, true)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open video writer for instance masks. Disabling.");
                enable_mask_video_writing_ = false;
            }
        }
        video_writers_initialized_ = true;
    }

    cv::Scalar getRandomTone(const cv::Scalar& base_color, int seed) {
        cv::RNG rng(seed); // Usar el ID de instancia o un índice como semilla
        double variation_range = 100.0; // Rango de variación para cada canal de color (0-255)

        double b = base_color[0];
        double g = base_color[1];
        double r = base_color[2];

        // Generar variaciones aleatorias, intentando mantener la luminosidad general
        // Podríamos hacer esto más sofisticado (ej. convirtiendo a HSV, variando H y S, y volviendo a BGR)
        // pero una simple variación aditiva puede ser suficiente visualmente.
        b += rng.uniform(-variation_range, variation_range);
        g += rng.uniform(-variation_range, variation_range);
        r += rng.uniform(-variation_range, variation_range);

        // Asegurar que los valores estén en el rango [0, 255]
        return cv::Scalar(
            cv::saturate_cast<uchar>(b),
            cv::saturate_cast<uchar>(g),
            cv::saturate_cast<uchar>(r)
        );
    }

        void writeInferredImagesToVideo(
        const std::array<cv::Mat, 3>& original_images,
        const std::vector<deploy::SegmentRes>& batch_results,
        const std::array<cv::Mat, 3>& instance_id_masks, 
        const std::array<cv::Size, 3>& original_sizes,
        const cv::Size& network_input_size) // network_input_size es net_input_size de generateInstanceIdMaskROI
    {
        (void)original_sizes; // Ya no se usa directamente aquí si instance_id_masks y original_images coinciden en tamaño

        if (class_colors_.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "Class colors vector is empty. Cannot color inferred images for video.");
            return;
        }

        std::vector<cv::Mat> frames_to_stitch;
        frames_to_stitch.reserve(3);

        // Estas variables de escala no se estaban usando, las comento o elimino para evitar warnings.
        // double scale_x_for_debug = static_cast<double>(original_images[0].cols) / network_input_size.width;
        // double scale_y_for_debug = static_cast<double>(original_images[0].rows) / network_input_size.height;

        for (size_t i = 0; i < 3; ++i) {
            if (original_images[i].empty() || i >= batch_results.size() || instance_id_masks[i].empty()) {
                RCLCPP_WARN(this->get_logger(), "Missing data for image %zu in writeInferredImagesToVideo. Using black frame.", i);
                frames_to_stitch.push_back(cv::Mat::zeros(video_frame_size_.height, video_frame_size_.width / 3, CV_8UC3));
                continue;
            }

            if (original_images[i].size() != instance_id_masks[i].size()) {
                RCLCPP_ERROR(this->get_logger(), "Image %zu: Mismatch between original image size (%dx%d) and instance_id_mask size (%dx%d). Skipping coloring.",
                             i, original_images[i].cols, original_images[i].rows,
                             instance_id_masks[i].cols, instance_id_masks[i].rows);
                cv::Mat display_image_resized_error = original_images[i].clone();
                if (display_image_resized_error.rows != video_frame_size_.height || display_image_resized_error.cols != video_frame_size_.width / 3) {
                     cv::resize(display_image_resized_error, display_image_resized_error, cv::Size(video_frame_size_.width / 3, video_frame_size_.height));
                }
                frames_to_stitch.push_back(display_image_resized_error);
                continue;
            }


            cv::Mat display_image = original_images[i].clone();
            const deploy::SegmentRes& result = batch_results[i];
            const cv::Mat& current_instance_id_mask = instance_id_masks[i]; 

            size_t num_instances = std::min({static_cast<size_t>(result.num), result.classes.size(), result.scores.size(), result.boxes.size()});

            // Estas variables de escala no se estaban usando aquí, las comento o elimino.
            // double current_scale_x = static_cast<double>(original_images[i].cols) / network_input_size.width;
            // double current_scale_y = static_cast<double>(original_images[i].rows) / network_input_size.height;


            for (size_t inst_idx = 0; inst_idx < num_instances; ++inst_idx) {
                int class_id = result.classes[inst_idx];
                cv::Scalar base_class_color = class_colors_[class_id % class_colors_.size()]; 
                uint16_t instance_pixel_id_val = static_cast<uint16_t>(inst_idx + 1); // ID de instancia (1, 2, ...)

                // Generar una tonalidad ligeramente diferente para esta instancia específica
                cv::Scalar toned_instance_color = getRandomTone(base_class_color, static_cast<int>(instance_pixel_id_val));

                cv::Mat single_instance_binary_mask;
                if (mask_encoding_ == "mono16") {
                    cv::compare(current_instance_id_mask, instance_pixel_id_val, single_instance_binary_mask, cv::CMP_EQ);
                } else { // mono8
                     cv::compare(current_instance_id_mask, static_cast<uchar>(instance_pixel_id_val), single_instance_binary_mask, cv::CMP_EQ);
                }
                
                if (cv::countNonZero(single_instance_binary_mask) > 0) {
                    display_image.setTo(toned_instance_color, single_instance_binary_mask); // Usar el color con tonalidad
                }

                // --- INICIO DEBUG: Dibujar Bounding Box ---
                // if (inst_idx < result.boxes.size()) {
                //     const deploy::Box& net_box = result.boxes[inst_idx];
                //     // Necesitarías current_scale_x y current_scale_y si descomentas esto
                //     double dbg_scale_x = static_cast<double>(original_images[i].cols) / network_input_size.width;
                //     double dbg_scale_y = static_cast<double>(original_images[i].rows) / network_input_size.height;
                //     cv::Rect orig_bbox(
                //         static_cast<int>(net_box.left * dbg_scale_x),
                //         static_cast<int>(net_box.top * dbg_scale_y),
                //         static_cast<int>((net_box.right - net_box.left) * dbg_scale_x),
                //         static_cast<int>((net_box.bottom - net_box.top) * dbg_scale_y)
                //     );
                //     orig_bbox &= cv::Rect(0, 0, display_image.cols, display_image.rows); 
                //     cv::rectangle(display_image, orig_bbox, cv::Scalar(0, 255, 255), 1); 
                // }
                // --- FIN DEBUG ---
            }
            
            if (display_image.rows != video_frame_size_.height || display_image.cols != video_frame_size_.width / 3) {
                cv::resize(display_image, display_image, cv::Size(video_frame_size_.width / 3, video_frame_size_.height));
            }
            frames_to_stitch.push_back(display_image);
        }

        if (frames_to_stitch.size() == 3) {
            cv::Mat combined_frame;
            cv::hconcat(frames_to_stitch, combined_frame);
            if (!combined_frame.empty() && video_writer_inferred_.isOpened()) {
                video_writer_inferred_.write(combined_frame);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "Could not prepare 3 frames for inferred video. Skipping frame.");
        }
    }
    
    void writeInstanceMasksToVideo(
        const std::array<cv::Mat, 3>& instance_id_masks, // CV_16UC1 o CV_8UC1, con IDs de instancia (1, 2, ...)
        const std::vector<deploy::SegmentRes>& batch_results,
        const std::array<cv::Size, 3>& original_sizes)
    {
        if (class_colors_.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "Class colors vector is empty. Cannot color instance masks for video.");
            return;
        }

        std::vector<cv::Mat> colored_masks_to_stitch;
        colored_masks_to_stitch.reserve(3);

        for (size_t i = 0; i < 3; ++i) {
            if (instance_id_masks[i].empty() || i >= batch_results.size()) {
                RCLCPP_WARN(this->get_logger(), "Missing data for image %zu in writeInstanceMasksToVideo. Using black frame.", i);
                colored_masks_to_stitch.push_back(cv::Mat::zeros(video_frame_size_.height, video_frame_size_.width / 3, CV_8UC3));
                continue;
            }

            const cv::Mat& current_id_mask = instance_id_masks[i]; // Máscara con IDs de instancia
            const deploy::SegmentRes& result = batch_results[i];
            
            // Crear la máscara coloreada con el tamaño original de la imagen de esa cámara
            cv::Mat colored_mask_display = cv::Mat::zeros(original_sizes[i], CV_8UC3); 
            if (original_sizes[i].width <=0 || original_sizes[i].height <=0){
                 RCLCPP_ERROR(this->get_logger(), "writeInstanceMasksToVideo: original_sizes[%zu] is invalid. Using black frame.", i);
                 colored_masks_to_stitch.push_back(cv::Mat::zeros(video_frame_size_.height, video_frame_size_.width / 3, CV_8UC3));
                 continue;
            }


            size_t num_instances_in_result = std::min({static_cast<size_t>(result.num), result.classes.size()});

            // Iterar sobre cada píxel de la máscara de IDs de instancia
            for (int r = 0; r < current_id_mask.rows; ++r) {
                for (int c = 0; c < current_id_mask.cols; ++c) {
                    uint16_t instance_id_from_mask = 0; // El ID de instancia (1, 2, 3...) leído de la máscara
                    if (mask_encoding_ == "mono16") {
                        instance_id_from_mask = current_id_mask.at<uint16_t>(r, c);
                    } else { // mono8
                        instance_id_from_mask = static_cast<uint16_t>(current_id_mask.at<uchar>(r, c));
                    }

                    // Si el píxel pertenece a una instancia (ID > 0)
                    if (instance_id_from_mask > 0) {
                        // El ID en la máscara es item_idx + 1. Así que item_idx es instance_id_from_mask - 1.
                        size_t item_idx = static_cast<size_t>(instance_id_from_mask - 1);

                        if (item_idx < num_instances_in_result) { // Asegurarse que el índice es válido
                            int class_id = result.classes[item_idx];
                            cv::Scalar base_color = class_colors_[class_id % class_colors_.size()];
                            // Usar instance_id_from_mask (que es único por instancia) como semilla para la tonalidad
                            cv::Scalar toned_color = getRandomTone(base_color, instance_id_from_mask); 
                            colored_mask_display.at<cv::Vec3b>(r, c) = cv::Vec3b(toned_color[0], toned_color[1], toned_color[2]);
                        }
                    }
                }
            }
            
            if (colored_mask_display.rows != video_frame_size_.height || colored_mask_display.cols != video_frame_size_.width / 3) {
                cv::resize(colored_mask_display, colored_mask_display, cv::Size(video_frame_size_.width / 3, video_frame_size_.height), 0, 0, cv::INTER_NEAREST);
            }
            colored_masks_to_stitch.push_back(colored_mask_display);
        }

        if (colored_masks_to_stitch.size() == 3) {
            cv::Mat combined_colored_mask_frame;
            cv::hconcat(colored_masks_to_stitch, combined_colored_mask_frame);
            if (!combined_colored_mask_frame.empty() && video_writer_instance_mask_.isOpened()) {
                video_writer_instance_mask_.write(combined_colored_mask_frame);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "Could not prepare 3 frames for mask video. Skipping frame.");
        }
    }
};

// Define and initialize the static member AFTER the class definition
std::chrono::steady_clock::time_point YoloBatchNode::last_publish_time = std::chrono::steady_clock::now();

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloBatchNode>();
    node->initSubscriptions();
    // Usar un MultiThreadedExecutor para permitir que los callbacks del grupo Reentrant
    // (y otros callbacks/timers en diferentes grupos) se ejecuten concurrentemente.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    RCLCPP_INFO(node->get_logger(), "Spinning node with MultiThreadedExecutor.");
    executor.spin();
    rclcpp::shutdown();
    return 0;
}