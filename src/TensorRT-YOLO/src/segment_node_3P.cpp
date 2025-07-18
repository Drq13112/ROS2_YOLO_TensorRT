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
#include <atomic>
#include <thread>
#include <condition_variable>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include <image_transport/image_transport.hpp>
#include "std_msgs/msg/u_int32.hpp"

#include <chrono>
#include <thread>

// Incluir las cabeceras de la librería deploy
#include "deploy/model.hpp"
#include "deploy/option.hpp"
#include "deploy/result.hpp"
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include "yolo_custom_interfaces/msg/instance_segmentation_info.hpp"

#include "chrono/ChronoTimer.hpp"

#define LOG_CUDA_ERROR(err, msg_prefix, logger)                              \
    if (err != cudaSuccess)                                                  \
    {                                                                        \
        RCLCPP_ERROR(logger, "%s: %s", msg_prefix, cudaGetErrorString(err)); \
    }

namespace fs = std::filesystem;
using namespace std::chrono_literals;

// Estructuras globales para medir diferencias de tiempo entre nodos
std::mutex g_callback_arrival_mutex;
std::map<std::string, timespec> g_latest_callback_times;

std::mutex g_publish_mutex;
std::map<std::string, timespec> g_latest_publish_times;

// Variables globales para la visualización en tiempo real
std::mutex g_display_mutex;
std::condition_variable g_display_cv;
std::map<std::string, cv::Mat> g_display_images;
std::map<std::string, bool> g_new_image_flags = {{"left", false}, {"front", false}, {"right", false}};
std::atomic<bool> g_stop_display_thread{false};

class YoloBatchNode : public rclcpp::Node
{
public:
    YoloBatchNode(const std::string &node_name,
                  const std::string &image_topic_name,
                  const std::string &output_topic_suffix,
                  bool enable_inferred_video,
                  bool enable_mask_video,
                  const std::string &video_path,
                  double fps,
                  const cv::Size &single_video_frame_size,
                  const std::string &image_transport_type = "compressed",
                  bool enable_measure_times = false,
                  bool enable_realtime_display = false)

        : Node(node_name),
          output_topic_suffix_(output_topic_suffix), // Guardar para nombres de archivo
          enable_inferred_video_writing_(enable_inferred_video),
          enable_mask_video_writing_(enable_mask_video),
          video_output_path_str_(video_path),
          video_fps_(fps),
          single_video_frame_size_(single_video_frame_size),
          image_transport_type_(image_transport_type),
          enable_measure_times_(enable_measure_times),
          enable_realtime_display_(enable_realtime_display)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing %s...", node_name.c_str());

        // ----------- Parámetros del nodo -----------
        this->declare_parameter<std::string>("engine_path", "yolo11m-seg.engine");
        this->declare_parameter<int>("input_width", 640);
        this->declare_parameter<int>("input_height", 416);
        this->declare_parameter<std::string>("mask_encoding", "mono8");
        this->declare_parameter<bool>("use_pinned_input_memory", true);
        this->declare_parameter<int>("input_channels", 3);

        auto engine_path = this->get_parameter("engine_path").get_value<std::string>();
        input_width_ = this->get_parameter("input_width").get_value<int>();
        input_height_ = this->get_parameter("input_height").get_value<int>();
        mask_encoding_ = this->get_parameter("mask_encoding").get_value<std::string>();
        use_pinned_input_memory_ = this->get_parameter("use_pinned_input_memory").get_value<bool>();
        input_channels_ = this->get_parameter("input_channels").get_value<int>();

        RCLCPP_INFO(this->get_logger(), "Engine path: %s", engine_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Input resized image size: %dx%d", input_width_, input_height_);
        RCLCPP_INFO(this->get_logger(), "Subscribing to: [%s]", image_topic_name.c_str());
        RCLCPP_INFO(this->get_logger(), "Reading topic in mode [%s]", image_transport_type_.c_str());
        RCLCPP_INFO(this->get_logger(), "Output topic suffix: [%s]", output_topic_suffix_.c_str());
        RCLCPP_INFO(this->get_logger(), "Writing inferred video to: [%s]", inferred_video_filename_.c_str());
        RCLCPP_INFO(this->get_logger(), "Writing mask video to: [%s]", mask_video_filename_.c_str());

        image_topic_name_ = image_topic_name;

        // Configuración de vídeo (nombres de archivo específicos del nodo)
        inferred_video_filename_ = "inferred_" + output_topic_suffix_ + ".avi";
        mask_video_filename_ = "mask_" + output_topic_suffix_ + ".avi";

        // --------- QoS profile ---------
        rclcpp::QoS qos_sensors(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
        qos_sensors.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        // qos_sensors.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        qos_sensors.keep_last(1);
        qos_sensors.durability_volatile();

        rclcpp::QoS qos_sensors_pub(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
        qos_sensors_pub.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        // qos_sensors_pub.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        qos_sensors_pub.keep_last(1);
        qos_sensors_pub.durability_volatile();

        // --------- Paleta de colores ---------
        class_colors_.push_back(cv::Scalar(255, 0, 0));     // Blue     -> Person
        class_colors_.push_back(cv::Scalar(0, 255, 0));     // Green    -> Car
        class_colors_.push_back(cv::Scalar(0, 0, 255));     // Red      -> Truck
        class_colors_.push_back(cv::Scalar(255, 255, 0));   // Yellow   -> Bus
        class_colors_.push_back(cv::Scalar(0, 255, 255));   // Cyan     -> Motorcycle
        class_colors_.push_back(cv::Scalar(255, 0, 255));   // Magenta
        class_colors_.push_back(cv::Scalar(192, 192, 192)); // Silver
        class_colors_.push_back(cv::Scalar(128, 128, 128)); // Gray
        class_colors_.push_back(cv::Scalar(128, 0, 0));     // Maroon
        class_colors_.push_back(cv::Scalar(128, 128, 0));   // Olive
        class_colors_.push_back(cv::Scalar(0, 128, 0));     // Dark Green
        class_colors_.push_back(cv::Scalar(128, 0, 128));   // Purple
        class_colors_.push_back(cv::Scalar(0, 128, 128));   // Teal
        class_colors_.push_back(cv::Scalar(0, 0, 128));     // Navy
        class_colors_.push_back(cv::Scalar(255, 165, 0));   // Orange
        class_colors_.push_back(cv::Scalar(255, 192, 203)); // Pink

        //  -------- Configuración del buffer de entrada ---------
        if (use_pinned_input_memory_)
        {
            single_image_pinned_bytes_ = static_cast<size_t>(input_width_ * input_height_ * input_channels_);
            cudaError_t err = cudaHostAlloc(&h_pinned_input_buffer_, single_image_pinned_bytes_, cudaHostAllocDefault);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(this->get_logger(), "[%s] Failed to allocate pinned host memory: %s. Disabling pinned memory.",
                             this->get_name(), cudaGetErrorString(err));
                h_pinned_input_buffer_ = nullptr;
                use_pinned_input_memory_ = false;
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "[%s] Allocated %zu bytes of pinned host memory.", this->get_name(), single_image_pinned_bytes_);
            }
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "[%s] Pinned input memory is disabled.", this->get_name());
            h_pinned_input_buffer_ = nullptr;
        }

        // -------- Inicializar inferencia --------
        deploy::InferOption option;
        option.enableSwapRB();

        // ----- Cargar el modelo de segmentación -----
        try
        {
            model_ = std::make_unique<deploy::SegmentModel>(engine_path, option);
            if (!model_)
            {
                throw std::runtime_error("Failed to load the engine for " + node_name);
            }
            RCLCPP_INFO(this->get_logger(), "Model loaded successfully for %s.", node_name.c_str());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error loading model for %s: %s", node_name.c_str(), e.what());
            rclcpp::shutdown(); // Consider just letting this node fail instead of shutting down all
            return;
        }

        // ------- Subscriptores -------

        image_sub_ = image_transport::create_subscription(
            this,
            image_topic_name_,
            std::bind(&YoloBatchNode::imageCallback, this, std::placeholders::_1),
            image_transport_type_,
            qos_sensors.get_rmw_qos_profile());

        // ------- Publicadores -------
        std::string instance_info_topic_name = "/segmentation/" + output_topic_suffix_ + "/instance_info";
        instance_info_pub_ = this->create_publisher<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(instance_info_topic_name, qos_sensors_pub);
        RCLCPP_INFO(this->get_logger(), "Publishing instance info to: [%s]", instance_info_topic_name.c_str());

        RCLCPP_INFO(this->get_logger(), "%s initialized successfully.", node_name.c_str());

        // ------- Iniciar el hilo de inferencia -------
        inference_thread_ = std::thread(&YoloBatchNode::inferenceLoop, this);
    }

    ~YoloBatchNode()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down %s...", this->get_name());
        if (inference_thread_.joinable())
        {
            inference_thread_.join();
        }
        if (video_writer_inferred_.isOpened())
        {
            video_writer_inferred_.release();
        }
        if (video_writer_instance_mask_.isOpened())
        {
            video_writer_instance_mask_.release();
        }
        if (h_pinned_input_buffer_)
        {
            cudaError_t err = cudaFreeHost(h_pinned_input_buffer_);
            LOG_CUDA_ERROR(err, ("[" + std::string(this->get_name()) + "] Destructor: Failed to free pinned host memory").c_str(), this->get_logger());
            h_pinned_input_buffer_ = nullptr;
        }
        RCLCPP_INFO(this->get_logger(), "%s shutdown complete.", this->get_name());
    }

    void getMetrics(double &read_freq, double &pub_freq) const
    {
        read_freq = reading_frequency_.load();
        pub_freq = publish_frequency_.load();
    }

private:
    // Variables para la sincronización de callbacks e inferencia
    std::mutex inference_mutex_;
    std::condition_variable inference_cv_;
    std::atomic<bool> stop_inference_thread_{false};
    std::thread inference_thread_;
    rclcpp::Time last_image_timestamp_;
    bool has_last_timestamp_ = false;

    // Estructura para almacenar imágenes con tiempos de entrada
    struct TimedImage
    {
        sensor_msgs::msg::Image::ConstSharedPtr msg; // Leer el número de secuencia de la imagen original
        timespec monotonic_entry_time;               // Time captured with clock_gettime(CLOCK_MONOTONIC, ...) in imageCallback
        timespec image_source_monotonic_capture_ts;  // To store the monotonic time from directory_publisher
        uint32_t source_image_seq;
    };

    // Cola de imágenes para inferencia
    std::queue<TimedImage> inference_queue_;

    // Parámetros
    int input_width_;
    int input_height_;
    std::string mask_encoding_;
    std::string image_topic_name_;
    std::string output_topic_suffix_;
    bool use_pinned_input_memory_ = true;
    std::string image_transport_type_ = "compressed";
    unsigned char *h_pinned_input_buffer_ = nullptr;
    size_t single_image_pinned_bytes_ = 0;
    int input_channels_ = 3;
    std::chrono::steady_clock::time_point last_actual_publish_time_;
    bool first_publish_done_ = false;

    // Variables para escritura de vídeo
    cv::VideoWriter video_writer_inferred_;
    cv::VideoWriter video_writer_instance_mask_;
    std::string video_output_path_str_;
    std::string inferred_video_filename_;
    std::string mask_video_filename_;
    double video_fps_;
    cv::Size single_video_frame_size_;
    bool enable_inferred_video_writing_ = false;
    bool enable_mask_video_writing_ = false;
    bool video_writers_initialized_ = false;
    bool enable_realtime_display_ = false;
    std::vector<cv::Scalar> class_colors_;
    bool enable_measure_times_ = false; // Por defecto, medir tiempos

    // Variables para métricas de rendimiento
    std::atomic<double> reading_frequency_{0.0}; // Hz (calculada por delta entre imágenes)
    std::atomic<double> publish_frequency_{0.0}; // Hz (calculada por intervalo entre publicaciones)

    // ROS Comms
    image_transport::Subscriber image_sub_;
    rclcpp::Publisher<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr instance_info_pub_;

    // ChronoTimer instances for metrics
    ChronoTimer timer_inter_callback_arrival_; // Time between arrivals in imageCallback
    ChronoTimer timer_queue_duration_;         // Time message spent in queue
    ChronoTimer timer_e2e_node_latency_;       // From callback entry to publish completion
    ChronoTimer timer_process_image_func_;     // Full duration of processImage function call
    ChronoTimer timer_inter_publish_;          // Time between publish calls

    // Modelo de segmentación
    std::unique_ptr<deploy::SegmentModel> model_;
    std::atomic<bool> stop_processing_thread_{false};

    // Estructura para almacenar estadísticas
    struct SimpleStats
    {
        double current_ms = 0.0, sum_ms = 0.0, mean_ms = 0.0, max_ms = 0.0;
        long count = 0;
        void record(double val)
        {
            current_ms = val;
            sum_ms += val;
            count++;
            mean_ms = sum_ms / count;
            if (count == 1 || val > max_ms)
                max_ms = val;
        }
    } stats_msg_age_;
    std::atomic<size_t> current_queue_size_{0};

    // Función para generar la máscara de ID de instancia
    cv::Mat generateInstanceIdMaskROI(const deploy::SegmentRes &result,
                                      const cv::Size &orig_size,
                                      const cv::Size &net_input_size)
    {
        cv::Mat instance_id_mask;
        if (this->mask_encoding_ == "mono16")
        {
            instance_id_mask = cv::Mat::zeros(orig_size, CV_16UC1);
        }
        else
        {
            instance_id_mask = cv::Mat::zeros(orig_size, CV_8UC1);
        }

        size_t num_detections = static_cast<size_t>(result.num);
        size_t num_items = std::min({num_detections, result.masks.size(),
                                     result.classes.size(), result.boxes.size()});

        if (num_detections > 0 && (num_detections != result.masks.size() ||
                                   num_detections != result.classes.size() ||
                                   num_detections != result.boxes.size()))
        {
            RCLCPP_WARN(this->get_logger(),
                        "[%s][generateInstanceIdMaskROI] Mismatch in result sizes. num: %d, masks: %zu, classes: %zu, boxes: %zu. Processing %zu items.",
                        this->get_name(), result.num, result.masks.size(), result.classes.size(), result.boxes.size(), num_items);
        }

        // Calcular las escalas para reescalar las ROIs de las cajas delimitadoras
        double scale_x = static_cast<double>(orig_size.width) / net_input_size.width;
        double scale_y = static_cast<double>(orig_size.height) / net_input_size.height;

        // RCLCPP_INFO(this->get_logger(), "Orige_size: %dx%d, Net input size: %dx%d, Scale: (%.2f, %.2f)",
        //             orig_size.width, orig_size.height, net_input_size.width, net_input_size.height, scale_x, scale_y);

        // Iterar sobre las mascaras, aplicar las ROIs y generar la máscara de ID de instancia
        for (size_t item_idx = 0; item_idx < num_items; ++item_idx)
        {
            if (result.masks[item_idx].data.empty() ||
                result.masks[item_idx].width <= 0 ||
                result.masks[item_idx].height <= 0)
            {
                RCLCPP_WARN(this->get_logger(),
                            "[%s][generateInstanceIdMaskROI] Item %zu has empty or invalid mask data.", this->get_name(), item_idx);
                continue;
            }

            // Se toma la caja del objeto de `result.boxes` (en coords. 640x416) y se
            // multiplica por `scale_x` y `scale_y` para obtener `orig_roi_rect`
            // Ahora sabemos dónde está la instancia en la imagen final de 1920x1200
            const deploy::Box &net_box = result.boxes[item_idx];
            cv::Rect orig_roi_rect(
                static_cast<int>(net_box.left * scale_x),
                static_cast<int>(net_box.top * scale_y),
                static_cast<int>((net_box.right - net_box.left) * scale_x),
                static_cast<int>((net_box.bottom - net_box.top) * scale_y));
            orig_roi_rect &= cv::Rect(0, 0, orig_size.width, orig_size.height);
            if (orig_roi_rect.width <= 0 || orig_roi_rect.height <= 0)
            {
                RCLCPP_WARN(this->get_logger(),
                            "[%s][generateInstanceIdMaskROI] Item %zu has zero or negative ROI size after scaling.", this->get_name(), item_idx);
                continue;
            }

            //  Es la máscara de baja resolución que da el modelo (ej: 160x104).
            cv::Mat raw_instance_mask(result.masks[item_idx].height,
                                      result.masks[item_idx].width, CV_8UC1,
                                      const_cast<void *>(static_cast<const void *>(result.masks[item_idx].data.data())));
            if (raw_instance_mask.empty())
            {
                RCLCPP_WARN(this->get_logger(),
                            "[%s][generateInstanceIdMaskROI] raw_instance_mask for item %zu is empty.", this->get_name(), item_idx);
                continue;
            }

            // Reescalar la máscara de instancia al tamaño de la ROI original
            double raw_scale_x_mask = static_cast<double>(raw_instance_mask.cols) / net_input_size.width;
            double raw_scale_y_mask = static_cast<double>(raw_instance_mask.rows) / net_input_size.height;
            cv::Rect raw_roi_on_mask(
                static_cast<int>(net_box.left * raw_scale_x_mask),
                static_cast<int>(net_box.top * raw_scale_y_mask),
                static_cast<int>((net_box.right - net_box.left) * raw_scale_x_mask),
                static_cast<int>((net_box.bottom - net_box.top) * raw_scale_y_mask));

            // Asegurarse de que la ROI esté dentro de los límites de la máscara
            raw_roi_on_mask &= cv::Rect(0, 0, raw_instance_mask.cols, raw_instance_mask.rows);

            if (raw_roi_on_mask.width <= 0 || raw_roi_on_mask.height <= 0)
            {
                RCLCPP_WARN(this->get_logger(),
                            "[%s][generateInstanceIdMaskROI] Item %zu has zero or negative ROI in raw mask.", this->get_name(), item_idx);
                continue;
            }

            // Se crea una "vista" (`submask`) que apunta solo a la porción de la
            // máscara cruda que contiene nuestro objeto. NO se copia memoria
            cv::Mat submask = raw_instance_mask(raw_roi_on_mask);

            // Reescalar la submáscara al tamaño de la ROI original
            cv::Mat resized_mask_for_roi;
            // cv::resize(submask, resized_mask_for_roi, orig_roi_rect.size(), 0, 0, cv::INTER_AREA);
            cv::resize(submask, resized_mask_for_roi, orig_roi_rect.size(), 0, 0, cv::INTER_NEAREST);
            cv::threshold(resized_mask_for_roi, resized_mask_for_roi, 0, 255, cv::THRESH_BINARY);

            // Se pinta el ID del objeto en la imagen de fondo, pero solo donde la máscara
            // redimensionada tiene píxeles de objeto (valor 255), es decir, los valores de la matriz que
            // reprsenatn dicho objeto son asigandos con el ID del objeto + 1 (para evitar el 0)

            uint16_t instance_pixel_value = static_cast<uint16_t>(item_idx + 1);
            cv::Mat output_roi = instance_id_mask(orig_roi_rect);

            if (this->mask_encoding_ == "mono16")
            {
                output_roi.setTo(instance_pixel_value, resized_mask_for_roi);
            }
            else
            {
                if (instance_pixel_value > 255)
                {
                    RCLCPP_WARN_ONCE(this->get_logger(),
                                     "[%s][generateInstanceIdMaskROI] Instance ID %u exceeds 255 for mono8 mask. Clamping. Consider using mono16.", this->get_name(), instance_pixel_value);
                    output_roi.setTo(static_cast<unsigned char>(255), resized_mask_for_roi);
                }
                else
                {
                    output_roi.setTo(static_cast<unsigned char>(instance_pixel_value), resized_mask_for_roi);
                }
            }
        }
        return instance_id_mask;
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        timespec monotonic_callback_entry_time;
        // Capturar timestamp monotónico inmediatamente al recibir el mensaje
        if (enable_measure_times_)
        {
            clock_gettime(CLOCK_MONOTONIC, &monotonic_callback_entry_time);

            // --- INICIO: Medición de diferencia de tiempo en la llegada de callbacks ---
            {
                std::lock_guard<std::mutex> lock(g_callback_arrival_mutex);
                g_latest_callback_times[this->output_topic_suffix_] = monotonic_callback_entry_time;

                for (const auto &pair : g_latest_callback_times)
                {
                    if (pair.first != this->output_topic_suffix_)
                    {
                        double time_diff_ms =
                            (static_cast<double>(monotonic_callback_entry_time.tv_sec) - static_cast<double>(pair.second.tv_sec)) * 1000.0 +
                            (static_cast<double>(monotonic_callback_entry_time.tv_nsec) - static_cast<double>(pair.second.tv_nsec)) / 1e6;

                        RCLCPP_INFO(this->get_logger(), "[%s] Callback Arrival Diff %s -> %s: %.4f ms",
                                    this->get_name(), this->output_topic_suffix_.c_str(), pair.first.c_str(), time_diff_ms);
                    }
                }
            }

            // --- FIN: Medición de diferencia de tiempo ---

            // Measure time between arrivals to this callback
            timer_inter_callback_arrival_.GetElapsedTime();
            if (timer_inter_callback_arrival_.cont > 0)
            {
                timer_inter_callback_arrival_.ComputeStats();
            }
            else
            {
                timer_inter_callback_arrival_.measured_time = 0;
                timer_inter_callback_arrival_.cont = 0;
                timer_inter_callback_arrival_.ComputeStats();
            }
            timer_inter_callback_arrival_.Reset();
        }

        TimedImage timed_msg;
        timed_msg.msg = msg;
        if (enable_measure_times_)
        {
            timed_msg.monotonic_entry_time = monotonic_callback_entry_time;
            timed_msg.image_source_monotonic_capture_ts.tv_sec = msg->header.stamp.sec;
            timed_msg.image_source_monotonic_capture_ts.tv_nsec = msg->header.stamp.nanosec;

            // Leer el número de secuencia desde frame_id y convertirlo
            try
            {
                // Usar std::stoul para uint32_t o std::stoull para uint64_t si el contador pudiera ser muy grande
                timed_msg.source_image_seq = static_cast<uint32_t>(std::stoul(msg->header.frame_id));
            }
            catch (const std::invalid_argument &ia)
            {
                RCLCPP_ERROR(this->get_logger(), "[%s] Invalid argument: Cannot convert frame_id ('%s') to sequence number. %s",
                             this->get_name(), msg->header.frame_id.c_str(), ia.what());
                timed_msg.source_image_seq = 0; // O algún valor de error
            }
            catch (const std::out_of_range &oor)
            {
                RCLCPP_ERROR(this->get_logger(), "[%s] Out of range: Cannot convert frame_id ('%s') to sequence number. %s",
                             this->get_name(), msg->header.frame_id.c_str(), oor.what());
                timed_msg.source_image_seq = 0; // O algún valor de error
            }

            // Calcular latencia de comunicación DirPub -> SegNode
            double latency_dirpub_to_segnode_ms =
                (monotonic_callback_entry_time.tv_sec - timed_msg.image_source_monotonic_capture_ts.tv_sec) * 1000.0 +
                (monotonic_callback_entry_time.tv_nsec - timed_msg.image_source_monotonic_capture_ts.tv_nsec) / 1e6;

            // Validación para detectar problemas de sincronización
            if (latency_dirpub_to_segnode_ms < 1000)
            {
                RCLCPP_INFO(this->get_logger(), "[%s] Latency (DirPub@%ld.%09ld -> SegNodeCallback@%ld.%09ld): %.3f ms",
                            this->get_name(),
                            timed_msg.image_source_monotonic_capture_ts.tv_sec, timed_msg.image_source_monotonic_capture_ts.tv_nsec,
                            monotonic_callback_entry_time.tv_sec, monotonic_callback_entry_time.tv_nsec,
                            latency_dirpub_to_segnode_ms);
            }
            else
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "[%s] Invalid latency measurement: %.3f ms. DirPub_ts=%ld.%09ld, SegNode_ts=%ld.%09ld",
                                     this->get_name(), latency_dirpub_to_segnode_ms,
                                     timed_msg.image_source_monotonic_capture_ts.tv_sec, timed_msg.image_source_monotonic_capture_ts.tv_nsec,
                                     monotonic_callback_entry_time.tv_sec, monotonic_callback_entry_time.tv_nsec);
            }
        }

        std::lock_guard<std::mutex> lock(inference_mutex_);
        inference_queue_.push(timed_msg);
        current_queue_size_.store(inference_queue_.size(), std::memory_order_relaxed);
        inference_cv_.notify_one();
    }

    // Hilo principal que se encarga de procesar la inferencia
    void inferenceLoop()
    {
        while (!stop_inference_thread_)
        {
            TimedImage timed_msg;
            long long loop_idle_time_us = 0;
            long long time_in_queue_us = 0;

            if (enable_measure_times_)
            {
                std::unique_lock<std::mutex> lock(inference_mutex_);

                auto t_before_wait = std::chrono::steady_clock::now();
                inference_cv_.wait(lock, [this]
                                   { return !inference_queue_.empty() || stop_inference_thread_; });
                auto t_after_wait = std::chrono::steady_clock::now();
                loop_idle_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_wait - t_before_wait).count();

                if (stop_inference_thread_)
                    break;

                timed_msg = inference_queue_.front();
                inference_queue_.pop();
                current_queue_size_.store(inference_queue_.size(), std::memory_order_relaxed);

                // Measure time in queue
                timer_queue_duration_.startTime = timed_msg.monotonic_entry_time; // Set start time to when it entered callback
                timer_queue_duration_.GetElapsedTime();                           // Measures up to now
                timer_queue_duration_.ComputeStats();
            }
            else
            {
                // Sincronización sin medir tiempos
                std::unique_lock<std::mutex> lock(inference_mutex_);
                inference_cv_.wait(lock, [this]
                                   { return !inference_queue_.empty() || stop_inference_thread_; });
                if (stop_inference_thread_)
                    break;

                timed_msg = inference_queue_.front();
                inference_queue_.pop();
                current_queue_size_.store(inference_queue_.size(), std::memory_order_relaxed);
            }

            if (enable_measure_times_ && enable_realtime_display_)
            {
                cv::waitKey(1);
            }
            if (enable_measure_times_)
            {
                processImage_timed(timed_msg, loop_idle_time_us, static_cast<long long>(timer_queue_duration_.measured_time * 1000.0));
            }
            else
            {
                processImage(timed_msg);
            }
        }
    }

    void processImage(const TimedImage &timed_msg)
    {

        // --- Lectura de imagen ---
        auto t_read_start = std::chrono::steady_clock::now();
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(timed_msg.msg, timed_msg.msg->encoding);

        // Validar que la imagen no esté vacía
        if (!cv_ptr || cv_ptr->image.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[%s] Received empty image on topic %s", this->get_name(), image_topic_name_.c_str());
            return;
        }

        // --- Procesamiento de imagen ---
        cv::Mat original_image = cv_ptr->image;
        // if (original_image.channels() == 1 && input_channels_ == 3)
        // {
        //     cv::cvtColor(original_image, original_image, cv::COLOR_GRAY2BGR);
        // }
        // if (original_image.channels() == 3 && input_channels_ == 1)
        // {
        //     cv::cvtColor(original_image, original_image, cv::COLOR_BGR2GRAY);
        // }
        if (timed_msg.msg->encoding == "bayer_rggb8")
        {
            cv::Mat rgb_image;
            cv::cvtColor(original_image, rgb_image, cv::COLOR_BayerRG2RGB);
            original_image = rgb_image;
        }

        cv::Size original_size = original_image.size();
        std_msgs::msg::Header current_header = timed_msg.msg->header;
        auto instance_info_msg = std::make_unique<yolo_custom_interfaces::msg::InstanceSegmentationInfo>();
        instance_info_msg->header = current_header;

        // --- Preprocesado ---
        cv::Mat resized_img;
        cv::Size network_input_target_size(input_width_, input_height_);
        if (original_size != network_input_target_size)
        {
            cv::resize(original_image, resized_img, network_input_target_size, 0, 0, cv::INTER_AREA);
            RCLCPP_WARN(this->get_logger(), "Input image does not match target size (%dx%d). Resizing to %dx%d.",
                        original_size.width, original_size.height,
                        network_input_target_size.width, network_input_target_size.height);
        }
        else
        {
            resized_img = original_image;
        }
        std::vector<deploy::Image> img_batch;
        unsigned char *image_data_ptr = resized_img.data;
        if (use_pinned_input_memory_ && h_pinned_input_buffer_)
        {
            if (resized_img.isContinuous() && (resized_img.total() * resized_img.elemSize() == single_image_pinned_bytes_))
            {
                std::memcpy(h_pinned_input_buffer_, resized_img.data, single_image_pinned_bytes_);
                image_data_ptr = h_pinned_input_buffer_;
            }
            else
            {
                RCLCPP_WARN_ONCE(this->get_logger(), "[%s] Resized image properties not suitable for pinned memory. Using its own data.", this->get_name());
            }
        }
        // Esto asegura que el backend de inferencia sepa que está manejando una imagen de 1 canal.
        img_batch.emplace_back(image_data_ptr, resized_img.cols, resized_img.rows);

        // ------- Inferencia --------
        std::vector<deploy::SegmentRes> results;

        model_->predict_async(img_batch);
        results = model_->get_results();

        if (results.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[%s] Inference returned no results.", this->get_name());
            return;
        }
        deploy::SegmentRes &result = results[0];

        // --- Postprocesado ---
        cv::Mat instance_id_mask_cv = generateInstanceIdMaskROI(result, original_size, network_input_target_size);

        cv_bridge::CvImage cv_img_mask_instance;
        cv_img_mask_instance.header = instance_info_msg->header;
        cv_img_mask_instance.encoding = this->mask_encoding_;
        // cv_img_mask_instance.image = instance_id_mask_cv;
        instance_info_msg->mask_width = instance_id_mask_cv.cols;
        instance_info_msg->mask_height = instance_id_mask_cv.rows;  
        instance_info_msg->mask_data.assign(instance_id_mask_cv.datastart, instance_id_mask_cv.dataend);

        // instance_info_msg->mask = *cv_img_mask_instance.toImageMsg();

        size_t num_detected_instances = static_cast<size_t>(result.num);
        num_detected_instances = std::min({num_detected_instances, result.scores.size(), result.classes.size()});
        instance_info_msg->scores.reserve(num_detected_instances);
        instance_info_msg->classes.reserve(num_detected_instances);
        for (size_t j = 0; j < num_detected_instances; ++j)
        {
            instance_info_msg->scores.push_back(result.scores[j]);
            instance_info_msg->classes.push_back(result.classes[j]);
        }

        // --- Publicación ---
        RCLCPP_INFO(this->get_logger(), "[%s] Publishing InstanceSegmentationInfo with shape: %dx%d, num_instances: %zu, packet_seq: %lu",
                    this->get_name(), instance_id_mask_cv.cols, instance_id_mask_cv.rows,
                    instance_info_msg->scores.size());
        instance_info_pub_->publish(std::move(instance_info_msg));
    }

    void processImage_timed(const TimedImage &timed_msg, long long loop_idle_time_us, long long time_in_queue_us_val)
    {
        timer_process_image_func_.Reset();                                  // Start timing for the entire function
        timer_e2e_node_latency_.startTime = timed_msg.monotonic_entry_time; // Start E2E latency from callback entry
        // Calcular latencia total desde DirPub hasta inicio de procesamiento
        timespec processing_start_time;
        clock_gettime(CLOCK_MONOTONIC, &processing_start_time);

        double total_latency_dirpub_to_processing_ms =
            (processing_start_time.tv_sec - timed_msg.image_source_monotonic_capture_ts.tv_sec) * 1000.0 +
            (processing_start_time.tv_nsec - timed_msg.image_source_monotonic_capture_ts.tv_nsec) / 1e6;

        // Calcular y mostrar el delta en ms entre la imagen actual y la anterior
        rclcpp::Time current_msg_time(timed_msg.msg->header.stamp.sec, timed_msg.msg->header.stamp.nanosec);
        double current_read_freq = 0.0;
        if (has_last_timestamp_)
        {
            auto delta_ns = (current_msg_time - last_image_timestamp_).nanoseconds();
            double delta_ms = static_cast<double>(delta_ns) / 1e6;
            current_read_freq = (delta_ms > 0) ? 1000.0 / delta_ms : 0.0; // Hz
            // RCLCPP_INFO(this->get_logger(), "[%s] Time delta between images: %.2f ms (%.2f Hz)",
            //             this->get_name(), delta_ms, current_read_freq);
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "[%s] First image received", this->get_name());
        }
        last_image_timestamp_ = current_msg_time;
        has_last_timestamp_ = true;

        // --- Lectura de imagen ---
        auto t_read_start = std::chrono::steady_clock::now();
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(timed_msg.msg, timed_msg.msg->encoding);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "[%s] cv_bridge exception during image read: %s",
                         this->get_name(), e.what());
            return;
        }
        auto t_read_end = std::chrono::steady_clock::now();
        auto t_read = std::chrono::duration_cast<std::chrono::microseconds>(t_read_end - t_read_start).count();

        if (!cv_ptr || cv_ptr->image.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[%s] Received empty image on topic %s", this->get_name(), image_topic_name_.c_str());
            return;
        }
        cv::Mat original_image = cv_ptr->image;

        // if (original_image.channels() == 1 && input_channels_ == 3)
        // {
        //     cv::cvtColor(original_image, original_image, cv::COLOR_GRAY2BGR);
        // }
        // if (original_image.channels() == 3 && input_channels_ == 1)
        // {
        //     cv::cvtColor(original_image, original_image, cv::COLOR_BGR2GRAY);
        // }
        if (timed_msg.msg->encoding == "bayer_rggb8")
        {
            cv::Mat rgb_image;
            cv::cvtColor(original_image, rgb_image, cv::COLOR_BayerRG2RGB);
            original_image = rgb_image;
        }

        cv::Size original_size = original_image.size();
        std_msgs::msg::Header current_header = timed_msg.msg->header;
        auto instance_info_msg = std::make_unique<yolo_custom_interfaces::msg::InstanceSegmentationInfo>();
        instance_info_msg->header = current_header;

        // --- Preprocesado ---
        auto t_pre_start = std::chrono::steady_clock::now();
        cv::Mat resized_img;
        cv::Size network_input_target_size(input_width_, input_height_);
        if (original_size != network_input_target_size)
        {
            cv::resize(original_image, resized_img, network_input_target_size, 0, 0, cv::INTER_AREA);
        }
        else
        {
            resized_img = original_image;
        }
        std::vector<deploy::Image> img_batch;
        unsigned char *image_data_ptr = resized_img.data;
        if (use_pinned_input_memory_ && h_pinned_input_buffer_)
        {
            if (resized_img.isContinuous() && (resized_img.total() * resized_img.elemSize() == single_image_pinned_bytes_))
            {
                std::memcpy(h_pinned_input_buffer_, resized_img.data, single_image_pinned_bytes_);
                image_data_ptr = h_pinned_input_buffer_;
            }
            else
            {
                RCLCPP_WARN_ONCE(this->get_logger(), "[%s] Resized image properties not suitable for pinned memory. Using its own data.", this->get_name());
            }
        }
        img_batch.emplace_back(image_data_ptr, resized_img.cols, resized_img.rows);
        auto t_pre_end = std::chrono::steady_clock::now();
        auto t_preproc = std::chrono::duration_cast<std::chrono::microseconds>(t_pre_end - t_pre_start).count();

        // --- Inferencia ---
        auto t_inf_start = std::chrono::steady_clock::now();
        std::vector<deploy::SegmentRes> results;

        timespec inference_start_ts;
        clock_gettime(CLOCK_MONOTONIC, &inference_start_ts);

        model_->predict_async(img_batch);
        results = model_->get_results();

        timespec inference_end_ts;
        clock_gettime(CLOCK_MONOTONIC, &inference_end_ts);

        // Populate new inference time fields
        instance_info_msg->processing_node_inference_start_time.sec = inference_start_ts.tv_sec;
        instance_info_msg->processing_node_inference_start_time.nanosec = inference_start_ts.tv_nsec;
        instance_info_msg->processing_node_inference_end_time.sec = inference_end_ts.tv_sec;
        instance_info_msg->processing_node_inference_end_time.nanosec = inference_end_ts.tv_nsec;

        auto t_inf_end = std::chrono::steady_clock::now();
        auto t_infer = std::chrono::duration_cast<std::chrono::microseconds>(t_inf_end - t_inf_start).count();

        if (results.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[%s] Inference returned no results.", this->get_name());
            return;
        }
        deploy::SegmentRes &result = results[0];

        // --- Postprocesado ---
        auto t_post_start = std::chrono::steady_clock::now();
        cv::Mat instance_id_mask_cv = generateInstanceIdMaskROI(result, original_size, network_input_target_size);

        // Populate the image source monotonic timestamp (T1_mono)
        instance_info_msg->image_source_monotonic_capture_time.sec = timed_msg.image_source_monotonic_capture_ts.tv_sec;
        instance_info_msg->image_source_monotonic_capture_time.nanosec = timed_msg.image_source_monotonic_capture_ts.tv_nsec;
        instance_info_msg->header.frame_id = timed_msg.msg->header.frame_id; // Propagar frame_id
        // Populate the monotonic entry time (T2_mono)
        instance_info_msg->processing_node_monotonic_entry_time.sec = timed_msg.monotonic_entry_time.tv_sec;
        instance_info_msg->processing_node_monotonic_entry_time.nanosec = timed_msg.monotonic_entry_time.tv_nsec;

        cv_bridge::CvImage cv_img_mask_instance;
        cv_img_mask_instance.header = instance_info_msg->header;
        cv_img_mask_instance.encoding = this->mask_encoding_;
        // cv_img_mask_instance.image = instance_id_mask_cv;
        instance_info_msg->mask_width = instance_id_mask_cv.cols;
        instance_info_msg->mask_height = instance_id_mask_cv.rows;  
        instance_info_msg->mask_data.assign(instance_id_mask_cv.datastart, instance_id_mask_cv.dataend);

        // instance_info_msg->mask = *cv_img_mask_instance.toImageMsg();

        size_t num_detected_instances = static_cast<size_t>(result.num);
        num_detected_instances = std::min({num_detected_instances, result.scores.size(), result.classes.size()});
        instance_info_msg->scores.reserve(num_detected_instances);
        instance_info_msg->classes.reserve(num_detected_instances);
        for (size_t j = 0; j < num_detected_instances; ++j)
        {
            instance_info_msg->scores.push_back(result.scores[j]);
            instance_info_msg->classes.push_back(result.classes[j]);
        }
        auto t_post_end = std::chrono::steady_clock::now();
        auto t_postproc = std::chrono::duration_cast<std::chrono::microseconds>(t_post_end - t_post_start).count();

        // --- Publicación ---
        auto t_pub_start = std::chrono::steady_clock::now();
        // Get current monotonic time for publishing this result (T3_mono)
        timespec ts_processing_node_publish;
        clock_gettime(CLOCK_MONOTONIC, &ts_processing_node_publish);

        // --- INICIO: Medición de diferencia de tiempo en la publicación ---
        {
            std::lock_guard<std::mutex> lock(g_publish_mutex);
            g_latest_publish_times[this->output_topic_suffix_] = ts_processing_node_publish;

            for (const auto &pair : g_latest_publish_times)
            {
                if (pair.first != this->output_topic_suffix_)
                {
                    double time_diff_ms =
                        (static_cast<double>(ts_processing_node_publish.tv_sec) - static_cast<double>(pair.second.tv_sec)) * 1000.0 +
                        (static_cast<double>(ts_processing_node_publish.tv_nsec) - static_cast<double>(pair.second.tv_nsec)) / 1e6;

                    RCLCPP_INFO(this->get_logger(), "[%s] Publication Diff %s -> %s: %.4f ms",
                                this->get_name(), this->output_topic_suffix_.c_str(), pair.first.c_str(), time_diff_ms);
                }
            }
        }
        // --- FIN: Medición de diferencia de tiempo ---

        instance_info_msg->processing_node_monotonic_publish_time.sec = ts_processing_node_publish.tv_sec;
        instance_info_msg->processing_node_monotonic_publish_time.nanosec = ts_processing_node_publish.tv_nsec;
        instance_info_msg->packet_sequence_number = static_cast<uint64_t>(timed_msg.source_image_seq);

        RCLCPP_INFO(this->get_logger(), "[%s] Publishing InstanceSegmentationInfo with shape: %dx%d, num_instances: %zu, packet_seq: %lu",
                    this->get_name(), instance_id_mask_cv.cols, instance_id_mask_cv.rows,
                    instance_info_msg->scores.size());

        instance_info_pub_->publish(std::move(instance_info_msg));
        auto t_pub_end = std::chrono::steady_clock::now();
        auto t_publish = std::chrono::duration_cast<std::chrono::microseconds>(t_pub_end - t_pub_start).count();

        // Measure E2E Node Latency (from callback entry to publish completion)
        timer_e2e_node_latency_.GetElapsedTime(); // Uses its startTime (monotonic_entry_time) and current time
        timer_e2e_node_latency_.ComputeStats();

        // Measure inter-publish time
        timer_inter_publish_.GetElapsedTime(); // Measures from last Reset() or construction
        if (timer_inter_publish_.cont > 0 || timer_e2e_node_latency_.cont == 1)
        { // Similar logic to inter_callback_arrival
            timer_inter_publish_.ComputeStats();
        }
        else
        {
            timer_inter_publish_.measured_time = 0;
            timer_inter_publish_.cont = 0;
            timer_inter_publish_.ComputeStats();
        }
        timer_inter_publish_.Reset(); // Reset for next publish interval

        // --- Visualización en tiempo real ---
        if (enable_realtime_display_)
        {
            cv::Mat display_image_ = createDisplayImage(original_image, result, instance_id_mask_cv, true);
            if (!display_image_.empty())
            {
                // En lugar de mostrar, se pasa al hilo de visualización
                std::lock_guard<std::mutex> lock(g_display_mutex);
                g_display_images[output_topic_suffix_] = display_image_;
                g_new_image_flags[output_topic_suffix_] = true;
                g_display_cv.notify_one();
            }
        }

        RCLCPP_INFO(this->get_logger(), "[%s] Video Writing", this->get_name());
        // Video Writing
        if (video_output_path_str_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[%s] Video output path is empty. Skipping video writing.", this->get_name());
            video_writers_initialized_ = false;
            return;
        }
        if (!video_writers_initialized_ && (enable_inferred_video_writing_ || enable_mask_video_writing_))
        {
            initializeVideoWriters();
        }
        if (video_writers_initialized_)
        {
            if (enable_inferred_video_writing_ && video_writer_inferred_.isOpened())
            {
                RCLCPP_INFO(this->get_logger(), "[%s] Writing inferred image to video...", this->get_name());
                writeInferredImageToVideo(original_image, result, instance_id_mask_cv, original_size, network_input_target_size);
            }
            if (enable_mask_video_writing_ && video_writer_instance_mask_.isOpened())
            {
                RCLCPP_INFO(this->get_logger(), "[%s] Writing instance mask to video...", this->get_name());
                writeInstanceMaskToVideo(instance_id_mask_cv, result, original_size);
            }
        }

        // Measure full processImage function duration
        timer_process_image_func_.GetElapsedTime();
        timer_process_image_func_.ComputeStats();

        // Calcular frecuencia de publicación real
        long long actual_inter_publish_us = 0;
        double actual_publish_freq = 0.0;
        if (first_publish_done_)
        {
            actual_inter_publish_us = std::chrono::duration_cast<std::chrono::microseconds>(t_pub_end - last_actual_publish_time_).count();
            if (actual_inter_publish_us > 0)
            {
                actual_publish_freq = 1e6 / static_cast<double>(actual_inter_publish_us);
            }
        }
        last_actual_publish_time_ = t_pub_end;
        first_publish_done_ = true;

        // Al final del procesamiento, calcular latencia total
        timespec processing_end_time;
        clock_gettime(CLOCK_MONOTONIC, &processing_end_time);

        double total_latency_dirpub_to_publish_ms =
            (processing_end_time.tv_sec - timed_msg.image_source_monotonic_capture_ts.tv_sec) * 1000.0 +
            (processing_end_time.tv_nsec - timed_msg.image_source_monotonic_capture_ts.tv_nsec) / 1e6;

        RCLCPP_INFO(this->get_logger(),
                    "[%s] Timings(us): Read=%ld, Pre=%ld, Infer=%ld, Post=%ld, PubCall=%ld. LoopIdle=%ld. QSize=%zu. "
                    "MsgAge(ms): Cur=%.2f,Avg=%.2f,Max=%.2f. "
                    "InterCB(ms): Cur=%.2f,Avg=%.2f,Max=%.2f. "
                    "QueueT(ms): Cur=%.2f,Avg=%.2f,Max=%.2f. "
                    "E2ELat(ms): Cur=%.2f,Avg=%.2f,Max=%.2f. "
                    "ProcFunc(ms): Cur=%.2f,Avg=%.2f,Max=%.2f. "
                    "InterPub(ms): Cur=%.2f,Avg=%.2f,Max=%.2f. "
                    "TotalDirPub->Pub(ms): %.2f",
                    this->get_name(),
                    t_read, t_preproc, t_infer, t_postproc, t_publish,
                    loop_idle_time_us, current_queue_size_.load(),
                    stats_msg_age_.current_ms, stats_msg_age_.mean_ms, stats_msg_age_.max_ms,
                    timer_inter_callback_arrival_.measured_time, timer_inter_callback_arrival_.mean_time, timer_inter_callback_arrival_.max_time,
                    timer_queue_duration_.measured_time, timer_queue_duration_.mean_time, timer_queue_duration_.max_time,
                    timer_e2e_node_latency_.measured_time, timer_e2e_node_latency_.mean_time, timer_e2e_node_latency_.max_time,
                    timer_process_image_func_.measured_time, timer_process_image_func_.mean_time, timer_process_image_func_.max_time,
                    timer_inter_publish_.measured_time, timer_inter_publish_.mean_time, timer_inter_publish_.max_time,
                    total_latency_dirpub_to_publish_ms);

        // Actualizar las variables de métrica:
        publish_frequency_.store(actual_publish_freq);
        reading_frequency_.store(current_read_freq);
    }

    void initializeVideoWriters()
    {
        if (video_writers_initialized_)
            return;

        fs::path output_dir(video_output_path_str_);
        if (!fs::exists(output_dir))
        {
            fs::create_directories(output_dir);
            RCLCPP_INFO(this->get_logger(), "[%s] Created video output directory: %s", this->get_name(), output_dir.string().c_str());
        }

        if (enable_inferred_video_writing_)
        {
            fs::path inferred_video_full_path = output_dir / inferred_video_filename_;
            RCLCPP_INFO(this->get_logger(), "[%s] Initializing inferred video writer: %s, Size: %dx%d, FPS: %.1f",
                        this->get_name(), inferred_video_full_path.string().c_str(),
                        single_video_frame_size_.width, single_video_frame_size_.height, video_fps_);
            if (!video_writer_inferred_.open(inferred_video_full_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), video_fps_, single_video_frame_size_, true))
            {
                RCLCPP_ERROR(this->get_logger(), "[%s] Failed to open video writer for inferred images. Disabling.", this->get_name());
                enable_inferred_video_writing_ = false;
            }
        }

        if (enable_mask_video_writing_)
        {
            fs::path mask_video_full_path = output_dir / mask_video_filename_;
            RCLCPP_INFO(this->get_logger(), "[%s] Initializing mask video writer: %s, Size: %dx%d, FPS: %.1f",
                        this->get_name(), mask_video_full_path.string().c_str(),
                        single_video_frame_size_.width, single_video_frame_size_.height, video_fps_);
            if (!video_writer_instance_mask_.open(mask_video_full_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), video_fps_, single_video_frame_size_, true))
            { // True for color
                RCLCPP_ERROR(this->get_logger(), "[%s] Failed to open video writer for instance masks. Disabling.", this->get_name());
                enable_mask_video_writing_ = false;
            }
        }
        video_writers_initialized_ = true;
    }

    cv::Scalar getRandomTone(const cv::Scalar &base_color, int seed)
    {
        cv::RNG rng(static_cast<uint64_t>(seed));
        double variation_range = 60.0;
        cv::Scalar toned_color;
        for (int i = 0; i < 3; ++i)
        {
            toned_color[i] = cv::saturate_cast<uchar>(base_color[i] + rng.uniform(-variation_range, variation_range));
        }
        return toned_color;
    }

    void writeInferredImageToVideo(
        const cv::Mat &original_image,    // Una sola imagen
        const deploy::SegmentRes &result, // Un solo resultado
        const cv::Mat &instance_id_mask,  // Una sola máscara de ID
        const cv::Size &original_size,    // Tamaño original de esta imagen
        const cv::Size &network_input_size)
    {
        if (class_colors_.empty())
        {
            RCLCPP_WARN_ONCE(this->get_logger(), "[%s] Class colors vector is empty. Cannot color inferred images for video.", this->get_name());
            return;
        }
        if (original_image.empty() || instance_id_mask.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[%s] Missing data for writeInferredImageToVideo. Skipping frame.", this->get_name());
            return;
        }
        if (original_image.size() != instance_id_mask.size())
        {
            RCLCPP_ERROR(this->get_logger(), "[%s] Mismatch between original image size (%dx%d) and instance_id_mask size (%dx%d). Skipping coloring.",
                         this->get_name(), original_image.cols, original_image.rows,
                         instance_id_mask.cols, instance_id_mask.rows);
            return;
        }

        cv::Mat display_image = original_image.clone();
        size_t num_instances = std::min({static_cast<size_t>(result.num), result.classes.size(), result.scores.size(), result.boxes.size()});

        for (size_t inst_idx = 0; inst_idx < num_instances; ++inst_idx)
        {
            int class_id = result.classes[inst_idx];
            cv::Scalar base_class_color = class_colors_[class_id % class_colors_.size()];
            uint16_t instance_pixel_id_val = static_cast<uint16_t>(inst_idx + 1);

            cv::Scalar toned_instance_color = getRandomTone(base_class_color, static_cast<int>(instance_pixel_id_val));
            cv::Mat single_instance_binary_mask;

            if (mask_encoding_ == "mono16")
            {
                cv::compare(instance_id_mask, instance_pixel_id_val, single_instance_binary_mask, cv::CMP_EQ);
            }
            else
            {
                cv::compare(instance_id_mask, static_cast<uchar>(instance_pixel_id_val), single_instance_binary_mask, cv::CMP_EQ);
            }

            if (cv::countNonZero(single_instance_binary_mask) > 0)
            {
                display_image.setTo(toned_instance_color, single_instance_binary_mask);
            }
        }

        if (display_image.size() != single_video_frame_size_)
        {
            cv::resize(display_image, display_image, single_video_frame_size_);
        }

        if (!display_image.empty() && video_writer_inferred_.isOpened())
        {
            video_writer_inferred_.write(display_image);
        }
    }

    void writeInstanceMaskToVideo(
        const cv::Mat &instance_id_mask,  // CV_16UC1 o CV_8UC1
        const deploy::SegmentRes &result, // Un solo resultado
        const cv::Size &original_size)    // Tamaño original de esta imagen
    {
        if (class_colors_.empty())
        {
            RCLCPP_WARN_ONCE(this->get_logger(), "[%s] Class colors vector is empty. Cannot color instance masks for video.", this->get_name());
            return;
        }
        if (instance_id_mask.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[%s] Missing data for writeInstanceMaskToVideo. Skipping frame.", this->get_name());
            return;
        }

        cv::Mat colored_mask_display = cv::Mat::zeros(original_size, CV_8UC3);
        size_t num_instances_in_result = std::min({static_cast<size_t>(result.num), result.classes.size()});

        for (int r = 0; r < instance_id_mask.rows; ++r)
        {
            for (int c = 0; c < instance_id_mask.cols; ++c)
            {
                uint16_t instance_id_from_mask = 0;
                if (mask_encoding_ == "mono16")
                {
                    instance_id_from_mask = instance_id_mask.at<uint16_t>(r, c);
                }
                else
                {
                    instance_id_from_mask = static_cast<uint16_t>(instance_id_mask.at<uchar>(r, c));
                }

                if (instance_id_from_mask > 0)
                {
                    size_t item_idx = static_cast<size_t>(instance_id_from_mask - 1);
                    if (item_idx < num_instances_in_result)
                    {
                        int class_id = result.classes[item_idx];
                        cv::Scalar base_color = class_colors_[class_id % class_colors_.size()];
                        cv::Scalar toned_color = getRandomTone(base_color, instance_id_from_mask);
                        colored_mask_display.at<cv::Vec3b>(r, c) = cv::Vec3b(toned_color[0], toned_color[1], toned_color[2]);
                    }
                }
            }
        }

        if (colored_mask_display.size() != single_video_frame_size_)
        {
            cv::resize(colored_mask_display, colored_mask_display, single_video_frame_size_, 0, 0, cv::INTER_NEAREST);
        }

        if (!colored_mask_display.empty() && video_writer_instance_mask_.isOpened())
        {
            video_writer_instance_mask_.write(colored_mask_display);
        }
    }

    cv::Mat createDisplayImage(
        const cv::Mat &original_image,
        const deploy::SegmentRes &result,
        const cv::Mat &instance_id_mask,
        bool resize_for_display)
    {
        if (class_colors_.empty() || original_image.empty() || instance_id_mask.empty())
        {
            return cv::Mat(); // Devuelve una matriz vacía si faltan datos
        }

        cv::Mat display_image = original_image.clone();
        size_t num_instances = std::min({static_cast<size_t>(result.num), result.classes.size(), result.scores.size(), result.boxes.size()});

        for (size_t inst_idx = 0; inst_idx < num_instances; ++inst_idx)
        {
            int class_id = result.classes[inst_idx];
            cv::Scalar base_class_color = class_colors_[class_id % class_colors_.size()];
            uint16_t instance_pixel_id_val = static_cast<uint16_t>(inst_idx + 1);

            cv::Scalar toned_instance_color = getRandomTone(base_class_color, static_cast<int>(instance_pixel_id_val));
            cv::Mat single_instance_binary_mask;

            if (mask_encoding_ == "mono16")
            {
                cv::compare(instance_id_mask, instance_pixel_id_val, single_instance_binary_mask, cv::CMP_EQ);
            }
            else
            {
                cv::compare(instance_id_mask, static_cast<uchar>(instance_pixel_id_val), single_instance_binary_mask, cv::CMP_EQ);
            }

            if (cv::countNonZero(single_instance_binary_mask) > 0)
            {
                // Mezclar el color de la máscara con la imagen original para un efecto de superposición
                cv::Mat colored_overlay;
                // Usar display_image.size() para el overlay, ya que ambas tienen el mismo tamaño (original) en este punto.
                cv::addWeighted(display_image, 0.6, cv::Mat(display_image.size(), display_image.type(), toned_instance_color), 0.4, 0, colored_overlay);
                // Copiar el resultado mezclado a la imagen de display, usando la máscara binaria.
                colored_overlay.copyTo(display_image, single_instance_binary_mask);
            }
        }

        // Redimensionar al final si es para visualización en tiempo real.
        if (resize_for_display && display_image.size() != single_video_frame_size_)
        {
            cv::resize(display_image, display_image, single_video_frame_size_);
        }

        return display_image;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    // Nodo temporal para obtener los parámetros de lanzamiento
    auto param_node = std::make_shared<rclcpp::Node>("yolo_param_node_for_main");

    // --- Configuración
    param_node->declare_parameter<bool>("enable_inferred_video", false);
    param_node->declare_parameter<bool>("enable_mask_video", false);
    param_node->declare_parameter<bool>("measure_times", false);
    param_node->declare_parameter<bool>("realtime_display", false);
    param_node->declare_parameter<std::string>("output_video_path", "/home/david/ros_videos/segment_node_3P_out");
    param_node->declare_parameter<double>("video_fps", 10.0);
    param_node->declare_parameter<int>("video_width", 1920);
    param_node->declare_parameter<int>("video_height", 1200);
    param_node->declare_parameter<std::string>("image_transport_type", "raw");
    param_node->declare_parameter<std::string>("left_camera_topic", "/camera_front_left/image_raw");
    param_node->declare_parameter<std::string>("front_camera_topic", "/camera_front/image_raw");
    param_node->declare_parameter<std::string>("right_camera_topic", "/camera_front_right/image_raw");

    // Obtener los valores de los parámetros
    bool enable_inferred_video_main = param_node->get_parameter("enable_inferred_video").as_bool();
    bool enable_mask_video_main = param_node->get_parameter("enable_mask_video").as_bool();
    bool MeasureTimes = param_node->get_parameter("measure_times").as_bool();
    bool RealtimeDisplay = param_node->get_parameter("realtime_display").as_bool();
    std::string output_video_path = param_node->get_parameter("output_video_path").as_string();
    double video_fps_main = param_node->get_parameter("video_fps").as_double();
    cv::Size single_cam_video_size(
        param_node->get_parameter("video_width").as_int(),
        param_node->get_parameter("video_height").as_int());
    std::string image_transport_type = param_node->get_parameter("image_transport_type").as_string();
    std::string left_topic = param_node->get_parameter("left_camera_topic").as_string();
    std::string front_topic = param_node->get_parameter("front_camera_topic").as_string();
    std::string right_topic = param_node->get_parameter("right_camera_topic").as_string();

    auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();

    auto left_node = std::make_shared<YoloBatchNode>("yolo_segment_node_left",
                                                     left_topic,
                                                     "left",
                                                     enable_inferred_video_main,
                                                     enable_mask_video_main,
                                                     output_video_path,
                                                     video_fps_main,
                                                     single_cam_video_size,
                                                     image_transport_type,
                                                     MeasureTimes,
                                                     RealtimeDisplay);

    auto front_node = std::make_shared<YoloBatchNode>("yolo_segment_node_front",
                                                      front_topic,
                                                      "front",
                                                      enable_inferred_video_main,
                                                      enable_mask_video_main,
                                                      output_video_path,
                                                      video_fps_main,
                                                      single_cam_video_size,
                                                      image_transport_type,
                                                      MeasureTimes,
                                                      RealtimeDisplay);

    auto right_node = std::make_shared<YoloBatchNode>("yolo_segment_node_right",
                                                      right_topic,
                                                      "right",
                                                      enable_inferred_video_main,
                                                      enable_mask_video_main,
                                                      output_video_path,
                                                      video_fps_main,
                                                      single_cam_video_size,
                                                      image_transport_type,
                                                      MeasureTimes,
                                                      RealtimeDisplay);

    executor->add_node(left_node);
    executor->add_node(front_node);
    executor->add_node(right_node);

    // Hilo para la visualización combinada
    std::thread display_thread;
    if (RealtimeDisplay)
    {
        display_thread = std::thread([]()
                                     {
            RCLCPP_INFO(rclcpp::get_logger("DisplayThread"), "Stitched display thread started.");
            cv::Mat stitched_image;
            while (!g_stop_display_thread.load())
            {
                cv::Mat left, front, right;
                {
                    std::unique_lock<std::mutex> lock(g_display_mutex);
                    g_display_cv.wait(lock, [] {
                        return (g_new_image_flags["left"] && g_new_image_flags["front"] && g_new_image_flags["right"]) || g_stop_display_thread.load();
                    });

                    if (g_stop_display_thread.load()) break;

                    // Copiar imágenes para liberarel lock rápidamente
                    left = g_display_images["left"].clone();
                    front = g_display_images["front"].clone();
                    right = g_display_images["right"].clone();

                    // Resetear flags
                    g_new_image_flags["left"] = false;
                    g_new_image_flags["front"] = false;
                    g_new_image_flags["right"] = false;
                }

                if (!left.empty() && !front.empty() && !right.empty())
                {
                    cv::Mat temp_h1, stitched_image;
                    cv::hconcat(left, front, temp_h1);
                    cv::hconcat(temp_h1, right, stitched_image);

                    // Redimensionar para una mejor visualización si es necesario
                    cv::Size display_size(1920, 1200 / 3); // Ajustar según el monitor
                    cv::resize(stitched_image, stitched_image, display_size, 0, 0, cv::INTER_AREA);

                    cv::imshow("Panoramic View", stitched_image);
                    cv::waitKey(1);
                }
            }
            cv::destroyWindow("Panoramic View");
            RCLCPP_INFO(rclcpp::get_logger("DisplayThread"), "Panoramic display thread finished."); });
    }

    // Hilo supervisor: cada 5 segundos muestra las métricas de cada nodo
    std::atomic<bool> stop_supervisor{false};

    std::thread supervisor_thread([&]()
                                  {
        while (!stop_supervisor.load()) {
            if(MeasureTimes){
                double left_read=0.0, left_pub=0.0;
                double front_read=0.0, front_pub=0.0;
                double right_read=0.0, right_pub=0.0;
                left_node->getMetrics(left_read, left_pub);
                front_node->getMetrics(front_read, front_pub);
                right_node->getMetrics(right_read, right_pub);
                RCLCPP_INFO(rclcpp::get_logger("Supervisor"),
                            "Supervisor: Left node: Read=%.2f Hz, Pub=%.2f Hz; Front node: Read=%.2f Hz, Pub=%.2f Hz; Right node: Read=%.2f Hz, Pub=%.2f Hz",
                            left_read, left_pub, front_read, front_pub, right_read, right_pub);
            } else {
                RCLCPP_INFO(rclcpp::get_logger("Supervisor"),
                            "Supervisor: Segment node is running");
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        } });

    RCLCPP_INFO(rclcpp::get_logger("main"), "Spinning nodes with MultiThreadedExecutor.");
    executor->spin();

    // // Al salir:
    stop_supervisor.store(true);
    if (supervisor_thread.joinable())
    {
        supervisor_thread.join();
    }
    if (RealtimeDisplay)
    {
        g_stop_display_thread.store(true); // Detener el hilo de visualización
        g_display_cv.notify_all();         // Despertar al hilo para que pueda terminar
        if (display_thread.joinable())
        {
            display_thread.join();
        }
    }

    rclcpp::shutdown();
    return 0;
}