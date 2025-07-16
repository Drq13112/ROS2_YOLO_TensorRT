#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <filesystem>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include <image_transport/image_transport.hpp>
#include "yolo_custom_interfaces/msg/pidnet_result.hpp"

#include "pidnet_segmentation_cpp/tensorrt_inference.h"
#include "pidnet_segmentation_cpp/ChronoTimer.hpp"

namespace fs = std::filesystem;

// --- Variables Globales para Sincronización y Visualización ---
std::mutex g_display_mutex;
std::condition_variable g_display_cv;
std::map<std::string, cv::Mat> g_display_images;
std::map<std::string, bool> g_new_image_flags = {{"left", false}, {"front", false}, {"right", false}};
std::atomic<bool> g_stop_display_thread{false};

class PIDNetNode : public rclcpp::Node
{
public:
    PIDNetNode(const std::string &node_name,
               const std::string &image_topic_name,
               const std::string &output_topic_suffix,
               const std::string &video_path,
               double fps,
               const cv::Size &video_frame_size,
               const std::string &image_transport_type = "raw",
               bool enable_measure_times = true,
               bool enable_realtime_display = true)
        : Node(node_name),
          output_topic_suffix_(output_topic_suffix),
          video_output_path_str_(video_path),
          video_fps_(fps),
          video_frame_size_(video_frame_size),
          image_transport_type_(image_transport_type),
          enable_measure_times_(enable_measure_times),
          enable_realtime_display_(enable_realtime_display)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing %s...", node_name.c_str());

        // --- Parámetros del Nodo ---
        this->declare_parameter<std::string>("engine_path", "pidnet_s_608x960.trt");
        this->declare_parameter<double>("overlay_alpha", 0.4);

        auto engine_path = this->get_parameter("engine_path").get_value<std::string>();
        overlay_alpha_ = this->get_parameter("overlay_alpha").get_value<double>();

        // --- Inicializar Motor de Inferencia ---
        try
        {
            inference_engine_ = std::make_unique<TensorRTInference>();
            if (!inference_engine_->loadEngine(engine_path))
            {
                throw std::runtime_error("Failed to load TensorRT engine from " + engine_path);
            }
            RCLCPP_INFO(this->get_logger(), "TensorRT engine loaded successfully from %s", engine_path.c_str());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error initializing inference engine: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        // --- QoS Profiles ---
        rclcpp::QoS qos_sensors(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
        qos_sensors.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        qos_sensors.keep_last(1);

        // --- Suscriptor ---
        image_sub_ = image_transport::create_subscription(
            this, image_topic_name,
            std::bind(&PIDNetNode::imageCallback, this, std::placeholders::_1),
            image_transport_type_, qos_sensors.get_rmw_qos_profile());

        // --- Publicadores ---
        std::string seg_map_topic = "/segmentation/" + output_topic_suffix_ + "/map";
        // std::string overlay_topic = "/segmentation/" + output_topic_suffix_ + "/overlay";
        // segmentation_pub_ = this->create_publisher<sensor_msgs::msg::Image>(seg_map_topic, qos_sensors);
        // overlay_pub_ = this->create_publisher<sensor_msgs::msg::Image>(overlay_topic, qos_sensors);
        segmentation_pub_ = this->create_publisher<yolo_custom_interfaces::msg::PidnetResult>(seg_map_topic, qos_sensors);

        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", image_topic_name.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing segmentation map to: %s", seg_map_topic.c_str());
        // RCLCPP_INFO(this->get_logger(), "Publishing overlay to: %s", overlay_topic.c_str());

        // --- Hilo de Inferencia ---
        inference_thread_ = std::thread(&PIDNetNode::inferenceLoop, this);

        // --- Crear eventos de CUDA para la medición ---
        cudaEventCreate(&gpu_timer_start_);
        cudaEventCreate(&gpu_timer_stop_);
        cudaEventCreate(&pre_start_);
        cudaEventCreate(&pre_stop_);
        cudaEventCreate(&infer_start_);
        cudaEventCreate(&infer_stop_);
        cudaEventCreate(&post_start_);
        cudaEventCreate(&post_stop_);
        RCLCPP_INFO(this->get_logger(), "%s initialized successfully.", node_name.c_str());
    }

    ~PIDNetNode()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down %s...", this->get_name());
        stop_inference_thread_.store(true);
        inference_cv_.notify_one();
        if (inference_thread_.joinable())
        {
            inference_thread_.join();
        }
        if (video_writer_overlay_.isOpened())
        {
            video_writer_overlay_.release();
        }

        // --- Destruir eventos de CUDA ---
        cudaEventDestroy(gpu_timer_start_);
        cudaEventDestroy(gpu_timer_stop_);
        cudaEventDestroy(pre_start_);
        cudaEventDestroy(pre_stop_);
        cudaEventDestroy(infer_start_);
        cudaEventDestroy(infer_stop_);
        cudaEventDestroy(post_start_);
        cudaEventDestroy(post_stop_);
        RCLCPP_INFO(this->get_logger(), "%s shutdown complete.", this->get_name());
    }

private:
    struct TimedImage
    {
        sensor_msgs::msg::Image::ConstSharedPtr msg;
        timespec monotonic_entry_time;               // Time captured with clock_gettime(CLOCK_MONOTONIC, ...) in imageCallback
        timespec image_source_monotonic_capture_ts;  // To store the monotonic time from directory_publisher
        uint32_t source_image_seq;
    };

    std::queue<TimedImage> inference_queue_;
    std::mutex inference_mutex_;
    std::condition_variable inference_cv_;
    std::thread inference_thread_;
    std::atomic<bool> stop_inference_thread_{false};

    // ROS Comms
    image_transport::Subscriber image_sub_;
    // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_pub_;
    rclcpp::Publisher<yolo_custom_interfaces::msg::PidnetResult>::SharedPtr segmentation_pub_;
    // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;

    // Inference Engine
    std::unique_ptr<TensorRTInference> inference_engine_;

    // Parameters
    std::string output_topic_suffix_;
    std::string image_transport_type_;
    double overlay_alpha_;
    bool enable_measure_times_;
    bool enable_realtime_display_;

    // Video Writing
    cv::VideoWriter video_writer_overlay_;
    std::string video_output_path_str_;
    double video_fps_;
    cv::Size video_frame_size_;
    bool video_writer_initialized_ = false;

    // Timers
    ChronoTimer timer_inter_callback_arrival_;
    ChronoTimer timer_queue_duration_;
    ChronoTimer timer_e2e_node_latency_;
    ChronoTimer timer_process_image_func_;
    ChronoTimer timer_inter_publish_;


    // --- NUEVO: Eventos de CUDA para mediciones precisas en GPU ---
    cudaEvent_t gpu_timer_start_, gpu_timer_stop_;
    cudaEvent_t pre_start_, pre_stop_;
    cudaEvent_t infer_start_, infer_stop_;
    cudaEvent_t post_start_, post_stop_;

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        TimedImage timed_msg;
        timed_msg.msg = msg;

        if (enable_measure_times_)
        {
            clock_gettime(CLOCK_MONOTONIC, &timed_msg.monotonic_entry_time);
            timer_inter_callback_arrival_.GetElapsedTime();
            timer_inter_callback_arrival_.ComputeStats();
            timer_inter_callback_arrival_.Reset();
        }

        {
            std::lock_guard<std::mutex> lock(inference_mutex_);
            inference_queue_.push(timed_msg);
        }
        inference_cv_.notify_one();
    }

    void inferenceLoop()
    {
        while (!stop_inference_thread_)
        {
            TimedImage timed_msg;
            {
                std::unique_lock<std::mutex> lock(inference_mutex_);
                inference_cv_.wait(lock, [this] {
                    return !inference_queue_.empty() || stop_inference_thread_;
                });

                if (stop_inference_thread_) break;

                timed_msg = inference_queue_.front();
                inference_queue_.pop();
            }

            if (enable_measure_times_)
            {
                timer_queue_duration_.startTime = timed_msg.monotonic_entry_time;
                timer_queue_duration_.GetElapsedTime();
                timer_queue_duration_.ComputeStats();
                processImageTimedGPU(timed_msg);
            }
            else
            {
                // processImage(timed_msg);
            }
        }
    }

    // void processImage(const TimedImage &timed_msg)
    // {
    //     // --- Lectura y Preprocesado ---
    //     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(timed_msg.msg, sensor_msgs::image_encodings::BGR8);
    //     if (!cv_ptr || cv_ptr->image.empty()) {
    //         RCLCPP_WARN(this->get_logger(), "Received empty image.");
    //         return;
    //     }
    //     cv::Mat input_frame = cv_ptr->image;
    //     cv::Mat preprocessed = inference_engine_->preprocess(input_frame);

    //     // --- Inferencia ---
    //     cv::Mat raw_output = inference_engine_->inference(preprocessed);

    //     // --- Postprocesado ---
    //     cv::Mat segmentation_map = inference_engine_->postprocess(raw_output, input_frame.size());
    //     cv::Mat colored_segmentation = inference_engine_->applyColormap(segmentation_map);
    //     cv::Mat overlay;
    //     cv::addWeighted(input_frame, 1.0 - overlay_alpha_, colored_segmentation, overlay_alpha_, 0, overlay);

    //     auto result_msg = std::make_unique<yolo_custom_interfaces::msg::PidnetResult>();


    //     // 2. Mapa de segmentación (imagen 8UC2)
    //     cv_bridge::CvImage cv_image(result_msg->header, "8UC2", segmentation_and_confidence_map);
    //     result_msg->segmentation_map = *cv_image.toImageMsg();

    //     segmentation_pub_->publish(std::move(result_msg));
    //     // auto overlay_msg = cv_bridge::CvImage(timed_msg.msg->header, "bgr8", overlay).toImageMsg();
    //     // overlay_pub_->publish(*overlay_msg);
    // }

    // void processImageTimed(const TimedImage &timed_msg)
    // {
    //     timer_process_image_func_.Reset();
    //     timer_e2e_node_latency_.startTime = timed_msg.monotonic_entry_time;

    //     // --- Lectura ---
    //     auto t_read_start = std::chrono::steady_clock::now();
    //     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(timed_msg.msg, sensor_msgs::image_encodings::BGR8);
    //     auto t_read_end = std::chrono::steady_clock::now();
    //     auto t_read = std::chrono::duration_cast<std::chrono::microseconds>(t_read_end - t_read_start).count();

    //     if (!cv_ptr || cv_ptr->image.empty()) {
    //         RCLCPP_WARN(this->get_logger(), "Received empty image.");
    //         return;
    //     }
    //     cv::Mat input_frame = cv_ptr->image;

    //     // --- Preprocesado ---
    //     auto t_pre_start = std::chrono::steady_clock::now();
    //     cv::Mat preprocessed = inference_engine_->preprocess(input_frame);
    //     auto t_pre_end = std::chrono::steady_clock::now();
    //     auto t_preproc = std::chrono::duration_cast<std::chrono::microseconds>(t_pre_end - t_pre_start).count();

    //     // --- Inferencia ---
    //     auto t_inf_start = std::chrono::steady_clock::now();
    //     cv::Mat raw_output = inference_engine_->inference(preprocessed);
    //     auto t_inf_end = std::chrono::steady_clock::now();
    //     auto t_infer = std::chrono::duration_cast<std::chrono::microseconds>(t_inf_end - t_inf_start).count();

    //     // --- Postprocesado ---
    //     auto t_post_start = std::chrono::steady_clock::now();
    //     cv::Mat segmentation_map = inference_engine_->postprocess(raw_output, input_frame.size());
    //     cv::Mat colored_segmentation = inference_engine_->applyColormap(segmentation_map);
    //     cv::Mat overlay;
    //     cv::addWeighted(input_frame, 1.0 - overlay_alpha_, colored_segmentation, overlay_alpha_, 0, overlay);
    //     auto t_post_end = std::chrono::steady_clock::now();
    //     auto t_postproc = std::chrono::duration_cast<std::chrono::microseconds>(t_post_end - t_post_start).count();

    //     // --- Publicación ---
    //     auto t_pub_start = std::chrono::steady_clock::now();
    //     // CORRECCIÓN: Crear y poblar el mensaje PidnetResult personalizado
    //     auto result_msg = std::make_unique<yolo_custom_interfaces::msg::PidnetResult>();

    //     // 1. Header y número de secuencia
    //     result_msg->header = timed_msg.msg->header;
    //     result_msg->packet_sequence_number = packet_sequence_counter_++;

    //     // 2. Mapa de segmentación (imagen 8UC2)
    //     cv_bridge::CvImage cv_image(result_msg->header, "8UC2", segmentation_and_confidence_map);
    //     result_msg->segmentation_map = *cv_image.toImageMsg();

    //     // 3. Timestamps
    //     // T1 (Entrada al callback) y T2 (Inicio del procesamiento en el hilo) son el mismo en esta arquitectura
    //     result_msg->image_source_monotonic_capture_time.sec = timed_msg.monotonic_entry_time.tv_sec;
    //     result_msg->image_source_monotonic_capture_time.nanosec = timed_msg.monotonic_entry_time.tv_nsec;
    //     result_msg->processing_node_monotonic_entry_time = result_msg->image_source_monotonic_capture_time;

    //     // T2a (Inicio de la inferencia)
    //     result_msg->processing_node_inference_start_time.sec = ts_infer_start.tv_sec;
    //     result_msg->processing_node_inference_start_time.nanosec = ts_infer_start.tv_nsec;

    //     // T2b (Fin de la inferencia)
    //     result_msg->processing_node_inference_end_time.sec = ts_infer_end.tv_sec;
    //     result_msg->processing_node_inference_end_time.nanosec = ts_infer_end.tv_nsec;

    //     // T3 (Justo antes de publicar)
    //     timespec ts_publish;
    //     clock_gettime(CLOCK_MONOTONIC, &ts_publish);
    //     result_msg->processing_node_monotonic_publish_time.sec = ts_publish.tv_sec;
    //     result_msg->processing_node_monotonic_publish_time.nanosec = ts_publish.tv_nsec;

    //     segmentation_pub_->publish(std::move(result_msg));
    //     auto t_pub_end = std::chrono::steady_clock::now();
    //     auto t_publish = std::chrono::duration_cast<std::chrono::microseconds>(t_pub_end - t_pub_start).count();

    //     timer_e2e_node_latency_.GetElapsedTime();
    //     timer_e2e_node_latency_.ComputeStats();
    //     timer_inter_publish_.GetElapsedTime();
    //     timer_inter_publish_.ComputeStats();
    //     timer_inter_publish_.Reset();
    //     timer_process_image_func_.GetElapsedTime();
    //     timer_process_image_func_.ComputeStats();

    //     // --- Visualización y Escritura en Video ---
    //     if (enable_realtime_display_) {
    //         cv::Mat display_image;
    //         cv::resize(overlay, display_image, cv::Size(640, 400)); // Resize for display
    //         std::lock_guard<std::mutex> lock(g_display_mutex);
    //         g_display_images[output_topic_suffix_] = display_image;
    //         g_new_image_flags[output_topic_suffix_] = true;
    //         g_display_cv.notify_one();
    //     }

    //     if (!video_output_path_str_.empty()) {
    //         if (!video_writer_initialized_) initializeVideoWriter();
    //         if (video_writer_overlay_.isOpened()) {
    //             cv::Mat video_frame;
    //             if (overlay.size() != video_frame_size_) {
    //                 cv::resize(overlay, video_frame, video_frame_size_);
    //             } else {
    //                 video_frame = overlay;
    //             }
    //             video_writer_overlay_.write(video_frame);
    //         }
    //     }

    //     RCLCPP_INFO(this->get_logger(),
    //         "[%s] Timings(us): Read=%ld, Pre=%ld, Infer=%ld, Post=%ld, Pub=%ld | "
    //         "InterCB(ms): Avg=%.2f,Max=%.2f | "
    //         "QueueT(ms): Avg=%.2f,Max=%.2f | "
    //         "E2ELat(ms): Avg=%.2f,Max=%.2f | "
    //         "InterPub(ms): Avg=%.2f,Max=%.2f",
    //         this->get_name(), t_read, t_preproc, t_infer, t_postproc, t_publish,
    //         timer_inter_callback_arrival_.mean_time, timer_inter_callback_arrival_.max_time,
    //         timer_queue_duration_.mean_time, timer_queue_duration_.max_time,
    //         timer_e2e_node_latency_.mean_time, timer_e2e_node_latency_.max_time,
    //         timer_inter_publish_.mean_time, timer_inter_publish_.max_time);
    // }

    void processImageTimedGPU(const TimedImage &timed_msg) {
        timespec ts_infer_start, ts_infer_end;
        timer_process_image_func_.Reset();
        timer_e2e_node_latency_.startTime = timed_msg.monotonic_entry_time;

        // --- Lectura (CPU) ---
        auto t_read_start = std::chrono::steady_clock::now();
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(timed_msg.msg, sensor_msgs::image_encodings::BGR8);
        auto t_read_end = std::chrono::steady_clock::now();
        float t_read = std::chrono::duration_cast<std::chrono::microseconds>(t_read_end - t_read_start).count();

        if (!cv_ptr || cv_ptr->image.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty image in GPU processing path.");
            return;
        }
        cv::Mat input_frame = cv_ptr->image;

        // --- Preprocesado (GPU) ---
        cv::Mat resized_frame;
        cv::resize(input_frame, resized_frame, cv::Size(1920, 1200));
        cudaEventRecord(pre_start_);
        inference_engine_->preprocess_gpu(resized_frame);
        cudaEventRecord(pre_stop_);

        // --- Inferencia (GPU) ---
        cudaEventRecord(infer_start_);
        clock_gettime(CLOCK_MONOTONIC, &ts_infer_start);
        inference_engine_->inference_gpu();
        clock_gettime(CLOCK_MONOTONIC, &ts_infer_end);
        cudaEventRecord(infer_stop_);

        // --- Postprocesado (GPU) ---
        cudaEventRecord(post_start_);
        // postprocess_gpu devuelve una única Mat de 2 canales (CV_8UC2)
        // cv::Mat segmentation_and_confidence_map = inference_engine_->postprocess_gpu(cv::Mat(), input_frame.size());
        cv::Mat segmentation_and_confidence_map = inference_engine_->postprocess_gpu_with_confidence(cv::Mat(), input_frame.size());
        cudaEventRecord(post_stop_);

        // --- Sincronizar y medir tiempos de GPU ---
        // Esperamos a que el último evento (post_stop_) se complete.
        cudaEventSynchronize(post_stop_);
        
        float t_preproc_ms = 0, t_infer_ms = 0, t_postproc_ms = 0;
        cudaEventElapsedTime(&t_preproc_ms, pre_start_, pre_stop_);
        cudaEventElapsedTime(&t_infer_ms, infer_start_, infer_stop_);
        cudaEventElapsedTime(&t_postproc_ms, post_start_, post_stop_);

        // Convertir a microsegundos para consistencia en el log
        long t_preproc = static_cast<long>(t_preproc_ms * 1000);
        long t_infer = static_cast<long>(t_infer_ms * 1000);
        long t_postproc = static_cast<long>(t_postproc_ms * 1000);

        // --- Operaciones en CPU post-GPU ---
        auto t_cpu_after_gpu_start = std::chrono::steady_clock::now();
        // Extraer solo el mapa de clases (canal 0) para colorear y crear el overlay
        cv::Mat segmentation_map_only;
        cv::Mat confidence_map_only;
        std::vector<cv::Mat> channels;
        cv::split(segmentation_and_confidence_map, channels);
        segmentation_map_only = channels[0];
        confidence_map_only = channels[1];
        cv::Mat colored_segmentation = inference_engine_->applyColormap(segmentation_map_only);
        cv::Mat overlay;
        cv::addWeighted(input_frame, 1.0 - overlay_alpha_, colored_segmentation, overlay_alpha_, 0, overlay);

        // --- Publicación ---
        auto t_pub_start = std::chrono::steady_clock::now();
        auto result_msg = std::make_unique<yolo_custom_interfaces::msg::PidnetResult>();

        // 1. Header y número de secuencia
        result_msg->header = timed_msg.msg->header;

        // 2. Mapa de segmentación (imagen 8UC2)
        cv_bridge::CvImage cv_image(result_msg->header, "8UC2", segmentation_and_confidence_map);
        result_msg->segmentation_map = *cv_image.toImageMsg();

        // 3. Timestamps
        // T1 (Entrada al callback) y T2 (Inicio del procesamiento en el hilo) son el mismo en esta arquitectura
        result_msg->image_source_monotonic_capture_time.sec = timed_msg.monotonic_entry_time.tv_sec;
        result_msg->image_source_monotonic_capture_time.nanosec = timed_msg.monotonic_entry_time.tv_nsec;
        result_msg->processing_node_monotonic_entry_time = result_msg->image_source_monotonic_capture_time;

        // T2a (Inicio de la inferencia)
        result_msg->processing_node_inference_start_time.sec = ts_infer_start.tv_sec;
        result_msg->processing_node_inference_start_time.nanosec = ts_infer_start.tv_nsec;

        // T2b (Fin de la inferencia)
        result_msg->processing_node_inference_end_time.sec = ts_infer_end.tv_sec;
        result_msg->processing_node_inference_end_time.nanosec = ts_infer_end.tv_nsec;

        // T3 (Justo antes de publicar)
        timespec ts_publish;
        clock_gettime(CLOCK_MONOTONIC, &ts_publish);
        result_msg->processing_node_monotonic_publish_time.sec = ts_publish.tv_sec;
        result_msg->processing_node_monotonic_publish_time.nanosec = ts_publish.tv_nsec;

        segmentation_pub_->publish(std::move(result_msg));

        auto t_pub_end = std::chrono::steady_clock::now();

        // auto overlay_msg = cv_bridge::CvImage(timed_msg.msg->header, "bgr8", overlay).toImageMsg();
        // overlay_pub_->publish(*overlay_msg);
        auto t_cpu_after_gpu_end = std::chrono::steady_clock::now();
        long t_publish = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_after_gpu_end - t_cpu_after_gpu_start).count();

        timer_e2e_node_latency_.GetElapsedTime();
        timer_e2e_node_latency_.ComputeStats();
        timer_inter_publish_.GetElapsedTime();
        timer_inter_publish_.ComputeStats();
        timer_inter_publish_.Reset();
        timer_process_image_func_.GetElapsedTime();
        timer_process_image_func_.ComputeStats();

        // --- Visualización y Escritura en Video ---
        if (enable_realtime_display_) {
            cv::Mat display_image;
            cv::resize(overlay, display_image, cv::Size(640, 400)); // Resize for display
            std::lock_guard<std::mutex> lock(g_display_mutex);
            g_display_images[output_topic_suffix_] = display_image;
            g_new_image_flags[output_topic_suffix_] = true;
            g_display_cv.notify_one();
        }

        if (!video_output_path_str_.empty()) {
            if (!video_writer_initialized_) initializeVideoWriter();
            if (video_writer_overlay_.isOpened()) {
                cv::Mat video_frame;
                if (overlay.size() != video_frame_size_) {
                    cv::resize(overlay, video_frame, video_frame_size_);
                } else {
                    video_frame = overlay;
                }
                video_writer_overlay_.write(video_frame);
            }
        }

        // Registrar tiempos
        RCLCPP_INFO(this->get_logger(),
            "[%s GPU] Timings(us): Read=%.0f, Pre=%ld, Infer=%ld, Post=%ld, Pub+CPU=%ld | "
            "InterCB(ms): Avg=%.2f,Max=%.2f | "
            "QueueT(ms): Avg=%.2f,Max=%.2f | "
            "E2ELat(ms): Avg=%.2f,Max=%.2f | "
            "InterPub(ms): Avg=%.2f,Max=%.2f",
            this->get_name(), t_read, t_preproc, t_infer, t_postproc, t_publish,
            timer_inter_callback_arrival_.mean_time, timer_inter_callback_arrival_.max_time,
            timer_queue_duration_.mean_time, timer_queue_duration_.max_time,
            timer_e2e_node_latency_.mean_time, timer_e2e_node_latency_.max_time,
            timer_inter_publish_.mean_time, timer_inter_publish_.max_time);
    }
    
    void initializeVideoWriter() {
        if (video_writer_initialized_) return;
        fs::path output_dir(video_output_path_str_);
        if (!fs::exists(output_dir)) {
            fs::create_directories(output_dir);
        }
        std::string video_filename = "overlay_" + output_topic_suffix_ + ".avi";
        fs::path video_full_path = output_dir / video_filename;
        
        RCLCPP_INFO(this->get_logger(), "Initializing video writer: %s", video_full_path.c_str());
        if (!video_writer_overlay_.open(video_full_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), video_fps_, video_frame_size_, true)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open video writer for overlay.");
        }
        video_writer_initialized_ = true;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto param_node = std::make_shared<rclcpp::Node>("pidnet_param_node");
    param_node->declare_parameter<bool>("measure_times", true);
    param_node->declare_parameter<bool>("realtime_display", true);
    param_node->declare_parameter<std::string>("output_video_path", "/home/david/ros_videos/pidnet_out");
    param_node->declare_parameter<double>("video_fps", 10.0);
    param_node->declare_parameter<int>("video_width", 1920);
    param_node->declare_parameter<int>("video_height", 1200);
    param_node->declare_parameter<std::string>("image_transport_type", "raw");
    param_node->declare_parameter<std::string>("left_camera_topic", "/camera_front_left/image_raw");
    param_node->declare_parameter<std::string>("front_camera_topic", "/camera_front/image_raw");
    param_node->declare_parameter<std::string>("right_camera_topic", "/camera_front_right/image_raw");

    bool measure_times = param_node->get_parameter("measure_times").as_bool();
    bool realtime_display = param_node->get_parameter("realtime_display").as_bool();
    std::string output_video_path = param_node->get_parameter("output_video_path").as_string();
    double video_fps = param_node->get_parameter("video_fps").as_double();
    cv::Size video_size(param_node->get_parameter("video_width").as_int(), param_node->get_parameter("video_height").as_int());
    std::string transport = param_node->get_parameter("image_transport_type").as_string();
    std::string left_topic = param_node->get_parameter("left_camera_topic").as_string();
    std::string front_topic = param_node->get_parameter("front_camera_topic").as_string();
    std::string right_topic = param_node->get_parameter("right_camera_topic").as_string();

    auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();

    auto left_node = std::make_shared<PIDNetNode>("pidnet_node_left", left_topic, "left", output_video_path, video_fps, video_size, transport, measure_times, realtime_display);
    auto front_node = std::make_shared<PIDNetNode>("pidnet_node_front", front_topic, "front", output_video_path, video_fps, video_size, transport, measure_times, realtime_display);
    auto right_node = std::make_shared<PIDNetNode>("pidnet_node_right", right_topic, "right", output_video_path, video_fps, video_size, transport, measure_times, realtime_display);

    executor->add_node(left_node);
    executor->add_node(front_node);
    executor->add_node(right_node);

    std::thread display_thread;
    if (realtime_display) {
        RCLCPP_INFO(rclcpp::get_logger("main"), "Starting display thread for PIDNet panoramic view...");
        display_thread = std::thread([]() {
            RCLCPP_INFO(rclcpp::get_logger("DisplayThread"), "Display thread started.");
            while (!g_stop_display_thread.load()) {
                cv::Mat left, front, right;
                {
                    std::unique_lock<std::mutex> lock(g_display_mutex);
                    g_display_cv.wait(lock, [] {
                        return (g_new_image_flags["left"] && g_new_image_flags["front"] && g_new_image_flags["right"]) || g_stop_display_thread.load();
                    });
                    if (g_stop_display_thread.load()) break;
                    left = g_display_images["left"].clone();
                    front = g_display_images["front"].clone();
                    right = g_display_images["right"].clone();
                    g_new_image_flags["left"] = g_new_image_flags["front"] = g_new_image_flags["right"] = false;
                }
                if (!left.empty() && !front.empty() && !right.empty()) {
                    cv::Mat stitched_image;
                    cv::hconcat(std::vector<cv::Mat>{left, front, right}, stitched_image);
                    cv::imshow("PIDNet Panoramic View", stitched_image);
                    cv::waitKey(1);
                }
            }
            cv::destroyWindow("PIDNet Panoramic View");
            RCLCPP_INFO(rclcpp::get_logger("DisplayThread"), "Display thread finished.");
        });
    }

    RCLCPP_INFO(rclcpp::get_logger("main"), "Spinning PIDNet nodes...");
    executor->spin();

    g_stop_display_thread.store(true);
    g_display_cv.notify_all();
    if (display_thread.joinable()) {
        display_thread.join();
    }

    rclcpp::shutdown();
    return 0;
}