#include "rclcpp/rclcpp.hpp"
#include "yolo_custom_interfaces/msg/instance_segmentation_info.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <fstream>   // Para std::ofstream
#include <iomanip>   // Para std::fixed, std::setprecision
#include <vector>    // Para std::vector (aunque no se usa directamente, es común)
#include <map>       // Para std::map
#include <string>    // Para std::string
#include <limits>    // Para std::numeric_limits
#include <cmath>     // Para std::sqrt (si se quisiera desviación estándar)
#include <filesystem>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread> // Para std::thread
#include <queue>  // Para std::queue
#include <condition_variable> // Para std::condition_variable
#include <atomic> // Para std::atomic
#include <sstream> // Para std::ostringstream

using namespace std::chrono_literals;

// Estructura para calcular y almacenar estadísticas de latencia
struct LatencyMetrics
{
    long count = 0;
    double sum_ms = 0.0;
    double sum_sq_ms = 0.0; // Suma de cuadrados para la varianza
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = std::numeric_limits<double>::min();
    double mean_ms = 0.0;
    double variance_ms = 0.0;

    void update(double value_ms)
    {
        count++;
        sum_ms += value_ms;
        sum_sq_ms += value_ms * value_ms;
        mean_ms = sum_ms / count;

        if (count > 1)
        {
            variance_ms = (sum_sq_ms - (sum_ms * sum_ms) / count) / (count - 1);
        }
        else
        {
            variance_ms = 0.0; // Varianza no definida para menos de 2 muestras
        }

        if (value_ms < min_ms)
            min_ms = value_ms;
        if (value_ms > max_ms)
            max_ms = value_ms;
    }

    void reset()
    {
        count = 0;
        sum_ms = 0.0;
        sum_sq_ms = 0.0;
        min_ms = std::numeric_limits<double>::max();
        max_ms = std::numeric_limits<double>::min();
        mean_ms = 0.0;
        variance_ms = 0.0;
    }
};

class FrequencySubscriber : public rclcpp::Node
{
public:
    FrequencySubscriber()
        : Node("result_frequency_subscriber"), stop_csv_writer_thread_(false)
    {
        // Inicializar thread que guarda las latencias en un archivo CSV

        csv_writer_thread_ = std::thread(&FrequencySubscriber::csvWriterLoop, this);

        // Abrir archivo CSV para escritura
        initialize_colors(); 

        auto callback =
            [this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg, const std::string &camera_id)
        {
            timespec ts_seg_sub_reception; // T4_mono
            clock_gettime(CLOCK_MONOTONIC, &ts_seg_sub_reception);

            timespec ts_image_source_capture; // T1_mono
            ts_image_source_capture.tv_sec = msg->image_source_monotonic_capture_time.sec;
            ts_image_source_capture.tv_nsec = msg->image_source_monotonic_capture_time.nanosec;

            timespec ts_processing_node_entry; // T2_mono
            ts_processing_node_entry.tv_sec = msg->processing_node_monotonic_entry_time.sec;
            ts_processing_node_entry.tv_nsec = msg->processing_node_monotonic_entry_time.nanosec;

            timespec ts_processing_node_publish; // T3_mono
            ts_processing_node_publish.tv_sec = msg->processing_node_monotonic_publish_time.sec;
            ts_processing_node_publish.tv_nsec = msg->processing_node_monotonic_publish_time.nanosec;

            // Calcular latencias individuales
            double lat_t1_t2_ms =
                (ts_processing_node_entry.tv_sec - ts_image_source_capture.tv_sec) * 1000.0 +
                (static_cast<double>(ts_processing_node_entry.tv_nsec) - static_cast<double>(ts_image_source_capture.tv_nsec)) / 1e6;

            double lat_t2_t3_ms =
                (ts_processing_node_publish.tv_sec - ts_processing_node_entry.tv_sec) * 1000.0 +
                (static_cast<double>(ts_processing_node_publish.tv_nsec) - static_cast<double>(ts_processing_node_entry.tv_nsec)) / 1e6;

            double lat_t3_t4_ms =
                (ts_seg_sub_reception.tv_sec - ts_processing_node_publish.tv_sec) * 1000.0 +
                (static_cast<double>(ts_seg_sub_reception.tv_nsec) - static_cast<double>(ts_processing_node_publish.tv_nsec)) / 1e6;
            
            double lat_t1_t4_ms_total =
                (ts_seg_sub_reception.tv_sec - ts_image_source_capture.tv_sec) * 1000.0 +
                (static_cast<double>(ts_seg_sub_reception.tv_nsec) - static_cast<double>(ts_image_source_capture.tv_nsec)) / 1e6;
            
            // Latencia acumulada T1->T3 (ya la tenías, pero la recalculo para consistencia si es necesario)
            double lat_t1_t3_ms =
                (ts_processing_node_publish.tv_sec - ts_image_source_capture.tv_sec) * 1000.0 +
                (static_cast<double>(ts_processing_node_publish.tv_nsec) - static_cast<double>(ts_image_source_capture.tv_nsec)) / 1e6;

            // Log visual (como antes)
            RCLCPP_INFO(this->get_logger(),
                        "[%s] Latency Breakdown (ms) Seq: %u\n"
                        "  T1 (DirPub ImgPub) ......: %ld.%09ld\n"
                        "  T2 (SegNode ImgRecep)....: %ld.%09ld  |  T1->T2 (Net+Queue): %.3f ms\n"
                        "  T3 (SegNode ResPub)......: %ld.%09ld  |  T2->T3 (Processing): %.3f ms\n"
                        "  T4 (SegSub ResRecep).....: %ld.%09ld  |  T3->T4 (Net+Queue): %.3f ms\n"
                        "  ----------------------------------------------------------------------\n"
                        "  Cumulative T1->T3 (DirPub->SegNodePub): %.3f ms\n"
                        "  TOTAL      T1->T4 (DirPub->SegSubRecep): %.3f ms",
                        
                        camera_id.c_str(), msg->header.stamp.nanosec, // Usando nanosec del header como un pseudo-seq
                        ts_image_source_capture.tv_sec, ts_image_source_capture.tv_nsec,
                        ts_processing_node_entry.tv_sec, ts_processing_node_entry.tv_nsec, lat_t1_t2_ms,
                        ts_processing_node_publish.tv_sec, ts_processing_node_publish.tv_nsec, lat_t2_t3_ms,
                        ts_seg_sub_reception.tv_sec, ts_seg_sub_reception.tv_nsec, lat_t3_t4_ms,
                        lat_t1_t3_ms,
                        lat_t1_t4_ms_total);


            std::ostringstream csv_line_stream;
            csv_line_stream << std::fixed << std::setprecision(3)
                            << camera_id << ","
                            << msg->header.stamp.nanosec << "," // Usando nanosec del header como un pseudo-seq
                            << ts_image_source_capture.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_image_source_capture.tv_nsec << ","
                            << ts_processing_node_entry.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_processing_node_entry.tv_nsec << ","
                            << ts_processing_node_publish.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_processing_node_publish.tv_nsec << ","
                            << ts_seg_sub_reception.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_seg_sub_reception.tv_nsec << ","
                            << lat_t1_t2_ms << ","
                            << lat_t2_t3_ms << ","
                            << lat_t3_t4_ms << ","
                            << lat_t1_t4_ms_total;
            

            std::lock_guard<std::mutex> csv_lock(csv_queue_mutex_);
            csv_data_queue_.push(csv_line_stream.str());
            csv_queue_cv_.notify_one(); // Notificar al hilo de escritura 
            
            
            // Comprobar y registrar latencias T3-T4 anómalas
            // if (lat_t3_t4_ms < 5.0 || lat_t3_t4_ms > 30.0)
            // if (lat_t3_t4_ms < 5.0)
            // {
            //     save_anomalous_frame_and_log(msg, camera_id, ts_processing_node_publish, ts_seg_sub_reception, lat_t3_t4_ms);
            //     RCLCPP_WARN(this->get_logger(), "[%s] Anomalous T3-T4 latency: %.3f ms (MsgT1: %u.%09u). Logged and image saved.",
            //                 camera_id.c_str(), lat_t3_t4_ms, msg->header.stamp.sec, msg->header.stamp.nanosec);
            // }

            // Actualizar estadísticas
            {
                std::lock_guard<std::mutex> stats_lock(metrics_mutex_);
                all_metrics_[camera_id]["T1_T2"].update(lat_t1_t2_ms);
                all_metrics_[camera_id]["T2_T3"].update(lat_t2_t3_ms);
                all_metrics_[camera_id]["T3_T4"].update(lat_t3_t4_ms);
                all_metrics_[camera_id]["T1_T4_Total"].update(lat_t1_t4_ms_total);
            }

            // Lógica original de conteo de frecuencia
            if (camera_id == "left")
            {
                std::lock_guard<std::mutex> lock(msg_count_mutex_left_);
                count_left_++;
            }
            else if (camera_id == "front")
            {
                std::lock_guard<std::mutex> lock(msg_count_mutex_front_);
                count_front_++;
            }
            else if (camera_id == "right")
            {
                std::lock_guard<std::mutex> lock(msg_count_mutex_right_);
                count_right_++;
            }
        };

        // Suscribirse a los tres tópicos referentes a resultados
        rclcpp::QoS qos_profile(1); // Puedes ajustar la QoS según necesidad
        // qos_profile.best_effort(); // Si prefieres best_effort para no bloquear tanto
        qos_profile.reliable();
        qos_profile.durability_volatile(); // Durabilidad VOLATILE
        qos_profile.keep_last(5); // Mantener el último mensaje recibido


        sub_left_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/left/instance_info", qos_profile,
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                callback(msg, "left");
            });

        sub_front_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/front/instance_info", qos_profile,
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                callback(msg, "front");
            });

        sub_right_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/right/instance_info", qos_profile,
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                callback(msg, "right");
            });

        report_timer_ = this->create_wall_timer(
            5s, std::bind(&FrequencySubscriber::report_metrics, this));

        last_report_time_ = this->now();
    }

    ~FrequencySubscriber()
    {   RCLCPP_INFO(this->get_logger(), "Shutting down FrequencySubscriber...");
        stop_csv_writer_thread_ = true;
        csv_queue_cv_.notify_all(); // Notificar al hilo de escritura CSV para que termine
        if (csv_writer_thread_.joinable())
        {
            csv_writer_thread_.join();
            RCLCPP_INFO(this->get_logger(), "CSV writer thread joined successfully.");
        }else
        {
            RCLCPP_WARN(this->get_logger(), "CSV writer thread was not joinable.");
        }
    }

private:

    void csvWriterLoop()
    {
        std::ofstream latency_log_file_;
        // Abrir el archivo CSV para escritura
        latency_log_file_.open("latency_log.csv", std::ios_base::out | std::ios_base::trunc);

        if(latency_log_file_.is_open())
        {
            latency_log_file_ << "camera_id,msg_seq,t1_sec,t1_nsec,t2_sec,t2_nsec,t3_sec,t3_nsec,t4_sec,t4_nsec,"
                              << "lat_t1_t2_ms,lat_t2_t3_ms,lat_t3_t4_ms,lat_t1_t4_total_ms\n";
            RCLCPP_INFO(this->get_logger(), "Logging latency data to latency_log.csv");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open latency_log.csv for writing!");
        }


        while(true){
            std::string line_to_write;
            {
                std::unique_lock<std::mutex> lock(csv_queue_mutex_);
                csv_queue_cv_.wait(lock, [this] {
                    return stop_csv_writer_thread_ || !csv_data_queue_.empty();
                });

                if (stop_csv_writer_thread_ && csv_data_queue_.empty())
                {
                    break; // Salir del bucle si se indica parar y la cola está vacía
                }

                if (!csv_data_queue_.empty())
                {
                    line_to_write = csv_data_queue_.front();
                    csv_data_queue_.pop();
                }
            } // Se libera el lock

            if (!line_to_write.empty())
            {
                latency_log_file_ << line_to_write << "\n";
            }
        }

        if (latency_log_file_.is_open())
        {
            latency_log_file_.close();
            RCLCPP_INFO(this->get_logger(), "CSV writer thread: Closed latency_log.csv");
        }
    }

    void initialize_colors() {
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
            class_colors_.push_back(cv::Scalar(255, 165, 0));   // Orange
            class_colors_.push_back(cv::Scalar(255, 192, 203)); // Pink
            // Añade más si es necesario
        }

    cv::Scalar getRandomTone(const cv::Scalar &base_color, int seed)
    {
        cv::RNG rng(static_cast<uint64_t>(seed)); // Usar uint64_t para el seed de RNG
        double variation_range = 60.0;
        cv::Scalar toned_color;
        for (int i = 0; i < 3; ++i)
        {
            toned_color[i] = cv::saturate_cast<uchar>(base_color[i] + rng.uniform(-variation_range, variation_range));
        }
        return toned_color;
    }

    void save_anomalous_frame_and_log(
        const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr& msg,
        const std::string& camera_id,
        const timespec& ts_t3_mono, // Timestamp de publicación del nodo de procesamiento
        const timespec& ts_t4_mono, // Timestamp de recepción en este subscriptor
        double lat_t3_t4_ms)
    {
        // 1. Convertir msg->mask a cv::Mat
        cv_bridge::CvImagePtr cv_ptr_mask;
        try {
            // El encoding de la máscara debe ser el mismo que el publicado por segment_node_3P
            // (ej. "mono16" o "mono8")
            cv_ptr_mask = cv_bridge::toCvCopy(msg->mask, msg->mask.encoding);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "[%s] cv_bridge exception for mask (MsgT1: %u.%09u): %s",
                         camera_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec, e.what());
            return;
        }

        if (!cv_ptr_mask || cv_ptr_mask->image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "[%s] Failed to convert mask to CvImage or mask is empty (MsgT1: %u.%09u).",
                         camera_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
            return;
        }
        const cv::Mat& instance_id_mask = cv_ptr_mask->image; // Debería ser CV_16UC1 o CV_8UC1

        // 2. Crear imagen coloreada
        cv::Mat colored_mask_display = cv::Mat::zeros(instance_id_mask.size(), CV_8UC3);
        size_t num_instances_in_msg = msg->classes.size();

        if (class_colors_.empty()) {
            RCLCPP_INFO(this->get_logger(), "[%s] Color palette is empty. Cannot color anomalous frame.", camera_id.c_str());
            return; // No se puede colorear
        }

        for (int r = 0; r < instance_id_mask.rows; ++r) {
            for (int c = 0; c < instance_id_mask.cols; ++c) {
                uint16_t instance_id_from_mask = 0;
                if (instance_id_mask.type() == CV_16UC1) { // mono16
                    instance_id_from_mask = instance_id_mask.at<uint16_t>(r, c);
                } else if (instance_id_mask.type() == CV_8UC1) { // mono8
                    instance_id_from_mask = static_cast<uint16_t>(instance_id_mask.at<uchar>(r, c));
                } else {
                    if (r == 0 && c == 0) { // Loguear solo una vez por imagen
                         RCLCPP_ERROR_ONCE(this->get_logger(), "[%s] Unexpected mask type: %d for MsgT1: %u.%09u. Cannot color.",
                                          camera_id.c_str(), instance_id_mask.type(), msg->header.stamp.sec, msg->header.stamp.nanosec);
                    }
                   // No se puede procesar, salir de los bucles de coloreado
                   goto end_coloring_loops;
                }

                if (instance_id_from_mask > 0) { // ID 0 es fondo
                    size_t item_idx = static_cast<size_t>(instance_id_from_mask - 1); // IDs son 1-based
                    if (item_idx < num_instances_in_msg && item_idx < msg->classes.size()) {
                        int class_id = msg->classes[item_idx];
                        cv::Scalar base_color = class_colors_[class_id % class_colors_.size()];
                        // Usar el ID de instancia para el seed del color asegura consistencia para la misma instancia
                        cv::Scalar toned_color = getRandomTone(base_color, instance_id_from_mask);
                        colored_mask_display.at<cv::Vec3b>(r, c) = cv::Vec3b(
                            static_cast<uchar>(toned_color[0]),
                            static_cast<uchar>(toned_color[1]),
                            static_cast<uchar>(toned_color[2])
                        );
                    }
                }
            }
        }
        end_coloring_loops:;


        // 3. Guardar imagen
        // Usar el timestamp T1 del header del mensaje para un nombre de archivo único
        anomalous_images_path_ = "/home/david/yolocpp_ws/anomalous_frames/";
        std::string image_filename = anomalous_images_path_ +
                                     camera_id + "_" +
                                     std::to_string(msg->header.stamp.sec) + "_" +
                                     std::to_string(msg->header.stamp.nanosec) + ".png";

        RCLCPP_INFO(this->get_logger(), "[%s] Saving anomalous frame image to %s (MsgT1: %u.%09u)",
                         camera_id.c_str(), image_filename.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
        try {
            if (!colored_mask_display.empty()) {
                cv::imwrite(image_filename, colored_mask_display);
                // RCLCPP_INFO(this->get_logger(), "[%s] Saved anomalous frame image to %s", camera_id.c_str(), image_filename.c_str());
            } else {
                RCLCPP_WARN(this->get_logger(), "[%s] Colored mask display is empty for MsgT1: %u.%09u. Not saving image.",
                            camera_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
            }
        } catch (const cv::Exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "[%s] OpenCV Exception saving image %s (MsgT1: %u.%09u): %s",
                         camera_id.c_str(), image_filename.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec, ex.what());
        }

        // 4. Registrar info adicional en el log CSV de anomalías
        if (anomalous_t3_t4_log_file_.is_open()) {
            std::lock_guard<std::mutex> anomalous_lock(anomalous_log_mutex_);
            anomalous_t3_t4_log_file_ << std::fixed << std::setprecision(3)
                                      << camera_id << ","
                                      << msg->header.stamp.sec << "," << msg->header.stamp.nanosec << "," // T1 del header del mensaje
                                      << ts_t3_mono.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t3_mono.tv_nsec << ","
                                      << ts_t4_mono.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t4_mono.tv_nsec << ","
                                      << lat_t3_t4_ms << ",\""; // Abrir comillas para listas

            // Clases
            for (size_t i = 0; i < msg->classes.size(); ++i) {
                anomalous_t3_t4_log_file_ << msg->classes[i] << (i == msg->classes.size() - 1 ? "" : ";");
            }
            anomalous_t3_t4_log_file_ << "\",\""; // Cerrar comillas de clases, abrir para scores

            // Scores
            for (size_t i = 0; i < msg->scores.size(); ++i) {
                anomalous_t3_t4_log_file_ << std::fixed << std::setprecision(2) << msg->scores[i] << (i == msg->scores.size() - 1 ? "" : ";");
            }
            anomalous_t3_t4_log_file_ << "\"\n"; // Cerrar comillas de scores y nueva línea
        }
    }
    void report_metrics() // Renombrado de report_frequency a report_metrics
    {
        auto now = this->now();
        double elapsed_sec = (now - last_report_time_).seconds();
        if (elapsed_sec <= 0) elapsed_sec = 1.0; // Evitar división por cero si el timer es muy rápido

        uint64_t current_left = 0, current_front = 0, current_right = 0;
        {
            std::lock_guard<std::mutex> lock(msg_count_mutex_left_);
            current_left = count_left_;
            count_left_ = 0;
        }
        {
            std::lock_guard<std::mutex> lock(msg_count_mutex_front_);
            current_front = count_front_;
            count_front_ = 0;
        }
        {
            std::lock_guard<std::mutex> lock(msg_count_mutex_right_);
            current_right = count_right_;
            count_right_ = 0;
        }

        double freq_left = current_left / elapsed_sec;
        double freq_front = current_front / elapsed_sec;
        double freq_right = current_right / elapsed_sec;

        RCLCPP_INFO(this->get_logger(),
                    "--- Metrics Report (last %.1f sec) ---", elapsed_sec);
        RCLCPP_INFO(this->get_logger(),
                    "Msg Frequencies -> Left: %.2f Hz, Front: %.2f Hz, Right: %.2f Hz",
                    freq_left, freq_front, freq_right);

        std::lock_guard<std::mutex> stats_lock(metrics_mutex_); // Proteger acceso a all_metrics_
        for (const auto &cam_pair : all_metrics_)
        {
            const std::string &camera_id = cam_pair.first;
            RCLCPP_INFO(this->get_logger(), "Latency Stats for Camera: [%s] (Cumulative)", camera_id.c_str());
            for (const auto &metric_pair : cam_pair.second)
            {
                const std::string &metric_name = metric_pair.first;
                const LatencyMetrics &metrics = metric_pair.second;
                if (metrics.count > 0) {
                    RCLCPP_INFO(this->get_logger(),
                                "  %s: Count=%ld, Mean=%.3f ms, Var=%.3f ms^2, Min=%.3f ms, Max=%.3f ms",
                                metric_name.c_str(), metrics.count, metrics.mean_ms, metrics.variance_ms,
                                metrics.min_ms, metrics.max_ms);
                } else {
                     RCLCPP_INFO(this->get_logger(), "  %s: No data yet.", metric_name.c_str());
                }
            }
        }
        RCLCPP_INFO(this->get_logger(), "--- End of Metrics Report ---");

        last_report_time_ = now;
    }

    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_left_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_front_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_right_;
    rclcpp::TimerBase::SharedPtr report_timer_;

    rclcpp::Time last_report_time_;

    // Para conteo de frecuencia de mensajes
    std::mutex msg_count_mutex_left_;
    std::mutex msg_count_mutex_front_;
    std::mutex msg_count_mutex_right_;
    uint64_t count_left_ = 0;
    uint64_t count_front_ = 0;
    uint64_t count_right_ = 0;

    // Para log en CSV
    // Variables para el hilo de escritura del CSV
    std::thread csv_writer_thread_;
    std::queue<std::string> csv_data_queue_;
    std::mutex csv_queue_mutex_;
    std::condition_variable csv_queue_cv_;
    std::atomic<bool> stop_csv_writer_thread_;
    std::ofstream anomalous_t3_t4_log_file_; 
    std::mutex anomalous_log_mutex_;
    std::string anomalous_images_path_ = "/home/david/yolocpp_ws/anomalous_frames"; // Ruta para guardar imágenes anómalas
    std::vector<cv::Scalar> class_colors_; // Paleta de colores

    // Para estadísticas de latencia
    std::map<std::string, std::map<std::string, LatencyMetrics>> all_metrics_;
    // Outer map key: camera_id ("left", "front", "right")
    // Inner map key: metric_name ("T1_T2", "T2_T3", "T3_T4", "T1_T4_Total")
    std::mutex metrics_mutex_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FrequencySubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}