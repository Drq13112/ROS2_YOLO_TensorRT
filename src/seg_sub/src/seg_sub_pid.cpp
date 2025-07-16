#include "rclcpp/rclcpp.hpp"
#include "yolo_custom_interfaces/msg/pidnet_result.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>

// Reutilizamos la estructura de métricas de latencia de seg_sub.cpp
struct LatencyMetrics { /* ... (código de LatencyMetrics idéntico a seg_sub.cpp) ... */ };

// --- Variables Globales para la Interfaz de Usuario ---
std::mutex g_filter_mutex;
int g_confidence_threshold = 30; // Umbral de confianza inicial (0-255)
// Mapa para activar/desactivar la visualización de cada clase
std::map<int, bool> g_class_visibility;
const int g_num_classes = 20; // Cityscapes tiene 20 clases (0-19)

// Nombres de las clases para la UI (Cityscapes)
const std::vector<std::string> g_class_names = {
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "background"
};

// Colormap (idéntico al del nodo de inferencia para consistencia)
const cv::Scalar g_colormap[20] = {
    {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153},
    {153, 153, 153}, {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152},
    {70, 130, 180}, {220, 20, 60}, {255, 0, 0}, {0, 0, 142}, {0, 0, 70},
    {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}, {0, 0, 0}
};

const std::map<char, int> g_key_to_class = {
    {'r', 0}, {'s', 1}, {'b', 2}, {'w', 3}, {'f', 4}, {'p', 5},
    {'l', 6}, {'t', 7}, {'v', 8}, {'e', 9}, {'k', 10}, // 'e' for tErrain, 'k' for sKy
    {'h', 11}, {'i', 12}, {'c', 13}, {'u', 14}, {'a', 15}, // 'h' for Human, 'i' for rIder, 'u' for trUck, 'a' for bus
    {'n', 16}, {'m', 17}, {'y', 18}  // 'n' for traiN, 'y' for bicYcle
};

// Callback para la barra de desplazamiento de confianza
void on_confidence_trackbar(int pos, void*) {
    std::lock_guard<std::mutex> lock(g_filter_mutex);
    g_confidence_threshold = pos;
}

// Callback para los checkboxes de las clases
void on_class_toggle(int state, void* userdata) {
    int class_id = *static_cast<int*>(userdata);
    std::lock_guard<std::mutex> lock(g_filter_mutex);
    g_class_visibility[class_id] = (state == 1);
}

double calculate_latency_ms(const timespec& start, const timespec& end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
}

class PIDNetVisualizer : public rclcpp::Node
{
public:
    PIDNetVisualizer() : Node("pidnet_visualizer")
    {
        // Inicializar visibilidad de clases
        for (int i = 0; i < g_num_classes; ++i) {
            g_class_visibility[i] = true;
        }

        // Crear ventanas y controles de OpenCV
        cv::namedWindow("PIDNet Visualization", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Controls", cv::WINDOW_NORMAL);
        cv::createTrackbar("Confidence >", "PIDNet Visualization", &g_confidence_threshold, 255, on_confidence_trackbar);


        RCLCPP_INFO(this->get_logger(), "Visualization controls ready.");
        RCLCPP_INFO(this->get_logger(), "Press keys (r, s, b, etc.) to toggle class visibility.");
        RCLCPP_INFO(this->get_logger(), "Press ESC to exit.");

        // Crear checkboxes para cada clase
        // for (int i = 0; i < g_num_classes; ++i) {
        //     // Necesitamos almacenar los IDs de clase para los callbacks
        //     class_ids_for_callbacks_.push_back(i);
        //     cv::createButton(g_class_names[i], on_class_toggle, &class_ids_for_callbacks_.back(), cv::QT_CHECKBOX, 1);
        // }

        auto callback = [this](const yolo_custom_interfaces::msg::PidnetResult::SharedPtr msg, const std::string& camera_id) {
            this->process_message(msg, camera_id);
        };

        // Suscripciones al nuevo tipo de mensaje
        rclcpp::QoS qos_profile(rclcpp::KeepLast(10));
        qos_profile.reliable();

        sub_left_ = this->create_subscription<yolo_custom_interfaces::msg::PidnetResult>(
            "/segmentation/left/map", qos_profile, [callback, this](const yolo_custom_interfaces::msg::PidnetResult::SharedPtr msg){ callback(msg, "left"); });

        sub_front_ = this->create_subscription<yolo_custom_interfaces::msg::PidnetResult>(
            "/segmentation/front/map", qos_profile, [callback, this](const yolo_custom_interfaces::msg::PidnetResult::SharedPtr msg){ callback(msg, "front"); });

        sub_right_ = this->create_subscription<yolo_custom_interfaces::msg::PidnetResult>(
            "/segmentation/right/map", qos_profile, [callback, this](const yolo_custom_interfaces::msg::PidnetResult::SharedPtr msg){ callback(msg, "right"); });
    }

    // CORRECCIÓN: Mover display_loop a la sección pública para que main pueda acceder a ella.
    void display_loop() {
        while (rclcpp::ok()) {
            std::map<std::string, cv::Mat> images_to_show;
            {
                std::lock_guard<std::mutex> lock(display_mutex_);
                images_to_show = display_images_;
            }

            if (!images_to_show.empty()) {
                cv::Mat left = images_to_show.count("left") ? images_to_show["left"] : cv::Mat::zeros(480, 640, CV_8UC3);
                cv::Mat front = images_to_show.count("front") ? images_to_show["front"] : cv::Mat::zeros(480, 640, CV_8UC3);
                cv::Mat right = images_to_show.count("right") ? images_to_show["right"] : cv::Mat::zeros(480, 640, CV_8UC3);

                // Redimensionar si es necesario para que quepan en la pantalla
                cv::resize(left, left, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
                cv::resize(front, front, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
                cv::resize(right, right, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);

                cv::Mat stitched_frame;
                // CORRECCIÓN: Crear un vector explícito para hconcat para mayor compatibilidad.
                std::vector<cv::Mat> images_to_concat = {left, front, right};
                cv::hconcat(images_to_concat, stitched_frame);
                cv::imshow("PIDNet Visualization", stitched_frame);
            }

            // cv::waitKey procesa los eventos de la UI de OpenCV (muy importante)
            int key = cv::waitKey(33); // Espera 33ms
            if (key == 27) { // ESC
                rclcpp::shutdown();
                break;
            } else if (key > 0) {
                // CORRECCIÓN: Lógica para activar/desactivar clases con teclas
                char pressed_key = static_cast<char>(key);
                if (g_key_to_class.count(pressed_key)) {
                    int class_id = g_key_to_class.at(pressed_key);
                    std::lock_guard<std::mutex> lock(g_filter_mutex);
                    g_class_visibility[class_id] = !g_class_visibility[class_id];
                    RCLCPP_INFO(this->get_logger(), "Toggled class '%s' to %s",
                        g_class_names[class_id].c_str(), g_class_visibility[class_id] ? "VISIBLE" : "HIDDEN");
                }
            }
        }
    }

private:
    // CORRECCIÓN: Usar el namespace correcto 'yolo_custom_interfaces'
    void process_message(const yolo_custom_interfaces::msg::PidnetResult::SharedPtr& msg, const std::string& camera_id)
    {
        // --- 1. Análisis de Latencia ---
        timespec ts_t4_recv;
        clock_gettime(CLOCK_MONOTONIC, &ts_t4_recv);

        // Extraer todos los timestamps del mensaje
        timespec ts_t0_img_pub = {msg->header.stamp.sec, msg->header.stamp.nanosec};
        timespec ts_t1_cb_entry = {msg->image_source_monotonic_capture_time.sec, msg->image_source_monotonic_capture_time.nanosec};
        timespec ts_t3_res_pub = {msg->processing_node_monotonic_publish_time.sec, msg->processing_node_monotonic_publish_time.nanosec};

        // Calcular latencias
        double lat_driver_to_seg = calculate_latency_ms(ts_t0_img_pub, ts_t1_cb_entry);
        double lat_seg_processing = calculate_latency_ms(ts_t1_cb_entry, ts_t3_res_pub);
        double lat_seg_to_viz = calculate_latency_ms(ts_t3_res_pub, ts_t4_recv);
        double lat_total = calculate_latency_ms(ts_t0_img_pub, ts_t4_recv);

        RCLCPP_INFO(this->get_logger(),
            "[%s] Latencies (ms) | Pkt#%lu | "
            "Driver->Seg: %7.3f | "
            "Seg Proc: %7.3f | "
            "Seg->Viz: %7.3f | "
            "Total E2E: %7.3f",
            camera_id.c_str(), msg->packet_sequence_number,
            lat_driver_to_seg, lat_seg_processing, lat_seg_to_viz, lat_total);


        // --- 2. Decodificación y Visualización con Filtros ---
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg->segmentation_map, "8UC2");
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        const cv::Mat& combined_map = cv_ptr->image;
        cv::Mat visualization = cv::Mat::zeros(combined_map.size(), CV_8UC3);

        std::map<int, bool> current_class_visibility;
        {
            std::lock_guard<std::mutex> lock(g_filter_mutex);
            current_class_visibility = g_class_visibility;
        }

        for (int r = 0; r < combined_map.rows; ++r) {
            for (int c = 0; c < combined_map.cols; ++c) {
                cv::Vec2b pixel_data = combined_map.at<cv::Vec2b>(r, c);
                uint8_t class_id = pixel_data[0];
                // uint8_t confidence = pixel_data[1]; // Confianza ya no se usa

                if (class_id < g_num_classes && current_class_visibility[class_id])
                {
                    const cv::Scalar& color = g_colormap[class_id];
                    visualization.at<cv::Vec3b>(r, c) = cv::Vec3b(color[0], color[1], color[2]);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(display_mutex_);
            display_images_[camera_id] = visualization;
        }
    }

    rclcpp::Subscription<yolo_custom_interfaces::msg::PidnetResult>::SharedPtr sub_left_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::PidnetResult>::SharedPtr sub_front_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::PidnetResult>::SharedPtr sub_right_;

    std::thread display_thread_;
    std::mutex display_mutex_;
    std::map<std::string, cv::Mat> display_images_;
    std::vector<int> class_ids_for_callbacks_; // Para que los punteros de los callbacks sean estables
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PIDNetVisualizer>();
    
    // Iniciar el hilo de visualización por separado para no bloquear el spin de ROS
    std::thread display_thread(&PIDNetVisualizer::display_loop, node.get());

    rclcpp::spin(node);
    
    display_thread.join();
    rclcpp::shutdown();
    return 0;
}
