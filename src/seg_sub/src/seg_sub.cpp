#include "rclcpp/rclcpp.hpp"
#include "yolo_custom_interfaces/msg/instance_segmentation_info.hpp"
#include <chrono>
#include <memory>
#include <mutex>

using namespace std::chrono_literals;

class FrequencySubscriber : public rclcpp::Node
{
public:
    FrequencySubscriber()
    : Node("result_frequency_subscriber")
    {
        auto callback = 
            [this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg, const std::string& camera_id)
            {
                timespec ts_seg_sub_reception;
                clock_gettime(CLOCK_MONOTONIC, &ts_seg_sub_reception); // T4_mono

                // Latency from segment_node_3P publish to seg_sub reception (T4_mono - T3_mono)
                timespec ts_processing_node_publish;
                ts_processing_node_publish.tv_sec = msg->processing_node_monotonic_publish_time.sec;
                ts_processing_node_publish.tv_nsec = msg->processing_node_monotonic_publish_time.nanosec;

                double latency_segnode_to_segsub_ms = 0.0;
                if (ts_processing_node_publish.tv_sec > 0) { // Check if timestamp is valid
                    latency_segnode_to_segsub_ms =
                        (ts_seg_sub_reception.tv_sec - ts_processing_node_publish.tv_sec) * 1000.0 +
                        (ts_seg_sub_reception.tv_nsec - ts_processing_node_publish.tv_nsec) / 1e6;
                }

                // Total latency from directory_publisher to seg_sub reception (T4_mono - T1_mono)
                timespec ts_image_source_capture;
                ts_image_source_capture.tv_sec = msg->image_source_monotonic_capture_time.sec;
                ts_image_source_capture.tv_nsec = msg->image_source_monotonic_capture_time.nanosec;
                
                double latency_total_ms = 0.0;
                if (ts_image_source_capture.tv_sec > 0) { // Check if timestamp is valid
                     latency_total_ms =
                        (ts_seg_sub_reception.tv_sec - ts_image_source_capture.tv_sec) * 1000.0 +
                        (ts_seg_sub_reception.tv_nsec - ts_image_source_capture.tv_nsec) / 1e6;
                }

                RCLCPP_INFO(this->get_logger(),
                            "[%s] Latencies: SegNodePub->SegSubRecep: %.3f ms, Total DirPub->SegSubRecep: %.3f ms. "
                            "DirPubMonoTS: %ld.%09ld, SegNodeMonoPubTS: %ld.%09ld, SegSubMonoRecepTS: %ld.%09ld",
                            camera_id.c_str(),
                            latency_segnode_to_segsub_ms,
                            latency_total_ms,
                            ts_image_source_capture.tv_sec, ts_image_source_capture.tv_nsec,
                            ts_processing_node_publish.tv_sec, ts_processing_node_publish.tv_nsec,
                            ts_seg_sub_reception.tv_sec, ts_seg_sub_reception.tv_nsec
                            );

                // Original frequency counting logic
                if (camera_id == "left") {
                    std::lock_guard<std::mutex> lock(mutex_left_);
                    count_left_++;
                } else if (camera_id == "front") {
                    std::lock_guard<std::mutex> lock(mutex_front_);
                    count_front_++;
                } else if (camera_id == "right") {
                    std::lock_guard<std::mutex> lock(mutex_right_);
                    count_right_++;
                }
            };

        // Suscribirse a los tres tópicos referentes a resultados
        sub_left_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/left/instance_info", 5, // Assuming original topic names
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg){
                callback(msg, "left");
            }
        );

        sub_front_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/front/instance_info", 5, // Assuming original topic names
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg){
                callback(msg, "front");
            }
        );

        sub_right_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/right/instance_info", 5, // Assuming original topic names
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg){
                callback(msg, "right");
            }
        );

        // Timer que cada 5 segundos reporta la frecuencia
        report_timer_ = this->create_wall_timer(
            5s, std::bind(&FrequencySubscriber::report_frequency, this));

        last_report_time_ = this->now();
    }

private:
    void report_frequency()
    {
        auto now = this->now();
        double elapsed_sec = (now - last_report_time_).seconds();

        uint64_t left = 0, front = 0, right = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_left_);
            left = count_left_;
            count_left_ = 0;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_front_);
            front = count_front_;
            count_front_ = 0;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_right_);
            right = count_right_;
            count_right_ = 0;
        }

        double freq_left = left / elapsed_sec;
        double freq_front = front / elapsed_sec;
        double freq_right = right / elapsed_sec;

        RCLCPP_INFO(this->get_logger(),
                    "Frequencies over last %.1f sec -> Left: %.2f Hz, Front: %.2f Hz, Right: %.2f Hz",
                    elapsed_sec, freq_left, freq_front, freq_right);

        last_report_time_ = now;
    }

    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_left_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_front_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_right_;
    rclcpp::TimerBase::SharedPtr report_timer_;

    rclcpp::Time last_report_time_;

    std::mutex mutex_left_;
    std::mutex mutex_front_;
    std::mutex mutex_right_;
    uint64_t count_left_ = 0;
    uint64_t count_front_ = 0;
    uint64_t count_right_ = 0;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FrequencySubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
