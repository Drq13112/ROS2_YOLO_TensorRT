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
        // Suscribirse a los tres tópicos referentes a resultados
        sub_left_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            // "/segmentation/left/instance_info", 5,
            "/segmentation/instance_info_1", 5,
            [this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                (void)msg;
                std::lock_guard<std::mutex> lock(mutex_left_);
                count_left_++;
            }
        );

        sub_front_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            // "/segmentation/front/instance_info", 5,
            "/segmentation/instance_info_2", 5,
            [this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                (void)msg;
                std::lock_guard<std::mutex> lock(mutex_front_);
                count_front_++;
            }
        );

        sub_right_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            // "/segmentation/right/instance_info", 5,
            "/segmentation/instance_info_3", 5,
            [this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                (void)msg;
                std::lock_guard<std::mutex> lock(mutex_right_);
                count_right_++;
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
