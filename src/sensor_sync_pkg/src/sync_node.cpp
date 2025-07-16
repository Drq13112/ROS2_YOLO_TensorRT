#include <memory>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <iomanip>
#include <sstream>

using std::placeholders::_1;

class SensorSyncNode : public rclcpp::Node
{
public:
    SensorSyncNode()
    : Node("sensor_sync_node")
    {
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rubyplus_points", 10,
            std::bind(&SensorSyncNode::lidar_callback, this, _1));
        cam_front_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera_front/image_raw", 10,
            std::bind(&SensorSyncNode::cam_front_callback, this, _1));
        cam_left_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera_front_left/image_raw", 10,
            std::bind(&SensorSyncNode::cam_left_callback, this, _1));
        cam_right_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera_front_right/image_raw", 10,
            std::bind(&SensorSyncNode::cam_right_callback, this, _1));
    }

private:
    // Guardar los últimos timestamps procesados para evitar repeticiones
    uint64_t last_lidar_ns_ = 0;
    uint64_t last_front_ns_ = 0;
    uint64_t last_left_ns_ = 0;
    uint64_t last_right_ns_ = 0;

    sensor_msgs::msg::PointCloud2::SharedPtr last_lidar_;
    sensor_msgs::msg::Image::SharedPtr last_cam_front_;
    sensor_msgs::msg::Image::SharedPtr last_cam_left_;
    sensor_msgs::msg::Image::SharedPtr last_cam_right_;

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        last_lidar_ = msg;
        // sync_data();
    }

    void cam_front_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        last_cam_front_ = msg;
        // sync_data();
    }

    void cam_left_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        last_cam_left_ = msg;
        // sync_data();
    }

    void cam_right_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        last_cam_right_ = msg;
        sync_data();
    }

    void sync_data()
    {
        if (last_lidar_ && last_cam_front_ && last_cam_left_ && last_cam_right_) {
            uint64_t lidar_ns  = static_cast<uint64_t>(last_lidar_->header.stamp.sec) * 1e9 + last_lidar_->header.stamp.nanosec;
            uint64_t front_ns  = static_cast<uint64_t>(last_cam_front_->header.stamp.sec) * 1e9 + last_cam_front_->header.stamp.nanosec;
            uint64_t left_ns   = static_cast<uint64_t>(last_cam_left_->header.stamp.sec) * 1e9 + last_cam_left_->header.stamp.nanosec;
            uint64_t right_ns  = static_cast<uint64_t>(last_cam_right_->header.stamp.sec) * 1e9 + last_cam_right_->header.stamp.nanosec;

            // Solo imprime si hay un cambio en cualquiera de los timestamps
            // if (lidar_ns == last_lidar_ns_ && front_ns == last_front_ns_ &&
            //     left_ns == last_left_ns_ && right_ns == last_right_ns_) {
            //     return;
            // }
            // last_lidar_ns_ = lidar_ns;
            // last_front_ns_ = front_ns;
            // last_left_ns_ = left_ns;
            // last_right_ns_ = right_ns;

            // Diferencias en milisegundos
            double diff_lidar_front = (double)(lidar_ns - front_ns) / 1e6;
            double diff_lidar_left  = (double)(lidar_ns - left_ns)  / 1e6;
            double diff_lidar_right = (double)(lidar_ns - right_ns) / 1e6;
            double diff_front_left  = (double)(front_ns - left_ns)  / 1e6;
            double diff_front_right = (double)(front_ns - right_ns) / 1e6;
            double diff_left_right  = (double)(left_ns - right_ns)  / 1e6;

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "\n================= Sensor Sync =================\n";
            oss << "Lidar timestamp:      " << lidar_ns  << " ns\n";
            oss << "Camera Front timestamp: " << front_ns  << " ns\n";
            oss << "Camera Left timestamp:  " << left_ns   << " ns\n";
            oss << "Camera Right timestamp: " << right_ns  << " ns\n";
            oss << "------------------------------------------------\n";
            oss << "Diff Lidar-Front:     " << diff_lidar_front << " ms\n";
            oss << "Diff Lidar-Left:      " << diff_lidar_left  << " ms\n";
            oss << "Diff Lidar-Right:     " << diff_lidar_right << " ms\n";
            oss << "Diff Front-Left:      " << diff_front_left  << " ms\n";
            oss << "Diff Front-Right:     " << diff_front_right << " ms\n";
            // oss << "Diff Left-Right:      " << diff_left_right  << " ms\n";
            oss << "================================================\n";

            RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_right_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SensorSyncNode>());
    rclcpp::shutdown();
    return 0;
}