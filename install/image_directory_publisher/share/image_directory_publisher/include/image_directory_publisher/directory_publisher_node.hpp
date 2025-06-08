#ifndef IMAGE_DIRECTORY_PUBLISHER_DIRECTORY_PUBLISHER_NODE_HPP
#define IMAGE_DIRECTORY_PUBLISHER_DIRECTORY_PUBLISHER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace image_directory_publisher {

class DirectoryPublisherNode : public rclcpp::Node {
public:
    DirectoryPublisherNode(const rclcpp::NodeOptions & options);
    void publish_images();

private:
    void load_images();
    void publish_image(const std::string & image_name, const std::string & camera_info_path);

    std::string directory_path_;
    std::string frame_id_;
    image_transport::CameraPublisher left_image_publisher_;
    image_transport::CameraPublisher front_image_publisher_;
    image_transport::CameraPublisher right_image_publisher_;
    sensor_msgs::msg::CameraInfo left_camera_info_;
    sensor_msgs::msg::CameraInfo front_camera_info_;
    sensor_msgs::msg::CameraInfo right_camera_info_;
    cv::Mat left_image_;
    cv::Mat front_image_;
    cv::Mat right_image_;
};

} // namespace image_directory_publisher

#endif // IMAGE_DIRECTORY_PUBLISHER_DIRECTORY_PUBLISHER_NODE_HPP