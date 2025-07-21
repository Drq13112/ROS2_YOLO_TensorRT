#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>   // C++17
#include "onnx_files/seg/YOLO11Seg.hpp" // Uncomment for YOLOv11

namespace fs = std::filesystem;

class YoloInferenceNode : public rclcpp::Node
{
  public:
  explicit YoloInferenceNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : Node("yolo_inference_node", options)
  {
    declare_parameter<std::string>("model_path",  "");
    declare_parameter<std::string>("labels_path", "");
    declare_parameter<std::string>("video_path",  "");
    declare_parameter<std::string>("output_dir", "/tmp");
    declare_parameter<bool>("use_gpu", true);

    get_parameter("model_path",  model_path_);
    get_parameter("labels_path", labels_path_);
    get_parameter("video_path",  video_path_);
    get_parameter("use_gpu",     isGPU_);
    get_parameter("output_dir",  output_dir_);

    RCLCPP_INFO(get_logger(), "Starting YoloInferenceNode");
    RCLCPP_INFO(get_logger(), "  model_path : %s", model_path_.c_str());
    RCLCPP_INFO(get_logger(), "  labels_path: %s", labels_path_.c_str());
    RCLCPP_INFO(get_logger(), "  video_path : %s", video_path_.c_str());
    RCLCPP_INFO(get_logger(), "  use_gpu    : %s", isGPU_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  output_dir : %s", output_dir_.c_str());

    if (!fs::exists(output_dir_)) {
      fs::create_directories(output_dir_);
      RCLCPP_INFO(get_logger(), "Created output directory.");
    }

    if (model_path_.empty() || labels_path_.empty() || video_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "model_path, labels_path, or video_path not supplied!");
      throw std::runtime_error("Missing model, labels, or video path");
    }

    try {
      detector_ = std::make_unique<YOLOv11SegDetector>(
          model_path_, labels_path_, isGPU_);
      RCLCPP_INFO(get_logger(), "Detector initialized successfully.");
    } catch (const std::exception& e) {
      RCLCPP_FATAL(get_logger(), "Failed to create detector: %s", e.what());
      throw;
    }

    /* --- ROS interfaces ------------------------------------------------- */
    image_sub_ = create_subscription<sensor_msgs::msg::Image>("input_image", rclcpp::SensorDataQoS(), 
      std::bind(&YoloInferenceNode::imageCallback, 
      this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /input_image, waiting for data…");

    runInferenceLoop();
  }


  private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr /*msg*/)
  {
    RCLCPP_DEBUG(get_logger(), "Image received, reading test file…");
  }
  void runInferenceLoop()
  {
    cv::VideoCapture cap(video_path_);
    if (!cap.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open video: %s", video_path_.c_str());
      return;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps    = 10;

    std::string output_path = output_dir_ + "/output_segmented.avi";
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width, height));

    if (!writer.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open VideoWriter for: %s", output_path.c_str());
      return;
    }

    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
      if (frame.empty()) break;

      auto t0 = std::chrono::high_resolution_clock::now();
      auto results = detector_->segment(frame);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - t0).count();

      RCLCPP_INFO(get_logger(), "Frame %d: %zu masks in %lld ms",
                  frame_count, results.size(), static_cast<long long>(ms));

      cv::Mat vis = frame.clone();
      detector_->drawSegmentationsAndBoxes(vis, results);
      writer.write(vis);
      frame_count++;
    }

    RCLCPP_INFO(get_logger(), "Video processing complete. Output saved to: %s", output_path.c_str());
  }

  /* ---------- member variables ----------------------------------------- */
  std::string  model_path_, labels_path_, video_path_, output_dir_;
  bool         isGPU_{true};

  std::unique_ptr<YOLOv11SegDetector> detector_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};

  /* ---------- main() --------------------------------------------------- */

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloInferenceNode>());
    rclcpp::shutdown();
    return 0;
}
