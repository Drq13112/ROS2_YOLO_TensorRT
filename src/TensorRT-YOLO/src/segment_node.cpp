/**
 * @file segment_node.cpp
 * @author laugh12321 (translated and adapted)
 * @brief Segment C++ example adapted for ROS 2
 * @date 2025-01-23
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <chrono> // Added for timing

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>

#include "deploy/model.hpp"
#include "deploy/option.hpp"
#include "deploy/result.hpp"

#include <NvInferPlugin.h> // Required for initLibNvInferPlugins
#include <NvInferPluginUtils.h>

#include <plugin/efficientIdxNMSPlugin/efficientIdxNMSPlugin.h> // Required for EfficientIdxNMSPlugin
#include <plugin/efficientIdxNMSPlugin/efficientIdxNMSParameters.h>
#include <plugin/efficientIdxNMSPlugin/efficientIdxNMSInference.h>

#include <plugin/efficientRotatedNMSPlugin/efficientRotatedNMSPlugin.h>
#include <plugin/efficientRotatedNMSPlugin/efficientRotatedNMSParameters.h>
#include <plugin/efficientRotatedNMSPlugin/efficientRotatedNMSInference.h>

#include <opencv2/videoio.hpp> // For cv::VideoCapture and cv::VideoWriter
#include "tensorrt_yolo/msg/segmentation_output.hpp" 
namespace fs = std::filesystem;

class SegmentNode : public rclcpp::Node
{
public:
  SegmentNode() : Node("segment_node")
  {
    // Declare parameters with default empty strings
    this->declare_parameter<std::string>("engine_path", "");
    this->declare_parameter<std::string>("input_path", "");
    this->declare_parameter<std::string>("output_path", "");
    this->declare_parameter<std::string>("label_path", "");

    // Get parameters
    this->get_parameter("engine_path", engine_path_);
    this->get_parameter("input_path", input_path_);
    this->get_parameter("output_path", output_path_);
    this->get_parameter("label_path", label_path_);
    RCLCPP_INFO(this->get_logger(), "Engine path: %s", engine_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Input path: %s", input_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Output path: %s", output_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Label path: %s", label_path_.c_str());


    if (engine_path_.empty() || input_path_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Parameters 'engine_path' and 'input_path' must be provided.");
      rclcpp::shutdown();
      return;
    }

    try {
      if (!fs::exists(engine_path_)) {
        throw std::runtime_error("Engine path does not exist: " + engine_path_);
      }
      if (!fs::exists(input_path_) || (!fs::is_regular_file(input_path_) && !fs::is_directory(input_path_))) {
        throw std::runtime_error("Input path does not exist or is not a regular file/directory: " + input_path_);
      }

      if (!output_path_.empty()) {
        if (label_path_.empty()) {
          throw std::runtime_error("Please provide a labels file using 'label_path' parameter.");
        }
        if (!fs::exists(label_path_)) {
          throw std::runtime_error("Label path does not exist: " + label_path_);
        }
        labels_ = generate_labels(label_path_);
        create_output_directory(output_path_);
      }

      deploy::InferOption option;
      option.enableSwapRB();

      if (!fs::is_regular_file(input_path_)) {
        option.enablePerformanceReport();
      }

      RCLCPP_INFO(this->get_logger(), "Loading model...");
      model_ = std::make_unique<deploy::SegmentModel>(engine_path_, option);
      if (!model_) {
        throw std::runtime_error("Failed to load model from path: " + engine_path_);
      }
      RCLCPP_INFO(this->get_logger(), "Model loaded successfully.");

      if (fs::is_regular_file(input_path_)) {
        RCLCPP_INFO(this->get_logger(), "Processing single image...");
        process_single_image(input_path_, output_path_, *model_, labels_);
      } else {
        auto image_files = get_images_in_directory(input_path_);
        if (image_files.empty()) {
          throw std::runtime_error("Failed to read images from path: " + input_path_);
        }
        RCLCPP_INFO(this->get_logger(), "Processing batch images...");
        process_batch_images(image_files, output_path_, *model_, labels_);
      }

      RCLCPP_INFO(this->get_logger(), "Inference completed.");

      if (option.enable_performance_report) {
        auto [throughput_str, gpu_latency_str, cpu_latency_str] = model_->performanceReport();
        RCLCPP_INFO(this->get_logger(), "%s", throughput_str.c_str());
        RCLCPP_INFO(this->get_logger(), "%s", gpu_latency_str.c_str());
        RCLCPP_INFO(this->get_logger(), "%s", cpu_latency_str.c_str());
      }

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
      rclcpp::shutdown();
    }
  }

private:
  std::string engine_path_;
  std::string input_path_;
  std::string output_path_;
  std::string label_path_;
  std::vector<std::string> labels_;
  std::unique_ptr<deploy::SegmentModel> model_;
  rclcpp::Publisher<tensorrt_yolo::msg::SegmentationOutput>::SharedPtr segmentation_pub_;

  // Get image files in directory
  std::vector<std::string> get_images_in_directory(const std::string& folder_path)
  {
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
      const auto extension = entry.path().extension().string();
      if (fs::is_regular_file(entry) &&
          (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp")) {
        image_files.push_back(entry.path().string());
      }
    }
    return image_files;
  }

  // Create output directory if not exists
  void create_output_directory(const std::string& output_path)
  {
    if (!fs::exists(output_path) && !fs::create_directories(output_path)) {
      throw std::runtime_error("Failed to create output directory: " + output_path);
    } else if (!fs::is_directory(output_path)) {
      throw std::runtime_error("Output path exists but is not a directory: " + output_path);
    }
  }

  // Generate labels from file
  std::vector<std::string> generate_labels(const std::string& label_file)
  {
    std::ifstream file(label_file);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open labels file: " + label_file);
    }

    std::vector<std::string> labels;
    std::string label;
    while (std::getline(file, label)) {
      labels.emplace_back(label);
    }
    return labels;
  }

  // Visualize inference results (segmentation task)
  void visualize(cv::Mat& image, deploy::SegmentRes& result, const std::vector<std::string>& labels)
  {
    for (size_t i = 0; i < result.num; ++i) {

      auto& box = result.boxes[i];
      int cls = result.classes[i];
      float score = result.scores[i];
      std::string label_str = (cls >= 0 && cls < labels.size()) ? labels[cls] : "CLS_ERR";
      std::string label_text = label_str + " " + cv::format("%.2f", score);

      int base_line;
      cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
      cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
      cv::rectangle(image, cv::Point(box.left, box.top - label_size.height - base_line), cv::Point(box.left + label_size.width, box.top), cv::Scalar(0, 255, 0), -1);
      cv::putText(image, label_text, cv::Point(box.left, box.top - base_line), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

      if (result.masks[i].data.empty()) continue;
      cv::Mat mask_cv(result.masks[i].height, result.masks[i].width, CV_8UC1, result.masks[i].data.data());
      
      // Resize mask to fit the bounding box, then place it on a full-size empty mask
      cv::Mat resized_obj_mask;
      cv::resize(mask_cv, resized_obj_mask, cv::Size(box.right - box.left, box.bottom - box.top));

      cv::Mat full_mask = cv::Mat::zeros(image.size(), CV_8UC1);
      if (box.left >= 0 && box.top >=0 && (box.left + resized_obj_mask.cols) <= image.cols && (box.top + resized_obj_mask.rows) <= image.rows) {
        resized_obj_mask.copyTo(full_mask(cv::Rect(box.left, box.top, resized_obj_mask.cols, resized_obj_mask.rows)));
      }

      // Apply color to mask
      cv::Mat colored_mask_region = image.clone();
      colored_mask_region.setTo(cv::Scalar(0,0,200), full_mask);
      cv::addWeighted(image, 0.7, colored_mask_region, 0.3, 0, image);

    }
  }

  void process_video_file(const std::string& video_path, const std::string& output_video_path,
                          deploy::SegmentModel& model, const std::vector<std::string>& labels)
  {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      throw std::runtime_error("Failed to open video: " + video_path);
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // Or use cap.get(cv::CAP_PROP_FOURCC)

    cv::VideoWriter video_writer;
    if (!output_video_path.empty()) {
      video_writer.open(output_video_path, fourcc, fps, cv::Size(frame_width, frame_height)); // Ensure output size matches processed frame
      if (!video_writer.isOpened()) {
         RCLCPP_WARN(this->get_logger(), "Failed to open VideoWriter for: %s. Output video will not be saved.", output_video_path.c_str());
      }
    }

    cv::Mat frame;
    int frame_idx = 0;
    while (rclcpp::ok()) {
      cap >> frame;
      if (frame.empty()) {
        break;
      }
      
      cv::Mat processed_frame = frame.clone(); // process_frame will modify this
      process_frame(processed_frame, frame_idx++, "", "", model, labels, &video_writer); // No individual file save for video frames

      // Display frame (optional)
      // cv::imshow("Processed Frame", processed_frame);
      // if (cv::waitKey(1) == 27) break; // ESC to quit
    }
    cap.release();
    if (video_writer.isOpened()) {
      video_writer.release();
    }
    // cv::destroyAllWindows();
    RCLCPP_INFO(this->get_logger(), "Video processing finished.");
  } 


  void process_frame(cv::Mat& image_to_process, int frame_idx,
                     const std::string& output_dir, const std::string& output_filename_base,
                     deploy::SegmentModel& model, const std::vector<std::string>& labels,
                     cv::VideoWriter* video_writer = nullptr)
  {
    auto total_frame_start_time = std::chrono::high_resolution_clock::now();

    // --- CPU Pre-processing (Resize) ---
    auto cpu_preproc_start_time = std::chrono::high_resolution_clock::now();
    const int expected_height = 640; // Or get from model if possible
    const int expected_width = 640;
    cv::Size expected_size(expected_width, expected_height);
    cv::Mat resized_image;
    if (image_to_process.cols != expected_width || image_to_process.rows != expected_height) {
      cv::resize(image_to_process, resized_image, expected_size);
    } else {
      resized_image = image_to_process.clone();
    }
    auto cpu_preproc_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_preproc_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_preproc_end_time - cpu_preproc_start_time);
    RCLCPP_INFO(this->get_logger(), "[Frame %d] CPU Preprocessing (resize): %.3f ms", frame_idx, cpu_preproc_duration.count() / 1000.0);

    deploy::Image img_for_model(resized_image.data, resized_image.cols, resized_image.rows);

    // --- Inference ---
    auto inference_start_time = std::chrono::high_resolution_clock::now();
    // Note: model.predict() includes GPU preproc, inference, and GPU postproc (NMS etc.)
    deploy::SegmentRes result = model.predict(img_for_model);
    auto inference_end_time = std::chrono::high_resolution_clock::now();
    auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time);
    RCLCPP_INFO(this->get_logger(), "[Frame %d] Model Predict (incl. GPU pre/post): %.3f ms", frame_idx, inference_duration.count() / 1000.0);

    // --- Post-processing (Generate Arrays for ROS Message) ---
    auto array_gen_start_time = std::chrono::high_resolution_clock::now();
    
    tensorrt_yolo::msg::SegmentationOutput seg_msg;
    seg_msg.header.stamp = this->get_clock()->now();
    seg_msg.header.frame_id = "segmentation_output"; // Or your desired frame_id
    seg_msg.image_height = resized_image.rows;
    seg_msg.image_width = resized_image.cols;

    seg_msg.class_id_map.assign(resized_image.rows * resized_image.cols, 0); // 0 for background
    seg_msg.instance_id_map.assign(resized_image.rows * resized_image.cols, 0); // 0 for background

    int valid_instance_count = 0;
    for (size_t i = 0; i < result.num; ++i) {
      valid_instance_count++;
    }

    seg_msg.instance_confidences.reserve(valid_instance_count);
    seg_msg.instance_class_ids.reserve(valid_instance_count);
    seg_msg.detected_instance_ids.reserve(valid_instance_count);

    int current_instance_id = 1; // Start instance IDs from 1
    for (size_t i = 0; i < result.num; ++i) {
      // if (result.scores[i] < conf_threshold_) continue;

      seg_msg.instance_confidences.push_back(result.scores[i]);
      seg_msg.instance_class_ids.push_back(result.classes[i]);
      seg_msg.detected_instance_ids.push_back(current_instance_id);

      if (result.masks[i].data.empty()) {
          current_instance_id++;
          continue;
      }

      cv::Mat raw_mask_cv(result.masks[i].height, result.masks[i].width, CV_8UC1, result.masks[i].data.data());
      cv::Mat resized_mask_for_map;
      // Resize the raw mask to the model input size (e.g., 640x640) before applying to map
      cv::resize(raw_mask_cv, resized_mask_for_map, cv::Size(resized_image.cols, resized_image.rows)); 

      for (int r = 0; r < resized_image.rows; ++r) {
        for (int c = 0; c < resized_image.cols; ++c) {
          if (resized_mask_for_map.at<uchar>(r, c) > 0) { // Assuming mask values > 0 mean foreground
            // Simple overwrite strategy for overlapping masks
            seg_msg.class_id_map[r * resized_image.cols + c] = result.classes[i];
            seg_msg.instance_id_map[r * resized_image.cols + c] = current_instance_id;
          }
        }
      }
      current_instance_id++;
    }
    auto array_gen_end_time = std::chrono::high_resolution_clock::now();
    auto array_gen_duration = std::chrono::duration_cast<std::chrono::microseconds>(array_gen_end_time - array_gen_start_time);
    RCLCPP_INFO(this->get_logger(), "[Frame %d] Postprocessing (Array Gen for ROS): %.3f ms", frame_idx, array_gen_duration.count() / 1000.0);

    // --- Publish ROS Message ---
    auto publish_start_time = std::chrono::high_resolution_clock::now();
    segmentation_pub_->publish(seg_msg);
    auto publish_end_time = std::chrono::high_resolution_clock::now();
    auto publish_duration = std::chrono::duration_cast<std::chrono::microseconds>(publish_end_time - publish_start_time);
    RCLCPP_INFO(this->get_logger(), "[Frame %d] ROS Publish Time: %.3f ms", frame_idx, publish_duration.count() / 1000.0);

    // --- Visualization (modifies image_to_process) ---
    // The visualize function should now use image_to_process (original frame size)
    // and the result which contains boxes relative to resized_image.
    // We need to scale boxes back or ensure visualize handles this.
    // For simplicity, let's visualize on resized_image and then resize it back if needed, or adapt visualize.
    // The current visualize function in the repo draws on the image passed to it.
    // Let's assume we visualize on `image_to_process` after scaling results, or visualize on `resized_image` and copy.
    
    // Create a display copy from the *original* frame dimensions if `image_to_process` was resized.
    // Or, if `image_to_process` is already the target size (640x640), use it directly.
    // The `visualize` function in the original code takes the `resized_image`.
    // We will draw on `resized_image` and if `video_writer` is active, resize `resized_image` back to original video dimensions.
    
    cv::Mat display_image = resized_image.clone(); // Visualize on the model input sized image
    visualize(display_image, result, labels_);


    if (video_writer && video_writer->isOpened()) {
        cv::Mat output_frame_for_video;
        // Resize `display_image` (which is 640x640 with annotations) back to original video dimensions
        cv::resize(display_image, output_frame_for_video, image_to_process.size());
        video_writer->write(output_frame_for_video);
    } else if (!output_dir.empty() && !output_filename_base.empty()) {
        fs::path output_file_path = fs::path(output_dir) / fs::path(output_filename_base);
        cv::imwrite(output_file_path.string(), display_image); // Save the 640x640 annotated image
        RCLCPP_INFO(this->get_logger(), "[Frame %d] Saved output image to: %s", frame_idx, output_file_path.string().c_str());
    }
    
    auto total_frame_end_time = std::chrono::high_resolution_clock::now();
    auto total_frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_frame_end_time - total_frame_start_time);
    RCLCPP_INFO(this->get_logger(), "[Frame %d] Total processing time: %lld ms", frame_idx, total_frame_duration.count());
  }


  // Process a single image
  void process_single_image(const std::string& image_path, const std::string& output_path, deploy::SegmentModel& model, const std::vector<std::string>& labels)
  {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image from path: " + image_path);
    }

    // Define expected model input dimensions (Height, Width)
    const int expected_height = 640;
    const int expected_width = 640;
    cv::Size expected_size(expected_width, expected_height);

    cv::Mat resized_image;
    if (image.cols != expected_width || image.rows != expected_height) {
      RCLCPP_INFO(this->get_logger(), "Original image size: %d x %d. Resizing to %d x %d.", image.cols, image.rows, expected_width, expected_height);
      cv::resize(image, resized_image, expected_size);
    } else {
      resized_image = image.clone(); // Use a clone if no resize to ensure visualize modifies a copy if needed
    }

    RCLCPP_INFO(this->get_logger(), "Processing image: %s. Input to model size: %d x %d", image_path.c_str(), resized_image.cols, resized_image.rows);
    
    deploy::Image img(resized_image.data, resized_image.cols, resized_image.rows);
    deploy::SegmentRes result; // Declare result outside the loop to store the last one

    const int num_iterations = 100;
    RCLCPP_INFO(this->get_logger(), "Starting %d inference iterations...", num_iterations);

    for (int i = 0; i < num_iterations; ++i) {
      // Start timing inference
      auto start_time = std::chrono::high_resolution_clock::now();

      result = model.predict(img); // Store the latest result

      visualize(resized_image, result, labels);

      // Stop timing inference and calculate duration
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      RCLCPP_INFO(this->get_logger(), "Iteration %d/%d: Inference time: %lld ms", i + 1, num_iterations, duration.count());
    }

    if (!output_path.empty()) {
      // Visualize and save the result of the last inference
      visualize(resized_image, result, labels);
      fs::path output_file_path = fs::path(output_path) / fs::path(image_path).filename();
      cv::imwrite(output_file_path.string(), resized_image);
      RCLCPP_INFO(this->get_logger(), "Saved output image (from last iteration) to: %s", output_file_path.string().c_str());
    }
  }

  // Process batch images
  void process_batch_images(const std::vector<std::string>& image_paths, const std::string& output_path, deploy::SegmentModel& model, const std::vector<std::string>& labels)
  {
    const int batch_size = model.batch_size();
    for (size_t i = 0; i < image_paths.size(); i += batch_size) {
      std::vector<cv::Mat> images;
      std::vector<deploy::Image> img_batch;
      std::vector<std::string> img_name_batch;

      for (size_t j = i; j < i + batch_size && j < image_paths.size(); ++j) {
        cv::Mat image = cv::imread(image_paths[j], cv::IMREAD_COLOR);
        if (image.empty()) {
          throw std::runtime_error("Failed to read image from path: " + image_paths[j]);
        }
        images.push_back(image);
        img_batch.emplace_back(image.data, image.cols, image.rows);
        img_name_batch.push_back(fs::path(image_paths[j]).filename().string());
      }

      auto results = model.predict(img_batch);

      if (!output_path.empty()) {
        for (size_t j = 0; j < images.size(); ++j) {
          visualize(images[j], results[j], labels);
          fs::path output_file_path = fs::path(output_path) / img_name_batch[j];
          cv::imwrite(output_file_path.string(), images[j]);
        }
      }
    }
  }
};

int main(int argc, char** argv)
{
  // Initialize TensorRT plugins.
  // The first argument is an optional logger (nullptr for default), second is namespace (empty for global).
  if (!initLibNvInferPlugins(nullptr, "")) {
      // This function often returns true, but errors during plugin loading might occur later.
      // Log a warning, but proceed, as the actual error will be caught during engine deserialization.
      fprintf(stderr, "Warning: initLibNvInferPlugins call completed. This does not guarantee all plugins are available.\n");
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SegmentNode>();
  // auto node = std::make_shared<SegmentNode>();
  // rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}