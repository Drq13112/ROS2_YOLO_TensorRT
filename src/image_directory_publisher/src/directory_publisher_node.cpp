#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp> // Para Publisher
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <filesystem>
#include <memory>
#include <algorithm> // Para std::sort y std::min
#include <vector>    // Para std::vector
#include <chrono>    // Para la medición de tiempo
#include <unordered_set> // Para el filtrado eficiente de archivos
#include <time.h> 
#include <string>

namespace image_directory_publisher {

class DirectoryPublisherNode : public rclcpp::Node {
public:
    DirectoryPublisherNode(const rclcpp::NodeOptions & options)
    : Node("directory_publisher_node", options), current_image_set_index_(0)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing DirectoryPublisherNode (Image Only)...");

        this->declare_parameter<std::string>("image_directory", "");
        this->get_parameter("image_directory", image_directory_);
        this->declare_parameter<std::string>("left_image_pattern", "left_*.png");
        this->get_parameter("left_image_pattern", left_image_pattern_);
        this->declare_parameter<std::string>("front_image_pattern", "front_*.png");
        this->get_parameter("front_image_pattern", front_image_pattern_);
        this->declare_parameter<std::string>("right_image_pattern", "right_*.png");
        this->get_parameter("right_image_pattern", right_image_pattern_);
        this->declare_parameter<double>("publish_rate", 10.0); // Hz
        double publish_rate = this->get_parameter("publish_rate").get_value<double>();
        this->declare_parameter<bool>("loop_playback", true);
        this->get_parameter("loop_playback", loop_playback_);
        this->declare_parameter<std::string>("frame_id_left", "camera_left_link");
        this->get_parameter("frame_id_left", frame_id_left_);
        this->declare_parameter<std::string>("frame_id_front", "camera_front_link");
        this->get_parameter("frame_id_front", frame_id_front_);
        this->declare_parameter<std::string>("frame_id_right", "camera_right_link");
        this->get_parameter("frame_id_right", frame_id_right_);


        RCLCPP_INFO(this->get_logger(), "Image directory: %s", image_directory_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publish rate: %.2f Hz", publish_rate);
        RCLCPP_INFO(this->get_logger(), "Loop playback: %s", loop_playback_ ? "true" : "false");

        populate_file_lists();

        if (left_image_files_.empty() && front_image_files_.empty() && right_image_files_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No image files found in the specified directory: %s. Shutting down.", image_directory_.c_str());
            rclcpp::shutdown();
            return;
        }

       
        // Crear un perfil QoS personalizado
        rclcpp::QoS custom_qos_profile(rclcpp::KeepLast(1)); // History: KEEP_LAST, Depth: 1
        custom_qos_profile.reliable();                      // Reliability: RELIABLE
        custom_qos_profile.durability_volatile();           // Durability: VOLATILE (implícito si no se especifica TRANSIENT_LOCAL)
                                                            // Para ser explícito:
                                                            // custom_qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);


        left_publisher_ = image_transport::create_publisher(this, "/camera_front_left/image_raw", custom_qos_profile.get_rmw_qos_profile());
        front_publisher_ = image_transport::create_publisher(this, "/camera_front/image_raw", custom_qos_profile.get_rmw_qos_profile());
        right_publisher_ = image_transport::create_publisher(this, "/camera_front_right/image_raw", custom_qos_profile.get_rmw_qos_profile());
        
        RCLCPP_INFO(this->get_logger(), "Publishing to /camera_front_left/image_raw, /camera_front/image_raw, /camera_front_right/image_raw");

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
            std::bind(&DirectoryPublisherNode::publish_all_images, this));
        
        RCLCPP_INFO(this->get_logger(), "DirectoryPublisherNode initialized successfully.");
    }

private:
    void populate_file_lists() {
        std::filesystem::path dir_path(image_directory_);
        if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
            RCLCPP_ERROR(this->get_logger(), "Image directory does not exist or is not a directory: %s", image_directory_.c_str());
            return;
        }

        auto collect_files = [&](const std::string& pattern, std::vector<std::string>& file_list, const std::string& camera_name) {
            // Simple pattern matching: assumes pattern is like "prefix_*.suffix"
            std::string prefix = pattern.substr(0, pattern.find('*'));
            std::string suffix = pattern.substr(pattern.find('*') + 1);
            std::unordered_set<std::string> allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"};


            for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    std::string extension = entry.path().extension().string();
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);


                    if (filename.rfind(prefix, 0) == 0 && // starts with prefix
                        filename.length() >= prefix.length() + suffix.length() && // long enough for suffix
                        filename.compare(filename.length() - suffix.length(), suffix.length(), suffix) == 0 && // ends with suffix
                        allowed_extensions.count(extension)) {
                        file_list.push_back(entry.path().string());
                    }
                }
            }
            std::sort(file_list.begin(), file_list.end()); // Sort files alphabetically (often corresponds to time)
            RCLCPP_INFO(this->get_logger(), "Found %zu images for %s camera with pattern '%s'", file_list.size(), camera_name.c_str(), pattern.c_str());
        };

        collect_files(left_image_pattern_, left_image_files_, "LEFT");
        collect_files(front_image_pattern_, front_image_files_, "FRONT");
        collect_files(right_image_pattern_, right_image_files_, "RIGHT");
    }

    void publish_image_set(
        const std::string& image_path,
        image_transport::Publisher& publisher,
        const std::string& packet_sequence_frame_id,
        const std::string& camera_name_log)
    {
        cv::Mat image_to_publish = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image_to_publish.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to read image: %s for %s camera", image_path.c_str(), camera_name_log.c_str());
            return;
        }

        std_msgs::msg::Header header;
        
        header.frame_id = std::to_string(current_image_set_index_);

        // Obtener el tiempo MONOTONIC actual y asignarlo al header
        timespec ts_monotonic_capture;
        clock_gettime(CLOCK_MONOTONIC, &ts_monotonic_capture);
        header.stamp.sec = ts_monotonic_capture.tv_sec;
        header.stamp.nanosec = ts_monotonic_capture.tv_nsec;

        // Convertir cv::Mat a sensor_msgs::msg::Image usando cv_bridge
        // Asumimos que la imagen es BGR8.
        auto msg_ptr = cv_bridge::CvImage(header, "bgr8", image_to_publish).toImageMsg();
    
        publisher.publish(std::move(msg_ptr)); // Publicar el mensaje único

        RCLCPP_INFO(this->get_logger(), "Published %s image: %s with MONOTONIC TS: %ld.%09ld",
                    camera_name_log.c_str(), image_path.c_str(), header.stamp.sec, header.stamp.nanosec);
    }

    void publish_all_images() {
        if (left_image_files_.empty() && front_image_files_.empty() && right_image_files_.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "No image files found to publish.");
            return;
        }

        size_t max_files = 0;
        if (!left_image_files_.empty()) max_files = std::max(max_files, left_image_files_.size());
        if (!front_image_files_.empty()) max_files = std::max(max_files, front_image_files_.size());
        if (!right_image_files_.empty()) max_files = std::max(max_files, right_image_files_.size());

        if (current_image_set_index_ >= max_files) {
            if (loop_playback_) {
                current_image_set_index_ = 0;
                RCLCPP_INFO(this->get_logger(), "Looping playback, restarting from the beginning.");
            } else {
                RCLCPP_INFO(this->get_logger(), "Finished publishing all images. Stopping timer.");
                if (timer_) { // Verificar si el timer existe antes de cancelarlo
                    timer_->cancel();
                }
                return;
            }
        }

        if (current_image_set_index_ < 1500) {
        

            if (!left_image_files_.empty() && current_image_set_index_ < left_image_files_.size()) {
                publish_image_set(left_image_files_[current_image_set_index_], left_publisher_, frame_id_left_, /* now, */ "LEFT");
                published_packet_count_1++;
            }
            if (!front_image_files_.empty() && current_image_set_index_ < front_image_files_.size()) {
                publish_image_set(front_image_files_[current_image_set_index_], front_publisher_, frame_id_front_, /* now, */ "FRONT");
                published_packet_count_2++;
            }
            if (!right_image_files_.empty() && current_image_set_index_ < right_image_files_.size()) {
                publish_image_set(right_image_files_[current_image_set_index_], right_publisher_, frame_id_right_, /* now, */ "RIGHT");
                published_packet_count_3++;
            }

            current_image_set_index_++;}
            RCLCPP_INFO(this->get_logger(), "Total image sets (packet 1) sent: %u", published_packet_count_1);
            RCLCPP_INFO(this->get_logger(), "Total image sets (packet 2) sent: %u", published_packet_count_2);
            RCLCPP_INFO(this->get_logger(), "Total image sets (packet 3) sent: %u", published_packet_count_3);

        
        
    }

    std::string image_directory_;
    std::string left_image_pattern_, front_image_pattern_, right_image_pattern_;
    bool loop_playback_;

    std::vector<std::string> left_image_files_;
    std::vector<std::string> front_image_files_;
    std::vector<std::string> right_image_files_;
    uint32_t current_image_set_index_ = 0;
    uint32_t published_packet_count_1 = 0;
    uint32_t published_packet_count_2 = 0;
    uint32_t published_packet_count_3 = 0;

    std::string frame_id_left_, frame_id_front_, frame_id_right_;

    image_transport::Publisher left_publisher_;
    image_transport::Publisher front_publisher_;
    image_transport::Publisher right_publisher_;

    rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace image_directory_publisher

RCLCPP_COMPONENTS_REGISTER_NODE(image_directory_publisher::DirectoryPublisherNode)