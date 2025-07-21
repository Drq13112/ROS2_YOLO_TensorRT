#include "rclcpp/rclcpp.hpp"
#include "yolo_custom_interfaces/msg/instance_segmentation_info.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <fstream> // Para std::ofstream
#include <iomanip> // Para std::fixed, std::setprecision
#include <vector>  // Para std::vector (aunque no se usa directamente, es común)
#include <map>     // Para std::map
#include <string>  // Para std::string
#include <limits>  // Para std::numeric_limits
#include <cmath>   // Para std::sqrt (si se quisiera desviación estándar)
#include <filesystem>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>             // Para std::thread
#include <queue>              // Para std::queue
#include <condition_variable> // Para std::condition_variable
#include <atomic>             // Para std::atomic
#include <sstream>            // Para std::ostringstream
#include <utility>            // Para std::pair
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <opencv2/calib3d.hpp>
#include <tf2/utils.h> // Required for tf2::durationFromSec

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

static bool timespec_less_than(const timespec &a, const timespec &b)
{
    if (a.tv_sec != b.tv_sec)
        return a.tv_sec < b.tv_sec;
    return a.tv_nsec < b.tv_nsec;
}

class SingleProjectionNode : public rclcpp::Node
{
public:
    SingleProjectionNode() : Node("single_projection_node")
    {
        // --- Declare and get parameters ---
        this->declare_parameter<std::string>("camera_id", "default");
        this->declare_parameter<std::string>("image_topic", "/segmentation/default/instance_info");
        this->declare_parameter<std::string>("lidar_topic", "/lidar/default/points");
        this->declare_parameter<std::string>("camera_frame_id", "default_camera_optical_frame");

        camera_id_ = this->get_parameter("camera_id").as_string();
        auto image_topic = this->get_parameter("image_topic").as_string();
        auto lidar_topic = this->get_parameter("lidar_topic").as_string();
        camera_frame_id_ = this->get_parameter("camera_frame_id").as_string();

        RCLCPP_INFO(this->get_logger(), "Initializing Node for camera_id: '%s'", camera_id_.c_str());
        RCLCPP_INFO(this->get_logger(), " -> Subscribing to Image: '%s'", image_topic.c_str());
        RCLCPP_INFO(this->get_logger(), " -> Subscribing to LiDAR: '%s'", lidar_topic.c_str());

        // --- Common Initialization ---
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        double fx = 1500.0, fy = 1500.0, cx = 1920.0 / 2.0, cy = 1200.0 / 2.0;
        camera_intrinsics_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        distortion_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);

        // --- Subscriptions ---
        rclcpp::QoS qos_profile(rclcpp::KeepLast(1));
        qos_profile.reliable();
        qos_profile.durability_volatile();

        rclcpp::QoS qos_lidar(rclcpp::KeepLast(1));
        qos_lidar.best_effort();
        qos_lidar.durability_volatile();

        sub_image_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            image_topic, qos_profile,
            std::bind(&SingleProjectionNode::imageCallback, this, std::placeholders::_1));

        sub_rubyplus_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic, qos_lidar,
            std::bind(&SingleProjectionNode::lidarCallback, this, std::placeholders::_1));

        sub_helios = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic, qos_lidar,
            std::bind(&SingleProjectionNode::lidarCallback, this, std::placeholders::_1));

        // --- Processing Timer ---
        processing_timer_ = this->create_wall_timer(20ms, std::bind(&SingleProjectionNode::processData, this));
    }

private:
    // Structure to hold the final matched data
    struct PointMatch
    {
        cv::Point3f point_3d_lidar_frame;
        cv::Point2f pixel_2d;
        int instance_id;
        int class_id;
        float score;
    };

    void imageCallback(const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        last_seg_info_ = msg;
    }
    void lidar_Ruby_Callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        last_pc_ruby_ = msg;
    }
    void lidar_Helios_Callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        last_pc_helios_ = msg;
    }

    // --- Processing Callback from Timer (Consumer) ---
    void processData()
    {
        yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr seg_info;
        sensor_msgs::msg::PointCloud2::SharedPtr pc;

        // Atomically get data and reset pointers to prevent reprocessing
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (!last_seg_info_ || !last_pc_ruby_ || !last_pc_helios_)
            {
                return; // Not enough data to process
            }
            {
                return; // Not enough data to process
            }
            seg_info = last_seg_info_;
            pc = last_pc_ruby_;
            last_seg_info_ = nullptr;
            last_pc_ruby_ = nullptr;
            last_pc_helios_ = nullptr;
        }

        // Check timestamp difference
        if (std::abs((rclcpp::Time(seg_info->header.stamp) - rclcpp::Time(pc->header.stamp)).seconds()) > 0.1)
        {
            RCLCPP_WARN(this->get_logger(), "Skipping projection. Timestamp diff > 100ms.");
            return;
        }

        // --- Heavy lifting ---
        try
        {
            cv_bridge::CvImagePtr cv_ptr_mask = cv_bridge::toCvCopy(seg_info->mask, seg_info->mask.encoding);
            cv::Mat instance_id_mask = cv_ptr_mask->image;

            const cv::Size target_size(1920, 1200);
            if (instance_id_mask.size() != target_size)
            {
                cv::resize(instance_id_mask, instance_id_mask, target_size, 0, 0, cv::INTER_NEAREST);
            }

            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                camera_frame_id_, pc->header.frame_id, tf2_ros::fromMsg(seg_info->header.stamp), tf2::durationFromSec(0.05));

            projectAndMatch(pc, transform, camera_id_, instance_id_mask, seg_info->classes, seg_info->scores);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform %s to %s: %s",
                        pc->header.frame_id.c_str(), camera_frame_id_.c_str(), ex.what());
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    // Helper to convert geometry_msgs::Transform to OpenCV rvec and tvec
    void transformToCV(const geometry_msgs::msg::Transform &tf, cv::Vec3d &rvec, cv::Vec3d &tvec)
    {
        // Translation
        tvec[0] = tf.translation.x;
        tvec[1] = tf.translation.y;
        tvec[2] = tf.translation.z;

        // Rotation from quaternion to rotation vector (Rodrigues)
        tf2::Quaternion q(
            tf.rotation.x,
            tf.rotation.y,
            tf.rotation.z,
            tf.rotation.w);
        tf2::Matrix3x3 R_tf(q);
        cv::Mat R_cv = cv::Mat::eye(3, 3, CV_64F);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                R_cv.at<double>(i, j) = R_tf[i][j];
            }
        }
        cv::Rodrigues(R_cv, rvec);
    }

    // NEW: Main projection and matching function
    void projectAndMatch(
        const sensor_msgs::msg::PointCloud2::SharedPtr &pc_msg,
        const geometry_msgs::msg::TransformStamped &transform,
        const std::string &camera_id,
        const cv::Mat &instance_mask,
        const std::vector<int32_t> &class_ids,
        const std::vector<float> &scores)
    {
        // 1. Convert transform to rvec and tvec for OpenCV
        cv::Vec3d rvec, tvec;
        transformToCV(transform.transform, rvec, tvec);

        // 2. Extract 3D points from PointCloud2 message
        std::vector<cv::Point3f> object_points;
        sensor_msgs::PointCloud2Iterator<float> iter_x(*pc_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*pc_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*pc_msg, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            // A simple filter to avoid points that are too close measuring the distance from the origin
            if (std::sqrt((*iter_x) * (*iter_x) + (*iter_y) * (*iter_y) + (*iter_z) * (*iter_z)) > 1.0)
            {
                object_points.emplace_back(*iter_x, *iter_y, *iter_z);
            }
        }

        if (object_points.empty())
        {
            return;
        }

        // 3. Project all points at once using OpenCV
        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points, rvec, tvec, camera_intrinsics_, distortion_coeffs_, image_points);

        // 4. Match projected points with segmentation mask
        std::vector<PointMatch> matches;
        for (size_t i = 0; i < image_points.size(); ++i)
        {
            cv::Point2f p = image_points[i];

            // Check if the projected point is within the image boundaries
            if (p.x >= 0 && p.x < instance_mask.cols && p.y >= 0 && p.y < instance_mask.rows)
            {

                // Get the instance ID from the mask (e.g., CV_16UC1 for uint16_t IDs)
                // Make sure the mask type is correct.
                uint16_t instance_id = instance_mask.at<uint16_t>(cv::Point(p.x, p.y));

                // Instance ID 0 is usually background, so we ignore it.
                // Also check if the ID is valid for the class/score arrays.
                if (instance_id > 0 && (instance_id - 1) < class_ids.size())
                {
                    PointMatch match;
                    match.point_3d_lidar_frame = object_points[i];
                    match.pixel_2d = p;
                    match.instance_id = instance_id;
                    match.class_id = class_ids[instance_id - 1];
                    match.score = scores[instance_id - 1];
                    matches.push_back(match);
                }
            }
        }

        // 5. Now you have the results in the 'matches' vector.
        // For now, we just log the count.
        // You can create a publisher here to publish this fused data.
        if (!matches.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Camera [%s] x LiDAR [%s]: Found %zu point-pixel matches.",
                        camera_id.c_str(), pc_msg->header.frame_id.c_str(), matches.size());
        }
    }

    // --- Members Variables ---
    std::string camera_id_;
    std::string camera_frame_id_;

    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_image_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_rubyplus_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_helios;
    rclcpp::TimerBase::SharedPtr processing_timer_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};

    std::mutex pc_mutex_ruby_;
    std::mutex pc_mutex_helios_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_pc_ruby_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_pc_helios_;

    cv::Mat camera_intrinsics_;
    cv::Mat distortion_coeffs_;

    rclcpp::TimerBase::SharedPtr report_timer_;
    rclcpp::Time last_report_time_;

    std::mutex data_mutex_;
    yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr last_seg_info_;

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
    std::vector<cv::Scalar> class_colors_;                                          // Paleta de colores

    // Para estadísticas de latencia
    std::map<std::string, std::map<std::string, LatencyMetrics>> all_metrics_;
    std::mutex metrics_mutex_;
    std::map<uint64_t, std::map<std::string, timespec>> pending_batch_t0_timestamps_;
    // For T4 reception spread calculation
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t4_timestamps_; // Key: packet_sequence_number
    std::map<uint64_t, double> batch_t4_reception_spread_ms_;                          // Key: packet_sequence_number
    std::map<uint64_t, std::pair<double, double>> calculated_batch_t0_offsets_;
    std::mutex batch_data_mutex_; // Kept for T0 offsets, consider merging if T0 spread replaces it

    // For T_X reception spread calculation
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t0_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t1_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t2_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t2a_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t2b_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t3_timestamps_;

    std::map<uint64_t, double> batch_t0_spread_ms_;
    std::map<uint64_t, double> batch_t1_spread_ms_;
    std::map<uint64_t, double> batch_t2_spread_ms_;
    std::map<uint64_t, double> batch_t2a_spread_ms_;
    std::map<uint64_t, double> batch_t2b_spread_ms_;
    std::map<uint64_t, double> batch_t3_spread_ms_;
    std::map<uint64_t, double> batch_t4_spread_ms_;
    std::mutex batch_spread_data_mutex_;

    // For Parallelism Metrics
    std::map<uint64_t, double> batch_sum_individual_inf_dur_ms_;
    std::map<uint64_t, double> batch_total_inf_span_ms_;
    std::map<uint64_t, double> batch_inf_overlap_time_ms_;
    std::map<uint64_t, double> batch_parallel_overlap_pct_;
    std::map<uint64_t, double> batch_inf_concurrency_factor_;
    std::map<uint64_t, bool> batch_parallelism_metrics_calculated_;

    std::mutex loss_tracking_mutex_;
    std::map<std::string, uint64_t> last_received_seq_num_;    // Key: camera_id, Value: último seq num recibido
    std::map<std::string, uint64_t> lost_packets_total_count_; // Key: camera_id, Value: total de paquetes perdidos acumulados
    std::map<std::string, bool> first_packet_received_flag_;   // Key: camera_id, para manejar el primer paquete

    // Para escritura de video
    std::mutex video_writer_mutex_;
    std::unique_ptr<cv::VideoWriter> video_writer_;
    std::string video_output_path_ = "/home/david/ros_videos/stitched_video.avi";
    std::map<uint64_t, std::map<std::string, cv::Mat>> frame_buffer_;
    static constexpr int VIDEO_FPS = 10;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SingleProjectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}