#pragma once

#include <iostream>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <math.h>
#include <filesystem>
#include <chrono>
#include <thread>

#include <mutex>
#include <condition_variable>
#include <atomic>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <pcl_ros/transforms.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  // tf2::fromMsg

#include "perception_obstacles/ego_vehicle/calculos_estado_coche.h"

#include "perception_obstacles/perception_utilities/time_data.h"
#include "perception_obstacles/perception_utilities/ChronoTimer.hpp"
#include "perception_obstacles/perception_utilities/helper_cuda.h"

#include "perception_obstacles/sensors_data/rubyplus_data.hpp"
#include "perception_obstacles/sensors_data/helios_data.hpp"

#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification.h"
#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification_utils.h"

#include "perception_obstacles/grid/grid_types.h"
#include "perception_obstacles/grid/particle_types.h"
#include "perception_obstacles/grid/grid_utils.cuh"
#include "perception_obstacles/grid/grid_inicializar_rejilla_cartesiana.h"

#include "perception_obstacles/grid/observed_occupancy_grid/observed_occupancy_grid.h"

#include "perception_obstacles/grid/offline_road_map/offline_road_map.cuh"

#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

/**
 * @brief Nodo principal del paquete. callbacks + main loop function
 *
 */
class PerceptionObstacles : public rclcpp::Node
{
  struct frame_parameters
  {
    std::string RubyPlus;
    std::string HeliosLeft;
    std::string HeliosRight;
    std::string odom;
    std::string world;
  };

  struct DATA_labels
  {
    int label_noise = -1;
    int label_suelo = 0;
    int label_obst = 1;
  };

public:
  PerceptionObstacles();
  ~PerceptionObstacles();

private:
  // -------------------- VARIABLES -------------------- //

  // Pointer to the thread that is gonna handle the main code (while)
  std::unique_ptr<std::thread> main_thread;

  // ----- Subscriptor and Publisher ----- //
  // ORDER IS IMPORTANT!!!! It sets the priority
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr RubyPlus_subscription;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr HeliosRight_subscription;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr HeliosLeft_subscription;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr localization_subscription;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_RB_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_Hr_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_Hl_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr obs_OG_pO_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr obs_OG_mO_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr obs_OG_mF_publisher_;

  // ---------- Transforms ---------- //
  std::shared_ptr<tf2_ros::TransformListener> transform_listener_{ nullptr };
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<geometry_msgs::msg::TransformStamped> RubyPlus_transform_{ nullptr };
  std::unique_ptr<geometry_msgs::msg::TransformStamped> HeliosLeft_transform_{ nullptr };
  std::unique_ptr<geometry_msgs::msg::TransformStamped> HeliosRight_transform_{ nullptr };
  std::unique_ptr<geometry_msgs::msg::TransformStamped> odom_transform_{ nullptr };
  frame_parameters frames_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_odom_world_broadcaster_;

  // ----- Synchronization ----- //
  // Mutex to block threads
  std::mutex mutex_;

  // Condition variable for waiting
  std::condition_variable RubyPlus_condition_;
  // std::condition_variable HeliosRight_condition_;
  // std::condition_variable HeliosLeft_condition_;
  std::condition_variable init_localization_condition_;

  // bool to avoid spurious wakeups
  bool flag_avoid_spurious_RubyPlus_callback_ = false;
  bool flag_HeliosRight_callback_available_ = false;
  bool flag_HeliosLeft_callback_available_ = false;
  bool flag_avoid_spurious_localization_callback_ = false;

  // ------------------------------ Code Variables ------------------------------ //
  EGO_VEH::INFO_ego h_info_coche_;
  EGO_VEH::INFO_ego h_info_coche_old_;
  EGO_VEH::INFO_ego* d_info_coche_;
  EGO_VEH::INFO_ego info_coche_modificable_callback_;
  EGO_VEH::INFO_ego debug_received_info_coche_;

  // ---------- Sensores & Observed Grid ---------- //
  // RubyPlus
  AUTOPIA_RubyPlus::MetaChannelData RB_data_;
  AUTOPIA_RubyPlus::parameters_channel_based param_CB_RB_;
  AUTOPIA_RubyPlus::PointCloud RB_pc_modificable_callback_;
  AUTOPIA_RubyPlus::PointCloud RB_pc_;
  AUTOPIA_RubyPlus::PointCloud* d_RB_pc_;

  GRID_TYPES::POLAR_OG RB_PolarOG_;
  GRID_TYPES::POLAR_OG* d_RB_PolarOG_;

  // Helios
  bool consider_PC_HeliosRight_ = false;
  bool consider_PC_HeliosLeft_ = false;
  bool valid_data_HeliosRight_ = false;
  bool valid_data_HeliosLeft_ = false;
  AUTOPIA_Helios::MetaChannelData Helios_data_;
  AUTOPIA_Helios::parameters_channel_based param_CB_Helios_;

  AUTOPIA_Helios::PointCloud Hr_pc_modificable_callback_;
  AUTOPIA_Helios::PointCloud Hr_pc_;
  AUTOPIA_Helios::PointCloud* d_Hr_pc_;

  GRID_TYPES::POLAR_OG_small Hr_PolarOG_;
  GRID_TYPES::POLAR_OG_small* d_Hr_PolarOG_;

  AUTOPIA_Helios::PointCloud Hl_pc_modificable_callback_;
  AUTOPIA_Helios::PointCloud Hl_pc_;
  AUTOPIA_Helios::PointCloud* d_Hl_pc_;

  GRID_TYPES::POLAR_OG_small Hl_PolarOG_;
  GRID_TYPES::POLAR_OG_small* d_Hl_PolarOG_;

  // * GRID
  GRID_TYPES::OG* h_grid_obs_;
  GRID_TYPES::OG* d_grid_obs_;

  // ---------- DOG ---------- //
  GRID_TYPES::CART_Data* h_grid_cart_data_;
  GRID_TYPES::CART_Data* d_grid_cart_data_;

  bool flag_particles = false;
  GRID_TYPES::DOG *h_grid_, *d_grid_;
  DYN_CLASS_OG::config *h_config_DOG_, *d_config_DOG_;
  PARTICLE_TYPES::PART_DOG *h_particles_, *d_particles_, *d_particles_sorted_, *d_particles_for_resampling_;

  float *d_random_pred, *d_random_particle_selection, *d_random_cell_selection, *d_random_asociacion,
      *d_random_vel_uniforme;

  // ---------- OFFLINE ROAD MAP ---------- //
  char *h_complete_road_map_, *d_complete_road_map_;
  OFF_ROAD_MAP::config *h_config_map_, *d_config_map_;
  GRID_TYPES::GRID_local_road *h_grid_road_, *d_grid_road_;

  // ---------- Other ---------- //
  DATA_labels labels_;
  size_t freeMem, totalMem;
  long int initial_rng_seed = 0;

  // ---------- Timers ---------- //
  DATA_times TIME_measurements_;

  // ---------- Streams ---------- //
  const int num_streams_ = 6;
  cudaStream_t streams_[6];

  // -------------------- FUNCTIONS -------------------- //

  // Callbacks publishers
  void RubyPlusCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_pointcloud);
  void HeliosRightCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_pointcloud);
  void HeliosLeftCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_pointcloud);
  void LocalizationCallback(const nav_msgs::msg::Odometry::SharedPtr msg_odom);

  int check_QoS_publisher_s(std::string topic_name);

  sensor_msgs::msg::PointCloud2 GeneratePointCloud2Message(const float x[], const float y[], const float z[],
                                                           const float intensity[], const float vert_ang[],
                                                           const int label[], const int channel_label[],
                                                           const int metaChannel[], const int n_points,
                                                           const std::string frame_id, const uint32_t sec,
                                                           const uint32_t nanosec);

  nav_msgs::msg::OccupancyGrid GenerateOGMessage(const float array2D[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
                                                 const int NC_X, const int NC_Y, const float MIN_X, const float MIN_Y,
                                                 const float RES, const uint32_t sec, const uint32_t nsec);

  bool publish_tf_odom_world = true;
  void PublishTransformOdomWorld(const double px_G, const double py_G, const double yaw_G, const uint32_t sec,
                                 const uint32_t nsec);

  // Parameters - Initialization
  void RubyPlusLoadParameters();
  void HeliosLoadParameters();

  void GetTransforms();
  std::unique_ptr<geometry_msgs::msg::TransformStamped> GetSensorTransform(std::string toFrameRel,
                                                                           std::string fromFrameRel);

  void Run();
};
