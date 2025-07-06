
#pragma once

#include <vector>
#include <stdio.h>
#include <math.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <filesystem>

// For quaternion to rotation matrix and For 1D arrangement used
#include "perception_obstacles/perception_utilities/utils.h"

namespace AUTOPIA_Helios
{
// ----------------------------- Helios Constant Data ----------------------------- //
const int N_ANGLES = 3600;  // originally 1800 (Changed for subchannel)
const int N_LAYERS = 16;    // originally 32
const int N_POINTS = N_ANGLES * N_LAYERS;

// ----------------------------- Helios Variables ----------------------------- //
typedef struct
{
  // MetaChannel stuff
  int rows;
  int cols;
  int total_points;
  int groups;
  //   int shift_metachannel;  // Jump for metachannel
  int n_metachannels;
  int n_points_metachannel;
  //   std::vector<int> index_order;
  std::vector<int> subchannel;
  std::vector<int> subchannel_layer;
  std::vector<int> idx1D;

} MetaChannelData;

typedef struct
{
  float threshold_height_is_obst = 10.0;
  float max_ground_height_within_distance = 0.4;
  float threshold_distance_for_max_ground_height = 5.0;

  float threshold_gradient_first_impact_is_obst = 30 * M_PI / 180.0;  // atan2(15 / 100) = 8.53

  float threshold_obst_height_dist = 0.5;

  float max_gradiente = 30 * M_PI / 180.0;

  // punto[2] < (puntoRef[2] + back2ground_diff_height_with_last_ground
  float back2ground_diff_height_with_last_ground = 0.25;

  // (punto[2] - puntoPrevio[2]) < back2ground_diff_height_with_last_ground
  float back2ground_diff_height_with_prev_obst = 0;

  float noise_radius = 0.5;
} parameters_channel_based;

typedef struct
{
  int LiDAR_ID = -1;
  float LiDAR_px = 0;
  float LiDAR_py = 0;
  float LiDAR_pz = 0;
  int n_points = 0;
  int n_layers = 0;
  bool referenced_to_odom = false;

  int64_t start_sec, start_nsec;
  int64_t end_sec, end_nsec;

  float x[AUTOPIA_Helios::N_POINTS];
  float y[AUTOPIA_Helios::N_POINTS];
  float z[AUTOPIA_Helios::N_POINTS];
  float intensity[AUTOPIA_Helios::N_POINTS];
  float vert_ang[AUTOPIA_Helios::N_POINTS];
  double timestamp[AUTOPIA_Helios::N_POINTS];
  int metaChannel[AUTOPIA_Helios::N_POINTS];
  int label[AUTOPIA_Helios::N_POINTS];
  int channel_label[AUTOPIA_Helios::N_POINTS];
  int label_reason[AUTOPIA_Helios::N_POINTS];

  float rotation_matrix[3][3];

} PointCloud;

// ----------------------------- Helios Functions ------------------------------ //

void initialize_metachannels_data(AUTOPIA_Helios::MetaChannelData* rb_data);

void initialize_pointcloud(AUTOPIA_Helios::PointCloud* pc, AUTOPIA_Helios::MetaChannelData* RB_data, const int LiDAR_ID,
                           const float px, const float py, const float pz, const int n_points, const int n_layers,
                           const geometry_msgs::msg::Transform& transform);

void transform_PointCloud2_to_HeliosPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg_pointcloud,
                                               AUTOPIA_Helios::PointCloud* RB_pc,
                                               AUTOPIA_Helios::MetaChannelData* RB_data);

}  // namespace AUTOPIA_Helios