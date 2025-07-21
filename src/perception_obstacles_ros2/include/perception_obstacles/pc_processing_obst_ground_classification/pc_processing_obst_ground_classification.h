#pragma once
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "perception_obstacles/sensors_data/rubyplus_data.hpp"
#include "perception_obstacles/sensors_data/helios_data.hpp"

#include "perception_obstacles/perception_utilities/time_data.h"
#include "perception_obstacles/perception_utilities/helper_cuda.h"
#include "perception_obstacles/perception_utilities/utils_cuda.cuh"

#include "perception_obstacles/ego_vehicle/calculos_estado_coche.h"

#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification_utils.h"

namespace OBST_GROUND
{

// ------------------------- CHANNEL BASED ------------------------- //
void host_pc_obst_ground_channel_based(int label[], int reason[], const float x[], const float y[], const float z[],
                                       const float intensity[], const int n_total_points, const int n_layers,
                                       const float LiDAR_height, const float threshold_height_is_obst,
                                       const float threshold_gradient_first_impact_is_obst, const float max_gradiente,
                                       const float max_ground_height_within_distance,
                                       const float threshold_distance_for_max_ground_height,
                                       const float back2ground_diff_height_with_last_ground,
                                       const float back2ground_diff_height_with_prev_obst, const float noise_radius,
                                       const int label_suelo, const int label_obst, const int label_noise);

void gpu_pc_obst_ground_channel_based(int d_label[], int d_label_reason[], const float d_x[], const float d_y[],
                                      const float d_z[], const float d_intensity[], const int n_total_points,
                                      const int n_layers, const float LiDAR_pz, const float threshold_height_is_obst,
                                      const float threshold_gradient_first_impact_is_obst, const float max_gradiente,
                                      const float max_ground_height_within_distance,
                                      const float threshold_distance_for_max_ground_height,
                                      const float back2ground_diff_height_with_last_ground,
                                      const float back2ground_diff_height_with_prev_obst, const float noise_radius,
                                      const float noise_x_min, const float noise_x_max, const float noise_y_min,
                                      const float noise_y_max, const int label_obst, const int label_suelo,
                                      const int label_noise);

// ------------------------- MEDIAN FILTER ------------------------- //
void reclassify_points_with_median_filter(int d_label[], const float d_x[], const float d_y[], const float d_z[],
                                          const int n_total_points, const float threshold_diff_ground,
                                          const int label_obstacle, const int label_ground, const int label_noise,
                                          const int iter);

void reclassify_points_with_median_filter_cart(int d_label[], const float d_x[], const float d_y[], const float d_z[],
                                               const int n_total_points, const float threshold_diff_ground,
                                               const int label_obstacle, const int label_ground, const int label_noise,
                                               const int iter);

// ------------------------- MAIN FUNCTION ------------------------- //
void pc_processing_core(AUTOPIA_RubyPlus::PointCloud* RB_pc, AUTOPIA_RubyPlus::PointCloud* d_RB_pc,
                        AUTOPIA_RubyPlus::parameters_channel_based* param_CB_RB, const bool consider_PC_HeliosRight,
                        const bool consider_PC_HeliosLeft, AUTOPIA_Helios::PointCloud* Hr_pc,
                        AUTOPIA_Helios::PointCloud* d_Hr_pc, AUTOPIA_Helios::PointCloud* Hl_pc,
                        AUTOPIA_Helios::PointCloud* d_Hl_pc, AUTOPIA_Helios::parameters_channel_based* param_CB_Helios,
                        const int label_obst, const int label_ground, const int label_noise,
                        EGO_VEH::INFO_ego* h_info_coche, EGO_VEH::INFO_ego* h_info_coche_old,
                        DATA_times* TIME_measurements, const int iter);

// ------------------------- DEBUG ------------------------- //
void write_file_pointcloud(const std::string name, const float x[], const float y[], const float z[],
                           const float vert_ang[], const float intensity[], const int channel[], const int label[],
                           const int channel_label[], const int label_reason[], const int n_points);

}  // namespace OBST_GROUND