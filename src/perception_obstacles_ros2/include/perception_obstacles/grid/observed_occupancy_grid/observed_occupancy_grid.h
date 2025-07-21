#pragma once
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <string>

#include "perception_obstacles/sensors_data/rubyplus_data.hpp"
#include "perception_obstacles/sensors_data/helios_data.hpp"

#include "perception_obstacles/grid/grid_types.h"
#include "perception_obstacles/grid/grid_utils.cuh"

#include "perception_obstacles/perception_utilities/time_data.h"
#include "perception_obstacles/perception_utilities/helper_cuda.h"
#include "perception_obstacles/perception_utilities/ChronoTimer.hpp"

namespace OBS_OG
{

// ----- INDEXES ----- //
void polar_compute_1D_angle_dist(int* i_1D, const int i_angle, const int i_dist, const int NC_DIST);

// ----- GPU ----- //
void compute_observed_occupancy_polar_gpu_simple_version(
    float d_grid_cart_data_mO[], float d_grid_cart_data_mF[], float d_grid_polar_mO[], float d_grid_polar_mF[],
    const float d_pc_x[], const float d_pc_y[], const float d_pc_z[], const float d_pc_intensity[],
    const int d_pc_label[], const bool pc_referenced_to_odom, const float lidar_px, const float lidar_py,
    const float lidar_pz, const int n_points, const int n_layers, const int label_obst, const int NC_X, const int NC_Y,
    const float CART_RES, const float centros_x[], const float centros_y[], const int NC_ANG, const int NC_DIST,
    const float MIN_ANG, const float MIN_DIST, const float RES_ANG, const float RES_DIST, const int iter);

void observed_occupancy_grid_raster_occupied_cart_only(
    float d_grid_cart_data_mO[], const float d_pc_x[], const float d_pc_y[], const float d_pc_z[],
    const float d_pc_intensity[], const int d_pc_label[], const bool LiDAR_referenced_to_odom, const float lidar_px,
    const float lidar_py, const float lidar_pz, const int n_points, const int label_obst, const int NC_X,
    const int NC_Y, const float MIN_X, const float MIN_Y, const float CART_RES, const int iter);

// ----- MAIN ----- //
void compute_observed_occupancy_core(GRID_TYPES::OG* d_grid, const GRID_TYPES::CART_Data* h_grid_cart,
                                     const GRID_TYPES::CART_Data* d_grid_cart, const GRID_TYPES::POLAR_OG* RB_PolarOG,
                                     GRID_TYPES::POLAR_OG* d_RB_PolarOG, const AUTOPIA_RubyPlus::PointCloud* RB_pc,
                                     const AUTOPIA_RubyPlus::PointCloud* d_RB_pc, const bool consider_helios_right,
                                     const GRID_TYPES::POLAR_OG_small* Hr_PolarOG,
                                     GRID_TYPES::POLAR_OG_small* d_Hr_PolarOG, const AUTOPIA_Helios::PointCloud* Hr_pc,
                                     const AUTOPIA_Helios::PointCloud* d_Hr_pc, const bool consider_helios_left,
                                     const GRID_TYPES::POLAR_OG_small* Hl_PolarOG,
                                     GRID_TYPES::POLAR_OG_small* d_Hl_PolarOG, const AUTOPIA_Helios::PointCloud* Hl_pc,
                                     const AUTOPIA_Helios::PointCloud* d_Hl_pc, const int label_obst,
                                     DATA_times* TIME_measurements, const int iter);

// ----- DEBUG ----- //
void write_files_polar_OG(const float d_grid_polar_mO[], const float d_grid_polar_mF[], const int NC_ANG,
                          const int NC_DIST, const int iter);
void write_files_observed_occupancy_grid(const float d_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
                                         const float d_mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const int iter);
}  // namespace OBS_OG