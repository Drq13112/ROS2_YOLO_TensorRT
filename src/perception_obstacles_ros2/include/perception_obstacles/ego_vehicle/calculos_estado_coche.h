#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <filesystem>

#include "perception_obstacles/perception_utilities/utils.h"
#include "perception_obstacles/perception_utilities/utils_cuda.cuh"

namespace EGO_VEH
{

typedef struct
{
  double px_G;
  double py_G;
  double vel;
  double yaw_G;
  double yaw_rate;

  float sin_yaw_G;
  float cos_yaw_G;

  double largo = 4.5;
  double ancho = 2.0;

  float ego_size_min_x = -0.8;
  float ego_size_max_x = 3.3;
  float ego_size_min_y = -1.1;
  float ego_size_max_y = 1.1;

  float delta_x;
  float delta_y;
  float delta_yaw;
  float sin_delta_yaw;
  float cos_delta_yaw;
  float delta_t;

  double tiempo;
  uint32_t sec;
  uint32_t nanosec;

} INFO_ego;

void predecir_estado_coche(EGO_VEH::INFO_ego* info_coche, const EGO_VEH::INFO_ego* info_coche_old,
                           const double target_time);

void calculo_delta_estado_coche(EGO_VEH::INFO_ego* info_coche, const EGO_VEH::INFO_ego* info_coche_old, const int iter,
                                const int iter_inicial);

void gpu_correct_ego_motion_displacement_for_pointcloud(float d_x[], float d_y[], double d_timestamp[],
                                                        const int n_total_points, const EGO_VEH::INFO_ego* h_info_coche,
                                                        const EGO_VEH::INFO_ego* h_info_coche_old);

void write_files_localization(const EGO_VEH::INFO_ego* info_coche, const bool start_file);

}  // namespace EGO_VEH
