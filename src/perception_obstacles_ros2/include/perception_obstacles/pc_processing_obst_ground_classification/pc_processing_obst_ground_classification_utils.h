#pragma once
#include <stdio.h>
#include <pcl_ros/transforms.hpp>
#include <iostream>
#include <vector>

namespace OBST_GROUND_UTILS
{

// ---------------------------------------- ROTATION ----------------------------------------
inline void apply_rotation_matrix(
  float * x, float * y, float * z, const float rotation_matrix[3][3])
{
  // Almacenar las coordenadas originales del punto
  float aux_x = *x;
  float aux_y = *y;
  float aux_z = *z;

  // Aplicar solo la rotación
  *x =
    rotation_matrix[0][0] * aux_x + rotation_matrix[0][1] * aux_y + rotation_matrix[0][2] * aux_z;
  *y =
    rotation_matrix[1][0] * aux_x + rotation_matrix[1][1] * aux_y + rotation_matrix[1][2] * aux_z;
  *z =
    rotation_matrix[2][0] * aux_x + rotation_matrix[2][1] * aux_y + rotation_matrix[2][2] * aux_z;
}

inline void rotate_pointcloud(
  float x[], float y[], float z[], const int n_points, const float rotation_matrix[3][3])
{
  for (int i = 0; i < n_points; i++) {
    apply_rotation_matrix(&x[i], &y[i], &z[i], rotation_matrix);
  }
}

}  // namespace OBST_GROUND_UTILS