#pragma once

#include <stdio.h>
#include <filesystem>
#include <iostream>

#include <math.h>

#include <string>

#include <yaml-cpp/yaml.h>

#include "perception_obstacles/grid/grid_types.h"
#include "perception_obstacles/grid/grid_utils.cuh"

#include "perception_obstacles/perception_utilities/utils.h"

namespace GRID_UTILS
{
// -------------------- INDEXES -------------------- //
void calculo_centro_celda(float* centro_celda_x, float* centro_celda_y, const int i_x, const int i_y, const int NC_X,
                          const int NC_Y, const float MIN_X, const float MIN_Y, const float RES);

// void calculo_indices_celda(int* i_x, int* i_y, float x, float y, const int NC_X, const int NC_Y, const float MIN_X,
//                            const float MIN_Y, const float RES);

void calculo_indices_celda(int* i_x, int* i_y, double x, double y, const int NC_X, const int NC_Y, const double MIN_X,
                           const double MIN_Y, const double RES);

int sub2ind(const int i_y, const int i_x, const int NC_X, const int NC_Y);

void ind2sub(const int idx, const int NC_X, const int NC_Y, int* i_y, int* i_x);

}  // namespace GRID_UTILS