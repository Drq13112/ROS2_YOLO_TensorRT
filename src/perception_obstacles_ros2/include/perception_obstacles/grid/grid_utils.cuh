#pragma once

#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include "perception_obstacles/grid/grid_types.h"
#include "perception_obstacles/grid/grid_utils.cuh"

#include "perception_obstacles/perception_utilities/utils.h"

namespace GRID_UTILS_CUDA
{

// GPU
__device__ inline void device_calculo_centro_celda(
  float * centro_celda_x, float * centro_celda_y, const int i_x, const int i_y, const int NC_X,
  const int NC_Y, const float MIN_X, const float MIN_Y, const float RES)
{
  *centro_celda_x = (float)i_x * RES + MIN_X + RES / 2.0;
  *centro_celda_y = (NC_Y - (float)i_y) * RES + MIN_Y - RES / 2.0;
}

// __device__ void device_calculo_indices_celda(
//   int * i_x, int * i_y, float x, float y, const int NC_X, const int NC_Y, const float MIN_X,
//   const float MIN_Y, const float RES)
// {
//   int indice_x, indice_y;

//   indice_x = ceil(round(((x - MIN_X) / RES) * 1e6) / 1e6) - 1;
//   indice_y = NC_Y - ceil(round(((y - MIN_Y) / RES) * 1e6) / 1e6);

//   if (indice_x < 0 || indice_x >= NC_X || indice_y < 0 || indice_y >= NC_Y) {
//     *i_x = -1;
//     *i_y = -1;
//   } else {
//     *i_x = indice_x;
//     *i_y = indice_y;
//   }
// }

__device__ inline void device_calculo_indices_celda(
  int * i_x, int * i_y, double x, double y, const int NC_X, const int NC_Y, const float MIN_X,
  const float MIN_Y, const float RES)
{
  int indice_x, indice_y;

  indice_x = ceil(round(((x - MIN_X) / RES) * 1e6) / 1e6) - 1;
  indice_y = NC_Y - ceil(round(((y - MIN_Y) / RES) * 1e6) / 1e6);

  if (indice_x < 0 || indice_x >= NC_X || indice_y < 0 || indice_y >= NC_Y) {
    *i_x = -1;
    *i_y = -1;
  } else {
    *i_x = indice_x;
    *i_y = indice_y;
  }
}

__device__ inline int device_sub2ind(const int i_y, const int i_x, const int NC_X, const int NC_Y)
{
  if (i_x < 0 || i_y < 0) {
    return -1;
  }
  return i_y * NC_X + i_x;
}

__device__ inline void device_ind2sub(
  const int idx, const int NC_X, const int NC_Y, int * i_y, int * i_x)
{
  if (idx == 1) {
    *i_x = -1;
    *i_y = -1;
  } else {
    *i_y = idx / NC_X;
    *i_x = idx % NC_X;
  }
}

__device__ inline int device_busqueda_binaria_ordenado_ascendente(
  const double datos[], const double numero_buscado, const int i_izq, const int i_der)
{
  int i_medio, idx;

  if (i_izq >= i_der) {
    if (datos[i_der] == numero_buscado) {
      idx = i_der + 1;
    } else {
      idx = i_izq;
    }
  } else {
    // NOTE busqueda binaria -> fix( (izq + der) / 2)
    i_medio = (i_izq + i_der) / 2;  // Como es un entero deberia hacer el fix el ya solito

    if (numero_buscado > datos[i_medio]) {
      idx = GRID_UTILS_CUDA::device_busqueda_binaria_ordenado_ascendente(
        datos, numero_buscado, i_medio + 1, i_der);
    } else if (numero_buscado < datos[i_medio]) {
      idx = GRID_UTILS_CUDA::device_busqueda_binaria_ordenado_ascendente(
        datos, numero_buscado, i_izq, i_medio);
    } else {
      idx = i_medio + 1;
    }
  }

  return idx;  // Devuelve el inmediatamente superior
}

}  // namespace GRID_UTILS_CUDA