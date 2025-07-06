#ifndef UTILS_CUDA_H_
#define UTILS_CUDA_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "perception_obstacles/perception_utilities/helper_cuda.h"

#include <string.h>
#include <math.h>

__device__ static void exp_2r_d(double * x)
{
  // p = 22. Se consigue reducir de ~7 a ~4.5ms
  *x = 1.0 + *x / 4194304;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
}

__device__ static void exp_2r_d(float * x)
{
  // p = 22. Se consigue reducir de ~7 a ~4.5ms
  *x = 1.0 + *x / 4194304;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
  *x *= *x;
}

__device__ static void device_restar_angulos(float * resta, const float ang1, const float ang2)
{
  float aux = ang1 - ang2 + M_PI;
  *resta = aux - floor(aux / (2 * M_PI)) * 2 * M_PI - M_PI;
}

__device__ static void device_restar_angulos(double * resta, const double ang1, const double ang2)
{
  double aux = ang1 - ang2 + M_PI;
  *resta = aux - floor(aux / (2 * M_PI)) * 2 * M_PI - M_PI;
}

// static float* bigger_size_needed_reAlloc_device_float(float* d_a, const int required_size, int* r_size)
// {
//   *r_size = std::max(required_size + 500, 2000);

//   checkCudaErrors(cudaFree(d_a));

//   // Esto no hace falta podriamos reservar espacio de memoria para el puntero d_a igualmente
//   float* ptr;
//   checkCudaErrors(cudaMalloc(&ptr, *r_size * sizeof(float)));

//   return ptr;
// }

// static bool* bigger_size_needed_reAlloc_device_bool(bool* d_a, const int required_size, int* r_size)
// {
//   *r_size = std::max(required_size + 500, 2000);
//   checkCudaErrors(cudaFree(d_a));

//   bool* ptr;
//   checkCudaErrors(cudaMalloc(&ptr, *r_size * sizeof(bool)));

//   return ptr;
// }

// static float* bigger_size_needed_reAlloc_host_float(float* h_a, const int required_size, int* r_size)
// {
//   *r_size = std::max(required_size + 500, 2000);
//   checkCudaErrors(cudaFreeHost(h_a));

//   float* ptr;
//   checkCudaErrors(cudaHostAlloc(&ptr, *r_size * sizeof(float), cudaHostAllocDefault));

//   return ptr;
// }

// static bool* bigger_size_needed_reAlloc_host_bool(bool* h_a, const int required_size, int* r_size)
// {
//   *r_size = std::max(required_size + 500, 2000);
//   checkCudaErrors(cudaFreeHost(h_a));

//   bool* ptr;
//   checkCudaErrors(cudaHostAlloc(&ptr, *r_size * sizeof(bool), cudaHostAllocDefault));

//   return ptr;
// }

// Matriz Rotacion Z
__device__ static void device_matrizRotacionZ(float * x, float * y, const float theta)
{
  float x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos(theta) - y_1 * sin(theta);
  *y = x_1 * sin(theta) + y_1 * cos(theta);
}
__device__ static void device_matrizRotacionZInversa(float * x, float * y, const float theta)
{
  float x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos(theta) + y_1 * sin(theta);
  *y = -x_1 * sin(theta) + y_1 * cos(theta);
}

__device__ static void device_matrizRotacionZ(double * x, double * y, const double theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos(theta) - y_1 * sin(theta);
  *y = x_1 * sin(theta) + y_1 * cos(theta);
}
__device__ static void device_matrizRotacionZInversa(double * x, double * y, const double theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos(theta) + y_1 * sin(theta);
  *y = -x_1 * sin(theta) + y_1 * cos(theta);
}

__device__ static void device_matrizRotacionZ_seno_coseno_precalculado(
  double * x, double * y, const double sin_theta, const double cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta - y_1 * sin_theta;
  *y = x_1 * sin_theta + y_1 * cos_theta;
}

__device__ static void device_matrizRotacionZInversa_seno_coseno_precalculado(
  double * x, double * y, const double sin_theta, const double cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta + y_1 * sin_theta;
  *y = -x_1 * sin_theta + y_1 * cos_theta;
}

__device__ static void device_matrizRotacionX_seno_coseno_precalculado(
  double * y, double * z, const double sin_theta, const double cos_theta)
{
  double y_1, z_1;

  y_1 = *y;
  z_1 = *z;

  *y = y_1 * cos_theta - z_1 * sin_theta;
  *z = y_1 * sin_theta + z_1 * cos_theta;
}

__device__ static void device_matrizRotacionXInversa_seno_coseno_precalculado(
  double * y, double * z, const double sin_theta, const double cos_theta)
{
  double y_1, z_1;

  y_1 = *y;
  z_1 = *z;

  *y = y_1 * cos_theta + z_1 * sin_theta;
  *z = -y_1 * sin_theta + z_1 * cos_theta;
}

__device__ static void device_matrizRotacionY_seno_coseno_precalculado(
  double * x, double * z, const double sin_theta, const double cos_theta)
{
  double x_1, z_1;

  x_1 = *x;
  z_1 = *z;

  *x = x_1 * cos_theta + z_1 * sin_theta;
  *z = -x_1 * sin_theta + z_1 * cos_theta;
}

__device__ static void device_matrizRotacionYInversa_seno_coseno_precalculado(
  double * x, double * z, const double sin_theta, const double cos_theta)
{
  double x_1, z_1;

  x_1 = *x;
  z_1 = *z;

  *x = x_1 * cos_theta - z_1 * sin_theta;
  *z = x_1 * sin_theta + z_1 * cos_theta;
}

// Matriz Rotacion Z

__device__ static void device_matrizRotacionZ_seno_coseno_precalculado(
  float * x, float * y, const float sin_theta, const float cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta - y_1 * sin_theta;
  *y = x_1 * sin_theta + y_1 * cos_theta;
}

__device__ static void device_matrizRotacionZInversa_seno_coseno_precalculado(
  float * x, float * y, const float sin_theta, const float cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta + y_1 * sin_theta;
  *y = -x_1 * sin_theta + y_1 * cos_theta;
}

__device__ static void device_matrizRotacionX_seno_coseno_precalculado(
  float * y, float * z, const float sin_theta, const float cos_theta)
{
  double y_1, z_1;

  y_1 = *y;
  z_1 = *z;

  *y = y_1 * cos_theta - z_1 * sin_theta;
  *z = y_1 * sin_theta + z_1 * cos_theta;
}

__device__ static void device_matrizRotacionXInversa_seno_coseno_precalculado(
  float * y, float * z, const float sin_theta, const float cos_theta)
{
  double y_1, z_1;

  y_1 = *y;
  z_1 = *z;

  *y = y_1 * cos_theta + z_1 * sin_theta;
  *z = -y_1 * sin_theta + z_1 * cos_theta;
}

__device__ static void device_matrizRotacionY_seno_coseno_precalculado(
  float * x, float * z, const float sin_theta, const float cos_theta)
{
  double x_1, z_1;

  x_1 = *x;
  z_1 = *z;

  *x = x_1 * cos_theta + z_1 * sin_theta;
  *z = -x_1 * sin_theta + z_1 * cos_theta;
}

__device__ static void device_matrizRotacionYInversa_seno_coseno_precalculado(
  float * x, float * z, const float sin_theta, const float cos_theta)
{
  double x_1, z_1;

  x_1 = *x;
  z_1 = *z;

  *x = x_1 * cos_theta - z_1 * sin_theta;
  *z = x_1 * sin_theta + z_1 * cos_theta;
}

__device__ static void device_interpolacion_lineal(
  float * y, const float x, const float x1, const float x2, const float y1, const float y2)
{
  if (x2 == x1) {
    *y = y1;
  } else {
    *y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
  }
}
__device__ static void device_interpolacion_lineal(
  double * y, const double x, const double x1, const double x2, const double y1, const double y2)
{
  if (x2 == x1) {
    *y = y1;
  } else {
    *y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
  }
}
__device__ static void device_interpolacion_lineal_angulos(
  double * ang, const double x, const double x0, const double x1, const double ang0,
  const double ang1)
{
  double a = (ang1 - ang0 + M_PI);
  double b = (2 * M_PI);
  double dif_ang = a - floor(a / b) * b - M_PI;
  *ang = ang0 + (x - x0) * dif_ang / (x1 - x0);
}

__device__ static void device_matrizRotacionZ_seno_coseno_precalculado(
  float * x, float * y, const double sin_theta, const double cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta - y_1 * sin_theta;
  *y = x_1 * sin_theta + y_1 * cos_theta;
}

__device__ static void device_matrizRotacionZInversa_seno_coseno_precalculado(
  float * x, float * y, const double sin_theta, const double cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta + y_1 * sin_theta;
  *y = -x_1 * sin_theta + y_1 * cos_theta;
}

__device__ static void d_matrizRotacion(
  double * x, double * y, double * z, const int eje_cambio, const double ang)
{
  double x_1, y_1, z_1;

  x_1 = *x;
  y_1 = *y;
  z_1 = *z;

  switch (eje_cambio) {
    case 1:  // Rotacion eje X
    {
      *y = y_1 * cos(ang) - z_1 * sin(ang);
      *z = y_1 * sin(ang) + z_1 * cos(ang);
      break;
    }
    case 2:  // Rotacion eje Y
    {
      *x = x_1 * cos(ang) + z_1 * sin(ang);
      *z = -x_1 * sin(ang) + z_1 * cos(ang);
      break;
    }
    case 3:  // Rotacion eje Z
    {
      *x = x_1 * cos(ang) - y_1 * sin(ang);
      *y = x_1 * sin(ang) + y_1 * cos(ang);
      break;
    }
  }
}

// __device__ static int kernel_sub2ind_haces(const int i_layer, const int i_beam, const int n_layers, const int
// n_beams)
// {
//   return i_beam * n_layers + i_layer;
// }

// __device__ static void kernel_ind2sub_haces(int* i_layer, int* i_beam, const int idx, const int n_layers,
//                                             const int n_beams)
// // const int sub, const int cols, const int rows, int *row, int *col)
// {
//   *i_beam = idx / n_layers;
//   *i_layer = idx % n_layers;

// #if DEBUG
//   int aux_idx;
//   aux_idx = kernel_sub2ind_haces(*i_layer, *i_beam, n_layers, n_beams);
//   if (aux_idx != idx)
//   {
//     printf(
//         "!\n!\n!\n kernel_ind2sub_haces - las funciones kernel_ind2sub y kernel_sub2ind_haces! NO "
//         "COINCIDEN \n"
//         "   idx recibido = %d -> i_beam y i_layer calculados [%d, %d] -> idx recalculado = %d\n!!\n",
//         idx, *i_beam, *i_layer, aux_idx);
//   }
// #endif
// }

__device__ static int device_sub2ind_polar(
  const int i_a, const int i_d, const int NC_ang, const int NC_dist)
{
#if DEBUG
  int aux_idx, aux_ia, aux_id;
  aux_idx = i_d * NC_ang + i_a;
  aux_id = aux_idx / NC_ang;
  aux_ia = aux_idx % NC_ang;
  if (aux_ia != i_a || aux_id != i_d) {
    printf(
      "\n!\n!\n! device_sub2ind_polar no coincide con device_ind2sub_polar \t "
      "Recibido ia_id = [%d, %d]; - calculado idx = %d; -> ia_id_inverso = [%d, %d] \t "
      "NC_ang = %d; NC_dist = %d; !\n!\n",
      i_a, i_d, aux_idx, aux_ia, aux_id, NC_ang, NC_dist);
  }
#endif

  return i_d * NC_ang + i_a;
}
__device__ static void device_ind2sub_polar(
  int * i_a, int * i_d, const int idx, const int NC_ang, const int NC_dist)
{
  *i_d = idx / NC_ang;
  *i_a = idx % NC_ang;

#if DEBUG
  int aux_idx;
  aux_idx = device_sub2ind_polar(*i_a, *i_d, NC_ang, NC_dist);
  if (aux_idx != idx) {
    printf(
      "!\n!\n!\n device_ind2sub_polar - las funciones kernel_ind2sub y device_sub2ind_polar! NO "
      "COINCIDEN \n!\n");
  }
#endif
}

// static void write_txt_float_1Dvector_into_2Dmatrix_device(
//   const float d_matrix[], const std::string name, const bool idx1D_by_cols, const int rows,
//   const int cols)
// {
//   float h_matrix[cols * rows];

//   checkCudaErrors(
//     cudaMemcpy(h_matrix, d_matrix, cols * rows * sizeof(float), cudaMemcpyDeviceToHost));

//   cudaDeviceSynchronize();

//   char s[200];
//   snprintf(s, sizeof(s), "output/%s.txt", name.c_str());

//   FILE * f;

//   f = fopen(s, "w");
//   if (f == NULL) {
//     printf("NO SE PUDO CREAR EL FICHERO %s\n", s);
//     exit(1);
//   }

//   int idx1D;
//   for (int i_r = 0; i_r < rows; i_r++) {
//     for (int i_c = 0; i_c < cols; i_c++) {
//       if (idx1D_by_cols) {
//         idx1D = i_r * cols + i_c;
//       } else {
//         idx1D = i_c * rows + i_r;
//       }

//       fprintf(f, "%f ", h_matrix[idx1D]);
//     }
//     fprintf(f, "\n");
//   }

//   fclose(f);
// }

static void write_txt_int_1Dvector_into_2Dmatrix_device(
  const int d_matrix[], const std::string name, const bool idx1D_by_cols, const int rows,
  const int cols)
{
  int h_matrix[cols * rows];

  checkCudaErrors(
    cudaMemcpy(h_matrix, d_matrix, cols * rows * sizeof(int), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  char s[200];
  snprintf(s, sizeof(s), "output/%s.txt", name.c_str());

  FILE * f;

  f = fopen(s, "w");
  if (f == NULL) {
    printf("NO SE PUDO CREAR EL FICHERO %s\n", s);
    exit(1);
  }

  int idx1D;
  for (int i_r = 0; i_r < rows; i_r++) {
    for (int i_c = 0; i_c < cols; i_c++) {
      if (idx1D_by_cols) {
        idx1D = i_r * cols + i_c;
      } else {
        idx1D = i_c * rows + i_r;
      }

      fprintf(f, "%d ", h_matrix[idx1D]);
    }
    fprintf(f, "\n");
  }

  fclose(f);
}

#endif /* UTILS_CUDA_H_ */