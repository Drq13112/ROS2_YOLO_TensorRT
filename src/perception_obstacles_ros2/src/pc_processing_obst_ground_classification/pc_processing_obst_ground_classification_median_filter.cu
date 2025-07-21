#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification.h"

// ---------------------------------------- INDEXES ---------------------------------------- //
inline __device__ void FM_device_compute_1D_angle_dist(
  int * i_1D, const int i_angle, const int i_dist, const int NC_DIST)
{
  *i_1D = i_angle * NC_DIST + i_dist;
}
inline __device__ void FM_device_compute_2D_angle_dist(
  int * i_angle, int * i_dist, const int i_1D, const int NC_DIST)
{
  *i_angle = i_1D / NC_DIST;
  *i_dist = i_1D % NC_DIST;
}

inline void FM_compute_1D_angle_dist(
  int * i_1D, const int i_angle, const int i_dist, const int NC_DIST)
{
  *i_1D = i_angle * NC_DIST + i_dist;
}

__device__ void FM_device_calcular_indice_celda_polar(
  int * i_a, int * i_d, const float ang, const float dist, const float MIN_ANG,
  const float MIN_DIST, const float RES_ANG, const float RES_DIST, const int NC_ANG,
  const int NC_DIST)
{
  *i_a = ceil(round(((ang - MIN_ANG) / RES_ANG) * 1e6) / 1e6) - 1;
  *i_d = ceil(round(((dist - MIN_DIST) / RES_DIST) * 1e6) / 1e6) - 1;

  if (*i_a < 0 || *i_a >= NC_ANG || *i_d < 0 || *i_d >= NC_DIST) {
    *i_a = -1;
    *i_d = -1;
  }
}

// ---------------------------------------- RASTERIZE POINTS ---------------------------------------- //
__global__ void gpu_rasterize_points_in_grid(
  float mean_ground_height[], int min_ground_height_intx100[], float cont_ground[],
  int min_obst_height_intx100[], float cont_obst[], const float x[], const float y[],
  const float z[], const int label[], const int n_points, const int label_obst,
  const int label_ground, const float MIN_ANG, const float MIN_DIST, const float RES_ANG,
  const float RES_DIST, const int NC_ANG, const int NC_DIST)
{
  // Get index from the division of blocks and grids
  int idx_sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_sample < n_points) {
    if (label[idx_sample] == label_ground || label[idx_sample] == label_obst) {
      float ang = atan2(y[idx_sample], x[idx_sample]);
      float dist = sqrt(x[idx_sample] * x[idx_sample] + y[idx_sample] * y[idx_sample]);

      int i_a, i_d, idx1D;
      FM_device_calcular_indice_celda_polar(
        &i_a, &i_d, ang, dist, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST, NC_ANG, NC_DIST);

      if (i_a == -1) {
        return;
      }

      FM_device_compute_1D_angle_dist(&idx1D, i_a, i_d, NC_DIST);

      int int_height = (int)(100 * (z[idx_sample]));
      if (label[idx_sample] == label_ground) {
        atomicAdd(&mean_ground_height[idx1D], z[idx_sample]);
        atomicAdd(&cont_ground[idx1D], 1.0f);
        atomicMin(&min_ground_height_intx100[idx1D], int_height);
      } else {
        atomicMin(&min_obst_height_intx100[idx1D], int_height);
        atomicAdd(&cont_obst[idx1D], 1.0f);
      }
    }
  }
}

// ---------------------------------------- MEDIAN FILTER ---------------------------------------- //
__global__ void global_filtro_mediana_suelo(
  float filtered_height[], int cont_cells_used[], const float mean_ground_height[],
  const int min_height_ground_intx100[], const float cont_ground[],
  const int min_height_obst_intx100[], const float cont_obst[], const int option_mean1_min2_both3,
  const float MIN_ANG, const float MIN_DIST, const float RES_ANG, const float RES_DIST,
  const int NC_ANG, const int NC_DIST)
{
  int idx1D_cell = blockIdx.x * blockDim.x + threadIdx.x;
  int i_a, i_d;
  FM_device_compute_2D_angle_dist(&i_a, &i_d, idx1D_cell, NC_DIST);

  if (i_a >= 0 && i_a < NC_ANG && i_d >= 0 && i_d < NC_DIST) {
    float data[9];  // kernel de 3x3 (aqui es donde se va el coste asi que vamos a dejarlo fijo)
    int sk = 1;
    // float data[25];  // kernel de 5x5 (aqui es donde se va el coste asi que vamos a dejarlo fijo)
    // int sk = 2;

    int idx2_1D, cont = 0;

    // Rellenamos el contenido
    for (int i2_a = i_a - sk; i2_a <= i_a + sk; i2_a++) {
      if (i2_a >= 0 && i2_a < NC_ANG) {
        for (int i2_d = i_d - sk; i2_d <= i_d + sk; i2_d++) {
          if (i2_d >= 0 && i2_d < NC_DIST) {
            FM_device_compute_1D_angle_dist(&idx2_1D, i2_a, i2_d, NC_DIST);

            // If there is info
            if (cont_ground[idx2_1D] > 0) {
              if (option_mean1_min2_both3 == 1) {
                data[cont] = mean_ground_height[idx2_1D];
              } else if (option_mean1_min2_both3 == 2) {
                data[cont] = ((float)min_height_ground_intx100[idx2_1D]) / 100.0;
              } else if (option_mean1_min2_both3 == 3) {
                data[cont] = (mean_ground_height[idx2_1D] +
                              ((float)min_height_ground_intx100[idx2_1D]) / 100.0) /
                             2.0;
              } else {
                printf("FM - no puede ser\n");
              }

              // If there are obstacle points beneath the lowest ground points... this ground height is probably wrong
              if (
                cont_obst[idx2_1D] > 0 &&
                min_height_ground_intx100[idx2_1D] > min_height_obst_intx100[idx2_1D]) {
                data[cont] = ((float)min_height_obst_intx100[idx2_1D]) / 100.0;
              }
              cont++;
            }
          }
        }
      }
    }

    // If there is info, fill the cell
    if (cont > 0) {
      // Ordenamos hasta la mitad incluido (porque no nos hace falta mas) (mitad incluido por si cont es par)
      // comparando el valor actual con cada uno de los siguientes valores
      // Ejemplo par:
      //   [4, 1, 3, 2, 9, 7] -> [1, 2, 3, 4, 7, 9] -> median = 3.5;
      //   ordenamos hasta: 6 / 2 = 3 -> data[0 : 3] = [1, 2, 3, 4] -> median = media(data[2], data[3]) = 3.5
      // Ejemplo impar
      //   [4, 1, 3, 2, 9] -> [1, 2, 3, 4, 7] -> median = 3;
      //   ordenamos hasta: 5 / 2 = 2 -> data[0 : 2] = [1, 2, 3] -> median = data[2] = 3

      int i_min_valor_local;
      float valor_temp_swap;
      int i_medio = cont / 2;
      for (int i_data = 0; i_data <= i_medio; i_data++) {
        // En principio el valor esta bien colocado
        i_min_valor_local = i_data;

        // Ahora empezamos a comparar con el siguiente valor
        for (int j = i_data + 1; j < cont; j++) {
          // Si el valor de esta nueva celda es menor que el guardado -> actualizamos
          if (data[j] < data[i_min_valor_local]) {
            i_min_valor_local = j;
          }
        }

        // Guardamos el valor menor
        valor_temp_swap = data[i_data];
        data[i_data] = data[i_min_valor_local];
        data[i_min_valor_local] = valor_temp_swap;
      }

      // guardamos el valor de en medio
      if (cont % 2 == 0) {
        filtered_height[idx1D_cell] = (data[i_medio - 1] + data[i_medio]) / 2.0;
      } else {
        filtered_height[idx1D_cell] = data[i_medio];
      }
      cont_cells_used[idx1D_cell] = cont;
    }
  }
}

// ---------------------------------------- RECOMPUTE LABELS ---------------------------------------- //
__global__ void global_reclassify_points_using_median_filter(
  int label[], const float ref_ground_height[], const int cont_points_used[], const float x[],
  const float y[], const float z[], const int n_points, const float threshold_diff_ground,
  const int label_obstacle, const int label_ground, const int label_noise, const float MIN_ANG,
  const float MIN_DIST, const float RES_ANG, const float RES_DIST, const int NC_ANG,
  const int NC_DIST)
{
  // Get index from the division of blocks and grids
  int idx_sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_sample < n_points) {
    if (label[idx_sample] != label_noise || true) {
      // This could be computed only once but... I'm trying to minimize the number of dynamic arrays created
      float ang = atan2(y[idx_sample], x[idx_sample]);
      float dist = sqrt(x[idx_sample] * x[idx_sample] + y[idx_sample] * y[idx_sample]);

      int i_a, i_d, idx1D;
      FM_device_calcular_indice_celda_polar(
        &i_a, &i_d, ang, dist, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST, NC_ANG, NC_DIST);
      if (i_a == -1) {
        return;
      }
      FM_device_compute_1D_angle_dist(&idx1D, i_a, i_d, NC_DIST);

      // If there is info, reclassify
      if (cont_points_used[idx1D] > 0) {
        // If the point is above the mean ground (+threshold) -> obstacle. If not, ground
        if ((z[idx_sample]) > (ref_ground_height[idx1D] + threshold_diff_ground)) {
          label[idx_sample] = label_obstacle;
        } else {
          label[idx_sample] = label_ground;
        }
      }
    }
  }
}

__global__ void global_solve_mean_1D_vector(
  float data_sum[], const float cont[], const int vector_size)
{
  int idx1D_cell = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx1D_cell < vector_size) {
    if (cont[idx1D_cell] > 0) {
      data_sum[idx1D_cell] /= cont[idx1D_cell];
    }
  }
}

__global__ void global_initialize_int_1D_vector(
  int array_int[], const int value, const int vector_size)
{
  int idx1D_cell = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx1D_cell < vector_size) {
    array_int[idx1D_cell] = value;
  }
}

// ---------------------------------------- CORE FUNCTION ---------------------------------------- //
void OBST_GROUND::reclassify_points_with_median_filter(
  int d_label[], const float d_x[], const float d_y[], const float d_z[], const int n_total_points,
  const float threshold_diff_ground, const int label_obstacle, const int label_ground,
  const int label_noise, const int iter)
{
  const static int NC_ANG = 96;  // 360 / NC_ANG = resolucion X grados
  const static float RES_ANG = 2 * M_PI / ((float)NC_ANG);
  const static float MIN_ANG = -M_PI;

  // const static int NC_DIST = 448;      // NC_DIST * RES metros = X metros
  // const static float RES_DIST = 0.25;  // TODO
  const static int NC_DIST = 256;     // NC_DIST * RES metros = X metros
  const static float RES_DIST = 0.5;  // TODO
  // const static int NC_DIST = 160;      // NC_DIST * RES metros = X metros
  // const static float RES_DIST = 0.75;  // TODO
  const static float MIN_DIST = 0.20;  // TODO

  int option_mean1_min2_both3 = 3;
  printf("PARAMETROS NO INICIALIZADOS\n");

  // Declare, reserve and initialize (potentially this could be done once... but lets try this way so it is easily resusable)
  float *d_FM_filtered_height, *d_FM_mean_ground_height, *d_FM_cont_ground, *d_FM_cont_obst;
  int *d_FM_cont_cells_used, *d_FM_min_ground_height_intx100, *d_FM_min_obst_height_intx100;

  checkCudaErrors(cudaMalloc((void **)&d_FM_filtered_height, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_FM_cont_cells_used, NC_ANG * NC_DIST * sizeof(int)));

  checkCudaErrors(cudaMalloc((void **)&d_FM_mean_ground_height, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_FM_cont_ground, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(
    cudaMalloc((void **)&d_FM_min_ground_height_intx100, NC_ANG * NC_DIST * sizeof(int)));

  checkCudaErrors(cudaMalloc((void **)&d_FM_cont_obst, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(
    cudaMalloc((void **)&d_FM_min_obst_height_intx100, NC_ANG * NC_DIST * sizeof(int)));

  // parallelization dimensions
  static dim3 blocks_beams(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grids_beams(
    (n_total_points + blocks_beams.x - 1) /
    blocks_beams.x);  // number of blocks: choose a number that ensures all data is processed

  static dim3 block_FM(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grid_FM(
    (NC_ANG * NC_DIST + block_FM.x - 1) /
    block_FM.x);  // number of blocks: choose a number that ensures all data is processed

  // Initialize
  checkCudaErrors(cudaMemset(d_FM_filtered_height, 0.0, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(cudaMemset(d_FM_cont_cells_used, 0, NC_ANG * NC_DIST * sizeof(int)));

  checkCudaErrors(cudaMemset(d_FM_mean_ground_height, 0.0, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(cudaMemset(d_FM_cont_ground, 0.0, NC_ANG * NC_DIST * sizeof(float)));
  global_initialize_int_1D_vector<<<grid_FM, block_FM>>>(
    d_FM_min_ground_height_intx100, 10000, NC_ANG * NC_DIST);

  checkCudaErrors(cudaMemset(d_FM_cont_obst, 0.0, NC_ANG * NC_DIST * sizeof(float)));
  global_initialize_int_1D_vector<<<grid_FM, block_FM>>>(
    d_FM_min_obst_height_intx100, 10000, NC_ANG * NC_DIST);

  // Rasterize points
  gpu_rasterize_points_in_grid<<<grids_beams, blocks_beams>>>(
    d_FM_mean_ground_height, d_FM_min_ground_height_intx100, d_FM_cont_ground,
    d_FM_min_obst_height_intx100, d_FM_cont_obst, d_x, d_y, d_z, d_label, n_total_points,
    label_obstacle, label_ground, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST, NC_ANG, NC_DIST);

  global_solve_mean_1D_vector<<<grid_FM, block_FM>>>(
    d_FM_mean_ground_height, d_FM_cont_ground, NC_ANG * NC_DIST);

  // Approximate ground height with median filter
  global_filtro_mediana_suelo<<<grid_FM, block_FM>>>(
    d_FM_filtered_height, d_FM_cont_cells_used, d_FM_mean_ground_height,
    d_FM_min_ground_height_intx100, d_FM_cont_ground, d_FM_min_obst_height_intx100, d_FM_cont_obst,
    option_mean1_min2_both3, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST, NC_ANG, NC_DIST);

  // Recompute labels
  global_reclassify_points_using_median_filter<<<grids_beams, blocks_beams>>>(
    d_label, d_FM_filtered_height, d_FM_cont_cells_used, d_x, d_y, d_z, n_total_points,
    threshold_diff_ground, label_obstacle, label_ground, label_noise, MIN_ANG, MIN_DIST, RES_ANG,
    RES_DIST, NC_ANG, NC_DIST);

  // Synchronize
  cudaDeviceSynchronize();

  // Debug
  /*
  std::string name;

  name = "FM_filtered_z_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(d_FM_filtered_height, name, true, NC_ANG, NC_DIST);

  name = "FM_grid_cont_cells_" + std::to_string(iter);
  write_txt_int_1Dvector_into_2Dmatrix_device(d_FM_cont_cells_used, name, true, NC_ANG, NC_DIST);

  name = "FM_grid_ground_mean_z_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(
    d_FM_mean_ground_height, name, true, NC_ANG, NC_DIST);

  name = "FM_grid_ground_min_z_" + std::to_string(iter);
  write_txt_int_1Dvector_into_2Dmatrix_device(
    d_FM_min_ground_height_intx100, name, true, NC_ANG, NC_DIST);

  name = "FM_grid_cont_ground_points_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(d_FM_cont_ground, name, true, NC_ANG, NC_DIST);

  name = "FM_grid_obst_min_z_" + std::to_string(iter);
  write_txt_int_1Dvector_into_2Dmatrix_device(
    d_FM_min_obst_height_intx100, name, true, NC_ANG, NC_DIST);

  name = "FM_grid_cont_obst_points_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(d_FM_cont_obst, name, true, NC_ANG, NC_DIST);
  */

  // Free data
  cudaFree(d_FM_filtered_height);
  d_FM_filtered_height = NULL;

  cudaFree(d_FM_cont_cells_used);
  d_FM_cont_cells_used = NULL;

  cudaFree(d_FM_mean_ground_height);
  d_FM_mean_ground_height = NULL;

  cudaFree(d_FM_cont_ground);
  d_FM_cont_ground = NULL;

  cudaFree(d_FM_min_ground_height_intx100);
  d_FM_min_ground_height_intx100 = NULL;

  cudaFree(d_FM_min_obst_height_intx100);
  d_FM_min_obst_height_intx100 = NULL;

  cudaFree(d_FM_cont_obst);
  d_FM_cont_obst = NULL;
}
