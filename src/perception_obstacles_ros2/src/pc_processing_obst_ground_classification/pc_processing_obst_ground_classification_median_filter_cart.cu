#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification.h"

// ---------------------------------------- INDEXES ---------------------------------------- //
inline __device__ int device_FM_sub2ind(
  const int i_y, const int i_x, const int NC_X, const int NC_Y)
{
  return i_x * NC_Y + i_y;
}

inline __device__ void device_FM_ind2sub(
  int * i_x, int * i_y, const int idx, const int NC_X, const int NC_Y)
{
  *i_x = idx / NC_Y;
  *i_y = idx % NC_Y;
}

inline void FM_compute_center_cell(
  float * centro_celda_x, float * centro_celda_y, const int i_x, const int i_y, const int NC_X,
  const int NC_Y, const float MIN_X, const float MIN_Y, const float RES)
{
  *centro_celda_x = (float)i_x * RES + MIN_X + RES / 2.0;
  *centro_celda_y = (NC_Y - (float)i_y) * RES + MIN_Y - RES / 2.0;
}

inline void FM_compute_index_cell(
  int * i_x, int * i_y, const float x, const float y, const int NC_X, const int NC_Y,
  const float MIN_X, const float MIN_Y, const float RES)
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

inline __device__ void device_FM_compute_index_cell(
  int * i_x, int * i_y, const float x, const float y, const int NC_X, const int NC_Y,
  const float MIN_X, const float MIN_Y, const float RES)
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

// ---------------------------------------- RASTERIZE POINTS ---------------------------------------- //
__global__ void gpu_rasterize_points_in_grid(
  float mean_ground_height[], int min_ground_height_intx100[], float cont_ground[],
  int min_obst_height_intx100[], float cont_obst[], const float x[], const float y[],
  const float z[], const int label[], const int n_points, const int label_obst,
  const int label_ground, const int NC_X, const int NC_Y, const float MIN_X, const float MIN_Y,
  const float RES)
{
  // Get index from the division of blocks and grids
  int idx_sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_sample < n_points) {
    if (label[idx_sample] == label_ground || label[idx_sample] == label_obst) {
      int i_x, i_y;
      device_FM_compute_index_cell(
        &i_x, &i_y, x[idx_sample], y[idx_sample], NC_X, NC_Y, MIN_X, MIN_Y, RES);

      if (i_x == -1) {
        return;
      }

      int idx1D = device_FM_sub2ind(i_x, i_y, NC_X, NC_Y);

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

__global__ void global_solve_mean_1Dvector(
  float data_sum[], const float cont[], const int vector_size)
{
  int idx1D_cell = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx1D_cell < vector_size) {
    if (cont[idx1D_cell] > 0) {
      data_sum[idx1D_cell] /= cont[idx1D_cell];
    }
  }
}

__global__ void fill_height_ground_under_ego(
  float mean_ground_height[], int min_obst_height_intx100[], float cont_ground[],
  const int ego_min_ix, const int ego_max_ix, const int ego_min_iy, const int ego_max_iy,
  const int NC_X, const int NC_Y)
{
  int idx1D_cell = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx1D_cell < NC_X * NC_Y) {
    int i_y = -1, i_x = -1;
    device_FM_ind2sub(&i_x, &i_y, idx1D_cell, NC_X, NC_Y);

    if (i_x > ego_min_ix && i_x < ego_max_ix && i_y > ego_min_iy && i_y > ego_max_iy) {
      mean_ground_height[idx1D_cell] = 0;
      min_obst_height_intx100[idx1D_cell] = 0;
      cont_ground[idx1D_cell] = 0;
    }
  }
}

__global__ void global_initialize_int_1Dvector(
  int array_int[], const int value, const int vector_size)
{
  int idx1D_cell = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx1D_cell < vector_size) {
    array_int[idx1D_cell] = value;
  }
}

// ---------------------------------------- MEDIAN FILTER ---------------------------------------- //
__global__ void global_filtro_mediana_suelo(
  float filtered_height[], int cont_cells_used[], const float mean_ground_height[],
  const int min_height_ground_intx100[], const float cont_ground[],
  const int min_height_obst_intx100[], const float cont_obst[], const int option_mean1_min2_both3,
  const int NC_X, const int NC_Y, const float MIN_X, const float MIN_Y, const float RES)
{
  int idx1D_cell = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = -1, i_x = -1;
  device_FM_ind2sub(&i_x, &i_y, idx1D_cell, NC_X, NC_Y);

  if (i_x >= 0 && i_x < NC_X && i_y >= 0 && i_y < NC_Y) {
    float data[9];  // kernel de 3x3 (aqui es donde se va el coste asi que vamos a dejarlo fijo)
    int sk = 1;
    // float data[25];  // kernel de 5x5 (aqui es donde se va el coste asi que vamos a dejarlo fijo)
    // int sk = 2;

    int idx2_1D, cont = 0;

    // Rellenamos el contenido
    for (int i2_y = i_y - sk; i2_y <= i_y + sk; i2_y++) {
      if (i2_y >= 0 && i2_y < NC_Y) {
        for (int i2_x = i_x - sk; i2_x <= i_x + sk; i2_x++) {
          if (i2_x >= 0 && i2_x < NC_X) {
            idx2_1D = device_FM_sub2ind(i2_y, i2_x, NC_X, NC_Y);

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
      for (int i_xata = 0; i_xata <= i_medio; i_xata++) {
        // En principio el valor esta bien colocado
        i_min_valor_local = i_xata;

        // Ahora empezamos a comparar con el siguiente valor
        for (int j = i_xata + 1; j < cont; j++) {
          // Si el valor de esta nueva celda es menor que el guardado -> actualizamos
          if (data[j] < data[i_min_valor_local]) {
            i_min_valor_local = j;
          }
        }

        // Guardamos el valor menor
        valor_temp_swap = data[i_xata];
        data[i_xata] = data[i_min_valor_local];
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
  const int label_obstacle, const int label_ground, const int label_noise, const int NC_X,
  const int NC_Y, const float MIN_X, const float MIN_Y, const float RES)
{
  // Get index from the division of blocks and grids
  int idx_sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_sample < n_points) {
    if (label[idx_sample] != label_noise) {
      int i_x, i_y;
      device_FM_compute_index_cell(
        &i_x, &i_y, x[idx_sample], y[idx_sample], NC_X, NC_Y, MIN_X, MIN_Y, RES);

      if (i_x == -1) {
        return;
      }

      int idx1D = device_FM_sub2ind(i_x, i_y, NC_X, NC_Y);

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

// ---------------------------------------- CORE FUNCTION ---------------------------------------- //
void OBST_GROUND::reclassify_points_with_median_filter_cart(
  int d_label[], const float d_x[], const float d_y[], const float d_z[], const int n_total_points,
  const float threshold_diff_ground, const int label_obstacle, const int label_ground,
  const int label_noise, const int iter)
{
  // const static int NC_X = 512;
  // const static int NC_Y = 512;
  // const static float RES = 0.5;
  // const static int NC_X = 256;
  // const static int NC_Y = 256;
  // const static float RES = 0.5;
  const static int NC_X = 192;
  const static int NC_Y = 192;
  const static float RES = 1;
  const static float MIN_X = -(float)NC_X * RES / 2.0;
  const static float MIN_Y = -(float)NC_Y * RES / 2.0;

  int option_mean1_min2_both3 = 3;
  printf("PARAMETROS NO INICIALIZADOS\n");

  // Declare, reserve and initialize (potentially this could be done once... but lets try this way so it is easily resusable)
  float *d_FM_filtered_height, *d_FM_mean_ground_height, *d_FM_cont_ground, *d_FM_cont_obst;
  int *d_FM_cont_cells_used, *d_FM_min_ground_height_intx100, *d_FM_min_obst_height_intx100;

  checkCudaErrors(cudaMalloc((void **)&d_FM_filtered_height, NC_Y * NC_X * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_FM_cont_cells_used, NC_Y * NC_X * sizeof(int)));

  checkCudaErrors(cudaMalloc((void **)&d_FM_mean_ground_height, NC_Y * NC_X * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_FM_cont_ground, NC_Y * NC_X * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_FM_min_ground_height_intx100, NC_Y * NC_X * sizeof(int)));

  checkCudaErrors(cudaMalloc((void **)&d_FM_cont_obst, NC_Y * NC_X * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_FM_min_obst_height_intx100, NC_Y * NC_X * sizeof(int)));

  // parallelization dimensions
  static dim3 blocks_beams(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grids_beams(
    (n_total_points + blocks_beams.x - 1) /
    blocks_beams.x);  // number of blocks: choose a number that ensures all data is processed

  static dim3 block_FM(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grid_FM(
    (NC_X * NC_Y + block_FM.x - 1) /
    block_FM.x);  // number of blocks: choose a number that ensures all data is processed

  // Initialize
  checkCudaErrors(cudaMemset(d_FM_filtered_height, 0.0, NC_Y * NC_X * sizeof(float)));
  checkCudaErrors(cudaMemset(d_FM_cont_cells_used, 0, NC_Y * NC_X * sizeof(int)));

  checkCudaErrors(cudaMemset(d_FM_mean_ground_height, 0.0, NC_Y * NC_X * sizeof(float)));
  checkCudaErrors(cudaMemset(d_FM_cont_ground, 0.0, NC_Y * NC_X * sizeof(float)));
  global_initialize_int_1Dvector<<<grid_FM, block_FM>>>(
    d_FM_min_ground_height_intx100, 10000, NC_Y * NC_X);

  checkCudaErrors(cudaMemset(d_FM_cont_obst, 0.0, NC_Y * NC_X * sizeof(float)));
  global_initialize_int_1Dvector<<<grid_FM, block_FM>>>(
    d_FM_min_obst_height_intx100, 10000, NC_Y * NC_X);

  // Rasterize points
  gpu_rasterize_points_in_grid<<<grids_beams, blocks_beams>>>(
    d_FM_mean_ground_height, d_FM_min_ground_height_intx100, d_FM_cont_ground,
    d_FM_min_obst_height_intx100, d_FM_cont_obst, d_x, d_y, d_z, d_label, n_total_points,
    label_obstacle, label_ground, NC_X, NC_Y, MIN_X, MIN_Y, RES);

  global_solve_mean_1Dvector<<<grid_FM, block_FM>>>(
    d_FM_mean_ground_height, d_FM_cont_ground, NC_Y * NC_X);

  float ego_min_x = -2.0, ego_max_x = 3.5, ego_min_y = -1.0, ego_max_y = 1.0;
  int ego_min_ix, ego_max_ix, ego_min_iy, ego_max_iy;
  printf("PARAMETROS NO INICIALIZADOS\n");

  FM_compute_index_cell(
    &ego_min_ix, &ego_min_iy, ego_min_x, ego_min_y, NC_X, NC_Y, MIN_X, MIN_Y, RES);
  FM_compute_index_cell(
    &ego_max_ix, &ego_max_iy, ego_max_x, ego_max_y, NC_X, NC_Y, MIN_X, MIN_Y, RES);

  fill_height_ground_under_ego<<<grid_FM, block_FM>>>(
    d_FM_mean_ground_height, d_FM_min_obst_height_intx100, d_FM_cont_ground, ego_min_ix, ego_max_ix,
    ego_min_iy, ego_max_iy, NC_X, NC_Y);

  // Approximate ground height with median filter
  global_filtro_mediana_suelo<<<grid_FM, block_FM>>>(
    d_FM_filtered_height, d_FM_cont_cells_used, d_FM_mean_ground_height,
    d_FM_min_ground_height_intx100, d_FM_cont_ground, d_FM_min_obst_height_intx100, d_FM_cont_obst,
    option_mean1_min2_both3, NC_X, NC_Y, MIN_X, MIN_Y, RES);

  // Recompute labels
  global_reclassify_points_using_median_filter<<<grids_beams, blocks_beams>>>(
    d_label, d_FM_filtered_height, d_FM_cont_cells_used, d_x, d_y, d_z, n_total_points,
    threshold_diff_ground, label_obstacle, label_ground, label_noise, NC_X, NC_Y, MIN_X, MIN_Y,
    RES);

  // Synchronize
  cudaDeviceSynchronize();

  // Debug

  /*
  std::string name;

  name = "FM_filtered_z_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(d_FM_filtered_height, name, true, NC_X, NC_Y);

  name = "FM_grid_cont_cells_" + std::to_string(iter);
  write_txt_int_1Dvector_into_2Dmatrix_device(d_FM_cont_cells_used, name, true, NC_X, NC_Y);

  name = "FM_grid_ground_mean_z_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(d_FM_mean_ground_height, name, true, NC_X, NC_Y);

  name = "FM_grid_ground_min_z_" + std::to_string(iter);
  write_txt_int_1Dvector_into_2Dmatrix_device(
    d_FM_min_ground_height_intx100, name, true, NC_X, NC_Y);

  name = "FM_grid_cont_ground_points_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(d_FM_cont_ground, name, true, NC_X, NC_Y);

  name = "FM_grid_obst_min_z_" + std::to_string(iter);
  write_txt_int_1Dvector_into_2Dmatrix_device(d_FM_min_obst_height_intx100, name, true, NC_X, NC_Y);

  name = "FM_grid_cont_obst_points_" + std::to_string(iter);
  write_txt_float_1Dvector_into_2Dmatrix_device(d_FM_cont_obst, name, true, NC_X, NC_Y);
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
