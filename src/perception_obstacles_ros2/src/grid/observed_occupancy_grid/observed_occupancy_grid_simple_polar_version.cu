#include "perception_obstacles/grid/observed_occupancy_grid/observed_occupancy_grid.h"

/* 
   --------------------------------------------------------------------------------------------
   ---------------------------------------- DISCLAIMER ----------------------------------------
   --------------------------------------------------------------------------------------------

   This code is not intended to be theoretically solid, but just a FAST and GENERIC strategy
   i.e. there are many thecnical decisions (e.g. average mean, fix values, etc.)
*/

// ---------------------------------------- INDEXES ---------------------------------------- //
inline __device__ void device_polar_compute_1D_angle_dist(
  int * i_1D, const int i_angle, const int i_dist, const int NC_DIST)
{
  *i_1D =
    i_angle * NC_DIST +
    i_dist;  // This order is intentional, because we are going to access the cells increasing the i_d index one by one
}
inline __device__ void device_increment_1D_by_i_d(int * i_1D, const int i_dist) { *i_1D += i_dist; }

inline __device__ void device_polar_compute_2D_angle_dist(
  int * i_angle, int * i_dist, const int i_1D, const int NC_DIST)
{
  *i_angle = i_1D / NC_DIST;
  *i_dist = i_1D % NC_DIST;
}

inline __device__ void device_polar_calcular_centro_angulo_celda_polar(
  float * ang, const int i_a, const float MIN_ANG, const float RES_ANG)
{
  *ang = (float)i_a * RES_ANG + MIN_ANG + RES_ANG / 2.0;
}

inline __device__ void device_polar_calcular_centro_distancia_celda_polar(
  float * dist, const int i_d, const float MIN_DIST, const float RES_DIST)
{
  *dist = (float)i_d * RES_DIST + MIN_DIST + RES_DIST / 2.0;
}

inline __device__ void device_polar_calcular_indice_celda_polar(
  int * i_a, int * i_d, bool * valid, const float ang, const float dist, const int NC_ANG,
  const int NC_DIST, const float MIN_ANG, const float MIN_DIST, const float RES_ANG,
  const float RES_DIST)
{
  *i_a = ceil(round(((ang - MIN_ANG) / RES_ANG) * 1e6) / 1e6) - 1;
  *i_d = ceil(round(((dist - MIN_DIST) / RES_DIST) * 1e6) / 1e6) - 1;

  *valid = true;
  if (*i_a < 0 || *i_a >= NC_ANG) {
    *i_a = -1;
    *valid = false;
  }
  if (*i_d < 0 || *i_d >= NC_DIST) {
    *i_d = -1;
    *valid = false;
  }

  // if (*i_a >= (NC_ANG - 1) || *i_a <= 0) {
  //   printf(
  //     "i_a = %d = (%f - %f) / %f - 1;\ni_d = %d = (%f - %f) / %f - 1;\n", *i_a, ang * 180 / M_PI,
  //     MIN_ANG * 180 / M_PI, RES_ANG * 180 / M_PI, *i_d, dist, MIN_DIST, RES_DIST);
  // }
}
void OBS_OG::polar_compute_1D_angle_dist(
  int * i_1D, const int i_angle, const int i_dist, const int NC_DIST)
{
  *i_1D = i_angle * NC_DIST + i_dist;
}
inline void polar_calcular_indice_celda_polar(
  int * i_a, int * i_d, const float ang, const float dist, const int NC_ANG, const int NC_DIST,
  const float MIN_ANG, const float MIN_DIST, const float RES_ANG, const float RES_DIST)
{
  *i_a = ceil(round(((ang - MIN_ANG) / RES_ANG) * 1e6) / 1e6) - 1;
  *i_d = ceil(round(((dist - MIN_DIST) / RES_DIST) * 1e6) / 1e6) - 1;

  if (*i_a < 0 || *i_a >= NC_ANG || *i_d < 0 || *i_d >= NC_DIST) {
    *i_a = -1;
    *i_d = -1;
  }
}

// ---------------------------------------- UTILS ---------------------------------------- //
inline __device__ void device_interpolacion_lineal(
  float * y, const float x, const float x1, const float x2, const float y1, const float y2)
{
  if (x2 == x1) {
    *y = y1;
  } else {
    *y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
  }
}

__device__ void device_compute_polar_indexes_from_xy(
  int * i_a, int * i_d, bool * valid, const float x, const float y, const int NC_ANG,
  const int NC_DIST, const float MIN_ANG, const float MIN_DIST, const float RES_ANG,
  const float RES_DIST)
{
  float dist = sqrt(x * x + y * y);
  float ang = atan2(y, x);

  device_polar_calcular_indice_celda_polar(
    i_a, i_d, valid, ang, dist, NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

  // if (*i_a >= (NC_ANG - 1) || *i_a <= 0) {
  //   printf("[%f, %f] -> [%f, %f] -> [%d, %d]\n", x, y, ang * 180 / M_PI, dist, *i_a, *i_d);
  // }
}

// ---------------------------------------- RASTERIZE ---------------------------------------- //
__global__ void global_rasterize_obstacle_points(
  float grid_sum_occ[], const float pc_x[], const float pc_y[], const float pc_z[],
  const int pc_label[], const int n_points, const bool pc_referenced_to_odom, const float lidar_px,
  const float lidar_py, const float lidar_pz, const int label_obst, const float v_impact,
  const float v_behind, const float max_obst_height, const float MIN_ANG, const float MIN_DIST,
  const float RES_ANG, const float RES_DIST, const int NC_ANG, const int NC_DIST)
{
  // Get index from the division of blocks and grids
  int idx_sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_sample < n_points) {
    // Only obstacle points
    if (pc_label[idx_sample] == label_obst) {
      float x = pc_x[idx_sample];
      float y = pc_y[idx_sample];
      float z = pc_z[idx_sample];
      if (pc_referenced_to_odom) {
        x -= lidar_px;
        y -= lidar_py;
        z -= lidar_pz;
      }

      // Discard high points
      if (z + lidar_pz > max_obst_height) {
        return;
      }

      // Compute polar index
      int i_a = -1, i_d = -1;
      bool valid = true;
      device_compute_polar_indexes_from_xy(
        &i_a, &i_d, &valid, x, y, NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

      if (valid == false) {
        return;
      }

      // Acumulate
      int idx1D = -1;
      device_polar_compute_1D_angle_dist(&idx1D, i_a, i_d, NC_DIST);
      atomicAdd(&grid_sum_occ[idx1D], v_impact);

      // Acumulate before
      if (i_d - 1 >= 0) {
        // device_polar_compute_1D_angle_dist(&idx1D, i_a, i_d - 1, NC_DIST);
        atomicAdd(&grid_sum_occ[idx1D - 1], v_behind);
      }

      // Acumulate behind
      if (i_d + 1 < NC_DIST) {
        // device_polar_compute_1D_angle_dist(&idx1D, i_a, i_d + 1, NC_DIST);
        atomicAdd(&grid_sum_occ[idx1D + 1], v_behind);
      }

      // Acumulate behind
      if (i_d + 2 < NC_DIST) {
        atomicAdd(&grid_sum_occ[idx1D + 2], v_behind / 3.0);
      }
    }
  }
}

__global__ void global_rasterize_free_space(
  float grid_cont_traversed_free_beams[], const float grid_sum_occ[], const float pc_x[],
  const float pc_y[], const float pc_z[], const int pc_label[], const float pc_intensity[],
  const int n_points, const bool pc_referenced_to_odom, const float lidar_px, const float lidar_py,
  const float lidar_pz, const float max_free_height, const float MIN_ANG, const float MIN_DIST,
  const float RES_ANG, const float RES_DIST, const int NC_ANG, const int NC_DIST)
{
  // Get index from the division of blocks and grids
  int idx_sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_sample < n_points) {
    float x = pc_x[idx_sample];
    float y = pc_y[idx_sample];
    float z = pc_z[idx_sample];
    if (pc_referenced_to_odom) {
      x -= lidar_px;
      y -= lidar_py;
      z -= lidar_pz;
    }

    // +++ Valid beams for free space +++ //

    // Only points with valid returns
    if (pc_intensity[idx_sample] <= 0) {
      return;
    }

    // If the lidar is above the maximum height for free and the impact to -> return, there is no chance of free space
    if (lidar_pz > max_free_height && (z + lidar_pz) > max_free_height) {
      return;
    }

    // +++ Compute polar index impact +++ //
    int ia_impact = -1, id_impact = -1;
    bool valid_impact = true;
    device_compute_polar_indexes_from_xy(
      &ia_impact, &id_impact, &valid_impact, x, y, NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG,
      RES_DIST);

    if (ia_impact == -1) {
      return;
    }

    // +++ Height equation +++ //
    float cell_height = 0;
    float height_at_id0 = 0.0;

    // Aproximate height at i_d = 0
    device_interpolacion_lineal(
      &height_at_id0, MIN_DIST, 0.0f, sqrt(x * x + y * y), lidar_pz, z + lidar_pz);

    // Line eq
    //   height_at_id0 = m * 0 + n;
    //   height_impact = m * id_impact + n;

    float m_height = (z + lidar_pz - height_at_id0) / ((float)id_impact);
    // float n_height = height_at_id0;

    // Fill space until impact
    int idx1D_id0 =
      ia_impact * NC_DIST;  // device_polar_compute_1D_angle_dist(&idx1D, ia_impact, i_d, NC_DIST);

    for (int i_d = 0; i_d < min(id_impact, NC_DIST); i_d++) {
      // If there are obstacle points break (we cannot guarantee free behind obstacles)
      if (grid_sum_occ[idx1D_id0 + i_d] > 0) {
        return;
      }

      // Check height limit
      cell_height = i_d * m_height + height_at_id0;

      // If it is within height limit -> Acumulate
      if (cell_height < max_free_height) {
        atomicAdd(&grid_cont_traversed_free_beams[idx1D_id0 + i_d], 1.0f);
      }
    }
  }
}

// ---------------------------------------- POLAR MASES ---------------------------------------- //
__global__ void global_convert_into_masses(
  float grid_polar_mO[], float grid_polar_mF[], const float grid_sum_occ[],
  const float grid_cont_traversed_free_beams[], const int NC_ANG, const int NC_DIST,
  const float num_beams_max_free)
{
  // Get index from the division of blocks and grids
  int idx_cell = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_cell < NC_ANG * NC_DIST) {
    // Occupied is copied
    grid_polar_mO[idx_cell] = min(1.0, grid_sum_occ[idx_cell]);

    // Free is copied by checking occupied (actually this should have already being managed)
    if (grid_polar_mO[idx_cell] == 0) {
      grid_polar_mF[idx_cell] =
        min(1.0, grid_cont_traversed_free_beams[idx_cell] / num_beams_max_free);
    } else {
      grid_polar_mF[idx_cell] = 0;
    }

    // Check for errors
    // if (
    //   grid_polar_mO[idx_cell] < 0 || grid_polar_mF[idx_cell] < 0 || grid_polar_mO[idx_cell] > 1 ||
    //   grid_polar_mF[idx_cell] > 1 || (grid_polar_mO[idx_cell] + grid_polar_mF[idx_cell]) > 1 ||
    //   isnan(grid_polar_mO[idx_cell]) || isnan(grid_polar_mF[idx_cell]) ||
    //   isinf(grid_polar_mO[idx_cell]) || isinf(grid_polar_mF[idx_cell])) {
    //   printf(
    //     "global_convert_into_masses - Error mO = %f; mf = %f;\n", grid_polar_mO[idx_cell],
    //     grid_polar_mF[idx_cell]);
    // }
  }
}

// ---------------------------------------- CARTESIAN ---------------------------------------- //

__device__ void device_aux_function_accumulate_masses(
  float * mO, float * mF, float * cont, const float grid_polar_mO[], const float grid_polar_mF[],
  const float x, const float y, const int NC_ANG, const int NC_DIST, const float MIN_ANG,
  const float MIN_DIST, const float RES_ANG, const float RES_DIST)
{
  int i_a = -1, i_d = -1, idx1D = -1;
  bool valid = true;
  device_compute_polar_indexes_from_xy(
    &i_a, &i_d, &valid, x, y, NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

  // printf("centro xy = [%f, %f] -> ia,id = [%d, %d]\n", x, y, i_a, i_d);

  if (valid == true) {
    device_polar_compute_1D_angle_dist(&idx1D, i_a, i_d, NC_DIST);

    // If there is info
    *mO = *mO + grid_polar_mO[idx1D];
    *mF = *mF + grid_polar_mF[idx1D];
    if (grid_polar_mO[idx1D] > 0 || grid_polar_mF[idx1D] > 0) {
      *cont = *cont + 1.0;
    }
  }
}

__global__ void global_transform_polar_grid_to_cartesian_grid(
  float grid_cart_mO[], float grid_cart_mF[], const float grid_polar_mO[],
  const float grid_polar_mF[], const float lidar_pos_2_refsys_x, const float lidar_pos_2_refsys_y,
  bool favour_occ, const int NC_X, const int NC_Y, const float CART_RES, const float centro_x[],
  const float centro_y[], const int NC_ANG, const int NC_DIST, const float MIN_ANG,
  const float MIN_DIST, const float RES_ANG, const float RES_DIST)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < NC_X * NC_Y) {
    int i_y = -1, i_x = -1;
    GRID_UTILS_CUDA::device_ind2sub(idx, NC_X, NC_Y, &i_y, &i_x);

    // Correct LiDAR position
    float cx = centro_x[i_x] - lidar_pos_2_refsys_x;
    float cy = centro_y[i_y] - lidar_pos_2_refsys_y;

    float mO = 0.0;
    float mF = 0.0;
    float cont = 0.0;

    device_aux_function_accumulate_masses(
      &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx, cy, NC_ANG, NC_DIST, MIN_ANG, MIN_DIST,
      RES_ANG, RES_DIST);

    // We should randomly sample num_samples_conversion at this distance
    int num_samples_conversion =
      ceil(CART_RES * CART_RES / (RES_ANG * RES_DIST) / sqrt(cx * cx + cy * cy));

    // But to avoid computational costs instead of random sampling a varying number...
    // we are gonna sample 9 points (including the center) if necessary
    if (num_samples_conversion > 2) {
      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx + CART_RES * 0.3, cy + CART_RES * 0.3,
        NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx + CART_RES * 0.3, cy - CART_RES * 0.3,
        NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx - CART_RES * 0.3, cy - CART_RES * 0.3,
        NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx - CART_RES * 0.3, cy + CART_RES * 0.3,
        NC_ANG, NC_DIST, MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);
    }
    if (num_samples_conversion > 5) {
      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx, cy + CART_RES * 0.35, NC_ANG, NC_DIST,
        MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx, cy - CART_RES * 0.35, NC_ANG, NC_DIST,
        MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx + CART_RES * 0.35, cy, NC_ANG, NC_DIST,
        MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);

      device_aux_function_accumulate_masses(
        &mO, &mF, &cont, grid_polar_mO, grid_polar_mF, cx - CART_RES * 0.35, cy, NC_ANG, NC_DIST,
        MIN_ANG, MIN_DIST, RES_ANG, RES_DIST);
    }

    // If there is info
    if (cont > 0) {
      mO /= cont;
      mF = min(1.0 - mO, mF / cont);
      if (favour_occ && mO > 0) {
        mF = 0.0;
      }

      // Check for errors
      // if (
      //   mO < 0 || mF < 0 || mO > 1 || mF > 1 || (mO + mF) > 1 || isnan(mO) || isnan(mF) ||
      //   isinf(mO) || isinf(mF)) {
      //   printf("Error mO = %f; mf = %f;\n", mO, mF);
      // }

      // Store it
      grid_cart_mO[idx] = mO;
      grid_cart_mF[idx] = mF;
    }
  }
}

// ---------------------------------------- MAIN CODE ---------------------------------------- //
void OBS_OG::compute_observed_occupancy_polar_gpu_simple_version(
  float d_grid_cart_data_mO[], float d_grid_cart_data_mF[], float d_grid_polar_mO[],
  float d_grid_polar_mF[], const float d_pc_x[], const float d_pc_y[], const float d_pc_z[],
  const float d_pc_intensity[], const int d_pc_label[], const bool pc_referenced_to_odom,
  const float lidar_px, const float lidar_py, const float lidar_pz, const int n_points,
  const int n_layers, const int label_obst, const int NC_X, const int NC_Y, const float CART_RES,
  const float centros_x[], const float centros_y[], const int NC_ANG, const int NC_DIST,
  const float MIN_ANG, const float MIN_DIST, const float RES_ANG, const float RES_DIST,
  const int iter)
{
  float max_obst_height = 2;  // TODO
  float max_free_height = 0.5;
  float v_impact = 0.5;
  float v_behind = 0.25;
  float num_beams_max_free = 20;
  bool polar2cart_favour_occ = false;
  printf("PARAMETROS NO INICIALIZADOS\n");

  bool measure_time = false;
  ChronoTimer time_raster_pc;
  ChronoTimer time_raster_beams;
  ChronoTimer time_masses;
  ChronoTimer time_polar2cart;
  if (measure_time) {
    cudaDeviceSynchronize();
  }

  // Initialize data (we could do this only once in the beginning of the code, but this is harder to reuse among projects)
  float * d_grid_sum_occ;
  float * d_grid_cont_traversed_free_beams;

  // Dimensions for parallelization
  dim3 blocks_beams(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  dim3 grids_beams((n_points + blocks_beams.x - 1) / blocks_beams.x);

  dim3 blocks_polar(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  dim3 grids_polar(
    (NC_ANG * NC_DIST + blocks_polar.x - 1) /
    blocks_polar.x);  // number of blocks: choose a number that ensures all data is processed

  dim3 blocks_cart(1024);
  dim3 grid_cart(
    (NC_X * NC_Y + blocks_cart.x - 1) /
    blocks_cart.x);  // number of blocks: choose a number that ensures all data is processed

  // ------------------------- Rasterize points ------------------------- //
  time_raster_pc.Reset();
  checkCudaErrors(cudaMalloc((void **)&d_grid_sum_occ, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(cudaMemset(d_grid_sum_occ, 0.0, NC_ANG * NC_DIST * sizeof(float)));
  global_rasterize_obstacle_points<<<grids_beams, blocks_beams>>>(
    d_grid_sum_occ, d_pc_x, d_pc_y, d_pc_z, d_pc_label, n_points, pc_referenced_to_odom, lidar_px,
    lidar_py, lidar_pz, label_obst, v_impact, v_behind, max_obst_height, MIN_ANG, MIN_DIST, RES_ANG,
    RES_DIST, NC_ANG, NC_DIST);
  if (measure_time) {
    cudaDeviceSynchronize();
    time_raster_pc.GetElapsedTime();
    printf("   Observation - Raster PC - measured time = %fms\n", time_raster_pc.measured_time);
  }

  // ------------------------- Rasterize free ------------------------- //
  time_raster_beams.Reset();
  checkCudaErrors(
    cudaMalloc((void **)&d_grid_cont_traversed_free_beams, NC_ANG * NC_DIST * sizeof(float)));
  checkCudaErrors(
    cudaMemset(d_grid_cont_traversed_free_beams, 0.0, NC_ANG * NC_DIST * sizeof(float)));
  global_rasterize_free_space<<<grids_beams, blocks_beams>>>(
    d_grid_cont_traversed_free_beams, d_grid_sum_occ, d_pc_x, d_pc_y, d_pc_z, d_pc_label,
    d_pc_intensity, n_points, pc_referenced_to_odom, lidar_px, lidar_py, lidar_pz, max_free_height,
    MIN_ANG, MIN_DIST, RES_ANG, RES_DIST, NC_ANG, NC_DIST);
  if (measure_time) {
    cudaDeviceSynchronize();
    time_raster_beams.GetElapsedTime();
    printf("   Observation - Raster beams - measured time = %f\n", time_raster_beams.measured_time);
  }

  //------------------------- Transform raster into masses ------------------------- //
  time_masses.Reset();
  global_convert_into_masses<<<grids_polar, blocks_polar>>>(
    d_grid_polar_mO, d_grid_polar_mF, d_grid_sum_occ, d_grid_cont_traversed_free_beams, NC_ANG,
    NC_DIST, num_beams_max_free);
  if (measure_time) {
    cudaDeviceSynchronize();
    time_masses.GetElapsedTime();
    printf("   Observation - Compute Masses - measured time = %fms\n", time_masses.measured_time);
  }

  // ------------------------- Transform to cartesian grid ------------------------- //
  time_polar2cart.Reset();
  // checkCudaErrors(cudaMemset(d_grid_cart_data_mO, 0.0, NC_X * NC_Y * sizeof(float))); // Esto realmente no hace falta porque se rellenan todas las celdas
  // checkCudaErrors(cudaMemset(d_grid_cart_data_mF, 0.0, NC_X * NC_Y * sizeof(float))); // Esto realmente no hace falta porque se rellenan todas las celdas
  global_transform_polar_grid_to_cartesian_grid<<<grid_cart, blocks_cart>>>(
    d_grid_cart_data_mO, d_grid_cart_data_mF, d_grid_polar_mO, d_grid_polar_mF, lidar_px, lidar_py,
    polar2cart_favour_occ, NC_X, NC_Y, CART_RES, centros_x, centros_y, NC_ANG, NC_DIST, MIN_ANG,
    MIN_DIST, RES_ANG, RES_DIST);

  if (measure_time) {
    cudaDeviceSynchronize();
    time_polar2cart.GetElapsedTime();
    printf("   Observation - Polar 2 Cart - measured time = %fms\n", time_polar2cart.measured_time);
  }

  // Synchronize
  cudaDeviceSynchronize();  // Not sure if I need to do this before cudaFree and leaveing the function

  // Free data
  cudaFree(d_grid_sum_occ);
  d_grid_sum_occ = NULL;
  cudaFree(d_grid_cont_traversed_free_beams);
  d_grid_cont_traversed_free_beams = NULL;
}
