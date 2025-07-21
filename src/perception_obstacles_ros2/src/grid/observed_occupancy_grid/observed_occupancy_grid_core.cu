#include "perception_obstacles/grid/observed_occupancy_grid/observed_occupancy_grid.h"

/* ---------------------------------------- CODE ORGANIZATION ---------------------------------------- 
* @brief All the main code should be organized here, e.g. if there are multiple sensors or they are of different 
* types or if it is computed in cpu or gpu
*/

// ---------------------------------------- New code is in 1D old code in 2D ---------------------------------------- //

__device__ void kernel_Dempster_combination(
  float * mO, float * mF, const float m1_O, const float m1_F, const float m2_O, const float m2_F)
{
  float m1_U = 1.0 - m1_O - m1_F;
  float m2_U = 1.0 - m2_O - m2_F;

  float conflict = m1_O * m2_F + m1_F * m2_O;

  *mO = 0;
  *mF = 0;
  if (conflict < 1.0) {
    *mO = (m1_O * m2_O + m1_U * m2_O + m1_O * m2_U);
    *mF = (m1_F * m2_F + m1_U * m2_F + m1_F * m2_U);

    *mO /= (1.0 - conflict);
    *mF /= (1.0 - conflict);
  }
}

__global__ void global_fuse_grids(
  float cart1D_mO[], float cart1D_mF[], const float RB_cart1D_mO[], const float RB_cart1D_mF[],
  const bool consider_helios_right, const float Hr_cart1D_mO[], const float Hr_cart1D_mF[],
  const bool consider_helios_left, const float Hl_cart1D_mO[], const float Hl_cart1D_mF[],
  const int n_total_cells)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_total_cells) {
    cart1D_mO[idx] = 0.0;
    cart1D_mF[idx] = 0.0;

    kernel_Dempster_combination(
      &cart1D_mO[idx], &cart1D_mF[idx], cart1D_mO[idx], cart1D_mF[idx], RB_cart1D_mO[idx],
      RB_cart1D_mF[idx]);

    if (consider_helios_right) {
      kernel_Dempster_combination(
        &cart1D_mO[idx], &cart1D_mF[idx], cart1D_mO[idx], cart1D_mF[idx], Hr_cart1D_mO[idx],
        Hr_cart1D_mF[idx]);
    }
    if (consider_helios_left) {
      kernel_Dempster_combination(
        &cart1D_mO[idx], &cart1D_mF[idx], cart1D_mO[idx], cart1D_mF[idx], Hl_cart1D_mO[idx],
        Hl_cart1D_mF[idx]);
    }

    if (consider_helios_right && consider_helios_left) {
      if (
        (cart1D_mO[idx] + cart1D_mF[idx]) > 1 || cart1D_mO[idx] < 0 || cart1D_mF[idx] < 0 ||
        cart1D_mO[idx] > 1 || cart1D_mF[idx] > 1 || isnan(cart1D_mO[idx]) ||
        isinf(cart1D_mO[idx]) || isnan(cart1D_mF[idx]) || isinf(cart1D_mF[idx])) {
        printf(
          "1D: idx = %d -> [%f, %f];               RB=[%f, %f]; Hr=[%f, %f]; Hl=[%f, %f]\n", idx,
          cart1D_mO[idx], cart1D_mF[idx], RB_cart1D_mO[idx], RB_cart1D_mF[idx], Hr_cart1D_mO[idx],
          Hr_cart1D_mF[idx], Hl_cart1D_mO[idx], Hl_cart1D_mF[idx]);
      }
    } else if (consider_helios_right && consider_helios_left == false) {
      if (
        (cart1D_mO[idx] + cart1D_mF[idx]) > 1 || cart1D_mO[idx] < 0 || cart1D_mF[idx] < 0 ||
        cart1D_mO[idx] > 1 || cart1D_mF[idx] > 1 || isnan(cart1D_mO[idx]) ||
        isinf(cart1D_mO[idx]) || isnan(cart1D_mF[idx]) || isinf(cart1D_mF[idx])) {
        printf(
          "1D: idx = %d -> [%f, %f];               RB=[%f, %f]; Hr=[%f, %f]; Hl=[-, -]\n", idx,
          cart1D_mO[idx], cart1D_mF[idx], RB_cart1D_mO[idx], RB_cart1D_mF[idx], Hr_cart1D_mO[idx],
          Hr_cart1D_mF[idx]);
      }
    } else if (consider_helios_right == false && consider_helios_left) {
      if (
        (cart1D_mO[idx] + cart1D_mF[idx]) > 1 || cart1D_mO[idx] < 0 || cart1D_mF[idx] < 0 ||
        cart1D_mO[idx] > 1 || cart1D_mF[idx] > 1 || isnan(cart1D_mO[idx]) ||
        isinf(cart1D_mO[idx]) || isnan(cart1D_mF[idx]) || isinf(cart1D_mF[idx])) {
        printf(
          "1D: idx = %d -> [%f, %f];               RB=[%f, %f]; Hr=[-, -]; Hl=[%f, %f]\n", idx,
          cart1D_mO[idx], cart1D_mF[idx], RB_cart1D_mO[idx], RB_cart1D_mF[idx], Hl_cart1D_mO[idx],
          Hl_cart1D_mF[idx]);
      }
    }
  }
}

// ---------------------------------------- New code (this) is in 1D old code (DOG 2025) in 2D ---------------------------------------- //
__global__ void global_transform_to_desired_format(
  float cart_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
  float cart_mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
  float cart_pO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const float cart1D_mO[],
  const float cart1D_mF[], const float max_mO, const float max_mF)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < GRID_TYPES::NC_X * GRID_TYPES::NC_Y) {
    int i_y = -1, i_x = -1;
    GRID_UTILS_CUDA::device_ind2sub(idx, GRID_TYPES::NC_X, GRID_TYPES::NC_Y, &i_y, &i_x);

    if (i_y != -1 && i_x != -1) {
      cart_mO[i_y][i_x] = cart1D_mO[idx];
      cart_mF[i_y][i_x] = cart1D_mF[idx];

      // Max values
      cart_mO[i_y][i_x] = min(max_mO, cart_mO[i_y][i_x]);
      cart_mF[i_y][i_x] = min(max_mF, cart_mF[i_y][i_x]);

      cart_pO[i_y][i_x] = cart_mO[i_y][i_x] + 0.5 * (1 - cart_mO[i_y][i_x] - cart_mF[i_y][i_x]);

      if (
        (cart1D_mO[idx] + cart1D_mF[idx]) > 1 || cart1D_mO[idx] < 0 || cart1D_mF[idx] < 0 ||
        cart_pO[i_y][i_x] < 0 || cart_pO[i_y][i_x] > 1 || cart1D_mO[idx] > 1 ||
        cart1D_mF[idx] > 1 || isnan(cart_pO[i_y][i_x]) || isinf(cart_pO[i_y][i_x])) {
        printf(
          "1D: idx = %d -> [%f, %f]; 2D: [%d, %d] -----> [%f, %f]  pO = %f\n", idx, cart1D_mO[idx],
          cart1D_mF[idx], i_x, i_y, cart_mO[i_y][i_x], cart_mF[i_y][i_x], cart_pO[i_y][i_x]);
      }
    }
  }
}

// ---------------------------------------- CORE FUNCTION ---------------------------------------- //
void OBS_OG::compute_observed_occupancy_core(
  GRID_TYPES::OG * d_grid, const GRID_TYPES::CART_Data * h_grid_cart,
  const GRID_TYPES::CART_Data * d_grid_cart, const GRID_TYPES::POLAR_OG * RB_PolarOG,
  GRID_TYPES::POLAR_OG * d_RB_PolarOG, const AUTOPIA_RubyPlus::PointCloud * RB_pc,
  const AUTOPIA_RubyPlus::PointCloud * d_RB_pc, const bool consider_helios_right,
  const GRID_TYPES::POLAR_OG_small * Hr_PolarOG, GRID_TYPES::POLAR_OG_small * d_Hr_PolarOG,
  const AUTOPIA_Helios::PointCloud * Hr_pc, const AUTOPIA_Helios::PointCloud * d_Hr_pc,
  const bool consider_helios_left, const GRID_TYPES::POLAR_OG_small * Hl_PolarOG,
  GRID_TYPES::POLAR_OG_small * d_Hl_PolarOG, const AUTOPIA_Helios::PointCloud * Hl_pc,
  const AUTOPIA_Helios::PointCloud * d_Hl_pc, const int label_obst, DATA_times * TIME_measurements,
  const int iter)
{
  bool include_only_helios_obstacle_points = true;
  printf("parametros NO INICIALIZADOS\n");

  // -------------------- INITIALIZE -------------------- //

  TIME_measurements->time_obsOG_mallocs.Reset();

  // Required variables (two (mO, mF) for each sensor)
  float *d_cart1D_mO, *d_cart1D_mF;
  float *d_RB_cart1D_mO, *d_RB_cart1D_mF;
  float *d_Hr_cart1D_mO, *d_Hr_cart1D_mF;
  float *d_Hl_cart1D_mO, *d_Hl_cart1D_mF;

  checkCudaErrors(
    cudaMalloc((void **)&d_cart1D_mO, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
  checkCudaErrors(
    cudaMalloc((void **)&d_cart1D_mF, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));

  checkCudaErrors(
    cudaMalloc((void **)&d_RB_cart1D_mO, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
  checkCudaErrors(
    cudaMalloc((void **)&d_RB_cart1D_mF, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));

  // Initialize as unknown
  checkCudaErrors(cudaMemset(d_grid->mO, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
  checkCudaErrors(cudaMemset(d_grid->mF, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));

  checkCudaErrors(
    cudaMemset(d_cart1D_mO, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
  checkCudaErrors(
    cudaMemset(d_cart1D_mF, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));

  checkCudaErrors(
    cudaMemset(d_RB_cart1D_mO, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
  checkCudaErrors(
    cudaMemset(d_RB_cart1D_mF, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));

  if (consider_helios_right) {
    checkCudaErrors(
      cudaMalloc((void **)&d_Hr_cart1D_mO, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
    checkCudaErrors(
      cudaMalloc((void **)&d_Hr_cart1D_mF, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
  }

  if (consider_helios_left) {
    checkCudaErrors(
      cudaMalloc((void **)&d_Hl_cart1D_mO, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
    checkCudaErrors(
      cudaMalloc((void **)&d_Hl_cart1D_mF, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float)));
  }

  cudaDeviceSynchronize();  // This is actually not needed by cudaMallocs, cudaMemset is blocking
  TIME_measurements->time_obsOG_mallocs.GetElapsedTime();
  TIME_measurements->time_obsOG_mallocs.ComputeStats();

  // -------------------- COMPUTE OGs -------------------- //
  // GPU simple version

  TIME_measurements->time_obsOG_RB.Reset();
  OBS_OG::compute_observed_occupancy_polar_gpu_simple_version(
    d_RB_cart1D_mO, d_RB_cart1D_mF, d_RB_PolarOG->grid_polar_mO, d_RB_PolarOG->grid_polar_mF,
    d_RB_pc->x, d_RB_pc->y, d_RB_pc->z, d_RB_pc->intensity, d_RB_pc->label,
    RB_pc->referenced_to_odom, RB_pc->LiDAR_px, RB_pc->LiDAR_py, RB_pc->LiDAR_pz, RB_pc->n_points,
    RB_pc->n_layers, label_obst, GRID_TYPES::NC_X, GRID_TYPES::NC_Y, GRID_TYPES::RES,
    d_grid_cart->centro_x, d_grid_cart->centro_y, RB_PolarOG->NC_ANG, RB_PolarOG->NC_DIST,
    RB_PolarOG->MIN_ANG, RB_PolarOG->MIN_DIST, RB_PolarOG->RES_ANG, RB_PolarOG->RES_DIST, iter);

  cudaDeviceSynchronize();  // For the time measurements
  TIME_measurements->time_obsOG_RB.GetElapsedTime();
  TIME_measurements->time_obsOG_RB.ComputeStats();

  TIME_measurements->time_obsOG_Hr.Reset();
  if (consider_helios_right) {
    if (include_only_helios_obstacle_points == false) {
      printf("ESTO NO ESTA PROBADO\n");
      exit(1);
      OBS_OG::compute_observed_occupancy_polar_gpu_simple_version(
        d_Hr_cart1D_mO, d_Hr_cart1D_mF, d_Hr_PolarOG->grid_polar_mO, d_Hr_PolarOG->grid_polar_mF,
        d_Hr_pc->x, d_Hr_pc->y, d_Hr_pc->z, d_Hr_pc->intensity, d_Hr_pc->label,
        Hr_pc->referenced_to_odom, Hr_pc->LiDAR_px, Hr_pc->LiDAR_py, Hr_pc->LiDAR_pz,
        Hr_pc->n_points, Hr_pc->n_layers, label_obst, GRID_TYPES::NC_X, GRID_TYPES::NC_Y,
        GRID_TYPES::RES, d_grid_cart->centro_x, d_grid_cart->centro_y, Hr_PolarOG->NC_ANG,
        Hr_PolarOG->NC_DIST, Hr_PolarOG->MIN_ANG, Hr_PolarOG->MIN_DIST, Hr_PolarOG->RES_ANG,
        Hr_PolarOG->RES_DIST, iter);
    } else {
      OBS_OG::observed_occupancy_grid_raster_occupied_cart_only(
        d_Hr_cart1D_mO, d_Hr_pc->x, d_Hr_pc->y, d_Hr_pc->z, d_Hr_pc->intensity, d_Hr_pc->label,
        Hr_pc->referenced_to_odom, Hr_pc->LiDAR_px, Hr_pc->LiDAR_py, Hr_pc->LiDAR_pz,
        Hr_pc->n_points, label_obst, GRID_TYPES::NC_X, GRID_TYPES::NC_Y, h_grid_cart->MIN_X,
        h_grid_cart->MIN_Y, GRID_TYPES::RES, iter);
    }
  }
  cudaDeviceSynchronize();  // For the time measurements
  TIME_measurements->time_obsOG_Hr.GetElapsedTime();
  TIME_measurements->time_obsOG_Hr.ComputeStats();

  TIME_measurements->time_obsOG_Hl.Reset();
  if (consider_helios_left) {
    if (include_only_helios_obstacle_points == false) {
      printf("ESTO NO ESTA PROBADO\n");
      exit(1);
      OBS_OG::compute_observed_occupancy_polar_gpu_simple_version(
        d_Hl_cart1D_mO, d_Hl_cart1D_mF, d_Hl_PolarOG->grid_polar_mO, d_Hl_PolarOG->grid_polar_mF,
        d_Hl_pc->x, d_Hl_pc->y, d_Hl_pc->z, d_Hl_pc->intensity, d_Hl_pc->label,
        Hl_pc->referenced_to_odom, Hl_pc->LiDAR_px, Hl_pc->LiDAR_py, Hl_pc->LiDAR_pz,
        Hl_pc->n_points, Hl_pc->n_layers, label_obst, GRID_TYPES::NC_X, GRID_TYPES::NC_Y,
        GRID_TYPES::RES, d_grid_cart->centro_x, d_grid_cart->centro_y, Hl_PolarOG->NC_ANG,
        Hl_PolarOG->NC_DIST, Hl_PolarOG->MIN_ANG, Hl_PolarOG->MIN_DIST, Hl_PolarOG->RES_ANG,
        Hl_PolarOG->RES_DIST, iter);
    } else {
      OBS_OG::observed_occupancy_grid_raster_occupied_cart_only(
        d_Hl_cart1D_mO, d_Hl_pc->x, d_Hl_pc->y, d_Hl_pc->z, d_Hl_pc->intensity, d_Hl_pc->label,
        Hl_pc->referenced_to_odom, Hl_pc->LiDAR_px, Hl_pc->LiDAR_py, Hl_pc->LiDAR_pz,
        Hl_pc->n_points, label_obst, GRID_TYPES::NC_X, GRID_TYPES::NC_Y, h_grid_cart->MIN_X,
        h_grid_cart->MIN_Y, GRID_TYPES::RES, iter);
    }
  }
  cudaDeviceSynchronize();  // For the time measurements
  TIME_measurements->time_obsOG_Hl.GetElapsedTime();
  TIME_measurements->time_obsOG_Hl.ComputeStats();

  // -------------------- FUSE OGs -------------------- //
  TIME_measurements->time_obsOG_fusion.Reset();

  // Copy to desired cart version
  dim3 blocks_cart(1024);
  dim3 grid_cart(
    (GRID_TYPES::NC_X * GRID_TYPES::NC_Y + blocks_cart.x - 1) /
    blocks_cart.x);  // number of blocks: choose a number that ensures all data is processed

  global_fuse_grids<<<grid_cart, blocks_cart>>>(
    d_cart1D_mO, d_cart1D_mF, d_RB_cart1D_mO, d_RB_cart1D_mF, consider_helios_right, d_Hr_cart1D_mO,
    d_Hr_cart1D_mF, consider_helios_left, d_Hl_cart1D_mO, d_Hl_cart1D_mF,
    GRID_TYPES::NC_X * GRID_TYPES::NC_Y);

  cudaDeviceSynchronize();  // For the time measurements
  TIME_measurements->time_obsOG_fusion.GetElapsedTime();
  TIME_measurements->time_obsOG_fusion.ComputeStats();

  // -------------------- STORE IT IN FINAL DESIRED FORMAT -------------------- //
  TIME_measurements->time_obsOG_final_format.Reset();
  float max_mO = 0.75;
  float max_mF = 0.5;
  printf("PARAMETROS NO INICIALIZADOS\n");

  global_transform_to_desired_format<<<grid_cart, blocks_cart>>>(
    d_grid->mO, d_grid->mF, d_grid->pO, d_cart1D_mO, d_cart1D_mF, max_mO, max_mF);

  cudaDeviceSynchronize();  // For the time measurements
  TIME_measurements->time_obsOG_final_format.GetElapsedTime();
  TIME_measurements->time_obsOG_final_format.ComputeStats();

  // Frees
  TIME_measurements->time_obsOG_frees.Reset();
  checkCudaErrors(cudaFree(d_cart1D_mO));
  checkCudaErrors(cudaFree(d_cart1D_mF));

  checkCudaErrors(cudaFree(d_RB_cart1D_mO));
  checkCudaErrors(cudaFree(d_RB_cart1D_mF));

  if (consider_helios_right) {
    checkCudaErrors(cudaFree(d_Hr_cart1D_mO));
    checkCudaErrors(cudaFree(d_Hr_cart1D_mF));
  }

  if (consider_helios_left) {
    checkCudaErrors(cudaFree(d_Hl_cart1D_mO));
    checkCudaErrors(cudaFree(d_Hl_cart1D_mF));
  }

  cudaDeviceSynchronize();  // This is the only one "needed" (not now because of the others, but in case the others were not used)
  TIME_measurements->time_obsOG_frees.GetElapsedTime();
  TIME_measurements->time_obsOG_frees.ComputeStats();
}