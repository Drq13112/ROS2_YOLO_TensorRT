#include "perception_obstacles/grid/observed_occupancy_grid/observed_occupancy_grid.h"

/* 
   --------------------------------------------------------------------------------------------
   ---------------------------------------- DISCLAIMER ----------------------------------------
   --------------------------------------------------------------------------------------------

   This code is not intended to be theoretically solid, but just a FAST and GENERIC strategy
   i.e. there are many thecnical decisions (e.g. average mean, fix values, etc.)
*/

// ---------------------------------------- RASTERIZE ---------------------------------------- //
__global__ void global_sum_obstacle_points(
  float grid_cont[], const float pc_x[], const float pc_y[], const float pc_z[],
  const int pc_label[], const int n_points, const bool LiDAR_referenced_to_odom,
  const float lidar_px, const float lidar_py, const float lidar_pz, const int label_obst,
  const float max_obst_height, const int NC_X, const int NC_Y, const float MIN_X, const float MIN_Y,
  const float RES)
{
  // Get index from the division of blocks and grids
  int idx_sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_sample < n_points) {
    // Only obstacle points
    if (pc_label[idx_sample] == label_obst) {
      float x = pc_x[idx_sample];
      float y = pc_y[idx_sample];
      float z = pc_z[idx_sample];
      if (LiDAR_referenced_to_odom == false) {
        x += lidar_px;
        y += lidar_py;
        z += lidar_pz;
      }

      // Discard high points
      if (z > max_obst_height) {
        return;
      }

      // Compute index
      int i_x = -1, i_y = -1;
      GRID_UTILS_CUDA::device_calculo_indices_celda(
        &i_x, &i_y, x, y, NC_X, NC_Y, MIN_X, MIN_Y, RES);

      if (i_x == -1) {
        return;
      }

      // Acumulate
      int idx1D = GRID_UTILS_CUDA::device_sub2ind(i_y, i_x, NC_X, NC_Y);
      atomicAdd(&grid_cont[idx1D], 1.0);
    }
  }
}

// ---------------------------------------- MASES ---------------------------------------- //
__global__ void global_convert_into_mO(
  float grid_mO[], const float num_points_max_occ, const int n_cells)
{
  // Get index from the division of blocks and grids
  int idx_cell = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_cell < n_cells) {
    // Occupied is copied
    grid_mO[idx_cell] = min(1.0, grid_mO[idx_cell] / num_points_max_occ);
  }
}

// ---------------------------------------- MAIN CODE ---------------------------------------- //
void OBS_OG::observed_occupancy_grid_raster_occupied_cart_only(
  float d_grid_cart_data_mO[], const float d_pc_x[], const float d_pc_y[], const float d_pc_z[],
  const float d_pc_intensity[], const int d_pc_label[], const bool LiDAR_referenced_to_odom,
  const float lidar_px, const float lidar_py, const float lidar_pz, const int n_points,
  const int label_obst, const int NC_X, const int NC_Y, const float MIN_X, const float MIN_Y,
  const float CART_RES, const int iter)
{
  float max_obst_height = 2;  // TODO
  float num_points_max_occ = 3;
  printf("PARAMETROS NO INICIALIZADOS\n");

  // Dimensions for parallelization
  dim3 blocks_beams(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  dim3 grids_beams((n_points + blocks_beams.x - 1) / blocks_beams.x);

  dim3 blocks_cart(1024);
  dim3 grid_cart(
    (NC_X * NC_Y + blocks_cart.x - 1) /
    blocks_cart.x);  // number of blocks: choose a number that ensures all data is processed

  // ------------------------- Rasterize points ------------------------- //
  global_sum_obstacle_points<<<grids_beams, blocks_beams>>>(
    d_grid_cart_data_mO, d_pc_x, d_pc_y, d_pc_z, d_pc_label, n_points, LiDAR_referenced_to_odom,
    lidar_px, lidar_py, lidar_pz, label_obst, max_obst_height, NC_X, NC_Y, MIN_X, MIN_Y, CART_RES);

  //------------------------- Transform raster into masses ------------------------- //
  global_convert_into_mO<<<grid_cart, blocks_cart>>>(
    d_grid_cart_data_mO, num_points_max_occ, NC_X * NC_Y);

  // Synchronize
  cudaDeviceSynchronize();  // Not sure if I need to do this before cudaFree and leaveing the function
}