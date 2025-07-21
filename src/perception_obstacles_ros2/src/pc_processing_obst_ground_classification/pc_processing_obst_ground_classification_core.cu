#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification.h"

__global__ void global_apply_translation_pc(
  float * x, float * y, float * z, bool * referenced_to_odom, const int n_total_points,
  const float LiDAR_px, const float LiDAR_py, const float LiDAR_pz)
{
  int i_p = blockIdx.x * blockDim.x + threadIdx.x;

  if (i_p == 0) {
    *referenced_to_odom = true;
  }

  if (i_p < n_total_points) {
    x[i_p] += LiDAR_px;
    y[i_p] += LiDAR_py;
    z[i_p] += LiDAR_pz;
  }
}

__global__ void global_store_data(
  float * target_pc_x, float * target_pc_y, float * target_pc_z, int * target_pc_label,
  const float * input_pc_x, const float * input_pc_y, const float * input_pc_z,
  const int * input_pc_label, const int starting_idx, const int n_points_corresponding_LiDAR)
{
  int idx_input = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_input < n_points_corresponding_LiDAR) {
    int idx_target = idx_input + starting_idx;
    target_pc_x[idx_target] = input_pc_x[idx_input];
    target_pc_y[idx_target] = input_pc_y[idx_input];
    target_pc_z[idx_target] = input_pc_z[idx_input];
    target_pc_label[idx_target] = input_pc_label[idx_input];
  }
}

__global__ void global_recover_data(
  int * target_pc_label, const int * input_pc_label, const int starting_idx,
  const int n_points_corresponding_LiDAR)
{
  int idx_target = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_target < n_points_corresponding_LiDAR) {
    target_pc_label[idx_target] = input_pc_label[idx_target + starting_idx];
  }
}

/** ----------------------------------------------------------------------------------------------------------------
 * @brief All the main code should be organized here, e.g. if there are multiple sensors or they are of different 
 * types or if it is computed in cpu or gpu
 */
void OBST_GROUND::pc_processing_core(
  AUTOPIA_RubyPlus::PointCloud * RB_pc, AUTOPIA_RubyPlus::PointCloud * d_RB_pc,
  AUTOPIA_RubyPlus::parameters_channel_based * param_CB_RB, const bool consider_PC_HeliosRight,
  const bool consider_PC_HeliosLeft, AUTOPIA_Helios::PointCloud * Hr_pc,
  AUTOPIA_Helios::PointCloud * d_Hr_pc, AUTOPIA_Helios::PointCloud * Hl_pc,
  AUTOPIA_Helios::PointCloud * d_Hl_pc, AUTOPIA_Helios::parameters_channel_based * param_CB_Helios,
  const int label_obst, const int label_ground, const int label_noise,
  EGO_VEH::INFO_ego * h_info_coche, EGO_VEH::INFO_ego * h_info_coche_old,
  DATA_times * TIME_measurements, const int iter)
{
  bool enumeration = true;

  static dim3 blocks_RB(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grids_RB(
    (RB_pc->n_points + blocks_RB.x - 1) /
    blocks_RB.x);  // number of blocks: choose a number that ensures all data is processed

  static dim3 blocks_Hr(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grids_Hr(
    (Hr_pc->n_points + blocks_Hr.x - 1) /
    blocks_Hr.x);  // number of blocks: choose a number that ensures all data is processed

  static dim3 blocks_Hl(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grids_Hl(
    (Hl_pc->n_points + blocks_Hl.x - 1) /
    blocks_Hl.x);  // number of blocks: choose a number that ensures all data is processed

  // -------------------- Rotate pointcloud -------------------- //

  // Apply rotation to fix it to parallel to the ground plane
  TIME_measurements->time_PCs_processing_rotation.Reset();
  OBST_GROUND_UTILS::rotate_pointcloud(
    RB_pc->x, RB_pc->y, RB_pc->z, RB_pc->n_points, RB_pc->rotation_matrix);
  if (consider_PC_HeliosRight) {
    OBST_GROUND_UTILS::rotate_pointcloud(
      Hr_pc->x, Hr_pc->y, Hr_pc->z, Hr_pc->n_points, Hr_pc->rotation_matrix);
  }
  if (consider_PC_HeliosLeft) {
    OBST_GROUND_UTILS::rotate_pointcloud(
      Hl_pc->x, Hl_pc->y, Hl_pc->z, Hl_pc->n_points, Hl_pc->rotation_matrix);
  }
  TIME_measurements->time_PCs_processing_rotation.GetElapsedTime();
  TIME_measurements->time_PCs_processing_rotation.ComputeStats();
  if (enumeration) {
    printf("\nPoint cloud rotated\n");
    printf(
      " - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_PCs_processing_rotation.measured_time,
      TIME_measurements->time_PCs_processing_rotation.mean_time,
      TIME_measurements->time_PCs_processing_rotation.max_time);
  }
  // -------------------- Copy PC Host to Device -------------------- //

  TIME_measurements->time_PCs_processing_host2device.Reset();
  checkCudaErrors(
    cudaMemcpy(d_RB_pc, RB_pc, sizeof(AUTOPIA_RubyPlus::PointCloud), cudaMemcpyHostToDevice));
  if (consider_PC_HeliosRight) {
    checkCudaErrors(
      cudaMemcpy(d_Hr_pc, Hr_pc, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyHostToDevice));
  }
  if (consider_PC_HeliosLeft) {
    checkCudaErrors(
      cudaMemcpy(d_Hl_pc, Hl_pc, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyHostToDevice));
  }
  cudaDeviceSynchronize();
  TIME_measurements->time_PCs_processing_host2device.GetElapsedTime();
  TIME_measurements->time_PCs_processing_host2device.ComputeStats();
  if (enumeration) {
    printf("\nPoint cloud copy host 2 device done\n");
    printf(
      " - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_PCs_processing_host2device.measured_time,
      TIME_measurements->time_PCs_processing_host2device.mean_time,
      TIME_measurements->time_PCs_processing_host2device.max_time);
  }

  // -------------------- CHANNEL-BASED -------------------- //

  TIME_measurements->time_PCs_processing_CB.Reset();

  // Classify
  OBST_GROUND::gpu_pc_obst_ground_channel_based(
    d_RB_pc->label, d_RB_pc->label_reason, d_RB_pc->x, d_RB_pc->y, d_RB_pc->z, d_RB_pc->intensity,
    RB_pc->n_points, RB_pc->n_layers, RB_pc->LiDAR_pz, param_CB_RB->threshold_height_is_obst,
    param_CB_RB->threshold_gradient_first_impact_is_obst,
    param_CB_RB->max_ground_height_within_distance,
    param_CB_RB->threshold_distance_for_max_ground_height, param_CB_RB->max_gradiente,
    param_CB_RB->back2ground_diff_height_with_last_ground,
    param_CB_RB->back2ground_diff_height_with_prev_obst, param_CB_RB->noise_radius,
    h_info_coche->ego_size_min_x - RB_pc->LiDAR_px, h_info_coche->ego_size_max_x - RB_pc->LiDAR_px,
    h_info_coche->ego_size_min_y - RB_pc->LiDAR_py, h_info_coche->ego_size_max_y - RB_pc->LiDAR_py,
    label_obst, label_ground, label_noise);

  // // Classify in host
  // OBST_GROUND::host_pc_obst_ground_channel_based(
  //   RB_pc->label, RB_pc->label_reason, RB_pc->x, RB_pc->y, RB_pc->z, RB_pc->intensity,
  //   RB_pc->n_points, RB_pc->n_layers, RB_pc->LiDAR_pz, param_CB_RB->threshold_height_is_obst,
  //   param_CB_RB->max_gradiente, param_CB_RB->back2ground_diff_height_with_last_ground, label_obst,
  //   label_ground, label_noise);
  // // Copy Host to Device
  // checkCudaErrors(cudaMemcpy(d_RB_pc, RB_pc, sizeof(AUTOPIA_RubyPlus::PointCloud), cudaMemcpyHostToDevice));
  // cudaDeviceSynchronize();

  if (consider_PC_HeliosRight) {
    OBST_GROUND::gpu_pc_obst_ground_channel_based(
      d_Hr_pc->label, d_Hr_pc->label_reason, d_Hr_pc->x, d_Hr_pc->y, d_Hr_pc->z, d_Hr_pc->intensity,
      Hr_pc->n_points, Hr_pc->n_layers, Hr_pc->LiDAR_pz, param_CB_Helios->threshold_height_is_obst,
      param_CB_Helios->threshold_gradient_first_impact_is_obst,
      param_CB_Helios->max_ground_height_within_distance,
      param_CB_Helios->threshold_distance_for_max_ground_height, param_CB_Helios->max_gradiente,
      param_CB_Helios->back2ground_diff_height_with_last_ground,
      param_CB_Helios->back2ground_diff_height_with_prev_obst, param_CB_Helios->noise_radius,
      h_info_coche->ego_size_min_x - Hr_pc->LiDAR_px,
      h_info_coche->ego_size_max_x - Hr_pc->LiDAR_px,
      h_info_coche->ego_size_min_y - Hr_pc->LiDAR_py,
      h_info_coche->ego_size_max_y - Hr_pc->LiDAR_py, label_obst, label_ground, label_noise);
  }

  if (consider_PC_HeliosLeft) {
    OBST_GROUND::gpu_pc_obst_ground_channel_based(
      d_Hl_pc->label, d_Hl_pc->label_reason, d_Hl_pc->x, d_Hl_pc->y, d_Hl_pc->z, d_Hl_pc->intensity,
      Hl_pc->n_points, Hl_pc->n_layers, Hl_pc->LiDAR_pz, param_CB_Helios->threshold_height_is_obst,
      param_CB_Helios->threshold_gradient_first_impact_is_obst,
      param_CB_Helios->max_ground_height_within_distance,
      param_CB_Helios->threshold_distance_for_max_ground_height, param_CB_Helios->max_gradiente,
      param_CB_Helios->back2ground_diff_height_with_last_ground,
      param_CB_Helios->back2ground_diff_height_with_prev_obst, param_CB_Helios->noise_radius,
      h_info_coche->ego_size_min_x - Hl_pc->LiDAR_px,
      h_info_coche->ego_size_max_x - Hl_pc->LiDAR_px,
      h_info_coche->ego_size_min_y - Hl_pc->LiDAR_py,
      h_info_coche->ego_size_max_y - Hl_pc->LiDAR_py, label_obst, label_ground, label_noise);
  }

  cudaDeviceSynchronize();
  TIME_measurements->time_PCs_processing_CB.GetElapsedTime();
  TIME_measurements->time_PCs_processing_CB.ComputeStats();
  if (enumeration) {
    printf("\nPoint cloud copy host 2 device done\n");
    printf(
      " - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_PCs_processing_CB.measured_time,
      TIME_measurements->time_PCs_processing_CB.mean_time,
      TIME_measurements->time_PCs_processing_CB.max_time);
  }

  // -------------------- CORRECT TRANSLATION POSITION LiDAR -------------------- //
  //                       (rotation must be applied before)                      //
  TIME_measurements->time_PCs_processing_translation.Reset();

  RB_pc->referenced_to_odom = true;
  global_apply_translation_pc<<<grids_RB, blocks_RB>>>(
    d_RB_pc->x, d_RB_pc->y, d_RB_pc->z, &d_RB_pc->referenced_to_odom, RB_pc->n_points,
    RB_pc->LiDAR_px, RB_pc->LiDAR_py, RB_pc->LiDAR_pz);

  if (consider_PC_HeliosRight) {
    Hr_pc->referenced_to_odom = true;
    global_apply_translation_pc<<<grids_Hr, blocks_Hr>>>(
      d_Hr_pc->x, d_Hr_pc->y, d_Hr_pc->z, &d_Hr_pc->referenced_to_odom, Hr_pc->n_points,
      Hr_pc->LiDAR_px, Hr_pc->LiDAR_py, Hr_pc->LiDAR_pz);
  }

  if (consider_PC_HeliosLeft) {
    Hl_pc->referenced_to_odom = true;
    global_apply_translation_pc<<<grids_Hl, blocks_Hl>>>(
      d_Hl_pc->x, d_Hl_pc->y, d_Hl_pc->z, &d_Hl_pc->referenced_to_odom, Hl_pc->n_points,
      Hl_pc->LiDAR_px, Hl_pc->LiDAR_py, Hl_pc->LiDAR_pz);
  }

  cudaDeviceSynchronize();
  TIME_measurements->time_PCs_processing_translation.GetElapsedTime();
  TIME_measurements->time_PCs_processing_translation.ComputeStats();
  if (enumeration) {
    printf("\nPoint cloud rotated\n");
    printf(
      " - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_PCs_processing_translation.measured_time,
      TIME_measurements->time_PCs_processing_translation.mean_time,
      TIME_measurements->time_PCs_processing_translation.max_time);
  }

  // -------------------- CORRECT POINT CLOUD -------------------- //
  TIME_measurements->time_PCs_correct_ego_motion.Reset();
  EGO_VEH::gpu_correct_ego_motion_displacement_for_pointcloud(
    d_RB_pc->x, d_RB_pc->y, d_RB_pc->timestamp, RB_pc->n_points, h_info_coche, h_info_coche_old);

  if (consider_PC_HeliosRight) {
    EGO_VEH::gpu_correct_ego_motion_displacement_for_pointcloud(
      d_Hr_pc->x, d_Hr_pc->y, d_Hr_pc->timestamp, Hr_pc->n_points, h_info_coche, h_info_coche_old);
  }

  if (consider_PC_HeliosLeft) {
    EGO_VEH::gpu_correct_ego_motion_displacement_for_pointcloud(
      d_Hl_pc->x, d_Hl_pc->y, d_Hl_pc->timestamp, Hl_pc->n_points, h_info_coche, h_info_coche_old);
  }

  cudaDeviceSynchronize();
  TIME_measurements->time_PCs_correct_ego_motion.GetElapsedTime();
  TIME_measurements->time_PCs_correct_ego_motion.ComputeStats();
  if (enumeration) {
    printf("\nPoint cloud rotated\n");
    printf(
      " - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_PCs_correct_ego_motion.measured_time,
      TIME_measurements->time_PCs_correct_ego_motion.mean_time,
      TIME_measurements->time_PCs_correct_ego_motion.max_time);
  }

  // -------------------- MEDIAN-FILTER -------------------- //

  TIME_measurements->time_PCs_processing_FM.Reset();

  // Enhance classification with median filter
  float threshold_diff_ground = 0.2;  // Care curbs and plants
  printf("PARAMETRO NO DEFINIDO\n");

  checkCudaErrors(cudaMemcpy(
    d_RB_pc->channel_label, d_RB_pc->label, RB_pc->n_points * sizeof(int),
    cudaMemcpyDeviceToDevice));

  if (consider_PC_HeliosRight) {
    checkCudaErrors(cudaMemcpy(
      d_Hr_pc->channel_label, d_Hr_pc->label, Hr_pc->n_points * sizeof(int),
      cudaMemcpyDeviceToDevice));
  }
  if (consider_PC_HeliosLeft) {
    checkCudaErrors(cudaMemcpy(
      d_Hl_pc->channel_label, d_Hl_pc->label, Hl_pc->n_points * sizeof(int),
      cudaMemcpyDeviceToDevice));
  }

  // +++++ Gather data +++++ //

  // Declare, reserve and initialize (potentially this could be done once... but lets try this way so it is easily resusable)
  float *d_x, *d_y, *d_z;
  int * d_label;
  int n_total_points = RB_pc->n_points + (int)consider_PC_HeliosRight * Hr_pc->n_points +
                       (int)consider_PC_HeliosLeft * Hl_pc->n_points;

  checkCudaErrors(cudaMalloc((void **)&d_x, n_total_points * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_y, n_total_points * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_z, n_total_points * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_label, n_total_points * sizeof(int)));

  int points_stored = 0;
  global_store_data<<<grids_RB, blocks_RB>>>(
    d_x, d_y, d_z, d_label, d_RB_pc->x, d_RB_pc->y, d_RB_pc->z, d_RB_pc->label, 0, RB_pc->n_points);
  points_stored += RB_pc->n_points;

  if (consider_PC_HeliosRight) {
    global_store_data<<<grids_Hr, blocks_Hr>>>(
      d_x, d_y, d_z, d_label, d_Hr_pc->x, d_Hr_pc->y, d_Hr_pc->z, d_Hr_pc->label, points_stored,
      Hr_pc->n_points);
    points_stored += Hr_pc->n_points;
  }

  if (consider_PC_HeliosLeft) {
    global_store_data<<<grids_Hl, blocks_Hl>>>(
      d_x, d_y, d_z, d_label, d_Hl_pc->x, d_Hl_pc->y, d_Hl_pc->z, d_Hl_pc->label, points_stored,
      Hl_pc->n_points);
    points_stored += Hl_pc->n_points;
  }

  if (points_stored != n_total_points) {
    printf("ERROR!! %d != %d\n", points_stored, n_total_points);
    exit(1);
  }

  // +++++ REAL MEDIAN FILTER +++++ //

  if (true) {
    OBST_GROUND::reclassify_points_with_median_filter(
      d_label, d_x, d_y, d_z, n_total_points, threshold_diff_ground, label_obst, label_ground,
      label_noise, iter);
  } else {
    OBST_GROUND::reclassify_points_with_median_filter_cart(
      d_label, d_x, d_y, d_z, n_total_points, threshold_diff_ground, label_obst, label_ground,
      label_noise, iter);
  }

  // +++++ Recover data +++++ //

  int points_recovered = 0;

  global_recover_data<<<grids_RB, blocks_RB>>>(d_RB_pc->label, d_label, 0, RB_pc->n_points);
  points_recovered += RB_pc->n_points;

  if (consider_PC_HeliosRight) {
    global_recover_data<<<grids_Hr, blocks_Hr>>>(
      d_Hr_pc->label, d_label, points_recovered, Hr_pc->n_points);
    points_recovered += Hr_pc->n_points;
  }

  if (consider_PC_HeliosLeft) {
    global_recover_data<<<grids_Hl, blocks_Hl>>>(
      d_Hl_pc->label, d_label, points_recovered, Hl_pc->n_points);
    points_recovered += Hl_pc->n_points;
  }

  if (points_recovered != n_total_points) {
    printf("ERROR!! %d != %d\n", points_recovered, n_total_points);
    exit(1);
  }
  cudaDeviceSynchronize();
  TIME_measurements->time_PCs_processing_FM.GetElapsedTime();
  TIME_measurements->time_PCs_processing_FM.ComputeStats();
  if (enumeration) {
    printf("\nPoint cloud rotated\n");
    printf(
      " - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_PCs_processing_FM.measured_time,
      TIME_measurements->time_PCs_processing_FM.mean_time,
      TIME_measurements->time_PCs_processing_FM.max_time);
  }

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_label);
}
