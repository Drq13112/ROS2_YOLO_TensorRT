#include "perception_obstacles/ego_vehicle/calculos_estado_coche.h"

// ---------------------------------------- BASIC EGO OPERATIONS ---------------------------------------- //

void EGO_VEH::predecir_estado_coche(
  EGO_VEH::INFO_ego * info_coche, const EGO_VEH::INFO_ego * info_coche_old,
  const double target_time)
{
  bool print_debug = false;

  if (print_debug) {
    printf(
      "info_coche recibido [%f, %f, %fº]     info_coche previous iter [%f, %f, %fº]     diff [%f, "
      "%f, %f^º]\n",
      info_coche->px_G, info_coche->py_G, info_coche->yaw_G * 180 / M_PI, info_coche_old->px_G,
      info_coche_old->py_G, info_coche_old->yaw_G * 180 / M_PI,
      info_coche->px_G - info_coche_old->px_G, info_coche->py_G - info_coche_old->py_G,
      (info_coche->yaw_G - info_coche_old->yaw_G) * 180 / M_PI);
  }

  // Tiempos de la vuelta y desfase
  double desfase_tiempo_lcm_ibeo = target_time - info_coche->tiempo;
  if (desfase_tiempo_lcm_ibeo < -0.06) {
    printf(
      "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Desfase ruby - localization negativo! -> "
      "desfase_tiempo_lcm_ibeo = %f\n",
      desfase_tiempo_lcm_ibeo);
    // printf("    Tiempo vuelta           = %f\n", target_time);
    // printf("    Tiempo lcm estado coche = %f\n", info_coche->tiempo);
  }
  if (desfase_tiempo_lcm_ibeo > 0.12) {
    printf(
      "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Desfase ruby - localization demasiado elevado: %f\n",
      desfase_tiempo_lcm_ibeo);
    // printf("    Tiempo ibeo           = %f\n", target_time);
    // printf("    Tiempo lcm estado coche = %f\n", info_coche->tiempo);
  }

  if (print_debug) {
    printf("Tiempo defase lcm-ibeo = %f\n", desfase_tiempo_lcm_ibeo);
  }

  // Calculamos un yaw rate
  // info_coche->yaw_rate = (info_coche->yaw_G - info_coche_old->yaw_G);
  // fixAngleRad(&info_coche->yaw_rate);
  // info_coche->yaw_rate /= (info_coche->tiempo - info_coche_old->tiempo);

  // Actualizamos la posicion del coche
  if (fabs(info_coche->yaw_rate) > 1e-5 && fabs(info_coche->vel) > 1e-3) {
    info_coche->px_G += info_coche->vel / info_coche->yaw_rate *
                        (sin(info_coche->yaw_G + info_coche->yaw_rate * desfase_tiempo_lcm_ibeo) -
                         sin(info_coche->yaw_G));
    info_coche->py_G += info_coche->vel / info_coche->yaw_rate *
                        (-cos(info_coche->yaw_G + info_coche->yaw_rate * desfase_tiempo_lcm_ibeo) +
                         cos(info_coche->yaw_G));
    info_coche->yaw_G += info_coche->yaw_rate * desfase_tiempo_lcm_ibeo;
    fixAngleRad(&info_coche->yaw_G);
  } else {
    info_coche->px_G += info_coche->vel * cos(info_coche->yaw_G) * desfase_tiempo_lcm_ibeo;
    info_coche->py_G += info_coche->vel * sin(info_coche->yaw_G) * desfase_tiempo_lcm_ibeo;
    //			info_coche->yaw_G = info_coche->yaw_G; si no hay yaw rate lo asumimos constante
    //			fixAngleRad(&info_coche->yaw_G);
  }
  info_coche->tiempo = target_time;
}

void EGO_VEH::calculo_delta_estado_coche(
  EGO_VEH::INFO_ego * info_coche, const EGO_VEH::INFO_ego * info_coche_old, const int iter,
  const int iter_inicial)
{
  if (iter <= iter_inicial) {
    info_coche->delta_x = 0.0;
    info_coche->delta_y = 0.0;
    info_coche->delta_yaw = 0.0;

    info_coche->delta_t = 0.0;
  } else {
    info_coche->delta_x = info_coche->px_G - info_coche_old->px_G;
    info_coche->delta_y = info_coche->py_G - info_coche_old->py_G;
    matrizRotacionZ(&info_coche->delta_x, &info_coche->delta_y, -info_coche_old->yaw_G);
    info_coche->delta_yaw = info_coche->yaw_G - info_coche_old->yaw_G;
    fixAngleRad(&info_coche->delta_yaw);

    info_coche->delta_t = info_coche->tiempo - info_coche_old->tiempo;
  }

  // Senos y cosenos
  info_coche->sin_yaw_G = sin(info_coche->yaw_G);
  info_coche->cos_yaw_G = cos(info_coche->yaw_G);
  info_coche->sin_delta_yaw = sin(info_coche->delta_yaw);
  info_coche->cos_delta_yaw = cos(info_coche->delta_yaw);
}

// ---------------------------------------- GPU CORRECT POINTCLOUD ---------------------------------------- //

__global__ void global_correct_ego_motion_points(
  float pc_x[], float pc_y[], double pc_timestamp[], const int n_total_points,
  const double t1_timestamp, const double t1_ego_px, const double t1_ego_py,
  const double t1_ego_yaw, const double t_lap_timestamp, const double t_lap_ego_px,
  const double t_lap_ego_py, const double t_lap_ego_yaw)
{
  // Get index from the division of blocks and grids
  int i_p = blockIdx.x * blockDim.x + threadIdx.x;

  if (i_p < n_total_points) {
    double beam_ego_x, beam_ego_y, beam_ego_yaw;
    double px = pc_x[i_p];
    double py = pc_y[i_p];

    // Compute the position of the ego vehicle at the time stamp of this point
    device_interpolacion_lineal(
      &beam_ego_x, pc_timestamp[i_p], t1_timestamp, t_lap_timestamp, t1_ego_px, t_lap_ego_px);

    device_interpolacion_lineal(
      &beam_ego_y, pc_timestamp[i_p], t1_timestamp, t_lap_timestamp, t1_ego_py, t_lap_ego_py);

    device_interpolacion_lineal_angulos(
      &beam_ego_yaw, pc_timestamp[i_p], t1_timestamp, t_lap_timestamp, t1_ego_yaw, t_lap_ego_yaw);

    // Local to global using the ego position when point was recorded
    device_matrizRotacionZ(&px, &py, beam_ego_yaw);
    px += beam_ego_x;
    py += beam_ego_y;

    // Global to local using the ego position at the end of the spin
    px -= t_lap_ego_px;
    py -= t_lap_ego_py;
    device_matrizRotacionZInversa(&px, &py, t_lap_ego_yaw);

    if ((isnan(px) && !isnan(py)) || (!isnan(px) && isnan(py))) {
      printf(
        "[%f < %f < %f];  [%f < %f < %f]; [%f < %f < %f]; [%f < %f < %f];"
        "[%f, %f] => [%f, %f]\n",
        t1_timestamp, pc_timestamp[i_p], t_lap_timestamp, t1_ego_px, beam_ego_x, t_lap_ego_px,
        t1_ego_py, beam_ego_y, t_lap_ego_py, t1_ego_yaw, beam_ego_yaw, t_lap_ego_yaw, pc_x[i_p],
        pc_y[i_p], px, py);
    }

    // Store
    pc_x[i_p] = px;
    pc_y[i_p] = py;
  }
}

void EGO_VEH::gpu_correct_ego_motion_displacement_for_pointcloud(
  float d_x[], float d_y[], double d_timestamp[], const int n_total_points,
  const EGO_VEH::INFO_ego * h_info_coche, const EGO_VEH::INFO_ego * h_info_coche_old)
{
  static dim3 blocks_points(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grids_points(
    (n_total_points + blocks_points.x - 1) /
    blocks_points.x);  // number of blocks: choose a number that ensures all data is processed

  global_correct_ego_motion_points<<<grids_points, blocks_points>>>(
    d_x, d_y, d_timestamp, n_total_points, h_info_coche_old->tiempo, h_info_coche_old->px_G,
    h_info_coche_old->py_G, h_info_coche_old->yaw_G, h_info_coche->tiempo, h_info_coche->px_G,
    h_info_coche->py_G, h_info_coche->yaw_G);
}