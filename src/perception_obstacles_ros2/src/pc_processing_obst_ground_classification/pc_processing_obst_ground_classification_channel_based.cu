#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification.h"

// ---------------------------------------- CPU ---------------------------------------- //

void OBST_GROUND::host_pc_obst_ground_channel_based(
  int label[], int reason[], const float x[], const float y[], const float z[],
  const float intensity[], const int n_total_points, const int n_layers, const float LiDAR_height,
  const float threshold_height_is_obst, const float threshold_gradient_first_impact_is_obst,
  const float max_ground_height_within_distance,
  const float threshold_distance_for_max_ground_height, const float max_gradiente,
  const float back2ground_diff_height_with_last_ground,
  const float back2ground_diff_height_with_prev_obst, const float noise_radius,
  const int label_suelo, const int label_obst, const int label_noise)
{
  int c_label = -1;
  float distPuntos, distSueloPunto, distSueloPuntoPrevio;
  float punto[3] = {0, 0, 0};
  float puntoPrevio[3] = {0, 0, 0};
  float puntoRef[3] = {0, 0, 0};
  float height_diff_respect_prev_point;
  float dx, dy, dz;
  float alpha;
  int size3d = 3 * sizeof(float);
  bool is_the_first_valid_point = true;

  int INTENSITY = -1;
  int IS_NAN = 0;
  int HIGH_HEIGHT = 1;
  int INHERITED = 2;
  int ALPHA = 3;
  int BACK_GROUND = 4;

  // Recorrer todos los puntos
  for (int i_p = 0; i_p < n_total_points; i_p++) {
    // Check first point and initialize virtual point
    if (i_p % n_layers == 0) {
      puntoPrevio[0] = 0.0;
      puntoPrevio[1] = 0.0;
      puntoPrevio[2] = 0.0;
      c_label = label_suelo;
      memcpy(&puntoRef, &puntoPrevio, size3d);
      is_the_first_valid_point = true;
    }

    if (intensity[i_p] <= 0) {
      label[i_p] = label_noise;
      reason[i_p] = INTENSITY;
      continue;
    }
    if (std::isnan(x[i_p])) {
      label[i_p] = label_noise;
      reason[i_p] = IS_NAN;
      continue;
    }

    // Get data of current point
    punto[0] = x[i_p];
    punto[1] = y[i_p];
    punto[2] = z[i_p] + LiDAR_height;

    // De momento hereda
    label[i_p] = c_label;
    reason[i_p] = INHERITED;

    if (
      punto[2] > threshold_height_is_obst ||
      (punto[2] > max_ground_height_within_distance &&
       (punto[0] * punto[0] + punto[1] * punto[1]) <
         threshold_distance_for_max_ground_height * threshold_distance_for_max_ground_height)) {
      // Es un obstaculo
      c_label = label_obst;
      label[i_p] = c_label;

      // Guardamos el punto PREVIO como referencia
      memcpy(&puntoRef, &puntoPrevio, size3d);
      reason[i_p] = HIGH_HEIGHT;

    } else {
      height_diff_respect_prev_point = punto[2] - puntoPrevio[2];

      // if the previous point was floor, check for obstacles
      if (c_label == label_suelo) {
        dx = punto[0] - puntoPrevio[0];
        dy = punto[1] - puntoPrevio[1];
        dz = punto[2] - puntoPrevio[2];
        distPuntos = sqrt(dx * dx + dy * dy + dz * dz);
        alpha = asin(height_diff_respect_prev_point / distPuntos);

        // Check if the point has came back -> We compute the distance from the previous point to the initial one and from this point to the initial one too
        // If the distance from the previous to the origin is bigger, the current point has came back -> thats a consecuence of an obstacle
        distSueloPunto = sqrt(punto[0] * punto[0] + punto[1] * punto[1] + punto[2] * punto[2]);
        distSueloPuntoPrevio = sqrt(
          puntoPrevio[0] * puntoPrevio[0] + puntoPrevio[1] * puntoPrevio[1] +
          puntoPrevio[2] * puntoPrevio[2]);

        //Is there an obstacle?
        if (
          (alpha > max_gradiente) || ((distSueloPunto - distSueloPuntoPrevio) < 0) ||
          (is_the_first_valid_point && alpha > threshold_gradient_first_impact_is_obst)) {
          // Hay obstaculo -> ya no hay suelo
          c_label = label_obst;
          reason[i_p] = ALPHA;

          // Guardamos el punto PREVIO como referencia
          memcpy(&puntoRef, &puntoPrevio, size3d);
        }
      } else {
        // If the height is smaller, check if the beam can be hitting the floor
        if (height_diff_respect_prev_point < back2ground_diff_height_with_prev_obst) {
          // If the current point is down a reference + range
          if (punto[2] < (puntoRef[2] + back2ground_diff_height_with_last_ground)) {
            // Hay suelo
            c_label = label_suelo;
            reason[i_p] = BACK_GROUND;

            // Guardamos este punto como referencia
            memcpy(&puntoRef, &punto, size3d);
          }
        }
      }
    }
    memcpy(&puntoPrevio, &punto, size3d);

    // Corregimos informacion del punto actual
    label[i_p] = c_label;

    is_the_first_valid_point = false;
  }
}

// ---------------------------------------- GPU ---------------------------------------- //
__device__ void kernel_check_obstacle(
  int * c_label, int * reason, const float punto[3], const float puntoPrevio[3],
  const float threshold_height_is_obst, const float max_ground_height_within_distance,
  const float threshold_distance_for_max_ground_height, const float max_gradiente,
  const bool is_the_first_valid_point, const float threshold_gradient_first_impact_is_obst,
  const int label_obstacle)
{
  // Calculations
  float distPuntos = sqrt(
    (punto[0] - puntoPrevio[0]) * (punto[0] - puntoPrevio[0]) +
    (punto[1] - puntoPrevio[1]) * (punto[1] - puntoPrevio[1]) +
    (punto[2] - puntoPrevio[2]) * (punto[2] - puntoPrevio[2]));

  float alpha = asin((punto[2] - puntoPrevio[2]) / distPuntos);

  float distSueloPunto = sqrt(punto[0] * punto[0] + punto[1] * punto[1] + punto[2] * punto[2]);
  float distSueloPuntoPrevio = sqrt(
    puntoPrevio[0] * puntoPrevio[0] + puntoPrevio[1] * puntoPrevio[1] +
    puntoPrevio[2] * puntoPrevio[2]);

  // Basic Individual Obstacle Constrictions //
  if (
    punto[2] > threshold_height_is_obst ||
    (punto[2] > max_ground_height_within_distance &&
     (punto[0] * punto[0] + punto[1] * punto[1]) <
       threshold_distance_for_max_ground_height * threshold_distance_for_max_ground_height)) {
    // Es un obstaculo
    *c_label = label_obstacle;

    *reason = 1;
  }

  // Check if the point has came back -> We compute the distance from the previous point to the initial one and from this point to the initial one too
  // If the distance from the previous to the origin is bigger, the current point has came back -> thats a consecuence of an obstacle
  if ((distSueloPunto - distSueloPuntoPrevio) < 0) {
    *c_label = label_obstacle;
    *reason = 5;
  }
  if (is_the_first_valid_point && alpha > threshold_gradient_first_impact_is_obst) {
    *c_label = label_obstacle;
    *reason = 4;
  }
  if (alpha > max_gradiente) {
    *c_label = label_obstacle;
    *reason = 3;
  }
}

__device__ void kernel_check_ground(
  int * c_label, int * reason, const float punto[3], const float puntoPrevio[3], float puntoRef[3],
  const float back2ground_diff_height_with_prev_obst,
  const float back2ground_diff_height_with_last_ground, const int label_suelo)
{
  // If the height is smaller, check if the beam can be hitting the floor
  if (punto[2] - puntoPrevio[2] < back2ground_diff_height_with_prev_obst) {
    // If the current point is down a reference + range
    if (punto[2] < (puntoRef[2] + back2ground_diff_height_with_last_ground)) {
      // Hay suelo
      *c_label = label_suelo;
      *reason = 6;

      // Guardamos este punto como referencia
      puntoRef[0] = punto[0];
      puntoRef[1] = punto[1];
      puntoRef[2] = punto[2];
    }
  }
}

__global__ void global_pc_obst_ground_channel_based_doubt_rollinglayers(
  int label[], int reason[], const float x[], const float y[], const float z[],
  const float intensity[], const int n_total_points, const int n_layers, const float LiDAR_height,
  const float threshold_height_is_obst, const float threshold_gradient_first_impact_is_obst,
  const float max_ground_height_within_distance,
  const float threshold_distance_for_max_ground_height, const float max_gradiente,
  const float back2ground_diff_height_with_last_ground,
  const float back2ground_diff_height_with_prev_obst, const float noise_radius,
  const float noise_x_min, const float noise_x_max, const float noise_y_min,
  const float noise_y_max, const int noisy_channels_up_to, const int label_obstacle,
  const int label_suelo, const int label_noise)
{
  // Get index from the division of blocks and grids
  int idx_thread = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread corresponds to the first point of a channel

  int idx_first_point = idx_thread * n_layers;

  if (idx_first_point < n_total_points) {
    int aux_label = -1;
    int label_potential_obstacle = label_obstacle * (-10);
    int ip_first_potential_obstacle = -1;
    int c_label = -1;
    float punto[3] = {0, 0, 0};
    float puntoPrevio[3] = {0, 0, 0};
    float puntoRef[3] = {0, 0, 0};
    bool is_the_first_valid_point = true;

    // Virtual point
    puntoPrevio[0] =
      noise_radius *
      cos(atan2(y[idx_first_point], x[idx_first_point]));  // at least start at noise range
    puntoPrevio[1] =
      noise_radius *
      sin(atan2(y[idx_first_point], x[idx_first_point]));  // at least start at noise range
    puntoPrevio[2] = 0.0;
    c_label = label_suelo;

    puntoRef[0] = puntoPrevio[0];
    puntoRef[1] = puntoPrevio[1];
    puntoRef[2] = puntoPrevio[2];

    // Recorrer todos los puntos empezando por el primer punto del channel
    for (int i_p = idx_first_point; i_p < (idx_first_point + n_layers); i_p++) {
      // Get data of current point
      punto[0] = x[i_p];
      punto[1] = y[i_p];
      punto[2] = z[i_p] + LiDAR_height;

      // +++++ Noise +++++ //
      if (idx_thread < noisy_channels_up_to) {
        label[i_p] = label_noise;
        reason[i_p] = -3;
        continue;
      }
      if (isnan(punto[0])) {
        label[i_p] = label_noise;
        reason[i_p] = 0;
        continue;
      }
      // if (intensity[i_p] <= 0) {
      //   label[i_p] = label_noise;
      //   reason[i_p] = -1;
      //   continue;
      // }
      if (
        punto[0] > noise_x_min && punto[0] < noise_x_max && punto[1] > noise_y_min &&
        punto[1] < noise_y_max) {
        label[i_p] = label_noise;
        reason[i_p] = -2;
        continue;
      }
      if ((punto[0] * punto[0] + punto[1] * punto[1]) < (noise_radius * noise_radius)) {
        label[i_p] = label_noise;
        reason[i_p] = -2;
        continue;
      }

      // +++++ Inherit +++++ //
      // De momento hereda
      label[i_p] = c_label;
      reason[i_p] = 2;

      // if the previous point was floor, check for obstacles
      if (c_label == label_suelo) {
        kernel_check_obstacle(
          &c_label, &reason[i_p], punto, puntoPrevio, threshold_height_is_obst,
          max_ground_height_within_distance, threshold_distance_for_max_ground_height,
          max_gradiente, is_the_first_valid_point, threshold_gradient_first_impact_is_obst,
          label_obstacle);

        if (c_label == label_obstacle) {
          // Guardamos el punto PREVIO como referencia
          puntoRef[0] = puntoPrevio[0];
          puntoRef[1] = puntoPrevio[1];
          puntoRef[2] = puntoPrevio[2];

          // Check if doubt obstacle point
          if (punto[2] - puntoRef[2] < back2ground_diff_height_with_last_ground) {
            ip_first_potential_obstacle = i_p;
            c_label = label_potential_obstacle;
          }
        }
      } else if (c_label == label_obstacle) {
        kernel_check_ground(
          &c_label, &reason[i_p], punto, puntoPrevio, puntoRef,
          back2ground_diff_height_with_prev_obst, back2ground_diff_height_with_last_ground,
          label_suelo);
      } else if (c_label == label_potential_obstacle) {
        // Check obstacle first
        kernel_check_obstacle(
          &aux_label, &reason[i_p], punto, puntoPrevio, threshold_height_is_obst,
          max_ground_height_within_distance, threshold_distance_for_max_ground_height,
          max_gradiente, is_the_first_valid_point, threshold_gradient_first_impact_is_obst,
          label_obstacle);

        // If it can be obstacle due to height and it fits with conditions of obstacle -> correct
        if (
          aux_label == label_obstacle &&
          (punto[2] - puntoRef[2]) > back2ground_diff_height_with_last_ground) {
          c_label = label_obstacle;

          // Correct previous points
          for (int ip2 = i_p; ip2 >= ip_first_potential_obstacle; ip2--) {
            label[ip2] = c_label;
          }
        }

        // If it is no obstacle check if it can be ground
        if (aux_label != label_obstacle) {
          kernel_check_ground(
            &aux_label, &reason[i_p], punto, puntoPrevio, puntoRef,
            back2ground_diff_height_with_prev_obst, back2ground_diff_height_with_last_ground,
            label_suelo);

          if (aux_label == label_suelo) {
            c_label = label_suelo;
            // Correct previous points
            for (int ip2 = i_p; ip2 >= ip_first_potential_obstacle; ip2--) {
              label[ip2] = c_label;
            }
          }
        }
      }

      puntoPrevio[0] = punto[0];
      puntoPrevio[1] = punto[1];
      puntoPrevio[2] = punto[2];

      // Corregimos informacion del punto actual
      label[i_p] = c_label;

      // If we have reach this point... the first valid point is set to false
      is_the_first_valid_point = false;
    }

    // If the end is reached without confirmation correct to ground
    if (c_label == label_potential_obstacle) {
      for (int ip2 = (idx_first_point + n_layers - 1); ip2 >= ip_first_potential_obstacle; ip2--) {
        label[ip2] = label_suelo;
      }
    }
  }
}

// Without Doubt
__global__ void global_pc_obst_ground_channel_based_rollinglayers(
  int label[], int reason[], const float x[], const float y[], const float z[],
  const float intensity[], const int n_total_points, const int n_layers, const float LiDAR_height,
  const float threshold_height_is_obst, const float threshold_gradient_first_impact_is_obst,
  const float max_ground_height_within_distance,
  const float threshold_distance_for_max_ground_height, const float max_gradiente,
  const float back2ground_diff_height_with_last_ground,
  const float back2ground_diff_height_with_prev_obst, const float noise_radius,
  const float noise_x_min, const float noise_x_max, const float noise_y_min,
  const float noise_y_max, const int noisy_channels_up_to, const int label_obstacle,
  const int label_suelo, const int label_noise)
{
  // Get index from the division of blocks and grids
  int idx_thread = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread corresponds to the first point of a channel

  int idx_first_point = idx_thread * n_layers;

  if (idx_first_point < n_total_points) {
    // bool obst_confirmed = false;
    int c_label = -1;
    float distPuntos, distSueloPunto, distSueloPuntoPrevio;
    float punto[3] = {0, 0, 0};
    float puntoPrevio[3] = {0, 0, 0};
    float puntoRef[3] = {0, 0, 0};
    float height_diff_respect_prev_point;
    float alpha;
    bool is_the_first_valid_point = true;

    // Virtual point
    puntoPrevio[0] =
      noise_radius *
      cos(atan2(y[idx_first_point], x[idx_first_point]));  // at least start at noise range
    puntoPrevio[1] =
      noise_radius *
      sin(atan2(y[idx_first_point], x[idx_first_point]));  // at least start at noise range
    puntoPrevio[2] = 0.0;
    c_label = label_suelo;

    puntoRef[0] = puntoPrevio[0];
    puntoRef[1] = puntoPrevio[1];
    puntoRef[2] = puntoPrevio[2];

    // Recorrer todos los puntos empezando por el primer punto del channel
    for (int i_p = idx_first_point; i_p < (idx_first_point + n_layers); i_p++) {
      // if (intensity[i_p] <= 0) {
      //   label[i_p] = label_noise;
      //   reason[i_p] = -1;
      //   continue;
      // }
      if (idx_thread < noisy_channels_up_to) {
        label[i_p] = label_noise;
        reason[i_p] = -3;
        continue;
      }
      if (isnan(x[i_p])) {
        label[i_p] = label_noise;
        reason[i_p] = 0;
        continue;
      }
      if (
        x[i_p] > noise_x_min && x[i_p] < noise_x_max && y[i_p] > noise_y_min &&
        y[i_p] < noise_y_max) {
        label[i_p] = label_noise;
        reason[i_p] = -2;
        continue;
      }
      if ((x[i_p] * x[i_p] + y[i_p] * y[i_p]) < (noise_radius * noise_radius)) {
        label[i_p] = label_noise;
        reason[i_p] = -2;
        continue;
      }

      // Get data of current point
      punto[0] = x[i_p];
      punto[1] = y[i_p];
      punto[2] = z[i_p] + LiDAR_height;

      // De momento hereda
      label[i_p] = c_label;
      reason[i_p] = 2;
      height_diff_respect_prev_point = punto[2] - puntoPrevio[2];

      if (
        punto[2] > threshold_height_is_obst ||
        (punto[2] > max_ground_height_within_distance &&
         (punto[0] * punto[0] + punto[1] * punto[1]) <
           threshold_distance_for_max_ground_height * threshold_distance_for_max_ground_height)) {
        // Es un obstaculo
        c_label = label_obstacle;

        // Guardamos el punto PREVIO como referencia
        puntoRef[0] = puntoPrevio[0];
        puntoRef[1] = puntoPrevio[1];
        puntoRef[2] = puntoPrevio[2];
        reason[i_p] = 1;
      } else {
        // if the previous point was floor, check for obstacles
        if (c_label == label_suelo) {
          distPuntos = sqrt(
            (punto[0] - puntoPrevio[0]) * (punto[0] - puntoPrevio[0]) +
            (punto[1] - puntoPrevio[1]) * (punto[1] - puntoPrevio[1]) +
            (punto[2] - puntoPrevio[2]) * (punto[2] - puntoPrevio[2]));
          alpha = asin(height_diff_respect_prev_point / distPuntos);

          // Check if the point has came back -> We compute the distance from the previous point to the initial one and from this point to the initial one too
          // If the distance from the previous to the origin is bigger, the current point has came back -> thats a consecuence of an obstacle
          distSueloPunto = sqrt(punto[0] * punto[0] + punto[1] * punto[1] + punto[2] * punto[2]);
          distSueloPuntoPrevio = sqrt(
            puntoPrevio[0] * puntoPrevio[0] + puntoPrevio[1] * puntoPrevio[1] +
            puntoPrevio[2] * puntoPrevio[2]);

          //Is there an obstacle?
          if (
            (alpha > max_gradiente) || ((distSueloPunto - distSueloPuntoPrevio) < 0) ||
            (is_the_first_valid_point && alpha > threshold_gradient_first_impact_is_obst)) {
            // Hay obstaculo -> ya no hay suelo
            c_label = label_obstacle;

            if ((distSueloPunto - distSueloPuntoPrevio) < 0) {
              reason[i_p] = 5;
            } else if (
              is_the_first_valid_point && alpha > threshold_gradient_first_impact_is_obst) {
              reason[i_p] = 4;
            } else {
              reason[i_p] = 3;
            }

            // Guardamos el punto PREVIO como referencia
            puntoRef[0] = puntoPrevio[0];
            puntoRef[1] = puntoPrevio[1];
            puntoRef[2] = puntoPrevio[2];
          }
        } else if (c_label == label_obstacle) {
          // if(obst_confirmed == false)
          // {
          //   if(punto[2] - puntoRef[2] > back2ground_diff_height_with_last_ground)
          //   {
          //     obst_confirmed = true;
          //   }
          // }

          // If the height is smaller, check if the beam can be hitting the floor
          if (height_diff_respect_prev_point < back2ground_diff_height_with_prev_obst) {
            // If the current point is down a reference + range
            if (punto[2] < (puntoRef[2] + back2ground_diff_height_with_last_ground)) {
              // Hay suelo
              c_label = label_suelo;
              reason[i_p] = 6;

              // Guardamos este punto como referencia
              puntoRef[0] = punto[0];
              puntoRef[1] = punto[1];
              puntoRef[2] = punto[2];
            }
          }
        }
      }
      puntoPrevio[0] = punto[0];
      puntoPrevio[1] = punto[1];
      puntoPrevio[2] = punto[2];

      // Corregimos informacion del punto actual
      label[i_p] = c_label;

      // If we have reach this point... the first valid point is set to false
      is_the_first_valid_point = false;
    }
  }
}

// ---------------------------------------- MAIN ---------------------------------------- //
void OBST_GROUND::gpu_pc_obst_ground_channel_based(
  int d_label[], int d_label_reason[], const float d_x[], const float d_y[], const float d_z[],
  const float d_intensity[], const int n_total_points, const int n_layers, const float LiDAR_pz,
  const float threshold_height_is_obst, const float threshold_gradient_first_impact_is_obst,
  const float max_ground_height_within_distance,
  const float threshold_distance_for_max_ground_height, const float max_gradiente,
  const float back2ground_diff_height_with_last_ground,
  const float back2ground_diff_height_with_prev_obst, const float noise_radius,
  const float noise_x_min, const float noise_x_max, const float noise_y_min,
  const float noise_y_max, const int label_obst, const int label_suelo, const int label_noise)
{
  static dim3 blocks_beams(
    1024);  // threads per block: multiples of 32 (warp size), max = 1024 (generally)
  static dim3 grids_beams(
    (n_total_points / n_layers + blocks_beams.x - 1) /
    blocks_beams.x);  // number of blocks: choose a number that ensures all data is processed

  int noisy_channels_up_to = 80;
  printf("Parametros sin inicializar\n");

  if (false) {
    global_pc_obst_ground_channel_based_rollinglayers<<<grids_beams, blocks_beams>>>(
      d_label, d_label_reason, d_x, d_y, d_z, d_intensity, n_total_points, n_layers, LiDAR_pz,
      threshold_height_is_obst, threshold_gradient_first_impact_is_obst,
      max_ground_height_within_distance, threshold_distance_for_max_ground_height, max_gradiente,
      back2ground_diff_height_with_last_ground, back2ground_diff_height_with_prev_obst,
      noise_radius, noise_x_min, noise_x_max, noise_y_min, noise_y_max, noisy_channels_up_to,
      label_obst, label_suelo, label_noise);
  } else {
    global_pc_obst_ground_channel_based_doubt_rollinglayers<<<grids_beams, blocks_beams>>>(
      d_label, d_label_reason, d_x, d_y, d_z, d_intensity, n_total_points, n_layers, LiDAR_pz,
      threshold_height_is_obst, threshold_gradient_first_impact_is_obst,
      max_ground_height_within_distance, threshold_distance_for_max_ground_height, max_gradiente,
      back2ground_diff_height_with_last_ground, back2ground_diff_height_with_prev_obst,
      noise_radius, noise_x_min, noise_x_max, noise_y_min, noise_y_max, noisy_channels_up_to,
      label_obst, label_suelo, label_noise);
  }
}
