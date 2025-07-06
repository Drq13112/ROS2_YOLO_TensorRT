#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

// ----------------------------------- Kernel correct ego motion ----------------------------------- //
__global__ void global_repsNPACT_particles_correct_ego_motion(
  PARTICLE_TYPES::PART_DOG * particles, const EGO_VEH::INFO_ego * info_coche,
  const float number_particles)
{
  int i_p = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_p < number_particles) {
    // Correct displacement
    particles->p_x[i_p] -= info_coche->delta_x;
    particles->p_y[i_p] -= info_coche->delta_y;
    device_matrizRotacionZInversa_seno_coseno_precalculado(
      &particles->p_x[i_p], &particles->p_y[i_p], info_coche->sin_delta_yaw,
      info_coche->cos_delta_yaw);

    // Transform speed (module remains, relative angle changes) + add noise of prediction
    float speed_module =
      sqrt(particles->v_x[i_p] * particles->v_x[i_p] + particles->v_y[i_p] * particles->v_y[i_p]);

    float angle = atan2(particles->v_y[i_p], particles->v_x[i_p]) - info_coche->delta_yaw;

    particles->v_x[i_p] = speed_module * cos(angle);
    particles->v_y[i_p] = speed_module * sin(angle);
  }
}

// ----------------------------------- Kernel predecir particulas ----------------------------------- //
__global__ void global_repsNPACT_particles_prediction(
  PARTICLE_TYPES::PART_DOG * particles, const float * d_numeros_aleatorios_normales,
  const DYN_CLASS_OG::config * config_DOG, const EGO_VEH::INFO_ego * info_coche,
  const int number_particles)
{
  int i_p = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_p < number_particles) {
    // Add velocity noise
    particles->v_x[i_p] += d_numeros_aleatorios_normales[i_p * 2] *
                           config_DOG->desviacion_vel_movimiento_particulas_ms_ciclo;
    particles->v_y[i_p] += d_numeros_aleatorios_normales[i_p * 2 + 1] *
                           config_DOG->desviacion_vel_movimiento_particulas_ms_ciclo;

    // THEN predict position (it is not the standard model, but we weight particles based on their displacement correcteness,
    // if we add noise after, we can end with particles in correct cells but with wrong velocity vectors (at least vel vectors that not fit this position))
    particles->p_x[i_p] += (particles->v_x[i_p] * info_coche->delta_t);
    particles->p_y[i_p] += (particles->v_y[i_p] * info_coche->delta_t);

    particles->peso_factor[i_p] *= config_DOG->probabilidad_supervivencia_particula;
    particles->new_born[i_p] = false;
  }
}

// ----------------------------------- Kernel compute indexes ----------------------------------- //
__global__ void global_repsNPACT_particles_compute_cell_indexes(
  PARTICLE_TYPES::PART_DOG * particles, const GRID_TYPES::CART_Data * grid_cart_data,
  const int number_particles)
{
  int i_p = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_p < number_particles) {
    GRID_UTILS_CUDA::device_calculo_indices_celda(
      &particles->indice_celda_x[i_p], &particles->indice_celda_y[i_p], particles->p_x[i_p],
      particles->p_y[i_p], grid_cart_data->NC_X, grid_cart_data->NC_Y, grid_cart_data->MIN_X,
      grid_cart_data->MIN_Y, grid_cart_data->RES);

    particles->indice_celda[i_p] = GRID_UTILS_CUDA::device_sub2ind(
      particles->indice_celda_y[i_p], particles->indice_celda_x[i_p], grid_cart_data->NC_X,
      grid_cart_data->NC_Y);

    if (particles->indice_celda_x[i_p] == -1 || particles->indice_celda_y[i_p] == -1) {
      particles->valida[i_p] = false;
      particles->peso_factor[i_p] = 0.0;
    }

    if (particles->indice_celda[i_p] < 0 && particles->valida[i_p]) {
      printf(
        "CUIDADOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO part valida con pos = [%f, %f] = [%d, %d] = "
        "%d;  peso = %f\n",
        particles->p_x[i_p], particles->p_y[i_p], particles->indice_celda_x[i_p],
        particles->indice_celda_y[i_p], particles->indice_celda[i_p], particles->peso_factor[i_p]);
    }
  }
}

// __global__ void global_compute_number_particles_within_cell(
//   GRID_TYPES::DOG * info_grid, PARTICLE_TYPES::PART_DOG * particles)
// {
//   int i_x = blockIdx.x * blockDim.x + threadIdx.x;
//   int i_y = blockIdx.y * blockDim.y + threadIdx.y;

//   int cnt = 0;
//   int i_lp = info_grid->indice_ultima_particula[i_y][i_x];
//   for (int ip = info_grid->indice_primera_particula[i_y][i_x]; ip < i_lp; ip++) {
//     // Aglomerar
//     if (particles->valida[ip]) {
//       cnt++;
//     }
//   }

//   info_grid->numero_particulas[i_y][i_x] = cnt;
// }

__global__ void global_compute_number_particles_within_cell_and_accumulate_occupied_mass(
  GRID_TYPES::DOG * info_grid, PARTICLE_TYPES::PART_DOG * particles,
  const GRID_TYPES::CART_Data * grid_cart_data)
{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x >= 0 && i_x < grid_cart_data->NC_X && i_y >= 0 && i_y < grid_cart_data->NC_Y) {
    info_grid->numero_particulas[i_y][i_x] = 0;
    info_grid->masa_pred_oc_factor[i_y][i_x] = 0.0;

    int i_lp = info_grid->indice_ultima_particula[i_y][i_x];
    for (int ip = info_grid->indice_primera_particula[i_y][i_x]; ip < i_lp; ip++) {
      if (particles->valida[ip]) {
        info_grid->masa_pred_oc_factor[i_y][i_x] += particles->peso_factor[ip];
        info_grid->numero_particulas[i_y][i_x]++;
      }
    }
  }
}

__global__ void global_repsNCxNCy_predict_masses_normalize_particles(
  PARTICLE_TYPES::PART_DOG * particles, GRID_TYPES::DOG * info_grid,
  const EGO_VEH::INFO_ego * info_coche, const GRID_TYPES::CART_Data * grid_cart_data,
  const DYN_CLASS_OG::config * config_DOG, const double factor_degradacion_masa_libre)
{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x < grid_cart_data->NC_X && i_y < grid_cart_data->NC_Y) {
    double sum_peso_celda_factor =
      info_grid->masa_pred_oc_factor[i_y][i_x];  // Es una variable de control, se inicializa asi

    //////////////////////////////////////////////////////////////////////////////////////
    // Masa predicha ocupada
    // - Por numero de particulas
    // - Limitado a un valor maximo
    if (info_grid->numero_particulas[i_y][i_x] > 0) {
      // Ajustamos la masa predicha al valor maximo establecido por parametro
      sum_peso_celda_factor =
        min(sum_peso_celda_factor, config_DOG->max_pred_certainty * config_DOG->factor);

      // Como hemos modificado la masa predicha -> reajustamos el peso de las particulas en consecuencia
      for (int i_p = info_grid->indice_primera_particula[i_y][i_x];
           i_p < info_grid->indice_ultima_particula[i_y][i_x];
           i_p++)  // Index of the first particle)
      {
        if (particles->valida[i_p]) {
          if (isnan(
                particles->peso_factor[i_p] * sum_peso_celda_factor /
                info_grid->masa_pred_oc_factor[i_y][i_x])) {
            printf(
              "PARTICULA NaN: %f = %f * %f / %f !!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
              particles->peso_factor[i_p] * sum_peso_celda_factor /
                info_grid->masa_pred_oc_factor[i_y][i_x],
              particles->peso_factor[i_p], sum_peso_celda_factor,
              info_grid->masa_pred_oc_factor[i_y][i_x]);
          }

          particles->peso_factor[i_p] = particles->peso_factor[i_p] * sum_peso_celda_factor /
                                        info_grid->masa_pred_oc_factor[i_y][i_x];
        }
      }

      // Guardamos la modificación
      info_grid->masa_pred_oc_factor[i_y][i_x] = sum_peso_celda_factor;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Masa predicha libre
    // - Rotacion por movimiento de Ego
    // - Degradacion por el paso del tiempo

    // Rotar el centro y ver en que celda estaba antes
    float centro_celda_x_1 = grid_cart_data->centro_x[i_x];
    float centro_celda_y_1 = grid_cart_data->centro_y[i_y];

    device_matrizRotacionZ_seno_coseno_precalculado(
      &centro_celda_x_1, &centro_celda_y_1, info_coche->sin_delta_yaw, info_coche->cos_delta_yaw);
    centro_celda_x_1 += info_coche->delta_x;
    centro_celda_y_1 += info_coche->delta_y;

    int i_x_1, i_y_1;

    GRID_UTILS_CUDA::device_calculo_indices_celda(
      &i_x_1, &i_y_1, centro_celda_x_1, centro_celda_y_1, grid_cart_data->NC_X,
      grid_cart_data->NC_Y, grid_cart_data->MIN_X, grid_cart_data->MIN_Y, grid_cart_data->RES);

    if (i_x_1 == -1) {
      info_grid->masa_pred_libre_factor[i_y][i_x] = 0.0;
    } else {
      // La masa libre se predice como una degradacion
      info_grid->masa_pred_libre_factor[i_y][i_x] =
        factor_degradacion_masa_libre * info_grid->masa_act_libre_factor[i_y_1][i_x_1];

      // sin embargo, debe respectar que m(O) + m(L) <= 1
      info_grid->masa_pred_libre_factor[i_y][i_x] = min(
        info_grid->masa_pred_libre_factor[i_y][i_x],
        config_DOG->factor - info_grid->masa_pred_oc_factor[i_y][i_x]);
    }
  }
}

// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //
// ----------------------------------- Main Function ----------------------------------- //
// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //
void DYN_CLASS_OG::dynamic_occupancy_prediction(
  const bool * flag_particles, PARTICLE_TYPES::PART_DOG * d_particles,
  PARTICLE_TYPES::PART_DOG * d_particles_sorted, GRID_TYPES::DOG * d_grid,
  const GRID_TYPES::CART_Data * h_grid_cart_data, const GRID_TYPES::CART_Data * d_grid_cart_data,
  const DYN_CLASS_OG::config * h_config_DOG, const DYN_CLASS_OG::config * d_config_DOG,
  const float * d_random_pred, const EGO_VEH::INFO_ego * h_info_coche,
  const EGO_VEH::INFO_ego * d_info_coche)
{
  // ---------- Predict particles ---------- //
  if (*flag_particles) {
    dim3 block_NP_ACT(512);
    dim3 grid_NP_ACT(PARTICLE_TYPES::NP_ACT / block_NP_ACT.x);

    // Check dimensions
    if (grid_NP_ACT.x * block_NP_ACT.x < PARTICLE_TYPES::NP_ACT) {
      printf("     !!! La configuración de grid_NP_ACT no cumple con NP_ACT\n");
      exit(1);
    }
    if (PARTICLE_TYPES::NP_ACT % block_NP_ACT.x != 0.0) {
      printf(
        "     !!! NP_ACT (%d) tiene que ser divisible entre block_NP_ACT.x (%d). Porque lo estas "
        "usando para definir grid_NP_ACT, si el resto no es cero te van a faltar bloques \n\n",
        PARTICLE_TYPES::NP_ACT, block_NP_ACT.x);
      exit(1);
    }

    // Correct ego motion
    global_repsNPACT_particles_correct_ego_motion<<<grid_NP_ACT, block_NP_ACT>>>(
      d_particles, d_info_coche, PARTICLE_TYPES::NP_ACT);

    // Predict particles
    global_repsNPACT_particles_prediction<<<grid_NP_ACT, block_NP_ACT>>>(
      d_particles, d_random_pred, d_config_DOG, d_info_coche, PARTICLE_TYPES::NP_ACT);

    // Compute new indexes
    global_repsNPACT_particles_compute_cell_indexes<<<grid_NP_ACT, block_NP_ACT>>>(
      d_particles, d_grid_cart_data, PARTICLE_TYPES::NP_ACT);
#if DEBUG_DOG
    if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
      printf(
        "dynamic_occupancy_prediction - prediccion - Los indices de las particulas estan mal "
        "\n");
      exit(1);
    }
#endif
    // Sort particles based on the new position
    gpu_sort_particles_with_thrust(d_particles, d_particles_sorted, d_grid, h_grid_cart_data);
#if DEBUG_DOG == 1
    if (!DYN_CLASS_OG::ordenar_particulas(
          d_particles, d_particles_sorted, d_grid, h_grid_cart_data)) {
      printf("dynamic_occupancy_prediction - sort - la ordenacion esta mal\n");
      exit(1);
    }
    if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
      printf("dynamic_occupancy_prediction - sort - Los indices de las particulas estan mal\n");
      exit(1);
    }
#endif
  }
  getLastCudaError("after predicted particles");

  // ---------- GRID ---------- //

  dim3 block_cells_particles(32, 16, 1);
  dim3 grid_cells_particles(
    (GRID_TYPES::NC_X + block_cells_particles.x - 1) / block_cells_particles.x,
    (GRID_TYPES::NC_Y + block_cells_particles.y - 1) / block_cells_particles.y, 1);

  // Assign particles to cells and compute predicted occupied mass
  global_compute_number_particles_within_cell_and_accumulate_occupied_mass<<<
    grid_cells_particles, block_cells_particles>>>(d_grid, d_particles, d_grid_cart_data);

#if DEBUG_DOG == 1
  printf("dynamic_occupancy_prediction check_cell_particle_number ... \n");
  check_cell_particle_number(d_particles, d_grid, h_grid_cart_data);
  printf("... debug particulas ordenadas check_cell_particle_number conseguido\n");
#endif

  // Masses prediction
  double factor_degradacion_masa_libre =
    exp(-h_info_coche->delta_t / h_config_DOG->tiempoMaxDegradacionLibertad);

  global_repsNCxNCy_predict_masses_normalize_particles<<<
    grid_cells_particles, block_cells_particles>>>(
    d_particles, d_grid, d_info_coche, d_grid_cart_data, d_config_DOG,
    factor_degradacion_masa_libre);

  cudaDeviceSynchronize();

  getLastCudaError("prediccion grid failed");
}