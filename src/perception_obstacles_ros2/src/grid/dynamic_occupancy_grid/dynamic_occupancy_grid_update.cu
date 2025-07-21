#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

// ----------------------------------- Update occupancy masses ----------------------------------- //
__global__ void global_repsNCxNCy_update_OG_masses(
  GRID_TYPES::DOG * info_grid, const GRID_TYPES::OG * obs_OG,
  const DYN_CLASS_OG::config * config_DOG)
{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x < GRID_TYPES::NC_X && i_y < GRID_TYPES::NC_Y) {
    // To avoid loosing precision, the masses are multiplied by a factor
    double masa_obs_oc_factor =
      min(config_DOG->factor - 1e-5, obs_OG->mO[i_y][i_x] * config_DOG->factor);

    double masa_obs_libre_factor =
      min(config_DOG->factor - 1e-5, obs_OG->mF[i_y][i_x] * config_DOG->factor);

    // Update based on Dempster Rule
    double conflicto_factor = masa_obs_oc_factor * info_grid->masa_pred_libre_factor[i_y][i_x] +
                              masa_obs_libre_factor * info_grid->masa_pred_oc_factor[i_y][i_x];

    conflicto_factor =
      max(1e-5, min(config_DOG->factor * config_DOG->factor - 1e-5, conflicto_factor));

    // Dempster rule free
    info_grid->masa_act_libre_copia[i_y][i_x] =
      config_DOG->factor *
      (info_grid->masa_pred_libre_factor[i_y][i_x] * masa_obs_libre_factor +
       info_grid->masa_pred_libre_factor[i_y][i_x] *
         (config_DOG->factor - masa_obs_oc_factor - masa_obs_libre_factor) +
       (config_DOG->factor - info_grid->masa_pred_oc_factor[i_y][i_x] -
        info_grid->masa_pred_libre_factor[i_y][i_x]) *
         masa_obs_libre_factor) /
      (config_DOG->factor * config_DOG->factor - conflicto_factor);

    // Dempster rule occupied
    info_grid->masa_act_oc_factor[i_y][i_x] =
      config_DOG->factor *
      (info_grid->masa_pred_oc_factor[i_y][i_x] * masa_obs_oc_factor +
       info_grid->masa_pred_oc_factor[i_y][i_x] *
         (config_DOG->factor - masa_obs_oc_factor - masa_obs_libre_factor) +
       (config_DOG->factor - info_grid->masa_pred_oc_factor[i_y][i_x] -
        info_grid->masa_pred_libre_factor[i_y][i_x]) *
         masa_obs_oc_factor) /
      (config_DOG->factor * config_DOG->factor - conflicto_factor);

    // Avoid noise
    info_grid->masa_act_libre_copia[i_y][i_x] =
      min(config_DOG->factor, max(0.0, info_grid->masa_act_libre_copia[i_y][i_x]));

    info_grid->masa_act_oc_factor[i_y][i_x] =
      min(config_DOG->factor, max(0.0, info_grid->masa_act_oc_factor[i_y][i_x]));
  }
}

// ----------------------------------- Finish free mass ----------------------------------- //
__global__ void global_repsNCxNCy_copy_free_mass_to_desired_format(GRID_TYPES::DOG * info_grid)
{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x < GRID_TYPES::NC_X && i_y < GRID_TYPES::NC_Y) {
    info_grid->masa_act_libre_factor[i_y][i_x] = info_grid->masa_act_libre_copia[i_y][i_x];
  }
}

// ----------------------------------- New and persistent masses ----------------------------------- //
__global__ void global_repsNCxNCy_new_and_persistent_masses(
  GRID_TYPES::DOG * info_grid, const GRID_TYPES::OG * obs_OG,
  const DYN_CLASS_OG::config * config_DOG)
{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x < GRID_TYPES::NC_X && i_y < GRID_TYPES::NC_Y) {
    // Masa nueva
    if (info_grid->numero_particulas[i_y][i_x] > 0) {
      if (config_DOG->inicializacion_particulas_nacimiento_solo_en_zonas_con_ocupacion_observada) {
        info_grid->masa_nueva_factor[i_y][i_x] = 0.0;
        if (obs_OG->mO[i_x][i_y] > config_DOG->valor_asumible_como_cero_para_codigo) {
          info_grid->masa_nueva_factor[i_y][i_x] =
            info_grid->masa_act_oc_factor[i_y][i_x] * config_DOG->probNacimiento *
            (config_DOG->factor - info_grid->masa_pred_oc_factor[i_y][i_x]) /
            (info_grid->masa_pred_oc_factor[i_y][i_x] +
             config_DOG->probNacimiento *
               (config_DOG->factor - info_grid->masa_pred_oc_factor[i_y][i_x]));
        }
      } else {
        info_grid->masa_nueva_factor[i_y][i_x] =
          info_grid->masa_act_oc_factor[i_y][i_x] * config_DOG->probNacimiento *
          (config_DOG->factor - info_grid->masa_pred_oc_factor[i_y][i_x]) /
          (info_grid->masa_pred_oc_factor[i_y][i_x] +
           config_DOG->probNacimiento *
             (config_DOG->factor - info_grid->masa_pred_oc_factor[i_y][i_x]));
      }
    } else {
      info_grid->masa_nueva_factor[i_y][i_x] = info_grid->masa_act_oc_factor[i_y][i_x];
    }

    int idx_xy = GRID_UTILS_CUDA::device_sub2ind(i_y, i_x, GRID_TYPES::NC_X, GRID_TYPES::NC_Y);
    info_grid->scan_masa_nueva_factor[idx_xy] = info_grid->masa_nueva_factor[i_y][i_x];

    // Masa persistente
    info_grid->masa_persistente_factor[i_y][i_x] =
      info_grid->masa_act_oc_factor[i_y][i_x] - info_grid->masa_nueva_factor[i_y][i_x];
  }
}

// ----------------------------------- Update particles ----------------------------------- //
__global__ void global_repsNCxNCy_update_particles_based_on_updated_occupied_mass(
  PARTICLE_TYPES::PART_DOG * particles, const GRID_TYPES::DOG * info_grid)
{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x < GRID_TYPES::NC_X && i_y < GRID_TYPES::NC_Y) {
    int i_p;
    if (info_grid->numero_particulas[i_y][i_x] > 0) {
      if (info_grid->masa_pred_oc_factor[i_y][i_x] == 0.0)  // Para no dividir entre 0
      {
        for (i_p = info_grid->indice_primera_particula[i_y][i_x];
             i_p < info_grid->indice_ultima_particula[i_y][i_x];
             i_p++)  // Index of the first particle)
        {
          if (particles->valida[i_p]) {
            printf("No deberia haber particulas validas aqui!!\n");
            particles->peso_factor[i_p] = 0.0;
            particles->valida[i_p] = false;
          }
        }
      } else  // Para no dividir entre 0
      {
        for (i_p = info_grid->indice_primera_particula[i_y][i_x];
             i_p < info_grid->indice_ultima_particula[i_y][i_x];
             i_p++)  // Index of the first particle)
        {
          // Update weight
          if (particles->valida[i_p]) {
            particles->peso_factor[i_p] = particles->peso_factor[i_p] *
                                          info_grid->masa_persistente_factor[i_y][i_x] /
                                          info_grid->masa_pred_oc_factor[i_y][i_x];
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //
// ----------------------------------- Main Function ----------------------------------- //
// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //
void DYN_CLASS_OG::dynamic_occupancy_grid_update(
  GRID_TYPES::DOG * d_grid, const GRID_TYPES::OG * d_obs_OG, PARTICLE_TYPES::PART_DOG * d_particles,
  const DYN_CLASS_OG::config * d_config_DOG)
{
  // Initializations
  cudaMemset(d_grid->scan_masa_nueva_factor, 0.0, GRID_TYPES::NC_P2 * sizeof(double));

  // Kernels
  dim3 blocks_cells(32, 16);
  dim3 grid_cells(
    (GRID_TYPES::NC_X + blocks_cells.x - 1) / blocks_cells.x,
    (GRID_TYPES::NC_Y + blocks_cells.y - 1) / blocks_cells.y);

  global_repsNCxNCy_update_OG_masses<<<grid_cells, blocks_cells>>>(d_grid, d_obs_OG, d_config_DOG);

  global_repsNCxNCy_copy_free_mass_to_desired_format<<<grid_cells, blocks_cells>>>(d_grid);

  global_repsNCxNCy_new_and_persistent_masses<<<grid_cells, blocks_cells>>>(
    d_grid, d_obs_OG, d_config_DOG);

  global_repsNCxNCy_update_particles_based_on_updated_occupied_mass<<<grid_cells, blocks_cells>>>(
    d_particles, d_grid);

  printf("ESTOY IGNORANDO V2X\n");
  printf("ESTOY IGNORANDO feedback - por hacerlo estructurado...\n");
}