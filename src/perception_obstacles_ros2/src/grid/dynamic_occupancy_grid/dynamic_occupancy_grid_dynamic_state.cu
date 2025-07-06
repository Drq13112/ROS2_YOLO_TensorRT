#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

__global__ void global_repsNCxNCy_calcular_estado_dinamico(
  GRID_TYPES::DOG * info_grid, const PARTICLE_TYPES::PART_DOG * particles,
  const DYN_CLASS_OG::config * config_DOG)

{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x < GRID_TYPES::NC_X && i_y < GRID_TYPES::NC_Y) {
    int contador_particulas = 0;

    // Dynamic state
    if (info_grid->masa_persistente_factor[i_y][i_x] > 0.0) {
      // Initialize to zero because we are going to use it as summation
      info_grid->vel_media_x[i_y][i_x] = 0.0;
      info_grid->vel_media_y[i_y][i_x] = 0.0;
      info_grid->vel_media_modulo[i_y][i_x] = 0.0;
      info_grid->vel_media_angulo[i_y][i_x] = 0.0;

      // Index of the first particle
      for (int i_p = info_grid->indice_primera_particula[i_y][i_x];
           i_p < info_grid->indice_ultima_particula[i_y][i_x];
           i_p++)  // Index of the first particle)
      {
        info_grid->mean_particulas_veces_remuestreadas[i_y][i_x] +=
          particles->veces_remuestreada[i_p];
        contador_particulas++;

        if (
          particles->valida[i_p] && particles->veces_remuestreada[i_p] >=
                                      config_DOG->calculo_vel_numero_minimo_de_veces_remuestreada) {
          // Start computing mean speed
          info_grid->vel_media_x[i_y][i_x] += (particles->v_x[i_p] * particles->peso_factor[i_p]);
          info_grid->vel_media_y[i_y][i_x] += (particles->v_y[i_p] * particles->peso_factor[i_p]);

          info_grid->vel_media_modulo[i_y][i_x] +=
            particles->peso_factor[i_p] * sqrt(
                                            particles->v_x[i_p] * particles->v_x[i_p] +
                                            particles->v_y[i_p] * particles->v_y[i_p]);
          info_grid->vel_media_angulo[i_y][i_x] +=
            particles->peso_factor[i_p] * atan2(particles->v_y[i_p], particles->v_x[i_p]);

          info_grid->sum_vel_peso_factor[i_y][i_x] += particles->peso_factor[i_p];
          info_grid->numero_particulas_velocidad[i_y][i_x]++;
        }
      }
      info_grid->mean_particulas_veces_remuestreadas[i_y][i_x] /= ((float)contador_particulas);

      // + Mean speed of the cell
      // ES IMPORTANTE ESTA SEPARACION, ESTO ES EL NUMERO DE PARTICULAS con velocidad NO SON TODAS LAS PARTICULAS (queremos las que han sido remuestreadas x veces)
      if (
        info_grid->numero_particulas_velocidad[i_y][i_x] == 0 ||
        info_grid->sum_vel_peso_factor[i_y][i_x] == 0.0) {
        info_grid->vel_media_x[i_y][i_x] = 0;
        info_grid->vel_media_y[i_y][i_x] = 0;
        info_grid->vel_sigma_x[i_y][i_x] = 0;
        info_grid->vel_sigma_y[i_y][i_x] = 0;
        info_grid->vel_cov_xy[i_y][i_x] = 0;

        info_grid->vel_media_modulo[i_y][i_x] = 0;
        info_grid->vel_media_angulo[i_y][i_x] = 0;
        info_grid->mean_particulas_veces_remuestreadas[i_y][i_x] = 0.0;
      } else {
        info_grid->info_vel_valida[i_y][i_x] = true;

        info_grid->vel_media_x[i_y][i_x] /= info_grid->sum_vel_peso_factor[i_y][i_x];
        info_grid->vel_media_y[i_y][i_x] /= info_grid->sum_vel_peso_factor[i_y][i_x];
        info_grid->vel_media_modulo[i_y][i_x] /= info_grid->sum_vel_peso_factor[i_y][i_x];
        info_grid->vel_media_angulo[i_y][i_x] /= info_grid->sum_vel_peso_factor[i_y][i_x];

        //				if(abs(sqrt(info_grid->vel_media_x[i_y][i_x] * info_grid->vel_media_x[i_y][i_x] + info_grid->vel_media_y[i_y][i_x] * info_grid->vel_media_y[i_y][i_x])
        //						- info_grid->vel_media_modulo[i_y][i_x]) > 1)
        //				{
        //					printf("componentes [%f, %f] = %f   mod = %f\n", info_grid->vel_media_x[i_y][i_x], info_grid->vel_media_y[i_y][i_x],
        //							sqrt(info_grid->vel_media_x[i_y][i_x] * info_grid->vel_media_x[i_y][i_x] + info_grid->vel_media_y[i_y][i_x] * info_grid->vel_media_y[i_y][i_x]),
        //							info_grid->vel_media_modulo[i_y][i_x]);
        //				}

        // + Compute sigma
        info_grid->vel_sigma_x[i_y][i_x] = 0.0;
        info_grid->vel_sigma_y[i_y][i_x] = 0.0;
        info_grid->vel_cov_xy[i_y][i_x] = 0.0;

        // double aux_mod, aux_ang, resta_ang;
        for (int i_p = info_grid->indice_primera_particula[i_y][i_x];
             i_p < info_grid->indice_ultima_particula[i_y][i_x];
             i_p++)  // Index of the first particle)
        {
          if (
            particles->valida[i_p] &&
            particles->veces_remuestreada[i_p] >=
              config_DOG->calculo_vel_numero_minimo_de_veces_remuestreada) {
            // Data for dynamic state
            info_grid->vel_sigma_x[i_y][i_x] +=
              ((particles->v_x[i_p] - info_grid->vel_media_x[i_y][i_x]) *
               (particles->v_x[i_p] - info_grid->vel_media_x[i_y][i_x]) *
               particles->peso_factor[i_p]);
            info_grid->vel_sigma_y[i_y][i_x] +=
              ((particles->v_y[i_p] - info_grid->vel_media_y[i_y][i_x]) *
               (particles->v_y[i_p] - info_grid->vel_media_y[i_y][i_x]) *
               particles->peso_factor[i_p]);

            //						info_grid->vel_sigma_x[i_y][i_x] += (particles->v_x[i_p] * particles->v_x[i_p] * particles->peso_factor[i_p]);
            //						info_grid->vel_sigma_y[i_y][i_x] += (particles->v_y[i_p] * particles->v_y[i_p] * particles->peso_factor[i_p]);

            //						info_grid->vel_cov_xy[i_y][i_x] += ( (particles->v_x[i_p] - info_grid->vel_media_x[i_y][i_x]) * (particles->v_y[i_p] - info_grid->vel_media_y[i_y][i_x]) * particles->peso_factor[i_p] );
            info_grid->vel_cov_xy[i_y][i_x] +=
              particles->v_x[i_p] * particles->v_y[i_p] * particles->peso_factor[i_p];

            // aux_mod = sqrt(
            //   particles->v_x[i_p] * particles->v_x[i_p] +
            //   particles->v_y[i_p] * particles->v_y[i_p]);
            // aux_ang = atan2(particles->v_y[i_p], particles->v_x[i_p]);
            // device_restar_angulos(&resta_ang, aux_ang, info_grid->vel_media_angulo[i_y][i_x]);
          }
        }

        info_grid->vel_sigma_x[i_y][i_x] =
          sqrt(info_grid->vel_sigma_x[i_y][i_x] / info_grid->sum_vel_peso_factor[i_y][i_x]);
        info_grid->vel_sigma_y[i_y][i_x] =
          sqrt(info_grid->vel_sigma_y[i_y][i_x] / info_grid->sum_vel_peso_factor[i_y][i_x]);

        //				info_grid->vel_sigma_x[i_y][i_x] = sqrt( info_grid->vel_sigma_x[i_y][i_x] / info_grid->sum_vel_peso_factor[i_y][i_x] - info_grid->vel_media_x[i_y][i_x] * info_grid->vel_media_x[i_y][i_x]);
        //				info_grid->vel_sigma_y[i_y][i_x] = sqrt( info_grid->vel_sigma_y[i_y][i_x] / info_grid->sum_vel_peso_factor[i_y][i_x] - info_grid->vel_media_y[i_y][i_x] * info_grid->vel_media_y[i_y][i_x]);

        info_grid->vel_cov_xy[i_y][i_x] =
          info_grid->vel_cov_xy[i_y][i_x] / info_grid->sum_vel_peso_factor[i_y][i_x] -
          info_grid->vel_media_x[i_y][i_x] * info_grid->vel_media_y[i_y][i_x];

        if (
          abs(info_grid->vel_cov_xy[i_y][i_x]) > 1e-5 && info_grid->vel_sigma_x[i_y][i_x] > 1e-5 &&
          info_grid->vel_sigma_y[i_y][i_x] > 1e-5) {
          if (info_grid->numero_particulas_velocidad[i_y][i_x] > 10) {
            info_grid->vel_mahalanobis[i_y][i_x] = sqrt(
              (-info_grid->vel_media_x[i_y][i_x] *
                 (info_grid->vel_cov_xy[i_y][i_x] * info_grid->vel_media_y[i_y][i_x] -
                  info_grid->vel_media_x[i_y][i_x] * info_grid->vel_sigma_y[i_y][i_x] *
                    info_grid->vel_sigma_y[i_y][i_x]) -
               info_grid->vel_media_y[i_y][i_x] *
                 (info_grid->vel_cov_xy[i_y][i_x] * info_grid->vel_media_x[i_y][i_x] -
                  info_grid->vel_media_y[i_y][i_x] * info_grid->vel_sigma_x[i_y][i_x] *
                    info_grid->vel_sigma_x[i_y][i_x])) /
              (-info_grid->vel_cov_xy[i_y][i_x] * info_grid->vel_cov_xy[i_y][i_x] +
               info_grid->vel_sigma_x[i_y][i_x] * info_grid->vel_sigma_x[i_y][i_x] *
                 info_grid->vel_sigma_y[i_y][i_x] * info_grid->vel_sigma_y[i_y][i_x]));
          } else {
            info_grid->vel_mahalanobis[i_y][i_x] = sqrt(
              info_grid->vel_media_x[i_y][i_x] / info_grid->vel_sigma_x[i_y][i_x] *
                info_grid->vel_media_x[i_y][i_x] / info_grid->vel_sigma_x[i_y][i_x] +
              info_grid->vel_media_y[i_y][i_x] / info_grid->vel_sigma_y[i_y][i_x] *
                info_grid->vel_media_y[i_y][i_x] / info_grid->vel_sigma_y[i_y][i_x]);
          }
        } else {
          info_grid->vel_mahalanobis[i_y][i_x] = 0.0;
        }
      }
    }
  }
}

void DYN_CLASS_OG::dynamic_occupancy_grid_dynamic_state(
  const bool * flag_particles, GRID_TYPES::DOG * d_grid,
  const PARTICLE_TYPES::PART_DOG * d_particles, const DYN_CLASS_OG::config * d_config_DOG)
{
  dim3 blocks(32, 16);
  dim3 grids(
    (GRID_TYPES::NC_X + blocks.x - 1) / blocks.x, (GRID_TYPES::NC_Y + blocks.y - 1) / blocks.y);

  checkCudaErrors(
    cudaMemset(d_grid->vel_media_x, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(
    cudaMemset(d_grid->vel_media_y, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(
    cudaMemset(d_grid->vel_sigma_x, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(
    cudaMemset(d_grid->vel_sigma_y, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(
    cudaMemset(d_grid->vel_cov_xy, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(
    cudaMemset(d_grid->vel_media_modulo, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(
    cudaMemset(d_grid->vel_media_angulo, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(cudaMemset(
    d_grid->mean_particulas_veces_remuestreadas, 0,
    GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(float)));

  checkCudaErrors(cudaMemset(
    d_grid->sum_vel_peso_factor, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double)));

  checkCudaErrors(cudaMemset(
    d_grid->numero_particulas_velocidad, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int)));

  checkCudaErrors(
    cudaMemset(d_grid->info_vel_valida, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(bool)));

  if (*flag_particles) {
    global_repsNCxNCy_calcular_estado_dinamico<<<grids, blocks>>>(
      d_grid, d_particles, d_config_DOG);
  }
}