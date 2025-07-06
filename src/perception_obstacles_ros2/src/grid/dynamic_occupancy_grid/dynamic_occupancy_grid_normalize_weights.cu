
#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

__global__ void global_compute_number_particles_within_cell(
  GRID_TYPES::DOG * info_grid, PARTICLE_TYPES::PART_DOG * particles)
{
  int i_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_x < GRID_TYPES::NC_X && i_y < GRID_TYPES::NC_Y) {
    int cnt = 0;
    for (int ip = info_grid->indice_primera_particula[i_y][i_x];
         ip < info_grid->indice_ultima_particula[i_y][i_x]; ip++) {
      if (particles->valida[ip]) {
        cnt++;
      }
    }

    info_grid->numero_particulas[i_y][i_x] = cnt;
  }
}

__global__ void global_igualar_peso_todas_particulas_respecto_masa_actualizada(
  const GRID_TYPES::DOG * info_grid, PARTICLE_TYPES::PART_DOG * particles)
{
  int i_p = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_p < PARTICLE_TYPES::NP_ACT) {
    if (particles->valida[i_p]) {
      int i_x, i_y;
      i_x = particles->indice_celda_x[i_p];
      i_y = particles->indice_celda_y[i_p];
      particles->peso_factor[i_p] =
        info_grid->masa_act_oc_factor[i_y][i_x] / ((float)info_grid->numero_particulas[i_y][i_x]);

#if DEBUG_DOG
      if (
        isnan(particles->peso_factor[i_p]) || isinf(particles->peso_factor[i_p]) ||
        particles->peso_factor[i_p] < 0) {
        printf(
          "global_igualar_peso_todas_particulas_respecto_masa_actualizada - particle %d wrong "
          "weight = %f = %f / %f   (pos = [%f, %f] => [%d, %d] = %d\n",
          i_p, particles->peso_factor[i_p], info_grid->masa_act_oc_factor[i_y][i_x],
          ((float)info_grid->numero_particulas[i_y][i_x]), particles->p_x[i_p], particles->p_y[i_p],
          particles->indice_celda_x[i_p], particles->indice_celda_y[i_p],
          particles->indice_celda[i_p]);
      }
#endif
    }
  }
}

void DYN_CLASS_OG::dynamic_occupancy_grid_normalize_weights(
  PARTICLE_TYPES::PART_DOG * d_particles, GRID_TYPES::DOG * d_grid,
  const GRID_TYPES::CART_Data * h_grid_cart_data)
{
  dim3 block_cells_particles(32, 16, 1);
  dim3 grid_cells_particles(
    (GRID_TYPES::NC_X + block_cells_particles.x - 1) / block_cells_particles.x,
    (GRID_TYPES::NC_Y + block_cells_particles.y - 1) / block_cells_particles.y, 1);

  dim3 block_NP_ACT(std::min(PARTICLE_TYPES::NP_ACT, 512));
  dim3 grid_NP_ACT(PARTICLE_TYPES::NP_ACT / block_NP_ACT.x);

  checkCudaErrors(
    cudaMemset(&d_grid->numero_particulas, 0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(int)));
  global_compute_number_particles_within_cell<<<grid_cells_particles, block_cells_particles>>>(
    d_grid, d_particles);

#if DEBUG_DOG
  if (!DYN_CLASS_OG::check_cell_particle_number(d_particles, d_grid, h_grid_cart_data)) {
    printf("Los indices no coinciden o el numero de particulas no coincide!\n ");
    exit(1);
  }
#endif

  global_igualar_peso_todas_particulas_respecto_masa_actualizada<<<grid_NP_ACT, block_NP_ACT>>>(
    d_grid, d_particles);
}