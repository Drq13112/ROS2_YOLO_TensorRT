#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

// cum_sum_peso tiene el tamanio de NP_P2, peroooo a partir de NP_TOT ya no hacemos nada
__global__ void global_repsNP_TOT_copiar_peso_particulas_para_cumsum(
  const PARTICLE_TYPES::PART_DOG * info_particles, double * cum_sum_peso)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < PARTICLE_TYPES::NP_TOT) {
    if (info_particles->valida[idx]) {
      cum_sum_peso[idx] = info_particles->peso_factor[idx];
    } else {
      cum_sum_peso[idx] = 0.0;
    }
  }
}

__global__ void global_resampling_en_base_al_peso(
  PARTICLE_TYPES::PART_DOG * info_particles,
  const PARTICLE_TYPES::PART_DOG * particles_for_resampling, const double * cum_sum_peso,
  const float * random_particle_selection)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i_p;

  if (idx < PARTICLE_TYPES::NP_ACT) {
    // // Find pseudo-randomly into which cell is going to be created this particle
    // peso_aleatorio = (double)random_cell_selection[idx] * peso_total;
    i_p = GRID_UTILS_CUDA::device_busqueda_binaria_ordenado_ascendente(
      cum_sum_peso,
      (double)random_particle_selection[idx] * cum_sum_peso[PARTICLE_TYPES::NP_TOT - 1], 0,
      PARTICLE_TYPES::NP_TOT - 1);

#if DEBUG_DOG
    if (i_p >= PARTICLE_TYPES::NP_TOT) {
      i_p = PARTICLE_TYPES::NP_TOT - 1;
      printf(
        "global_resampling_en_base_al_peso - no se cumple: %d < %d (NP_TOT) CORREGIDO A NP_TOT - "
        "1 \t num buscado %.20f,max num %.20f; num rand = %.20f\n",
        i_p, PARTICLE_TYPES::NP_TOT,
        (double)random_particle_selection[idx] * cum_sum_peso[PARTICLE_TYPES::NP_TOT - 1],
        cum_sum_peso[PARTICLE_TYPES::NP_TOT - 1], (double)random_particle_selection[idx]);
    }
#endif

    //			printf("Particula resampleada: %d \t peso %f \t v = %f cumsum %f \n", i_p, info_particles->peso_factor[i_p], (double)random_particle_selection[idx] * cum_sum_peso[NP_TOT - 1], cum_sum_peso[i_p]);

    info_particles->valida[idx] = true;

    info_particles->p_x[idx] = particles_for_resampling->p_x[i_p];
    info_particles->p_y[idx] = particles_for_resampling->p_y[i_p];
    info_particles->v_x[idx] = particles_for_resampling->v_x[i_p];
    info_particles->v_y[idx] = particles_for_resampling->v_y[i_p];
    info_particles->peso_factor[idx] = particles_for_resampling->peso_factor[i_p];

    info_particles->veces_remuestreada[idx] = particles_for_resampling->veces_remuestreada[i_p] + 1;
    info_particles->new_born[idx] = particles_for_resampling->new_born[i_p] + 1;

    info_particles->indice_celda_x[idx] = particles_for_resampling->indice_celda_x[i_p];
    info_particles->indice_celda_y[idx] = particles_for_resampling->indice_celda_y[i_p];
    info_particles->indice_celda[idx] = particles_for_resampling->indice_celda[i_p];

    info_particles->indices_ordenados[idx] = particles_for_resampling->indices_ordenados[i_p];
#if DEBUG_DOG
    int debug_i_celda = GRID_UTILS_CUDA::device_sub2ind(
      info_particles->indice_celda_y[idx], info_particles->indice_celda_x[idx], GRID_TYPES::NC_X,
      GRID_TYPES::NC_Y);
    if (
      debug_i_celda != info_particles->indice_celda[idx] || info_particles->indice_celda[idx] < 0 ||
      info_particles->indice_celda[idx] >= GRID_TYPES::NC_X * GRID_TYPES::NC_Y ||
      isnan((float)info_particles->indice_celda[idx]) ||
      isinf((float)info_particles->indice_celda[idx])) {
      printf(
        "Error particula (%d) resampleada [%f, %f] = [%d, %d] = %d (= %d debug)    Original: "
        "valida = %d, remuestreada = %d, newborn = %d, peso = %f\n",
        idx, info_particles->p_x[idx], info_particles->p_y[idx],
        info_particles->indice_celda_x[idx], info_particles->indice_celda_y[idx],
        info_particles->indice_celda[idx], debug_i_celda, particles_for_resampling->valida[i_p],
        particles_for_resampling->veces_remuestreada[i_p],
        (int)particles_for_resampling->new_born[i_p], particles_for_resampling->peso_factor[i_p]);
    }
#endif
    // info_particles->id_track[idx] = particles_for_resampling->id_track[i_p];

    // info_particles->class_bayes_prob_vehiculo[idx] =
    //   particles_for_resampling->class_bayes_prob_vehiculo[i_p];
    // info_particles->class_bayes_prob_peaton[idx] =
    //   particles_for_resampling->class_bayes_prob_peaton[i_p];
    // info_particles->class_bayes_prob_muro[idx] =
    //   particles_for_resampling->class_bayes_prob_muro[i_p];
    // info_particles->class_bayes_prob_otro[idx] =
    //   particles_for_resampling->class_bayes_prob_otro[i_p];

    // info_particles->class_bayes_veces_updated[idx] =
    //   particles_for_resampling->class_bayes_veces_updated[i_p];
    // info_particles->class_bayes_veces_consecutivas_no_updated[idx] =
    //   particles_for_resampling->class_bayes_veces_consecutivas_no_updated[i_p];
  }
}

// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------- MAIN CODIGO ----------------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
void DYN_CLASS_OG::dynamic_occupancy_grid_resampling(
  GRID_TYPES::DOG * d_grid, PARTICLE_TYPES::PART_DOG * d_particles,
  PARTICLE_TYPES::PART_DOG * d_particles_for_resampling,
  const GRID_TYPES::CART_Data * d_grid_cart_data, const DYN_CLASS_OG::config * d_config_DOG,
  const float * d_random_particle_selection)
{
  // ---------- Cum Sum Particle's Weight ---------- //
  dim3 block_NP_TOT(512);
  dim3 grid_NP_TOT(PARTICLE_TYPES::NP_TOT / block_NP_TOT.x);

  // Vector 1D that is gona have the cum sum
  double * d_cum_sum_peso;
  checkCudaErrors(cudaMalloc((void **)&d_cum_sum_peso, PARTICLE_TYPES::NP_P2 * sizeof(double)));

  // Point the weights
  thrust::device_ptr<double> d_weight_ptr = thrust::device_pointer_cast(d_particles->peso_factor);
  // Point the result
  thrust::device_ptr<double> d_cumsum_ptr = thrust::device_pointer_cast(d_cum_sum_peso);

  // Cum sum
  thrust::inclusive_scan(d_weight_ptr, d_weight_ptr + PARTICLE_TYPES::NP_P2, d_cumsum_ptr);

  // ---------- Resampling ---------- //
  dim3 block_NP_ACT(std::min(PARTICLE_TYPES::NP_ACT, 512));
  dim3 grid_NP_ACT(PARTICLE_TYPES::NP_ACT / block_NP_ACT.x);

  // Copy the set of particles to support the resampling
  checkCudaErrors(cudaMemcpy(
    d_particles_for_resampling, d_particles, sizeof(PARTICLE_TYPES::PART_DOG),
    cudaMemcpyDeviceToDevice));

  // Resample based on weight
  global_resampling_en_base_al_peso<<<grid_NP_ACT, block_NP_ACT>>>(
    d_particles, d_particles_for_resampling, d_cum_sum_peso, d_random_particle_selection);

  cudaFree(d_cum_sum_peso);
}
