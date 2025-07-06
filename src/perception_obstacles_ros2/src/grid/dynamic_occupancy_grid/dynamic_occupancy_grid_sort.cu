
#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

// Por lo que entiendo, este kernel (no es originalmente mio)
// - Hacer la copia al array ordenado
// - Establecer el indice de la ultima particula en la celda
__global__ void global_create_sorted_particle_array(
  const PARTICLE_TYPES::PART_DOG * particles, PARTICLE_TYPES::PART_DOG * particles_sorted,
  GRID_TYPES::DOG * info_grid)
{
  __shared__ int sharedHash[33];  // blockSize + 1 elements

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int hash;

  //	if (index < PARTICLE_TYPES::NP_ACT)
  {
    hash = particles->indice_celda[index];
    sharedHash[threadIdx.x + 1] = hash;

    if (index > 0 && threadIdx.x == 0) {
      sharedHash[0] = particles->indice_celda[index - 1];
    }
  }
  __syncthreads();

  //	if (index < PARTICLE_TYPES::NP_ACT)
  {
    if (index == 0 || hash != sharedHash[threadIdx.x]) {
      *(&info_grid->indice_primera_particula[0][0] + hash) = index;
      if (index > 0) {
        *(&info_grid->indice_ultima_particula[0][0] + sharedHash[threadIdx.x]) = index;
      }
    }

    if (index == PARTICLE_TYPES::NP_ACT - 1) {
      *(&info_grid->indice_ultima_particula[0][0] + hash) = index + 1;
    }

    // Now use the sorted index to reorder the pos and vel data
    int sortedIndex = particles->indices_ordenados[index];

    // Datos que ya han sido ordenados por thrust (y el puntero)
    particles_sorted->indices_ordenados[index] = sortedIndex;
    particles_sorted->indice_celda[index] = particles->indice_celda[index];

    // Copias para ordenar (TOOOOOOOODO EL RESTO DE VARIABLES)
    particles_sorted->p_x[index] = particles->p_x[sortedIndex];
    particles_sorted->p_y[index] = particles->p_y[sortedIndex];
    particles_sorted->v_x[index] = particles->v_x[sortedIndex];
    particles_sorted->v_y[index] = particles->v_y[sortedIndex];
    particles_sorted->peso_factor[index] = particles->peso_factor[sortedIndex];
    particles_sorted->vel_likelihood[index] = particles->vel_likelihood[sortedIndex];

    particles_sorted->indice_celda_x[index] = particles->indice_celda_x[sortedIndex];
    particles_sorted->indice_celda_y[index] = particles->indice_celda_y[sortedIndex];
#if DEBUG_DOG == 1
    int debug_i_celda = GRID_UTILS_CUDA::device_sub2ind(
      particles_sorted->indice_celda_y[index], particles_sorted->indice_celda_x[index],
      GRID_TYPES::NC_X, GRID_TYPES::NC_Y);
    if (debug_i_celda != particles_sorted->indice_celda[index]) {
      printf(
        "CUIDADO!!!! sorted part = [%f, %f] = [%d, %d] = %d != debug %d     original = [%d, %d] = "
        "%d\n",
        particles_sorted->p_x[index], particles_sorted->p_y[index],
        particles_sorted->indice_celda_x[index], particles_sorted->indice_celda_y[index],
        particles_sorted->indice_celda[index], debug_i_celda,
        particles->indice_celda_x[sortedIndex], particles->indice_celda_y[sortedIndex],
        particles->indice_celda[sortedIndex]);
    }
#endif

    particles_sorted->valida[index] = particles->valida[sortedIndex];
    particles_sorted->veces_remuestreada[index] = particles->veces_remuestreada[sortedIndex];
    particles_sorted->new_born[index] = particles->new_born[sortedIndex];
  }
}

void DYN_CLASS_OG::gpu_sort_particles_with_thrust(
  PARTICLE_TYPES::PART_DOG * d_particles, PARTICLE_TYPES::PART_DOG * d_particles_sorted,
  GRID_TYPES::DOG * d_grid, const GRID_TYPES::CART_Data * h_grid_cart_data)

{
  printf("aqui habra que añadir las copias de tracking y de bayes\n");

#if DEBUG_DOG == 1
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf("gpu_sort_particles_with_thrust - STARTING - Los indices de las particulas estan mal\n");
    exit(1);
  }
#endif

  // 1. Initialize data
  checkCudaErrors(cudaMemset(
    d_grid->indice_primera_particula, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int)));

  checkCudaErrors(cudaMemset(
    d_grid->indice_ultima_particula, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int)));

  // 2. Ordenacion con thrust  por celda
  thrust::device_ptr<int> d_indices_ordenados_beg(d_particles->indices_ordenados);
  thrust::device_ptr<int> d_indices_celda_beg(d_particles->indice_celda);

  // thrust::sequence fills with a sequence of numbers (0 : NP_ACT-1)
  thrust::sequence(d_indices_ordenados_beg, d_indices_ordenados_beg + PARTICLE_TYPES::NP_ACT);

  // thrust::sort_by_key sorts both d_indices_celda_beg and d_indices_ordenados_beg in ascending order of d_indices_celda_beg
  thrust::sort_by_key(
    d_indices_celda_beg, d_indices_celda_beg + PARTICLE_TYPES::NP_ACT, d_indices_ordenados_beg);

  // 3. Crear el array de particulas ordenadas
  checkCudaErrors(cudaMemcpy(
    d_particles_sorted, d_particles, sizeof(PARTICLE_TYPES::PART_DOG), cudaMemcpyDeviceToDevice));

  int block = 32;
  int grid = (PARTICLE_TYPES::NP_ACT + block - 1) / block;
  global_create_sorted_particle_array<<<grid, block>>>(d_particles, d_particles_sorted, d_grid);
  getLastCudaError("kernel_ordenar_masaPredOcupada_2 failed");

  // 4. Copia de nuevo al array original
  // old version kernel_copy_sorted_part<<<grid, block>>>(d_particles, d_particles_sorted);
  checkCudaErrors(cudaMemcpy(
    d_particles, d_particles_sorted, sizeof(PARTICLE_TYPES::PART_DOG), cudaMemcpyDeviceToDevice));

  cudaDeviceSynchronize();
  getLastCudaError("kernel_copy_sorted failed");

#if DEBUG_DOG == 1
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf("gpu_sort_particles_with_thrust - END - Los indices de las particulas estan mal\n");
    exit(1);
  }
#endif
}
