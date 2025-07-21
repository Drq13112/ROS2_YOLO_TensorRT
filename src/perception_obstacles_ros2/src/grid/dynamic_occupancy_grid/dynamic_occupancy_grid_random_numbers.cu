#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

// Kernel generar numeros aleatorios UNIFORMES para el desplazamiento de particulas
__global__ void global_generar_numeros_random_uniformes(
  float * randoms, const int num_reps, const int seed, const int size)
{
  int idx;

  curandState_t state;

  curand_init(
    seed + threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x * num_reps + 1, &state);

  for (int i_rep = 0; i_rep < num_reps; i_rep++) {
    idx = (blockIdx.x * blockDim.x + threadIdx.x) * num_reps + i_rep;
    if (idx < size) {
      randoms[idx] = curand_uniform(&state);
    }
  }
}

// Kernel generar numeros aleatorios normales
__global__ void global_generar_numeros_random_normales(
  float * randoms, const int num_reps, const int seed, const int size)
{
  //  Ejemplo
  //	   Bloques 3
  //	   Threads por bloque 5
  //	   Repeticiones por thread 3
  //
  //	   (blockIdx.x * blockDim.x + threadIdx.x) * num_reps + i_rep
  //
  //	   Bloque 0, Hilo 0 -> idx: 0, 1, 2
  //
  //	   Bloque 0, Hilo 1 -> idx: 3, 4, 5
  //	    - i_rep = 0 -> (0 * 5 + 1) * 3 + 0 = 3
  //	    - i_rep = 1 -> (0 * 5 + 1) * 3 + 1 = 4
  //	    - i_rep = 2 -> (0 * 5 + 1) * 3 + 2 = 5
  //
  //	   Bloque 0, Hilo 4 ->
  //	    - i_rep = 0 -> (0 * 5 + 4) * 3 + 0 = 12
  //	    - i_rep = 1 -> (0 * 5 + 4) * 3 + 1 = 13
  //	    - i_rep = 2 -> (0 * 5 + 4) * 3 + 2 = 14
  //
  //	   Bloque 1, Hilo 0 ->
  //	    - i_rep = 0 -> (1 * 5 + 0) * 3 + 0 = 15
  //	    - i_rep = 1 -> (1 * 5 + 0) * 3 + 1 = 16
  //	    - i_rep = 2 -> (1 * 5 + 0) * 3 + 2 = 17
  //
  //	   Bloque 1, Hilo 1 ->
  //	    - i_rep = 0 -> (1 * 5 + 1) * 3 + 0 = 18
  //	    - i_rep = 1 -> (1 * 5 + 1) * 3 + 1 = 19
  //	    - i_rep = 2 -> (1 * 5 + 1) * 3 + 2 = 20

  int idx;

  curandState_t state;

  curand_init(
    seed + threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x * num_reps + 1, &state);

  for (int i_rep = 0; i_rep < num_reps; i_rep++) {
    idx = (blockIdx.x * blockDim.x + threadIdx.x) * num_reps + i_rep;

    if (idx < size) {
      randoms[idx] = curand_normal(&state);
    }
  }
}

void DYN_CLASS_OG::dynamic_occupancy_grid_random_numbers_for_DOG_initialization(
  float * d_random_asociacion_inicializar, float * d_random_cell_selection_inicializar,
  float * d_random_vel_uniforme_inicializar, const long int rng_seed)
{
  dim3 grid_size_randoms_NP_ACT(4);
  dim3 block_size_randoms_NP_ACT(64);
  int num_reps_random_NP_ACT =
    PARTICLE_TYPES::NP_ACT / (grid_size_randoms_NP_ACT.x * block_size_randoms_NP_ACT.x) +
    1;  // 157;

  dim3 grid_size_randoms_2NP_ACT(4);
  dim3 block_size_randoms_2NP_ACT(128);
  int num_reps_random_2NP_ACT =
    PARTICLE_TYPES::NP_ACT / (grid_size_randoms_NP_ACT.x * block_size_randoms_NP_ACT.x) +
    1;  // 157;

  global_generar_numeros_random_uniformes<<<grid_size_randoms_NP_ACT, block_size_randoms_NP_ACT>>>(
    d_random_asociacion_inicializar, num_reps_random_NP_ACT, rng_seed + 1, PARTICLE_TYPES::NP_ACT);

  global_generar_numeros_random_uniformes<<<grid_size_randoms_NP_ACT, block_size_randoms_NP_ACT>>>(
    d_random_cell_selection_inicializar, num_reps_random_NP_ACT, rng_seed + 2,
    PARTICLE_TYPES::NP_ACT);

  global_generar_numeros_random_uniformes<<<
    grid_size_randoms_2NP_ACT, block_size_randoms_2NP_ACT>>>(
    d_random_vel_uniforme_inicializar, num_reps_random_2NP_ACT, rng_seed + 3,
    2 * PARTICLE_TYPES::NP_ACT);

  getLastCudaError("after global_generar_numeros_random_uniformes");
}

void DYN_CLASS_OG::dynamic_occupancy_grid_random_numbers_for_DOG(
  float * d_random_pred, float * d_random_particle_selection, float * d_random_cell_selection,
  float * d_random_asociacion, float * d_random_vel_uniforme, const long int rng_seed,
  const cudaStream_t streams[])
{
  dim3 grid_size_randoms_NP_ACT(4);
  dim3 block_size_randoms_NP_ACT(64);
  int num_reps_random_NP_ACT =
    PARTICLE_TYPES::NP_ACT / (grid_size_randoms_NP_ACT.x * block_size_randoms_NP_ACT.x) +
    1;  // 157;

  dim3 grid_size_randoms_2NP_ACT(4);
  dim3 block_size_randoms_2NP_ACT(128);
  int num_reps_random_2NP_ACT =
    PARTICLE_TYPES::NP_ACT / (grid_size_randoms_NP_ACT.x * block_size_randoms_NP_ACT.x) +
    1;  // 157;

  dim3 grid_size_randoms_NP_NEW(1);
  dim3 block_size_randoms_NP_NEW(32);
  int num_reps_random_NP_NEW = (PARTICLE_TYPES::NP_NEW / block_size_randoms_NP_NEW.x) + 1;

  dim3 grid_size_randoms_NP_NEW_2(1);
  dim3 block_size_randoms_NP_NEW_2(64);
  int num_reps_random_NP_NEW_2 = ((2 * PARTICLE_TYPES::NP_NEW) / block_size_randoms_NP_NEW_2.x) + 1;

  // --------------- Elegir celdas para nuevas particulas --------------- //
  // Numeros random seleccionar celda de forma aleatoria
  global_generar_numeros_random_uniformes<<<
    grid_size_randoms_NP_NEW, block_size_randoms_NP_NEW, 0, streams[0]>>>(
    d_random_cell_selection, num_reps_random_NP_NEW, rng_seed + 5, PARTICLE_TYPES::NP_NEW);

  // --------------- Estado Nuevas partículas --------------- //

  // Ruido posicion celda (lo eliminamos)
  //   global_generar_numeros_random_uniformes<<<
  //     grid_size_randoms_NP_NEW_2, block_size_randoms_NP_NEW_2, 0, streams[3]>>>(
  //     d_random_pos_celda, num_reps_random_NP_NEW_2, rng_seed + 8,
  //     NP_NEW * 2);  // TODO esto es realmente necesario? Quiza nos suda la polla esto

  // Ruido velocidad para nuevas particulas
  global_generar_numeros_random_uniformes<<<
    grid_size_randoms_NP_NEW_2, block_size_randoms_NP_NEW_2, 0, streams[1]>>>(
    d_random_vel_uniforme, num_reps_random_NP_NEW_2, rng_seed + 9, PARTICLE_TYPES::NP_NEW * 2);

  // global_generar_numeros_random_normales<<<
  //   grid_size_randoms_NP_NEW_2, block_size_randoms_NP_NEW_2, 0, streams[2]>>>(
  //   d_random_vel_normal, num_reps_random_NP_NEW_2, rng_seed+ 10, PARTICLE_TYPES::NP_NEW * 2);

  // Random por si se asocia la particula a la medida de velocidad
  global_generar_numeros_random_uniformes<<<
    grid_size_randoms_NP_NEW, block_size_randoms_NP_NEW, 0, streams[3]>>>(
    d_random_asociacion, num_reps_random_NP_NEW, rng_seed + 6, PARTICLE_TYPES::NP_NEW);

  // --------------- Predicción (es pa la próxima iteración en verdad) --------------- //
  global_generar_numeros_random_normales<<<
    grid_size_randoms_2NP_ACT, block_size_randoms_2NP_ACT, 0, streams[4]>>>(
    d_random_pred, num_reps_random_2NP_ACT, rng_seed + 4, PARTICLE_TYPES::NP_ACT * 2);

  // --------------- Resampling --------------- //
  global_generar_numeros_random_uniformes<<<
    grid_size_randoms_NP_ACT, block_size_randoms_NP_ACT, 0, streams[5]>>>(
    d_random_particle_selection, num_reps_random_NP_ACT, rng_seed + 11, PARTICLE_TYPES::NP_ACT);
}