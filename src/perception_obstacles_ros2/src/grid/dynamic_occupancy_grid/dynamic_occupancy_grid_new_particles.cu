#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

// Randomly spread particles based on the new mass
__global__ void global_repartir_particulas_nuevas(
  GRID_TYPES::DOG * info_grid, PARTICLE_TYPES::PART_DOG * particles,
  const float * random_cell_selection, const int NC_X, const int NC_Y, const int NP_ACT,
  const int NC_P2, const int particle_first_valid_index, const int particle_last_valid_index)
{
  // Vamos a recorrer particulas -> idx corresponde a cada una de las nuevas particulas que se crean
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_new = idx + particle_first_valid_index;
  int i_x, i_y, i_celda;
  double peso_total;

  if (idx_new >= particle_first_valid_index && idx_new < particle_last_valid_index) {
    peso_total = info_grid->scan_masa_nueva_factor[NC_P2 - 1];

    // Find pseudo-randomly into which cell is going to be created this particle
    // peso_aleatorio = (double)random_cell_selection[idx] * peso_total;
    i_celda = GRID_UTILS_CUDA::device_busqueda_binaria_ordenado_ascendente(
      info_grid->scan_masa_nueva_factor, (double)random_cell_selection[idx] * peso_total, 0,
      NC_P2 - 1);

    GRID_UTILS_CUDA::device_ind2sub(i_celda, NC_X, NC_Y, &i_y, &i_x);

    // Save a counter with the total number of new particles in this cell
    atomicAdd(&info_grid->numero_particulas_nuevas[i_y][i_x], 1);

    // Save inside the particle, to which cell it belongs too
    particles->indice_celda_x[idx_new] = i_x;
    particles->indice_celda_y[idx_new] = i_y;
    particles->indice_celda[idx_new] = i_celda;
#if DEBUG_DOG == 1
    int debug_i_celda;
    debug_i_celda = GRID_UTILS_CUDA::device_sub2ind(
      particles->indice_celda_y[idx_new], particles->indice_celda_x[idx_new], GRID_TYPES::NC_X,
      GRID_TYPES::NC_Y);
    if (debug_i_celda != particles->indice_celda[idx_new]) {
      printf(
        "error!! particula nueva indices guardado [%d, %d] = %d != %d\n",
        particles->indice_celda_x[idx_new], particles->indice_celda_y[idx_new],
        particles->indice_celda[idx_new], debug_i_celda);
    }
#endif
  }
}

// Initialize particles
__global__ void global_inicializar_particulas_nuevas(
  const DYN_CLASS_OG::config * config_DOG, const GRID_TYPES::DOG * info_grid,
  PARTICLE_TYPES::PART_DOG * particles, const GRID_TYPES::CART_Data * grid_cart_data,
  const float * random_asociacion, const float * random_vel_uniforme, const int NP_ACT,
  const int particle_first_valid_index, const int particle_last_valid_index)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_new = idx + particle_first_valid_index;
  int i_x, i_y;

  if (idx_new >= particle_first_valid_index && idx_new < particle_last_valid_index) {
    // Inside which cell is this particle?
    i_x = particles->indice_celda_x[idx_new];
    i_y = particles->indice_celda_y[idx_new];

    // Al new particles are valid
    particles->valida[idx_new] = true;
    particles->new_born[idx_new] = true;
    particles->veces_remuestreada[idx_new] = 0;

    // The weight of this particle is proportional to the number of particles and the new mass
    particles->peso_factor[idx_new] = info_grid->masa_nueva_factor[i_y][i_x] /
                                      ((float)info_grid->numero_particulas_nuevas[i_y][i_x]);

    // The particle achieves the position of the cell (lets ignore noise within it... assuming cells are small enough)
    particles->p_x[idx_new] = grid_cart_data->centro_x[i_x];
    particles->p_y[idx_new] = grid_cart_data->centro_y[i_y];
#if DEBUG_DOG == 1
    int debug_ix, debug_iy, debug_i_celda;

    GRID_UTILS_CUDA::device_calculo_indices_celda(
      &debug_ix, &debug_iy, particles->p_x[idx_new], particles->p_y[idx_new], GRID_TYPES::NC_X,
      GRID_TYPES::NC_Y, grid_cart_data->MIN_X, grid_cart_data->MIN_Y, grid_cart_data->RES);
    debug_i_celda =
      GRID_UTILS_CUDA::device_sub2ind(debug_iy, debug_ix, GRID_TYPES::NC_X, GRID_TYPES::NC_Y);

    if (
      debug_i_celda != particles->indice_celda[idx_new] ||
      debug_ix != particles->indice_celda_x[idx_new] ||
      debug_iy != particles->indice_celda_y[idx_new]) {
      printf(
        "error!! particula nueva posicion [%f, %f] indices guardados [%d, %d] = %d != indices "
        "debug [%d, %d] = %d\n",
        particles->p_x[idx_new], particles->p_y[idx_new], particles->indice_celda_x[idx_new],
        particles->indice_celda_y[idx_new], particles->indice_celda[idx_new], debug_ix, debug_iy,
        debug_i_celda);
    }
#endif

    // 1. Realimentacion
    /*
    if (
      d->realimentacion_activa && d->realimentacion_feed_inicializacion_particulas &&
      ((info_grid->probAsociacion_cluster[i_y][i_x] > 0.00001 &&
        (d->realimentacion_por_clusters_activa ||
         d->analizar_resultados_falsear_feedback_con_ground_truth)) ||
       (info_grid->track_fixed_box_probAsociacion_inicializacion[i_y][i_x] > 0.001 &&
        d->realimentacion_por_fixed_box_activa) ||
       (d->realimentacion_muro_estatico_por_bayes_activa &&
        info_grid->class_bayes_hay_info_prob[i_y][i_x] &&
        info_grid->class_bayes_prob_mayor[i_y][i_x] == d->class_bayes_tipo_muro))) {
      bool usar_realimentacion = true;
      if (d->mapa_ruta_activo && d_map->solo_realimentar_celdas_dentro_carretera) {
        if (map_ego->type[idx_route_map] != d_map->type_carretera) {
          usar_realimentacion = false;
        }
      }

      if (usar_realimentacion) {
        double aux_pA = 0.0, aux_vx, aux_vy;
        if (
          info_grid->track_fixed_box_probAsociacion_inicializacion[i_y][i_x] > 0.001 &&
          d->realimentacion_por_fixed_box_activa) {
          aux_vx = info_grid->track_fixed_box_vel_inferida_x[i_y][i_x];
          aux_vy = info_grid->track_fixed_box_vel_inferida_y[i_y][i_x];

          aux_pA = info_grid->track_fixed_box_probAsociacion_inicializacion[i_y][i_x];
          //					aux_pA = min(d->realimentacion_inicializacion_particulas_pA_fixed_box_max, max(0.0, aux_pA)); esto ya esta incluido

          aux_inicializacion_particulas_sigma_vel =
            d->realimentacion_inicializacion_particulas_sigma_vel_fixed_box;
          //					printf("kernel_crear_particulas_roughnening - box %f -> %f\n", info_grid->track_fixed_box_probAsociacion[i_y][i_x], aux_pA);

          if (
            aux_pA > d->realimentacion_inicializacion_particulas_pA_fixed_box_max * 1.00001 ||
            aux_pA < -1e-4) {
            printf(
              "no puede ser, has calculado una P(A) inicializacion fixed box que no esta dentro de "
              "los rangos \t se deberia cumplir: [%f < %f < %f]\n",
              0.0, aux_pA, d->realimentacion_inicializacion_particulas_pA_fixed_box_max);
          }
        } else if (
          info_grid->class_bayes_hay_info_prob[i_y][i_x] &&
          info_grid->class_bayes_prob_mayor[i_y][i_x] == d->class_bayes_tipo_muro) {
          aux_vx = 0.0;
          aux_vy = 0.0;
          //					aux_pA = d->realimentacion_mapa_objetos_grandes_fuera_carretera_pA;
          aux_pA = info_grid->class_bayes_prob_muro[i_y][i_x];
          aux_inicializacion_particulas_sigma_vel =
            d->realimentacion_mapa_objetos_grandes_fuera_carretera_sigma;
        } else {
          aux_vx = info_grid->velocidad_inferida_x[i_y][i_x];
          aux_vy = info_grid->velocidad_inferida_y[i_y][i_x];

          aux_inicializacion_particulas_sigma_vel =
            d->realimentacion_inicializacion_particulas_sigma_vel_cluster;
          //					if(d->realimentacion_por_clusters_CGFalse_CCTrue)
          //					{
          //						aux_pA = min(d->realimentacion_inicializacion_particulas_pA_cluster_matched_max, max(d->realimentacion_inicializacion_particulas_pA_cluster_matched_min, info_grid->probAsociacion_cluster[i_y][i_x]));
          //					}
          //					else
          //					{
          //						aux_pA = max(d->realimentacion_inicializacion_particulas_pA_cluster_matched_min, info_grid->probAsociacion_cluster[i_y][i_x] * (d->realimentacion_inicializacion_particulas_pA_cluster_matched_max - d->realimentacion_inicializacion_particulas_pA_cluster_matched_min) + d->realimentacion_inicializacion_particulas_pA_cluster_matched_min);
          //					}

          aux_pA = info_grid->probAsociacion_cluster[i_y][i_x];
          aux_pA = min(
            d->realimentacion_inicializacion_particulas_pA_cluster_matched_max, max(0.0, aux_pA));

          if (
            aux_pA > d->realimentacion_inicializacion_particulas_pA_cluster_matched_max ||
            aux_pA < 0) {
            printf(
              "no puede ser, has calculado una P(A) inicializacion cluster que no esta dentro de "
              "los rangos \t se deberia cumplir: [%f < %f < %f]\n",
              0, aux_pA, d->realimentacion_update_pA_cluster_matched_max);
          }

          if (d->realimentacion_mapa_objetos_grandes_fuera_carretera_son_estaticos) {
            if (info_grid->es_objeto_grande_fuera_carretera[i_y][i_x]) {
              aux_pA = d->realimentacion_mapa_objetos_grandes_fuera_carretera_pA;
              aux_inicializacion_particulas_sigma_vel =
                d->realimentacion_mapa_objetos_grandes_fuera_carretera_sigma;
            }
          }

          //					printf("cluster %f -> %f\n", info_grid->probAsociacion_cluster[i_y][i_x], aux_pA);
        }

        //				if(d->mapa_ruta_activo && d_map->realimentacion_limitar_velocidad_inferida_a_maximo_zona)
        //				{
        //					if(map_ego->type[idx_route_map] == d_map->type_carretera)
        //					{
        //						aux_vx = min(abs(aux_vx), d_map->max_velocidad_carretera * 1.2) * (aux_vx / abs(aux_vx));
        //						aux_vy = min(abs(aux_vy), d_map->max_velocidad_carretera * 1.2) * (aux_vy / abs(aux_vy));
        //					}
        //					else
        //					{
        //						aux_vx = min(abs(aux_vx), d_map->max_velocidad_offroad * 1.2) * (aux_vx / abs(aux_vx));
        //						aux_vy = min(abs(aux_vy), d_map->max_velocidad_offroad * 1.2) * (aux_vy / abs(aux_vy));
        //					}
        //				}

        if (
          d->realimentacion_si_hay_track_vehiculo_NO_hay_feed_por_cluster &&
          info_grid->tipo_objeto[i_y][i_x] != 0) {
          aux_pA = 0.0;
        }

        if (aux_pA > random_asociacion[idx]) {
          particles->v_x[idx_new] =
            aux_vx + aux_inicializacion_particulas_sigma_vel * random_vel_normal[idx * 2];
          particles->v_y[idx_new] =
            aux_vy + aux_inicializacion_particulas_sigma_vel * random_vel_normal[idx * 2 + 1];

          //					printf("Inicializacion por realimentacion: vel inferida [%f %f] vel aleatoria [%f, %f]\n", info_grid->velocidad_inferida_x[i_y][i_x], info_grid->velocidad_inferida_y[i_y][i_x], particles->v_x[idx_new], particles->v_y[idx_new]);
          return;
        }
      }
    }
    */

    // Si hemos llegado aquí -> inicialización aleatoria
    particles->v_x[idx_new] =
      random_vel_uniforme[idx * 2] * 2 * config_DOG->max_vel - config_DOG->max_vel;
    particles->v_y[idx_new] =
      random_vel_uniforme[idx * 2 + 1] * 2 * config_DOG->max_vel - config_DOG->max_vel;
  }
}

void compute_cum_sum_masa_libre(GRID_TYPES::DOG * d_grid)
{
  // Vector 1D que contiene la masa nueva
  double * d_copy_masa_nueva_factor;
  checkCudaErrors(
    cudaMalloc((void **)&d_copy_masa_nueva_factor, GRID_TYPES::NC_P2 * sizeof(double)));

  // Copia de la masa nueva
  cudaMemcpy(
    d_copy_masa_nueva_factor, d_grid->scan_masa_nueva_factor, GRID_TYPES::NC_P2 * sizeof(double),
    cudaMemcpyDeviceToDevice);

  // Punteros que va a usar el inclusive
  thrust::device_ptr<double> d_masa_nueva_ptr =
    thrust::device_pointer_cast(d_copy_masa_nueva_factor);
  thrust::device_ptr<double> d_cumsum_ptr =
    thrust::device_pointer_cast(d_grid->scan_masa_nueva_factor);

  // Cum sum
  thrust::inclusive_scan(d_masa_nueva_ptr, d_masa_nueva_ptr + GRID_TYPES::NC_P2, d_cumsum_ptr);

  cudaFree(d_copy_masa_nueva_factor);
}

// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------- MAIN CODIGO ----------------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
void DYN_CLASS_OG::dynamic_occupancy_grid_compute_new_particles(
  GRID_TYPES::DOG * d_grid, PARTICLE_TYPES::PART_DOG * d_particles,
  const GRID_TYPES::CART_Data * d_grid_cart_data, const DYN_CLASS_OG::config * d_config_DOG,
  const float * d_random_cell_selection, const float * d_random_asociacion,
  const float * d_random_vel_uniforme)
{
  // ---------- Cum sum New mass ---------- //
  compute_cum_sum_masa_libre(d_grid);

  // ---------- New Particles ---------- //
  dim3 block_NP_NEW(std::min(PARTICLE_TYPES::NP_NEW, 128));
  dim3 grid_NP_NEW(PARTICLE_TYPES::NP_NEW / block_NP_NEW.x);

  checkCudaErrors(cudaMemset(
    d_grid->numero_particulas_nuevas, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int)));

  // Guardamos las particulas en la estructura de PARTICLES pero detras del set activo (NP_ACT), i.e. desde NP_ACT hasta el número total NP_TOT = NP_ACT + NP_NEW
  global_repartir_particulas_nuevas<<<grid_NP_NEW, block_NP_NEW>>>(
    d_grid, d_particles, d_random_cell_selection, GRID_TYPES::NC_X, GRID_TYPES::NC_Y,
    PARTICLE_TYPES::NP_ACT, GRID_TYPES::NC_P2, PARTICLE_TYPES::NP_ACT, PARTICLE_TYPES::NP_TOT);

  global_inicializar_particulas_nuevas<<<grid_NP_NEW, block_NP_NEW>>>(
    d_config_DOG, d_grid, d_particles, d_grid_cart_data, d_random_asociacion, d_random_vel_uniforme,
    PARTICLE_TYPES::NP_ACT, PARTICLE_TYPES::NP_ACT, PARTICLE_TYPES::NP_TOT);
}

// ------------------------------ MAIN CODIGO INICIALIZAR TODAS ----------------------------- //
// Es lo mismo que el otro pero cambia el número de particulas y por lo tanto el tamaño de los vectores y eso
void DYN_CLASS_OG::dynamic_occupancy_grid_compute_new_particles_all(
  GRID_TYPES::DOG * d_grid, PARTICLE_TYPES::PART_DOG * d_particles,
  const GRID_TYPES::CART_Data * d_grid_cart_data, const DYN_CLASS_OG::config * d_config_DOG,
  const float * d_random_asociacion, const float * d_random_cell_selection,
  const float * d_random_vel_uniforme)
{
  // ---------- Cum sum New mass ---------- //
  compute_cum_sum_masa_libre(d_grid);

  // ---------- New Particles ---------- //
  dim3 block_NP_ACT(512);
  dim3 grid_NP_ACT(PARTICLE_TYPES::NP_ACT / block_NP_ACT.x);

  checkCudaErrors(cudaMemset(
    d_grid->numero_particulas_nuevas, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int)));

  global_repartir_particulas_nuevas<<<grid_NP_ACT, block_NP_ACT>>>(
    d_grid, d_particles, d_random_cell_selection, GRID_TYPES::NC_X, GRID_TYPES::NC_Y,
    PARTICLE_TYPES::NP_ACT, GRID_TYPES::NC_P2, 0, PARTICLE_TYPES::NP_ACT);

  global_inicializar_particulas_nuevas<<<grid_NP_ACT, block_NP_ACT>>>(
    d_config_DOG, d_grid, d_particles, d_grid_cart_data, d_random_asociacion, d_random_vel_uniforme,
    PARTICLE_TYPES::NP_ACT, 0, PARTICLE_TYPES::NP_ACT);

  cudaDeviceSynchronize();
}