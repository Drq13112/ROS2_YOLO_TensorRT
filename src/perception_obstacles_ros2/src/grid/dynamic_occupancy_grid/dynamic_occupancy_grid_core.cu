#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

void DYN_CLASS_OG::leer_fichero_configuracion_DOG(DYN_CLASS_OG::config * config_DOG)
{
  printf("   Leyendo ficheros mapa... \n");
  std::string config_file_path = std::string(CONFIG_DIR) + "/config_DOG.yaml";

  YAML::Node data_yaml = YAML::LoadFile(config_file_path);

  config_DOG->factor = data_yaml["factor"].as<float>();

  config_DOG->valor_asumible_como_cero_para_codigo =
    data_yaml["valor_asumible_como_cero_para_codigo"].as<float>();

  config_DOG->desviacion_vel_movimiento_particulas_ms_ciclo =
    data_yaml["desviacion_vel_movimiento_particulas_ms_ciclo"].as<float>();

  config_DOG->probabilidad_supervivencia_particula =
    data_yaml["probabilidad_supervivencia_particula"].as<float>();

  config_DOG->tiempoMaxDegradacionLibertad = data_yaml["tiempoMaxDegradacionLibertad"].as<float>();

  config_DOG->max_pred_certainty = data_yaml["max_pred_certainty"].as<float>();

  config_DOG->probNacimiento = data_yaml["probNacimiento"].as<float>();

  config_DOG->desviacion_vel_movimiento_particulas_ms_ciclo =
    data_yaml["desviacion_vel_movimiento_particulas_ms_ciclo"].as<float>();

  config_DOG->max_vel = data_yaml["max_vel_kmh"].as<float>() / 3.6;

  config_DOG->calculo_vel_numero_minimo_de_veces_remuestreada =
    data_yaml["calculo_vel_numero_minimo_de_veces_remuestreada"].as<float>();

  config_DOG->inicializacion_particulas_nacimiento_solo_en_zonas_con_ocupacion_observada =
    (bool)data_yaml["inicializacion_particulas_nacimiento_solo_en_zonas_con_ocupacion_observada"]
      .as<int>();

  config_DOG->threshold_celda_ocupada = data_yaml["threshold_celda_ocupada"].as<float>();

  config_DOG->write_ficheros_masa_ocupacion =
    (bool)data_yaml["write_ficheros_masa_ocupacion"].as<int>();

  printf("   ... parametros DOG leidos\n\n");
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------- CORE ------------------------------------------------------------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- //

void DYN_CLASS_OG::compute_dynamic_occupancy_grid_core(
  bool * flag_particles, PARTICLE_TYPES::PART_DOG * d_particles,
  PARTICLE_TYPES::PART_DOG * d_particles_sorted,
  PARTICLE_TYPES::PART_DOG * d_particles_for_resampling, GRID_TYPES::DOG * d_grid,
  const DYN_CLASS_OG::config * h_config_DOG, const DYN_CLASS_OG::config * d_config_DOG,
  const GRID_TYPES::OG * d_obs_OG, const GRID_TYPES::CART_Data * h_grid_cart_data,
  const GRID_TYPES::CART_Data * d_grid_cart_data, const EGO_VEH::INFO_ego * h_info_coche,
  const EGO_VEH::INFO_ego * d_info_coche, float * d_random_pred,
  float * d_random_particle_selection, float * d_random_cell_selection, float * d_random_asociacion,
  float * d_random_vel_uniforme, const long int rng_seed, const cudaStream_t * streams,
  DATA_times * TIME_measurements, const int iteration)
{
  bool enumeration = true;

  if (enumeration) {
    printf("\n   Entrando compute_dynamic_occupancy_grid_core \n");
  }

#if DEBUG_DOG
  printf("DEBUG_DOG ACTIVO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif
  if (PARTICLE_TYPES::NP_ACT + PARTICLE_TYPES::NP_NEW > PARTICLE_TYPES::NP_TOT) {
    printf("     !!! NP_TOT tiene que ser la suma de NP_ACT + NP_NEW");
    exit(1);
  }

  if ((PARTICLE_TYPES::NP_P2 & (PARTICLE_TYPES::NP_P2 - 1)) != 0) {
    printf(
      "     !!! NO PUEDES CALCULAR EL PESO ACUMULADO CON UN NUMERO QUE NO SEA POTENCIA DE 2 \n\n");
    exit(1);
  }

  if ((PARTICLE_TYPES::NP_ACT % 32 != 0)) {
    printf("     !!! NP_ACT tiene que ser divisible entre el tamaño del warp (32) \n\n");
    exit(1);
  }

#if DEBUG_DOG
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - starting - Los indices de las particulas estan mal\n");
    exit(1);
  }
#endif

  // -------------------- Random numbers -------------------- //
  TIME_measurements->time_DOG_random_numbers.Reset();

  if (*flag_particles) {
    DYN_CLASS_OG::dynamic_occupancy_grid_random_numbers_for_DOG(
      d_random_pred, d_random_particle_selection, d_random_cell_selection, d_random_asociacion,
      d_random_vel_uniforme, rng_seed, streams);
  }

  cudaDeviceSynchronize();  // Just for time measuring
  TIME_measurements->time_DOG_random_numbers.GetElapsedTime();
  TIME_measurements->time_DOG_random_numbers.ComputeStats();

  if (enumeration) {
    printf("\n   Random numbers calculados\n");
    printf(
      "     - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_DOG_random_numbers.measured_time,
      TIME_measurements->time_DOG_random_numbers.mean_time,
      TIME_measurements->time_DOG_random_numbers.max_time);
  }

  // -------------------- Prediction -------------------- //
  TIME_measurements->time_DOG_prediction.Reset();

  DYN_CLASS_OG::dynamic_occupancy_prediction(
    flag_particles, d_particles, d_particles_sorted, d_grid, h_grid_cart_data, d_grid_cart_data,
    h_config_DOG, d_config_DOG, d_random_pred, h_info_coche, d_info_coche);

  cudaDeviceSynchronize();  // Just for time measuring
  TIME_measurements->time_DOG_prediction.GetElapsedTime();
  TIME_measurements->time_DOG_prediction.ComputeStats();

#if DEBUG_DOG
  if (!DYN_CLASS_OG::ordenar_particulas(
        d_particles, d_particles_sorted, d_grid, h_grid_cart_data)) {
    printf("compute_dynamic_occupancy_grid_core - Prediction - La ordenacion no es correcta\n!");
    exit(1);
  }
  if (!DYN_CLASS_OG::comprobar_peso_particula(
        d_particles, d_grid->masa_pred_oc_factor, d_grid, h_config_DOG->factor)) {
    printf(
      "compute_dynamic_occupancy_grid_core - Prediction - Hay particulas predichas validas con "
      "peso despues predecir < 0 o > 1 o nan !\n ");
    printf("   o no coincide la suma \n ");
    exit(1);
  }
  if (!DYN_CLASS_OG::check_cell_particle_number(d_particles, d_grid, h_grid_cart_data)) {
    printf("compute_dynamic_occupancy_grid_core - Prediction - Los indices no coinciden !\n ");
    exit(1);
  }
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - Prediction - Los indices de las particulas estan mal "
      "\n");
    exit(1);
  }
#endif
  if (enumeration) {
    printf("\n   Prediccion calculada\n");
    printf(
      "     - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_DOG_prediction.measured_time,
      TIME_measurements->time_DOG_prediction.mean_time,
      TIME_measurements->time_DOG_prediction.max_time);
  }

  if (h_config_DOG->write_ficheros_masa_ocupacion) {
    DYN_CLASS_OG::write_files_occupancy(
      d_grid->masa_pred_oc_factor, d_grid->masa_pred_libre_factor, iteration, h_config_DOG,
      "prediccion");
  }

  // -------------------- Update -------------------- //
  TIME_measurements->time_DOG_update.Reset();

  DYN_CLASS_OG::dynamic_occupancy_grid_update(d_grid, d_obs_OG, d_particles, d_config_DOG);
#if DEBUG_DOG
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - update - Los indices de las particulas estan mal "
      "\n");
    exit(1);
  }
#endif
  cudaDeviceSynchronize();  // Just for time measuring
  TIME_measurements->time_DOG_update.GetElapsedTime();
  TIME_measurements->time_DOG_update.ComputeStats();

  if (enumeration) {
    printf("\n   Update calculado\n");
    printf(
      "     - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_DOG_update.measured_time,
      TIME_measurements->time_DOG_update.mean_time, TIME_measurements->time_DOG_update.max_time);
  }

  if (h_config_DOG->write_ficheros_masa_ocupacion) {
    DYN_CLASS_OG::write_files_occupancy(
      d_grid->masa_act_oc_factor, d_grid->masa_act_libre_factor, iteration, h_config_DOG, "update");
  }

#if DEBUG_DOG
  if (*flag_particles) {
    if (!DYN_CLASS_OG::comprobar_peso_particula(
          d_particles, d_grid->masa_persistente_factor, d_grid, h_config_DOG->factor)) {
      printf("Hay particulas validas con peso despues actualizar < 0 o > 1 o nan !\n ");
      printf("   o no coincide la suma \n ");
      exit(1);
    }
    if (!DYN_CLASS_OG::check_cell_particle_number(d_particles, d_grid, h_grid_cart_data)) {
      printf("Los indices no coinciden o el numero de particulas no coincide!\n ");
      exit(1);
    }
    if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
      printf(
        "compute_dynamic_occupancy_grid_core - time_DOG_equalize_weights - Los indices de las "
        "particulas estan mal\n");
      exit(1);
    }
  }
#endif

  // -------------------- Dynamic State -------------------- //
  TIME_measurements->time_DOG_velocity.Reset();

  DYN_CLASS_OG::dynamic_occupancy_grid_dynamic_state(
    flag_particles, d_grid, d_particles, d_config_DOG);

  cudaDeviceSynchronize();  // Just for time measuring
  TIME_measurements->time_DOG_velocity.GetElapsedTime();
  TIME_measurements->time_DOG_velocity.ComputeStats();

  if (enumeration) {
    printf("\n   Dynamic State calculado\n");
    printf(
      "     - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_DOG_velocity.measured_time,
      TIME_measurements->time_DOG_velocity.mean_time,
      TIME_measurements->time_DOG_velocity.max_time);
  }

  // -------------------- Resampling and new particles -------------------- //
  TIME_measurements->time_DOG_resampling_roughtening.Reset();

  if (*flag_particles) {
    DYN_CLASS_OG::dynamic_occupancy_grid_compute_new_particles(
      d_grid, d_particles, d_grid_cart_data, d_config_DOG, d_random_asociacion,
      d_random_cell_selection, d_random_vel_uniforme);

    DYN_CLASS_OG::dynamic_occupancy_grid_resampling(
      d_grid, d_particles, d_particles_for_resampling, d_grid_cart_data, d_config_DOG,
      d_random_particle_selection);

  } else {
    printf(
      "------------------------- NO HAY PARTICULAS Y ESTAMOS REGENERANDOLAS (no tiene porque ser "
      "incorrecto) ------------------------- \n");

    *flag_particles = true;

    float * d_random_cell_selection_inicializar;
    cudaMalloc(
      (float **)&d_random_cell_selection_inicializar, PARTICLE_TYPES::NP_ACT * sizeof(float));

    float * d_random_vel_uniforme_inicializar;
    cudaMalloc(
      (float **)&d_random_vel_uniforme_inicializar, 2 * PARTICLE_TYPES::NP_ACT * sizeof(float));

    float * d_random_asociacion_inicializar;
    cudaMalloc((float **)&d_random_asociacion_inicializar, PARTICLE_TYPES::NP_ACT * sizeof(float));

    DYN_CLASS_OG::dynamic_occupancy_grid_random_numbers_for_DOG_initialization(
      d_random_asociacion_inicializar, d_random_cell_selection_inicializar,
      d_random_vel_uniforme_inicializar, rng_seed);

    DYN_CLASS_OG::dynamic_occupancy_grid_compute_new_particles_all(
      d_grid, d_particles, d_grid_cart_data, d_config_DOG, d_random_asociacion_inicializar,
      d_random_cell_selection_inicializar, d_random_vel_uniforme_inicializar);

    cudaFree(d_random_cell_selection_inicializar);
    d_random_cell_selection_inicializar = NULL;
    cudaFree(d_random_vel_uniforme_inicializar);
    d_random_vel_uniforme_inicializar = NULL;
    cudaFree(d_random_asociacion_inicializar);
    d_random_asociacion_inicializar = NULL;
  }

  cudaDeviceSynchronize();  // Just for time measuring
  TIME_measurements->time_DOG_resampling_roughtening.GetElapsedTime();
  TIME_measurements->time_DOG_resampling_roughtening.ComputeStats();

#if DEBUG_DOG
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - resampling - Los indices de las particulas estan mal "
      "\n");
    exit(1);
  }
#endif

  if (enumeration) {
    printf("\n   Roughening y Resampling calculado\n");
    printf(
      "     - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_DOG_resampling_roughtening.measured_time,
      TIME_measurements->time_DOG_resampling_roughtening.mean_time,
      TIME_measurements->time_DOG_resampling_roughtening.max_time);
  }

  // -------------------- Equalize Particles' Weights -------------------- //
  TIME_measurements->time_DOG_equalize_weights.Reset();

  gpu_sort_particles_with_thrust(d_particles, d_particles_sorted, d_grid, h_grid_cart_data);

#if DEBUG_DOG
  if (!DYN_CLASS_OG::ordenar_particulas(
        d_particles, d_particles_sorted, d_grid, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - time_DOG_equalize_weights - La ordenacion no es "
      "correcta\n!");
    exit(1);
  }
  if (!DYN_CLASS_OG::check_cell_particle_number(d_particles, d_grid, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - time_DOG_equalize_weights - Los indices no coinciden "
      "o el numero de particulas no coincide!\n ");
    exit(1);
  }
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - time_DOG_equalize_weights - Los indices de las "
      "particulas estan mal\n");
    exit(1);
  }
#endif

  DYN_CLASS_OG::dynamic_occupancy_grid_normalize_weights(d_particles, d_grid, h_grid_cart_data);

  cudaDeviceSynchronize();  // Just for time measuring
  TIME_measurements->time_DOG_equalize_weights.GetElapsedTime();
  TIME_measurements->time_DOG_equalize_weights.ComputeStats();

  if (enumeration) {
    printf("\n   Igualar peso particulas calculado\n");
    printf(
      "     - elapsed time = %fms, mean = %fms, max = %fms\n",
      TIME_measurements->time_DOG_equalize_weights.measured_time,
      TIME_measurements->time_DOG_equalize_weights.mean_time,
      TIME_measurements->time_DOG_equalize_weights.max_time);
  }

#if DEBUG_DOG
  if (!DYN_CLASS_OG::comprobar_peso_particula(
        d_particles, d_grid->masa_act_oc_factor, d_grid, h_config_DOG->factor)) {
    printf("Hay particulas validas con peso despues igualar los pesos < 0 o > 1 o nan !\n ");
    printf("   o no coincide la suma \n ");
    exit(1);
  }
  if (!DYN_CLASS_OG::check_particle_cell_indexes(d_particles, h_grid_cart_data)) {
    printf(
      "compute_dynamic_occupancy_grid_core - normalize - Los indices de las particulas estan "
      "mal\n");
    exit(1);
  }
#endif

  // ------------------------------------------------------------------------------------------------------------------------------------------------------------------ //
  printf("ESTOY IGNORANDO LO SIGUIENTE!\n");
  /*
    static std::vector<int> ids_tracks_particulas;
	ids_tracks_particulas.clear();
	static std::vector<int> ids_tracks_pred;
	ids_tracks_pred.clear();

	if (d->matching_track_particulas)
	{
		for (int i_t = 0; i_t < tracks.size(); i_t++)
		{
			if (tracks[i_t].vivo && tracks[i_t].veces_visto > d->matching_track_particulas_edad_minima)
			{
				ids_tracks_particulas.push_back(tracks[i_t].id);
			}
			if (tracks[i_t].vivo && tracks[i_t].veces_visto > d->asociar_track_celda_min_veces_visto)
			{
				ids_tracks_pred.push_back(tracks[i_t].id);
			}
		}
	}
    */

  cudaDeviceSynchronize();
}