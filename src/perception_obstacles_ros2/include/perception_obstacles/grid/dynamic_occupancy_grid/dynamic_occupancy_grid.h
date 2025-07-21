#pragma once
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <curand.h>
#include <curand_kernel.h>

#include "perception_obstacles/perception_utilities/utils.h"

#include "perception_obstacles/grid/particle_types.h"
#include "perception_obstacles/grid/grid_types.h"
#include "perception_obstacles/grid/grid_utils.cuh"
#include "perception_obstacles/grid/grid_utils.h"

#include "perception_obstacles/ego_vehicle/calculos_estado_coche.h"

#include "perception_obstacles/perception_utilities/time_data.h"
#include "perception_obstacles/perception_utilities/helper_cuda.h"
#include "perception_obstacles/perception_utilities/ChronoTimer.hpp"

namespace DYN_CLASS_OG
{

typedef struct
{
  double factor = 1e5;

  float valor_asumible_como_cero_para_codigo = 0;

  float desviacion_vel_movimiento_particulas_ms_ciclo = 1.0;
  float tiempoMaxDegradacionLibertad = 0.16;
  float probabilidad_supervivencia_particula = 0.7;
  float max_pred_certainty = 0.99;
  float probNacimiento = 0.02;

  float max_vel = 15;  // m/s

  float calculo_vel_numero_minimo_de_veces_remuestreada;

  bool inicializacion_particulas_nacimiento_solo_en_zonas_con_ocupacion_observada = true;

  float threshold_celda_ocupada = 0.3;

  bool write_ficheros_masa_ocupacion = false;
} config;

void leer_fichero_configuracion_DOG(DYN_CLASS_OG::config* config_DOG);

void gpu_sort_particles_with_thrust(PARTICLE_TYPES::PART_DOG* d_particles, PARTICLE_TYPES::PART_DOG* d_particles_sorted,
                                    GRID_TYPES::DOG* d_grid, const GRID_TYPES::CART_Data* h_grid_cart_data);

void dynamic_occupancy_grid_random_numbers_for_DOG(float* d_random_pred, float* d_random_particle_selection,
                                                   float* d_random_cell_selection, float* d_random_asociacion,
                                                   float* d_random_vel_uniforme, const long int rng_seed,
                                                   const cudaStream_t streams[]);

void dynamic_occupancy_grid_random_numbers_for_DOG_initialization(float* d_random_asociacion_inicializar,
                                                                  float* d_random_cell_selection_inicializar,
                                                                  float* d_random_vel_uniforme_inicializar,
                                                                  const long int rng_seed);

void dynamic_occupancy_prediction(const bool* flag_particles, PARTICLE_TYPES::PART_DOG* d_particles,
                                  PARTICLE_TYPES::PART_DOG* d_particles_sorted, GRID_TYPES::DOG* d_grid,
                                  const GRID_TYPES::CART_Data* h_grid_cart_data,
                                  const GRID_TYPES::CART_Data* d_grid_cart_data,
                                  const DYN_CLASS_OG::config* h_config_DOG, const DYN_CLASS_OG::config* d_config_DOG,
                                  const float* d_random_pred, const EGO_VEH::INFO_ego* h_info_coche,
                                  const EGO_VEH::INFO_ego* d_info_coche);

void dynamic_occupancy_grid_update(GRID_TYPES::DOG* d_grid, const GRID_TYPES::OG* d_obs_OG,
                                   PARTICLE_TYPES::PART_DOG* d_particles, const DYN_CLASS_OG::config* d_config_DOG);

void dynamic_occupancy_grid_dynamic_state(const bool* flag_particles, GRID_TYPES::DOG* d_grid,
                                          const PARTICLE_TYPES::PART_DOG* d_particles,
                                          const DYN_CLASS_OG::config* d_config_DOG);

void dynamic_occupancy_grid_compute_new_particles(GRID_TYPES::DOG* d_grid, PARTICLE_TYPES::PART_DOG* d_particles,
                                                  const GRID_TYPES::CART_Data* d_grid_cart_data,
                                                  const DYN_CLASS_OG::config* d_config_DOG,
                                                  const float* d_random_cell_selection,
                                                  const float* d_random_asociacion, const float* d_random_vel_uniforme);

void dynamic_occupancy_grid_compute_new_particles_all(GRID_TYPES::DOG* d_grid, PARTICLE_TYPES::PART_DOG* d_particles,
                                                      const GRID_TYPES::CART_Data* d_grid_cart_data,
                                                      const DYN_CLASS_OG::config* d_config_DOG,
                                                      const float* d_random_asociacion,
                                                      const float* d_random_cell_selection,
                                                      const float* d_random_vel_uniforme);

void dynamic_occupancy_grid_resampling(GRID_TYPES::DOG* d_grid, PARTICLE_TYPES::PART_DOG* d_particles,
                                       PARTICLE_TYPES::PART_DOG* d_particles_for_resampling,
                                       const GRID_TYPES::CART_Data* d_grid_cart_data,
                                       const DYN_CLASS_OG::config* d_config_DOG,
                                       const float* d_random_particle_selection);

void dynamic_occupancy_grid_normalize_weights(PARTICLE_TYPES::PART_DOG* d_particles, GRID_TYPES::DOG* d_grid,
                                              const GRID_TYPES::CART_Data* h_grid_cart_data);

// ----- CORE ----- //
void compute_dynamic_occupancy_grid_core(
    bool* flag_particles, PARTICLE_TYPES::PART_DOG* d_particles, PARTICLE_TYPES::PART_DOG* d_particles_sorted,
    PARTICLE_TYPES::PART_DOG* d_particles_for_resampling, GRID_TYPES::DOG* d_grid,
    const DYN_CLASS_OG::config* h_config_DOG, const DYN_CLASS_OG::config* d_config_DOG, const GRID_TYPES::OG* d_obs_OG,
    const GRID_TYPES::CART_Data* h_grid_cart_data, const GRID_TYPES::CART_Data* d_grid_cart_data,
    const EGO_VEH::INFO_ego* h_info_coche, const EGO_VEH::INFO_ego* d_info_coche, float* d_random_pred,
    float* d_random_particle_selection, float* d_random_cell_selection, float* d_random_asociacion,
    float* d_random_vel_uniforme, const long int rng_seed, const cudaStream_t* streams, DATA_times* TIME_measurements,
    const int iteration);

// ----- DEBUG ----- //
bool ordenar_particulas(const PARTICLE_TYPES::PART_DOG* d_particles, const PARTICLE_TYPES::PART_DOG* d_particles_sorted,
                        const GRID_TYPES::DOG* d_grid, const GRID_TYPES::CART_Data* host_grid_cart_data);

bool comprobar_peso_particula(const PARTICLE_TYPES::PART_DOG* d_particles,
                              const double d_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const GRID_TYPES::DOG* d_grid,
                              const double factor_perdida_precision);

bool check_cell_particle_number(const PARTICLE_TYPES::PART_DOG* d_particles, const GRID_TYPES::DOG* d_grid,
                                const GRID_TYPES::CART_Data* grid_cart_data);

bool check_particle_cell_indexes(const PARTICLE_TYPES::PART_DOG* d_particles,
                                 const GRID_TYPES::CART_Data* grid_cart_data);

// ----- WRITE FILES ----- //
void write_files_DOG_color(const GRID_TYPES::DOG* d_grid, const int iteration, const DYN_CLASS_OG::config* config_DOG);

void write_files_occupancy(const double d_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
                           const double d_mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const int i_iter,
                           const DYN_CLASS_OG::config* config_DOG, const std::string nombre);

void write_files_num_particles(const GRID_TYPES::DOG* d_dog, const int i_iter, const DYN_CLASS_OG::config* config_DOG);

}  // namespace DYN_CLASS_OG