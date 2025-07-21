#pragma once

#include <stdio.h>
#include <iostream>

#include <fstream>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <yaml-cpp/yaml.h>

#include "perception_obstacles/perception_utilities/helper_cuda.h"

#include "perception_obstacles/grid/grid_types.h"
#include "perception_obstacles/grid/grid_utils.cuh"

#include "perception_obstacles/ego_vehicle/calculos_estado_coche.h"

namespace OFF_ROAD_MAP
{

typedef struct
{
  int NC_X;  // Columnas
  int NC_Y;  // Filas
  int NC_XY;

  std::string nombre_mapa;
  std::string path_mapa;

  // Datos relativos al mapa
  float lado_celda;
  double UTM_origen_mapa_x;
  double UTM_origen_mapa_y;
  double UTM_origen_mapa_heading;
  double UTM_origen_mapa_heading_seno, UTM_origen_mapa_heading_coseno;

  int type_carretera;
  int type_off_road;

  // Modificadores Clustering
  bool decision_tecnica_no_clusterizar_offroad;

} config;

// Liberar memoria
void free_allocated_memory(
  OFF_ROAD_MAP::config * h_config_map, OFF_ROAD_MAP::config * d_config_map,
  char * h_complete_road_map, char * d_complete_road_map, GRID_TYPES::GRID_local_road * h_grid_road,
  GRID_TYPES::GRID_local_road * d_grid_road);

// inicializacion
void leer_fichero_configuracion_mapa_ruta(OFF_ROAD_MAP::config * config_map);

void initialize_offline_road_map(
  OFF_ROAD_MAP::config * h_config_map, OFF_ROAD_MAP::config * d_config_map,
  char * h_complete_road_map, char * d_complete_road_map, GRID_TYPES::GRID_local_road * h_grid_road,
  GRID_TYPES::GRID_local_road * d_grid_road);

void carga_mapa(char * h_complete_road_map, const OFF_ROAD_MAP::config * config_map);

int route_map_sub2ind(const int i_x, const int i_y, const int NC_X_map);

void route_map_ind2sub(int * i_x, int * i_y, const int idx, const int NC_X_map);

void calculo_indices_celda_offline_road_map(
  int * i_x, int * i_y, bool * conseguido, const float x, const float y,
  const OFF_ROAD_MAP::config * config_map);

__device__ int kernel_sub2ind(const int i_x, const int i_y, const int NC_X_map);

__device__ void kernel_ind2sub(int * i_x, int * i_y, const int idx, const int NC_X_map);

__device__ void device_calculo_indices_celda_offline_road_map(
  int * i_x, int * i_y, bool * conseguido, const float x, const float y,
  const OFF_ROAD_MAP::config * config_map);

__global__ void copia_mapa_a_rejilla_local_straightforward(
  GRID_TYPES::GRID_local_road * d_grid_road, const char * d_complete_road_map,
  const EGO_VEH::INFO_ego * d_info_coche, const float * centro_x, const float * centro_y,
  const OFF_ROAD_MAP::config * config_map);

void get_offline_road_map_core(
  GRID_TYPES::GRID_local_road * d_grid_road, const char * d_complete_road_map,
  const OFF_ROAD_MAP::config * d_config_map, const EGO_VEH::INFO_ego * d_info_coche,
  const GRID_TYPES::CART_Data * d_grid_cart);

}  // namespace OFF_ROAD_MAP
