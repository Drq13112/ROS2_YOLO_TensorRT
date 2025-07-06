#pragma once
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <string>

namespace GRID_TYPES
{
// ----------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------- //
// ------------------------------------------------ CARTESIAN GRID ------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------- //

// Variables that define the size of the grid (they are initialized like this because we need constant variables)
const int NC_X = 512;  // Multiple of 32
const int NC_Y = 512;  // Multiple of 32
const int NC_XY = GRID_TYPES::NC_X * GRID_TYPES::NC_Y;
const float RES = 0.2;
const int NC_P2 = 262144;  // must be first power of two that is bigger or equal to NC_X*NC_Y

typedef struct
{
  float MIN_X = -50.0;  // TODO: initialize this
  float MIN_Y = -50.0;

  float RES = GRID_TYPES::RES;

  int NC_X = GRID_TYPES::NC_X;
  int NC_Y = GRID_TYPES::NC_Y;

  float centro_x[GRID_TYPES::NC_X];
  float centro_y[GRID_TYPES::NC_Y];
} CART_Data;

typedef struct
{
  float mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  float mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  float pO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

} OG;

typedef struct
{
  double masa_pred_oc_factor[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];  // factor es que ha sido multiplicado por un factor,
                                                                   // e.g. masa_pred * 1000
  double masa_pred_libre_factor[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  double masa_act_oc_factor[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double masa_act_libre_factor[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double masa_act_libre_copia[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  double masa_nueva_factor[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double masa_persistente_factor[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];  // factor quiere decir que ha sido multiplicado
                                                                       // por un factor, por ejemplo masa_pred * 1000
  double scan_masa_nueva_factor[GRID_TYPES::NC_P2];

  int numero_particulas[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  int indice_primera_particula[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  int indice_ultima_particula[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  float mean_particulas_veces_remuestreadas[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  bool info_vel_valida[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_media_x[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_media_y[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_sigma_x[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_sigma_y[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_cov_xy[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_mahalanobis[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_media_modulo[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double vel_media_angulo[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double sum_vel_peso_factor[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  int numero_particulas_velocidad[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  int numero_particulas_nuevas[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

} DOG;

typedef struct
{
  bool es_carretera[GRID_TYPES::NC_XY];
} GRID_local_road;

// ----------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------- POLAR GRID --------------------------------------------------- //
// ----------------------------------------------------------------------------------------------------------------- //

// Variables that define the size of the grid (they are initialized like this because we need constant variables)
const int OG_NC_ANG = 960;    // 704;    // 1024;   // 360 / FM_NC_ang = resolucion X grados
const int OG_NC_DIST = 1024;  // FM_NC_dist * X metros = 80 metros
typedef struct
{
  float RES_ANG = 2 * M_PI / GRID_TYPES::OG_NC_ANG;
  float MIN_ANG = -M_PI;
  float MIN_DIST = 1.0;  // TODO
  float RES_DIST = 0.1;  // TODO
  int NC_ANG = GRID_TYPES::OG_NC_ANG;
  int NC_DIST = GRID_TYPES::OG_NC_DIST;

  float grid_polar_mO[GRID_TYPES::OG_NC_ANG * GRID_TYPES::OG_NC_DIST];
  float grid_polar_mF[GRID_TYPES::OG_NC_ANG * GRID_TYPES::OG_NC_DIST];
} POLAR_OG;

// Variables that define the size of the grid (they are initialized like this because we need constant variables)
const int OG_NC_ANG_small = 640;   // 1024;   // 360 / FM_NC_ang = resolucion X grados
const int OG_NC_DIST_small = 256;  // FM_NC_dist * X metros
typedef struct
{
  float RES_ANG = 2 * M_PI / GRID_TYPES::OG_NC_ANG_small;
  float MIN_ANG = -M_PI;
  float MIN_DIST = 0.5;  // TODO
  float RES_DIST = 0.1;  // TODO
  int NC_ANG = GRID_TYPES::OG_NC_ANG_small;
  int NC_DIST = GRID_TYPES::OG_NC_DIST_small;

  float grid_polar_mO[GRID_TYPES::OG_NC_ANG_small * GRID_TYPES::OG_NC_DIST_small];
  float grid_polar_mF[GRID_TYPES::OG_NC_ANG_small * GRID_TYPES::OG_NC_DIST_small];
} POLAR_OG_small;

}  // namespace GRID_TYPES