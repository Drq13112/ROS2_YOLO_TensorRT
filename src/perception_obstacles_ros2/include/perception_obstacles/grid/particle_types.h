#pragma once
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <string>

namespace PARTICLE_TYPES
{

// const int NP_P2 = 4194304;   // 2^22 = 2097152 tiene que ser una potencia de dos superior a NP_TOT
// const int NP_TOT = 4194304;  // NP_TOT = NP_ACT + NP_NEW
// const int NP_ACT = 3670016;  //
// const int NP_NEW = 524288;   // 1024 * 2^9

// const int  NP_P2     = 2097152  ; // 2^21 = 2097152 tiene que ser una potencia de dos superior a NP_TOT
// const int  NP_TOT    = 2097152  ; // NP_TOT = NP_ACT + NP_NEW
// const int  NP_ACT    = 1966080  ; //
// const int  NP_NEW    =  131072  ; // 1024 * 128

// const int NP_P2 = 1048576;   // 2^20
// const int NP_TOT = 1048576;  // Total number of particles NP_TOT = NP_ACT + NP_NEW
// const int NP_ACT = 983040;   // Number of active particles: (NP_P2 - NP_NEW) % 1024 == true
// const int NP_NEW = 65536;    // Number of new particles 1024 * (32 * 2)

// const int NP_P2 = 524288;   // 2^19
// const int NP_TOT = 524288;  // Total number of particles NP_TOT = NP_ACT + NP_NEW
// const int NP_ACT = 458752;  // Number of active particles: (NP_P2 - NP_NEW) % 1024 == true
// const int NP_NEW = 65536;   // Number of new particles 1024 * (32 * 2)

const int NP_P2 = 262144;   // 2^18
const int NP_TOT = 262144;  // Total number of particles NP_TOT = NP_ACT + NP_NEW
const int NP_ACT = 229376;  // Number of active particles: (NP_P2 - NP_NEW) % 1024 == true
const int NP_NEW = 32768;   // Number of new particles 1024 * (32)

// const int  NP_P2     = 131072  ; // SHOULD AT LEAST WORK WITH THIS (due to past implementations)
// const int  NP_TOT    = 131072  ; // Total number of particles
// const int  NP_ACT    =  98304  ; // Number of active particles
// const int  NP_NEW    =  32768  ; // Number of new particles

// const int NP_P2 = 65536;   //
// const int NP_TOT = 65536;  // Total number of particles
// const int NP_ACT = 57344;  // Number of active particles
// const int NP_NEW = 8192;   // Number of new particles

typedef struct
{
  double p_x[NP_TOT];
  double p_y[NP_TOT];
  double v_x[NP_TOT];
  double v_y[NP_TOT];
  double peso_factor[NP_TOT];  // peso_factor indica que es el peso multiplicado por un factor. Por ejemplo: peso * 1000
  double vel_likelihood[NP_TOT];

  int indice_celda_x[NP_TOT];
  int indice_celda_y[NP_TOT];
  int indice_celda[NP_TOT];       // Indice unidimensional
  int indices_ordenados[NP_TOT];  // TODO Type cambiar nombre a algo como: idx_siguiente_part_en_celda

  bool valida[NP_TOT];
  bool new_born[NP_TOT];
  int veces_remuestreada[NP_TOT];

} PART_DOG;

}  // namespace PARTICLE_TYPES