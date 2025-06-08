#pragma once

#include <vector>
#include <chrono>
#include <math.h>

/**
 * @brief Clase para simular un cronómetro.
 * Cuando se crea inicia el contador
 * Se reinicia el contador cuando llama a Reset
 * Cuando llamamos a GetElapsedTime devuelve el tiempo que transcurrido
 *
 */
class ChronoTimer
{
public:
  ChronoTimer();
  void Reset();
  void GetElapsedTime();
  void ComputeStats();
  double measured_time;
  double mean_time = 0;
  double max_time = 0;

  struct timespec startTime;
  struct timespec lastTime;
  int cont = 0;
};
