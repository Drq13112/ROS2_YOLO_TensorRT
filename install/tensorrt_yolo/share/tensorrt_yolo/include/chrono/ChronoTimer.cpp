#include "ChronoTimer.hpp"
#include <ctime>  // Para clock_gettime
#include <iostream>

/**
 * @brief Constructor por defecto, guarda el tiempo en el que se creó
 *
 */
ChronoTimer::ChronoTimer()
{
  clock_gettime(CLOCK_MONOTONIC, &startTime);
}

/**
 * @brief Actualiza el tiempo al actual
 *
 */
void ChronoTimer::Reset()
{
  clock_gettime(CLOCK_MONOTONIC, &startTime);
}

/**
 * @brief Devuelve el tiempo transcurrido desde la última vez que se estableció el tiempo inicial
 *
 * @return double
 */
void ChronoTimer::GetElapsedTime()
{
  clock_gettime(CLOCK_MONOTONIC, &lastTime);

  // Convertir el tiempo a milisegundos
  measured_time = (lastTime.tv_sec - startTime.tv_sec) * 1000.0 + (lastTime.tv_nsec - startTime.tv_nsec) / 1e6;
}

/**
 * @brief Calcula la media
 */
void ChronoTimer::ComputeStats()
{
  cont++;

  if (cont > 1)
  {
    mean_time = ((mean_time * (cont - 1)) + measured_time) / cont;
  }
  else
  {
    mean_time = measured_time;
  }

  max_time = std::max(max_time, measured_time);
}
