#pragma once

#include <iomanip>
#include <iostream>
//#include <math.h>
#include <cmath>
#include <limits>
#include <string>    // std::string
#include <iostream>  // std::cout
#include <sstream>
#include <vector>
#include <cstring>
#include <pcl_ros/transforms.hpp>

// void interpolacion_lineal(float* y, const float x, const float x1, const float x2, const float y1,
//                                  const float y2);

// void interpolacion_lineal(double* y, const double x, const double x1, const double x2, const double y1,
//                                  const double y2);

// void interpolacion_lineal_angulos(float* ang, const float x, const float x0, const float x1, const double
// ang0, const double ang1);

// void interpolacion_lineal_angulos(double* ang, const double x, const double x0, const double x1,
//                                          const double ang0, const double ang1);

// void restar_angulos(float* resta, const float ang1, const float ang2);

// double TimeSpecToSeconds(struct timespec* ts);

// double TimeDiffSeconds(struct timespec* tend, struct timespec* tstart);

// void print_matrix(std::vector<std::vector<double> >& matrix);

// void matrizRotacion(double* x, double* y, double* z, const int eje_cambio, const double ang);

void matrizRotacionZ(double* x, double* y, const double theta);

void matrizRotacionZ_seno_coseno_precalculado(double* x, double* y, const double sin_theta, const double cos_theta);

void matrizRotacionZ_seno_coseno_precalculado(float* x, float* y, const float sin_theta, const float cos_theta);

void matrizRotacionZInversa_seno_coseno_precalculado(double* x, double* y, const double sin_theta,
                                                     const double cos_theta);

void matrizRotacionZInversa_seno_coseno_precalculado(float* x, float* y, const float sin_theta, const float cos_theta);

void matrizRotacionZ(float* x, float* y, const float theta);

void fixAngleRad(double* ang);

void fixAngleRad(float* ang);

/*
void calcular_media_vector_1D(double* media, const double* x, const int num_puntos);

void calcular_varianza_1D(double* var, const double media, const double* x, const int num_puntos);

void printf_tiempos_medio_maximo_minimo(const std::vector<double>* tiempos, const std::string texto,
                                               const double factor_milliseconds);
*/

void compute_rotation_matrix(const geometry_msgs::msg::Transform& transform, float rotation_matrix[3][3]);

void compute_1D_layers_angles(int* i_1D, const int i_layer, const int i_angle, const int N_LAYERS);

void compute_2D_layers_angles(int* i_layer, int* i_angle, const int i_1D, const int N_LAYERS);

void calculoRGB_angulo(const double ang, double* R, double* G, double* B);

void pignistic_transformation(double* pO, const double mO, const double mL);
