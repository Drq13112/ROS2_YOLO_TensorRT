#include "perception_obstacles/perception_utilities/utils.h"

/*
double exp_2(double x)
{
  //    int p, i;
  //
  //    p = 20; // Probado: minimo 20 para empezar a tener una aproximacion decente (y solo se reduce de 7 a 5ms la obs)
  //    x = 1.0 + x / (2 << p);
  //    for (i = 0; i < p; i++)
  //        x *= x;
  //
  //    return x;

  //    // p = 21. Con este 'p' no hay diferencias con matlab EN ESTE LOG. Se consigue reducir de ~7 a ~4ms
  //    x = 1.0 + x / 2097152;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x;
  //    return x;

  // p = 22. Se consigue reducir de ~7 a ~4.5ms
  x = 1.0 + x / 4194304;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  return x;

  //    // p = 25. Se consigue reducir de ~7 a ~5ms
  //    x = 1.0 + x / 33554432;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x; x *= x; x *= x; x *= x;
  //    x *= x;
  //    return x;
}

double exp_2(float x)
{
  // p = 22. Se consigue reducir de ~7 a ~4.5ms
  x = 1.0 + x / 4194304;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  return x;
}
*/

// void interpolacion_lineal(float* y, const float x, const float x1, const float x2, const float y1,
//                                  const float y2)
// {
//   if (x2 == x1)
//   {
//     *y = y1;
//   }
//   else
//   {
//     *y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
//   }
// }

// void interpolacion_lineal(double* y, const double x, const double x1, const double x2, const double y1,
//                                  const double y2)
// {
//   if (x2 == x1)
//   {
//     *y = y1;
//   }
//   else
//   {
//     *y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
//   }
// }

// void interpolacion_lineal_angulos(float* ang, const float x, const float x0, const float x1, const double
// ang0,
//                                          const double ang1)
// {
//   float a = (ang1 - ang0 + M_PI);
//   float b = (2 * M_PI);
//   float dif_ang = a - floor(a / b) * b - M_PI;
//   *ang = ang0 + (x - x0) * dif_ang / (x1 - x0);
// }

// void interpolacion_lineal_angulos(double* ang, const double x, const double x0, const double x1,
//                                          const double ang0, const double ang1)
// {
//   double a = (ang1 - ang0 + M_PI);
//   double b = (2 * M_PI);
//   double dif_ang = a - floor(a / b) * b - M_PI;
//   *ang = ang0 + (x - x0) * dif_ang / (x1 - x0);
// }

// void restar_angulos(float* resta, const float ang1, const float ang2)
// {
//   float aux = ang1 - ang2 + M_PI;
//   *resta = aux - floor(aux / (2 * M_PI)) * 2 * M_PI - M_PI;
// }

// template <typename T>
// std::string to_string_with_precision(const T a_value, const int n = 6)
// {
//   std::ostringstream out;
//   out << std::setprecision(n) << a_value;
//   return out.str();
// }

// double TimeSpecToSeconds(struct timespec* ts)
// {
//   return (double)ts->tv_sec + (double)ts->tv_nsec / 1e9;
// }

// double TimeDiffSeconds(struct timespec* tend, struct timespec* tstart)
// {
//   double time_taken;
//   time_taken = tend->tv_sec - tstart->tv_sec;
//   time_taken = time_taken + (tend->tv_nsec - tstart->tv_nsec) * 1e-9;
//   return time_taken;
// }

// void print_matrix(std::vector<std::vector<double> >& matrix)
// {
//   printf("\n");
//   for (int i = 0; i < (int)matrix.size(); i++)
//   {
//     for (int j = 0; j < (int)matrix[i].size(); j++)
//     {
//       printf("%.4f\t", matrix[i][j]);
//     }
//     printf("\n");
//   }
// }

// void matrizRotacion(double* x, double* y, double* z, const int eje_cambio, const double ang)
// {
//   double x_1, y_1, z_1;

//   x_1 = *x;
//   y_1 = *y;
//   z_1 = *z;

//   switch (eje_cambio)
//   {
//     case 1:  // Rotacion eje X
//     {
//       *y = y_1 * cos(ang) - z_1 * sin(ang);
//       *z = y_1 * sin(ang) + z_1 * cos(ang);
//       break;
//     }
//     case 2:  // Rotacion eje Y
//     {
//       *x = x_1 * cos(ang) + z_1 * sin(ang);
//       *z = -x_1 * sin(ang) + z_1 * cos(ang);
//       break;
//     }
//     case 3:  // Rotacion eje Z
//     {
//       *x = x_1 * cos(ang) - y_1 * sin(ang);
//       *y = x_1 * sin(ang) + y_1 * cos(ang);
//       break;
//     }
//   }
// }

void matrizRotacionZ(double* x, double* y, const double theta)
{
  double x_1, y_1;

  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos(theta) - y_1 * sin(theta);
  *y = x_1 * sin(theta) + y_1 * cos(theta);
}

void matrizRotacionZ_seno_coseno_precalculado(double* x, double* y, const double sin_theta, const double cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta - y_1 * sin_theta;
  *y = x_1 * sin_theta + y_1 * cos_theta;
}
void matrizRotacionZ_seno_coseno_precalculado(float* x, float* y, const float sin_theta, const float cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta - y_1 * sin_theta;
  *y = x_1 * sin_theta + y_1 * cos_theta;
}

void matrizRotacionZInversa_seno_coseno_precalculado(double* x, double* y, const double sin_theta,
                                                     const double cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta + y_1 * sin_theta;
  *y = -x_1 * sin_theta + y_1 * cos_theta;
}
void matrizRotacionZInversa_seno_coseno_precalculado(float* x, float* y, const float sin_theta, const float cos_theta)
{
  double x_1, y_1;
  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos_theta + y_1 * sin_theta;
  *y = -x_1 * sin_theta + y_1 * cos_theta;
}

void matrizRotacionZ(float* x, float* y, const float theta)
{
  double x_1, y_1;

  x_1 = *x;
  y_1 = *y;

  *x = x_1 * cos(theta) - y_1 * sin(theta);
  *y = x_1 * sin(theta) + y_1 * cos(theta);
}

void fixAngleRad(double* ang)
{
  while (*ang > M_PI)
  {
    *ang -= 2 * M_PI;
  }

  while (*ang <= -M_PI)
  {
    *ang += 2 * M_PI;
  }
}
void fixAngleRad(float* ang)
{
  while (*ang > M_PI)
  {
    *ang -= 2 * M_PI;
  }

  while (*ang <= -M_PI)
  {
    *ang += 2 * M_PI;
  }
}

/*
void calcular_media_vector_1D(double* media, const double* x, const int num_puntos)
{
  for (int i = 0; i < num_puntos; i++)
  {
    *media += x[i];
  }

  *media /= num_puntos;
}

void calcular_varianza_1D(double* var, const double media, const double* x, const int num_puntos)
{
  for (int i = 0; i < num_puntos; i++)
  {
    *var += ((x[i] - media) * (x[i] - media));
  }
  *var /= (num_puntos - 1);
}

void printf_tiempos_medio_maximo_minimo(const std::vector<double>* tiempos, const std::string texto,
                                               const double factor_milliseconds)
{
  double media = 0.0, maximo = -100000.0, minimo = 100000.0;
  int cont_tiempos = 0;

  for (int i = 0; i < (int)tiempos->size(); i++)
  {
    if (tiempos->at(i) > 0)
    {
      cont_tiempos++;

      media += tiempos->at(i);

      if (tiempos->at(i) < minimo)
      {
        minimo = tiempos->at(i);
      }

      if (tiempos->at(i) > maximo)
      {
        maximo = tiempos->at(i);
      }
    }
  }
  media /= cont_tiempos;

  std::cout << texto << " media " << media * factor_milliseconds << " \t \t min " << minimo * factor_milliseconds
            << " \t max " << maximo * factor_milliseconds << " [ms] " << std::endl;
}
*/
void compute_rotation_matrix(const geometry_msgs::msg::Transform& transform, float rotation_matrix[3][3])
{
  // 1. Extraer el quaternion y convertirlo a una matriz de rotación
  tf2::Quaternion q(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);

  tf2::Matrix3x3 aux_R(q);

  // 2. Guardar solo la parte de rotación (3x3)
  std::memset(rotation_matrix, 0, 3 * 3 * sizeof(float));

  rotation_matrix[0][0] = aux_R[0][0];
  rotation_matrix[0][1] = aux_R[0][1];
  rotation_matrix[0][2] = aux_R[0][2];

  rotation_matrix[1][0] = aux_R[1][0];
  rotation_matrix[1][1] = aux_R[1][1];
  rotation_matrix[1][2] = aux_R[1][2];

  rotation_matrix[2][0] = aux_R[2][0];
  rotation_matrix[2][1] = aux_R[2][1];
  rotation_matrix[2][2] = aux_R[2][2];

  printf("Quaternion [%f, %f, %f, %f] to rotation matrix: \n", transform.rotation.x, transform.rotation.y,
         transform.rotation.z, transform.rotation.w);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      printf("   %.4f ", rotation_matrix[i][j]);
    }
    printf("\n");
  }
}

void compute_1D_layers_angles(int* i_1D, const int i_layer, const int i_angle, const int N_LAYERS)
{
  *i_1D = i_angle * N_LAYERS + i_layer;
}
void compute_2D_layers_angles(int* i_layer, int* i_angle, const int i_1D, const int N_LAYERS)
{
  *i_angle = i_1D / N_LAYERS;
  *i_layer = i_1D % N_LAYERS;
}

void calculoRGB_angulo(const double ang, double* R, double* G, double* B)
{
  double aux_ang = ang, valor_min, valor_max, normalizacion;
  fixAngleRad(&aux_ang);

  if (aux_ang >= 0 && aux_ang < M_PI / 2.0)
  {
    valor_min = 0;
    valor_max = M_PI / 2.0;
    normalizacion = (aux_ang - valor_min) / (valor_max - valor_min);

    *R = 0;
    *G = 1.0 - normalizacion;
    *B = normalizacion;
  }
  else if (aux_ang >= M_PI / 2.0 && aux_ang <= M_PI)
  {
    valor_min = M_PI / 2.0;
    valor_max = M_PI;
    normalizacion = (aux_ang - valor_min) / (valor_max - valor_min);

    *R = normalizacion;
    *G = 0;
    *B = 1 - normalizacion;
  }
  else if (aux_ang >= -M_PI && aux_ang < -M_PI / 2.0)
  {
    valor_min = -M_PI;
    valor_max = -M_PI / 2.0;
    normalizacion = (aux_ang - valor_min) / (valor_max - valor_min);

    *R = 1;
    *G = normalizacion;
    *B = 0;
  }
  else
  {
    valor_min = -M_PI / 2.0;
    valor_max = 0;
    normalizacion = (aux_ang - valor_min) / (valor_max - valor_min);

    *R = 1.0 - normalizacion;
    *G = 1;
    *B = 0;
  }
}

void pignistic_transformation(double* pO, const double mO, const double mL)
{
  *pO = mO + 0.5 * (1.0 - mO - mL);
}
