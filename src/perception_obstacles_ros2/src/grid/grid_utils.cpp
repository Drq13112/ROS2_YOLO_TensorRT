#include "perception_obstacles/grid/grid_utils.cuh"

namespace GRID_UTILS
{
// -------------------- INDEXES -------------------- //
void calculo_centro_celda(float* centro_celda_x, float* centro_celda_y, const int i_x, const int i_y, const int NC_X,
                          const int NC_Y, const float MIN_X, const float MIN_Y, const float RES)
{
  *centro_celda_x = (float)i_x * RES + MIN_X + RES / 2.0;
  *centro_celda_y = ((float)NC_Y - (float)i_y) * RES + MIN_Y - RES / 2.0;

  // printf("X = %f = %f * %f + %f + %f\n", *centro_celda_x, (float)i_x, RES, MIN_X, RES / 2.0);
  // printf(
  //   "Y = %f = (%f - %f) * %f + %f - %f\n", *centro_celda_y, (float)NC_Y, (float)i_y, RES, MIN_Y,
  //   RES / 2.0);
}

void calculo_indices_celda(int* i_x, int* i_y, float x, float y, const int NC_X, const int NC_Y, const float MIN_X,
                           const float MIN_Y, const float RES)
{
  int indice_x, indice_y;

  indice_x = ceil(round(((x - MIN_X) / RES) * 1e6) / 1e6) - 1;
  indice_y = NC_Y - ceil(round(((y - MIN_Y) / RES) * 1e6) / 1e6);

  if (indice_x < 0 || indice_x >= NC_X || indice_y < 0 || indice_y >= NC_Y)
  {
    *i_x = -1;
    *i_y = -1;
  }
  else
  {
    *i_x = indice_x;
    *i_y = indice_y;
  }
}

void calculo_indices_celda(int* i_x, int* i_y, double x, double y, const int NC_X, const int NC_Y, const double MIN_X,
                           const double MIN_Y, const double RES)
{
  int indice_x, indice_y;

  indice_x = ceil(round(((x - MIN_X) / RES) * 1e6) / 1e6) - 1;
  indice_y = NC_Y - ceil(round(((y - MIN_Y) / RES) * 1e6) / 1e6);

  if (indice_x < 0 || indice_x >= NC_X || indice_y < 0 || indice_y >= NC_Y)
  {
    *i_x = -1;
    *i_y = -1;
  }
  else
  {
    *i_x = indice_x;
    *i_y = indice_y;
  }
}

int sub2ind(const int i_y, const int i_x, const int NC_X, const int NC_Y)
{
  if (i_x < 0 || i_y < 0)
  {
    return -1;
  }
  return i_y * NC_X + i_x;
}

void ind2sub(const int idx, const int NC_X, const int NC_Y, int* i_y, int* i_x)
{
  if (idx == -1)
  {
    *i_y = idx / NC_X;
    *i_x = idx % NC_X;
  }
  else
  {
    *i_x = -1;
    *i_y = -1;
  }
}

}  // namespace GRID_UTILS