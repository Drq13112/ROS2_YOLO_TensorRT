#include "perception_obstacles/grid/grid_inicializar_rejilla_cartesiana.h"

void grid_inicializar_rejilla_cartesiana(GRID_TYPES::CART_Data* grid_cart_data)
{
  printf("   Leyendo ficheros rejilla cartesiana ... \n");
  std::string config_file_path = std::string(CONFIG_DIR) + "/config_cart_grid.yaml";

  YAML::Node data_yaml = YAML::LoadFile(config_file_path);

  printf(" Datos rejilla cartesiana:\n");

  grid_cart_data->NC_X = GRID_TYPES::NC_X;
  grid_cart_data->NC_Y = GRID_TYPES::NC_Y;
  grid_cart_data->RES = GRID_TYPES::RES;
  grid_cart_data->MIN_X = data_yaml["grid_local_min_x"].as<float>();
  grid_cart_data->MIN_Y = data_yaml["grid_local_min_y"].as<float>();

  printf("  - MIN_X = %f\n", grid_cart_data->MIN_X);
  printf("  - MIN_Y = %f\n", grid_cart_data->MIN_Y);
  printf("\n\n");

  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++)
  {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++)
    {
      GRID_UTILS::calculo_centro_celda(&grid_cart_data->centro_x[i_x], &grid_cart_data->centro_y[i_y], i_x, i_y,
                                       grid_cart_data->NC_X, grid_cart_data->NC_Y, grid_cart_data->MIN_X,
                                       grid_cart_data->MIN_Y, grid_cart_data->RES);
    }
  }
}