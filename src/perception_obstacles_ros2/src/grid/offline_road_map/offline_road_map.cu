#include "perception_obstacles/grid/offline_road_map/offline_road_map.cuh"

// +++++++++++++++++++++++++++++++++++++ FREE MEMORY +++++++++++++++++++++++++++++++++++++ //
void OFF_ROAD_MAP::free_allocated_memory(
  OFF_ROAD_MAP::config * h_config_map, OFF_ROAD_MAP::config * d_config_map,
  char * h_complete_road_map, char * d_complete_road_map, GRID_TYPES::GRID_local_road * h_grid_road,
  GRID_TYPES::GRID_local_road * d_grid_road)
{
  cudaFreeHost(h_complete_road_map);
  cudaFree(d_complete_road_map);

  cudaFreeHost(h_config_map);
  cudaFree(d_config_map);

  cudaFreeHost(h_grid_road);
  cudaFree(d_grid_road);
}

// +++++++++++++++++++++++++++++++++++++ INITIALIZATION +++++++++++++++++++++++++++++++++++++ //
void OFF_ROAD_MAP::initialize_offline_road_map(
  OFF_ROAD_MAP::config * h_config_map, OFF_ROAD_MAP::config * d_config_map,
  char * h_complete_road_map, char * d_complete_road_map, GRID_TYPES::GRID_local_road * h_grid_road,
  GRID_TYPES::GRID_local_road * d_grid_road)
{
  printf(
    "ESTO POR ALGUNA RAZÓN NO FUNCIONA ... no entiendo porque en el main si, si es lo mismo\n");
  exit(1);
  printf("\n\nInitializing offline road map ... \n");

  // 1. Data config
  // Reserve data
  checkCudaErrors(cudaMallocHost((void **)&h_config_map, sizeof(OFF_ROAD_MAP::config)));
  checkCudaErrors(cudaMalloc((void **)&d_config_map, sizeof(OFF_ROAD_MAP::config)));
  cudaDeviceSynchronize();

  // Read config file
  OFF_ROAD_MAP::leer_fichero_configuracion_mapa_ruta(h_config_map);

  // Copy to device
  checkCudaErrors(
    cudaMemcpy(d_config_map, h_config_map, sizeof(OFF_ROAD_MAP::config), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  // 2. Load grid map
  // CUIDADO ES IMPORTANTE leer_fichero_configuracion_mapa_ruta ANTES DE RESERVAR, YA QUE SE INICIALIZA config_map->NC_XY

  // Reserve
  checkCudaErrors(
    cudaMallocHost((void **)&h_complete_road_map, h_config_map->NC_XY * sizeof(char)));
  checkCudaErrors(cudaMalloc((void **)&d_complete_road_map, h_config_map->NC_XY * sizeof(char)));
  cudaDeviceSynchronize();

  // Read map
  OFF_ROAD_MAP::carga_mapa(h_complete_road_map, h_config_map);

  // Copy to device
  checkCudaErrors(cudaMemcpy(
    d_complete_road_map, h_complete_road_map, h_config_map->NC_XY * sizeof(char),
    cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  // // Local map
  checkCudaErrors(cudaMallocHost((void **)&h_grid_road, sizeof(GRID_TYPES::GRID_local_road)));
  checkCudaErrors(cudaMalloc((void **)&d_grid_road, sizeof(GRID_TYPES::GRID_local_road)));
  cudaDeviceSynchronize();

  std::memset(
    &h_grid_road->es_carretera, false, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(bool));
  checkCudaErrors(cudaMemset(
    &d_grid_road->es_carretera, false, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(bool)));

  cudaDeviceSynchronize();

  printf("... offline road map initialized \n\n");
}

void OFF_ROAD_MAP::leer_fichero_configuracion_mapa_ruta(OFF_ROAD_MAP::config * config_map)
{
  printf("   Leyendo ficheros mapa... \n");
  // std::string config_file_path = "/config/config_offline_road_map.yaml";
  std::string config_file_path = std::string(CONFIG_DIR) + "/config_offline_road_map.yaml";

  YAML::Node data_yaml = YAML::LoadFile(config_file_path);

  printf("      DATOS MAPA:\n");

  std::string nombre_mapa = data_yaml["nombre_mapa"].as<std::string>();
  config_map->nombre_mapa = nombre_mapa;
  printf("       - Nombre: %s\n", config_map->nombre_mapa.c_str());

  std::string path_mapa = data_yaml[config_map->nombre_mapa + "_path_mapa_ruta"].as<std::string>();
  config_map->path_mapa = path_mapa;
  printf("       - path_mapa: %s\n", config_map->path_mapa.c_str());

  config_map->NC_X = data_yaml[config_map->nombre_mapa + "_NC_X_route_map"].as<int>();
  config_map->NC_Y = data_yaml[config_map->nombre_mapa + "_NC_Y_route_map"].as<int>();
  config_map->NC_XY = config_map->NC_X * config_map->NC_Y;
  printf("       - Celdas: [%d, %d]\n", config_map->NC_X, config_map->NC_Y);

  config_map->UTM_origen_mapa_x =
    data_yaml[config_map->nombre_mapa + "_UTM_origen_mapa_x"].as<double>();
  config_map->UTM_origen_mapa_y =
    data_yaml[config_map->nombre_mapa + "_UTM_origen_mapa_y"].as<double>();
  config_map->UTM_origen_mapa_heading =
    data_yaml[config_map->nombre_mapa + "_UTM_origen_mapa_heading"].as<double>();
  printf(
    "       - Origen: [%f, %f, %fgrad]\n", config_map->UTM_origen_mapa_x,
    config_map->UTM_origen_mapa_y, config_map->UTM_origen_mapa_heading);

  config_map->lado_celda = data_yaml[config_map->nombre_mapa + "_lado_celda"].as<float>();
  printf("       - Lado celda: %f\n", config_map->lado_celda);

  config_map->UTM_origen_mapa_heading_seno = sin(config_map->UTM_origen_mapa_heading);
  config_map->UTM_origen_mapa_heading_coseno = cos(config_map->UTM_origen_mapa_heading);

  config_map->type_off_road = data_yaml["type_off_road"].as<int>();
  config_map->type_carretera = data_yaml["type_carretera"].as<int>();

  printf("   ... parametros mapa ruta leidos\n\n");
}

void OFF_ROAD_MAP::carga_mapa(char * h_complete_road_map, const OFF_ROAD_MAP::config * config_map)
{
  printf("Leyendo mapa ruta ...\n");

  char path_mapa[200];
  std::string line;

  std::string aux_map_dir = std::string(MAP_DIR);

  snprintf(
    path_mapa, sizeof(path_mapa), "%s/%smapa.bin", aux_map_dir.c_str(),
    config_map->path_mapa.c_str());

  FILE * f = fopen(path_mapa, "rb");
  if (f == NULL) {
    printf("ERROR opening file path_mapa\n  %s\n\n", path_mapa);
    exit(1);
  }

  printf("    leyendo mapa %s\n", path_mapa);

  size_t result;
  for (int i = 0; i < config_map->NC_XY; i++) {
    result = fread(&h_complete_road_map[i], 1, sizeof(char), f);
  }
  fclose(f);

  printf("    ... mapa %s cargado! \n", path_mapa);
}

// +++++++++++++++++++++++++++++++++++++ CALCULATIONS +++++++++++++++++++++++++++++++++++++ //
void OFF_ROAD_MAP::calculo_indices_celda_offline_road_map(
  int * i_x, int * i_y, bool * conseguido, const float x, const float y,
  const OFF_ROAD_MAP::config * config_map)
{
  *i_x = ceil(round(((config_map->UTM_origen_mapa_x - x) / config_map->lado_celda) * 1e6) / 1e6);
  *i_y = ceil(round(((y - config_map->UTM_origen_mapa_y) / config_map->lado_celda) * 1e6) / 1e6);

  *conseguido = true;
  if (*i_x < 0 || *i_x >= config_map->NC_X || *i_y < 0 || *i_y >= config_map->NC_Y) {
    *conseguido = false;
  }
}

void OFF_ROAD_MAP::route_map_ind2sub(int * i_x, int * i_y, const int idx, const int NC_X_map)
{
  *i_y = idx / NC_X_map;
  *i_x = idx % NC_X_map;
}
int OFF_ROAD_MAP::route_map_sub2ind(const int i_x, const int i_y, const int NC_X_map)
{
  return NC_X_map * i_y + i_x;
}

__device__ void OFF_ROAD_MAP::kernel_ind2sub(
  int * i_x, int * i_y, const int idx, const int NC_X_map)
{
  *i_y = idx / NC_X_map;
  *i_x = idx % NC_X_map;
}
__device__ int OFF_ROAD_MAP::kernel_sub2ind(const int i_x, const int i_y, const int NC_X_map)
{
  return NC_X_map * i_y + i_x;
}

__device__ void OFF_ROAD_MAP::device_calculo_indices_celda_offline_road_map(
  int * i_x, int * i_y, bool * conseguido, const float x, const float y,
  const OFF_ROAD_MAP::config * config_map)
{
  *i_x = ceil(round(((config_map->UTM_origen_mapa_x - x) / config_map->lado_celda) * 1e6) / 1e6);
  *i_y = ceil(round(((y - config_map->UTM_origen_mapa_y) / config_map->lado_celda) * 1e6) / 1e6);

  *conseguido = true;
  //	if(*i_x < 0 || *i_x >= OFF_ROAD_MAP::NC_X || *i_y < 0 || *i_y >= OFF_ROAD_MAP::NC_Y){
  if (*i_x < 0 || *i_x >= config_map->NC_X || *i_y < 0 || *i_y >= config_map->NC_Y) {
    *conseguido = false;
  }
}

__global__ void OFF_ROAD_MAP::copia_mapa_a_rejilla_local_straightforward(
  GRID_TYPES::GRID_local_road * grid_road, const char * complete_road_map,
  const EGO_VEH::INFO_ego * info_coche, const float * centro_x, const float * centro_y,
  const OFF_ROAD_MAP::config * config_map)
{
  int i_x_map, i_y_map, idx_map, i_x_ego, i_y_ego, idx_ego;
  double x_mapa, y_mapa;  // Importante double porque vamos a ir a UTM

  i_x_ego = blockIdx.x * blockDim.x + threadIdx.x;
  i_y_ego = blockIdx.y * blockDim.y + threadIdx.y;

  // Convertir celda al S.Ref del mapa
  x_mapa = centro_x[i_x_ego];
  y_mapa = centro_y[i_y_ego];

  device_matrizRotacionZ_seno_coseno_precalculado(
    &x_mapa, &y_mapa, info_coche->sin_yaw_G, info_coche->cos_yaw_G);
  x_mapa += info_coche->px_G;
  y_mapa += info_coche->py_G;

  // printf("[%f, %f, %f]    [%d, %d] -> [%f, %f] -> [%f, %f]\n", info_coche->px_G, info_coche->py_G, info_coche->yaw_G, i_x_ego, i_y_ego, info_grid->centro_x[i_x_ego], info_grid->centro_y[i_y_ego], x_mapa, y_mapa);

  // 1D local grid
  idx_ego = GRID_UTILS_CUDA::device_sub2ind(i_y_ego, i_x_ego, GRID_TYPES::NC_X, GRID_TYPES::NC_Y);

  // Calcular indice de la celda
  bool conseguido;
  OFF_ROAD_MAP::device_calculo_indices_celda_offline_road_map(
    &i_x_map, &i_y_map, &conseguido, x_mapa, y_mapa, config_map);

  if (conseguido == false) {
    grid_road->es_carretera[idx_ego] = false;
    return;
  }

  // Rellenamos el contenido de la celda
  idx_map = OFF_ROAD_MAP::kernel_sub2ind(i_x_map, i_y_map, config_map->NC_X);

  if (static_cast<int>(complete_road_map[idx_map]) == config_map->type_carretera) {
    grid_road->es_carretera[idx_ego] = true;
  } else {
    grid_road->es_carretera[idx_ego] = false;
  }
}

void OFF_ROAD_MAP::get_offline_road_map_core(
  GRID_TYPES::GRID_local_road * d_grid_road, const char * d_complete_road_map,
  const OFF_ROAD_MAP::config * d_config_map, const EGO_VEH::INFO_ego * d_info_coche,
  const GRID_TYPES::CART_Data * d_grid_cart)
{
  static dim3 block_rejilla_cart(32, 16, 1);
  static dim3 grid_rejilla_cart(
    (GRID_TYPES::NC_X + block_rejilla_cart.x - 1) / block_rejilla_cart.x,
    (GRID_TYPES::NC_Y + block_rejilla_cart.y - 1) / block_rejilla_cart.y, 1);

  OFF_ROAD_MAP::
    copia_mapa_a_rejilla_local_straightforward<<<grid_rejilla_cart, block_rejilla_cart>>>(
      d_grid_road, d_complete_road_map, d_info_coche, d_grid_cart->centro_x, d_grid_cart->centro_y,
      d_config_map);

  // To finish and ensure synchronization
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "get_offline_road_map_core - Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}
