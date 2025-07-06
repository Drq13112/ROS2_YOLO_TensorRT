#include "perception_obstacles/grid/observed_occupancy_grid/observed_occupancy_grid.h"

// ---------------------------------------- DEBUG ---------------------------------------- //

void OBS_OG::write_files_observed_occupancy_grid(
  const float d_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
  const float d_mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const int iter)
{
  float mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  float mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  checkCudaErrors(cudaMemcpy(
    mO, d_mO, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(
    mF, d_mF, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  std::string folderName = std::filesystem::current_path().string() + "/output";
  if (std::filesystem::exists(folderName) == false) {
    if (std::filesystem::create_directory(folderName)) {
      std::cout << "Folder '" << folderName << "' created successfully." << std::endl;
    } else {
      std::cerr << "Error: Could not create the folder '" << folderName << "'." << std::endl;
    }
  }

  char s_mO[300], s_mF[300];
  snprintf(s_mO, sizeof(s_mO), "%s/mO_%d.txt", folderName.c_str(), iter);
  snprintf(s_mF, sizeof(s_mF), "%s/mF_%d.txt", folderName.c_str(), iter);
  FILE *f_mO, *f_mF;

  f_mO = fopen(s_mO, "w");
  if (f_mO == NULL) {
    printf("NO SE PUDO CREAR EL FICHERO %s\n", s_mO);
    exit(1);
  }
  f_mF = fopen(s_mF, "w");

  bool hay_error = false;
  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      fprintf(f_mO, "%f ", mO[i_y][i_x]);
      fprintf(f_mF, "%f ", mF[i_y][i_x]);

      if (
        (mO[i_y][i_x] + mF[i_y][i_x]) > 1 || mO[i_y][i_x] < 0 || mF[i_y][i_x] < 0 ||
        mO[i_y][i_x] > 1 || mF[i_y][i_x] > 1 || isnan(mF[i_y][i_x]) || isinf(mF[i_y][i_x])) {
        printf("ERROR: [%d, %d] -> [%f, %f]; \n", i_x, i_y, mO[i_y][i_x], mF[i_y][i_x]);
        hay_error = true;
      }
    }
    fprintf(f_mO, "\n");
    fprintf(f_mF, "\n");
  }

  fclose(f_mO);
  fclose(f_mF);

  printf("Observed OG files written\n");
  if (hay_error) {
    exit(1);
  }
}

void OBS_OG::write_files_polar_OG(
  const float d_grid_polar_mO[], const float d_grid_polar_mF[], const int NC_ANG, const int NC_DIST,
  const int iter)
{
  float grid_polar_mO[NC_ANG * NC_DIST];
  float grid_polar_mF[NC_ANG * NC_DIST];

  // float * grid_polar_mO = new float[NC_ANG * NC_DIST];
  // float * grid_polar_mF = new float[NC_ANG * NC_DIST];

  checkCudaErrors(cudaMemcpy(
    grid_polar_mO, d_grid_polar_mO, NC_ANG * NC_DIST * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(
    grid_polar_mF, d_grid_polar_mF, NC_ANG * NC_DIST * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  char s_mO[200], s_mF[200];
  snprintf(s_mO, sizeof(s_mO), "output/polar_mO_%d.txt", iter);
  snprintf(s_mF, sizeof(s_mF), "output/polar_mF_%d.txt", iter);
  FILE *f_mO, *f_mF;

  f_mO = fopen(s_mO, "w");
  if (f_mO == NULL) {
    printf("NO SE PUDO CREAR EL FICHERO %s\n", s_mO);
    exit(1);
  }
  f_mF = fopen(s_mF, "w");

  int idx1D;
  for (int i_a = 0; i_a < NC_ANG; i_a++) {
    for (int i_d = 0; i_d < NC_DIST; i_d++) {
      OBS_OG::polar_compute_1D_angle_dist(&idx1D, i_a, i_d, NC_DIST);

      fprintf(f_mO, "%f ", grid_polar_mO[idx1D]);
      fprintf(f_mF, "%f ", grid_polar_mF[idx1D]);
    }
    fprintf(f_mO, "\n");
    fprintf(f_mF, "\n");
  }

  fclose(f_mO);
  fclose(f_mF);

  // free(grid_polar_mO);
  // grid_polar_mO = NULL;
  // free(grid_polar_mF);
  // grid_polar_mF = NULL;
  printf("Ficheros rejilla observada polar escritos\n");
}