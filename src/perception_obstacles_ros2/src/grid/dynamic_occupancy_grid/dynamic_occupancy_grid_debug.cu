#include "perception_obstacles/grid/dynamic_occupancy_grid/dynamic_occupancy_grid.h"

// ------------------------------------------------------------------------------------------------------ //
// El UNICO momento en el que las particulas pueden tener peso mayor que 1 es despues de aglomerar las estaticas (durante la prediccion)
bool DYN_CLASS_OG::comprobar_peso_particula(
  const PARTICLE_TYPES::PART_DOG * d_particles,
  const double d_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const GRID_TYPES::DOG * d_grid,
  const double factor_perdida_precision)
{
  printf("DEBUG - comprobar_peso_particula\n");

  cudaDeviceSynchronize();

  PARTICLE_TYPES::PART_DOG * host_particles;
  GRID_TYPES::DOG * host_grid;
  double host_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double check_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  int check_n_part[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  checkCudaErrors(cudaMallocHost((void **)&host_particles, sizeof(PARTICLE_TYPES::PART_DOG)));
  checkCudaErrors(cudaMallocHost((void **)&host_grid, sizeof(GRID_TYPES::DOG)));

  cudaMemcpy(host_particles, d_particles, sizeof(PARTICLE_TYPES::PART_DOG), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_grid, d_grid, sizeof(GRID_TYPES::DOG), cudaMemcpyDeviceToHost);
  cudaMemcpy(
    host_mO, d_mO, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double), cudaMemcpyDeviceToHost);

  memset(check_mO, 0.0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(double));
  memset(check_n_part, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int));

  int i_y, i_x;

  cudaDeviceSynchronize();

  bool correcto = true;
  for (int i_p = 0; i_p < PARTICLE_TYPES::NP_ACT; i_p++) {
    if (host_particles->valida[i_p]) {
      i_x = host_particles->indice_celda_x[i_p];
      i_y = host_particles->indice_celda_y[i_p];
      check_mO[i_y][i_x] += host_particles->peso_factor[i_p];
      check_n_part[i_y][i_x]++;
      if (
        host_particles->peso_factor[i_p] < 0 ||
        host_particles->peso_factor[i_p] > factor_perdida_precision ||
        isnan(host_particles->peso_factor[i_p]) || isinf(host_particles->peso_factor[i_p])) {
        printf(
          "debug_fp_comprobar_peso_particula - %d -> Peso particula valida = %.16f\t [%.16f, "
          "%.16f], [%.16f, %.16f] -> cell [%d, %d] (num gpu particles = %d)\n",
          i_p, host_particles->peso_factor[i_p], host_particles->p_x[i_p], host_particles->p_y[i_p],
          host_particles->v_x[i_p], host_particles->v_y[i_p], host_particles->indice_celda_x[i_p],
          host_particles->indice_celda_y[i_p], host_grid->numero_particulas[i_y][i_x]);
        correcto = false;

        break;
      }
    }
  }
  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      if (
        std::abs(check_mO[i_y][i_x] - host_mO[i_y][i_x]) > 1e-5 &&
        host_grid->numero_particulas[i_y][i_x] > 0) {
        printf(
          "valor_ocupacion[%d, %d] = %f NO coincide con la suma del peso de las part = %f (num gpu "
          "particles = %d    check nº particles = %d)\n",
          i_y, i_x, host_mO[i_y][i_x], check_mO[i_y][i_x], host_grid->numero_particulas[i_y][i_x],
          check_n_part[i_y][i_x]);
        correcto = false;
        break;
      }
    }
    if (correcto == false) {
      break;
    }
  }

  cudaFreeHost(host_particles);
  cudaFreeHost(host_grid);
  cudaDeviceSynchronize();
  printf("DEBUG - comprobar_peso_particula hecho\n");

  return correcto;
}

// ------------------------------------------------------------------------------------------------------ //
bool DYN_CLASS_OG::check_particle_cell_indexes(
  const PARTICLE_TYPES::PART_DOG * d_particles, const GRID_TYPES::CART_Data * grid_cart_data)
{
  printf("DEBUG - check_particle_cell_indexes\n");

  cudaDeviceSynchronize();
  bool correcto = true;

  PARTICLE_TYPES::PART_DOG * host_particles;
  checkCudaErrors(cudaMallocHost((void **)&host_particles, sizeof(PARTICLE_TYPES::PART_DOG)));
  cudaMemcpy(host_particles, d_particles, sizeof(PARTICLE_TYPES::PART_DOG), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int i_x, i_y, i_celda;
  int cont_printf = 0;
  for (int i_p = 0; i_p < PARTICLE_TYPES::NP_ACT; i_p++) {
    if (host_particles->valida[i_p]) {
      GRID_UTILS::calculo_indices_celda(
        &i_x, &i_y, host_particles->p_x[i_p], host_particles->p_y[i_p], grid_cart_data->NC_X,
        grid_cart_data->NC_Y, grid_cart_data->MIN_X, grid_cart_data->MIN_Y, grid_cart_data->RES);

      i_celda = GRID_UTILS::sub2ind(i_y, i_x, GRID_TYPES::NC_X, GRID_TYPES::NC_Y);

      if (
        i_x != host_particles->indice_celda_x[i_p] && i_y != host_particles->indice_celda_y[i_p] ||
        i_celda != host_particles->indice_celda[i_p] || host_particles->indice_celda_x[i_p] < 0 ||
        host_particles->indice_celda_y[i_p] < 0 || host_particles->indice_celda[i_p] < 0 ||
        isnan(host_particles->indice_celda_x[i_p]) || isnan(host_particles->indice_celda_y[i_p]) ||
        isnan(host_particles->indice_celda[i_p]) || isinf(host_particles->indice_celda_x[i_p]) ||
        isinf(host_particles->indice_celda_y[i_p]) || isinf(host_particles->indice_celda[i_p])) {
        printf(
          "part[%d] pos = [%f, %f] sus indices guardados en gpu = [%d, %d] = %d   !=   cpu_debug = "
          "[%d, %d] = %d  (veces remuestreada = %d, new born = %d valida = %d, peso = %f)\n",
          i_p, host_particles->p_x[i_p], host_particles->p_y[i_p],
          host_particles->indice_celda_x[i_p], host_particles->indice_celda_y[i_p],
          host_particles->indice_celda[i_p], i_x, i_y, i_celda,
          host_particles->veces_remuestreada[i_p], (int)host_particles->new_born[i_p],
          (int)host_particles->valida[i_p], host_particles->peso_factor[i_p]);
        correcto = false;
        cont_printf++;
        if (cont_printf > 5) {
          break;
        }
      }
    }
  }

  cudaFreeHost(host_particles);
  cudaDeviceSynchronize();
  printf("DEBUG - check_particle_cell_indexes hecho\n");

  return correcto;
}

bool DYN_CLASS_OG::check_cell_particle_number(
  const PARTICLE_TYPES::PART_DOG * d_particles, const GRID_TYPES::DOG * d_grid,
  const GRID_TYPES::CART_Data * grid_cart_data)
{
  printf("DEBUG - check_cell_particle_number\n");

  cudaDeviceSynchronize();
  bool correcto = true;

  PARTICLE_TYPES::PART_DOG * host_particles;
  checkCudaErrors(cudaMallocHost((void **)&host_particles, sizeof(PARTICLE_TYPES::PART_DOG)));
  cudaMemcpy(host_particles, d_particles, sizeof(PARTICLE_TYPES::PART_DOG), cudaMemcpyDeviceToHost);

  GRID_TYPES::DOG * host_grid;
  checkCudaErrors(cudaMallocHost((void **)&host_grid, sizeof(GRID_TYPES::DOG)));
  cudaMemcpy(host_grid, d_grid, sizeof(GRID_TYPES::DOG), cudaMemcpyDeviceToHost);
  int check_n_part[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  memset(check_n_part, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int));

  int i_x, i_y;
  for (int i_p = 0; i_p < PARTICLE_TYPES::NP_ACT; i_p++) {
    if (host_particles->valida[i_p]) {
      GRID_UTILS::calculo_indices_celda(
        &i_x, &i_y, host_particles->p_x[i_p], host_particles->p_y[i_p], grid_cart_data->NC_X,
        grid_cart_data->NC_Y, grid_cart_data->MIN_X, grid_cart_data->MIN_Y, grid_cart_data->RES);

      check_n_part[i_y][i_x]++;
    }
  }

  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      if (check_n_part[i_y][i_x] != check_n_part[i_y][i_x]) {
        printf(
          "n particulas [%d, %d]: gpu = [%d] vs check [%d]\n", i_x, i_y,
          host_grid->numero_particulas[i_y][i_x], check_n_part[i_y][i_x]);
        correcto = false;
      }
    }
  }

  cudaFreeHost(host_particles);
  printf("DEBUG - check_cell_particle_number hecho\n");

  cudaFreeHost(host_grid);

  cudaDeviceSynchronize();
  return correcto;
}

// ------------------------------------------------------------------------------------------------------ //
bool DYN_CLASS_OG::ordenar_particulas(
  const PARTICLE_TYPES::PART_DOG * d_particles, const PARTICLE_TYPES::PART_DOG * d_particles_sorted,
  const GRID_TYPES::DOG * d_grid, const GRID_TYPES::CART_Data * host_grid_cart_data)
{
  printf("DEBUG - ordenar_particulas\n");
  cudaDeviceSynchronize();
  bool correcto = true;

  PARTICLE_TYPES::PART_DOG * host_particles;
  PARTICLE_TYPES::PART_DOG * host_particles_sorted;
  GRID_TYPES::DOG * host_grid;

  checkCudaErrors(cudaMallocHost((void **)&host_particles, sizeof(PARTICLE_TYPES::PART_DOG)));
  checkCudaErrors(
    cudaMallocHost((void **)&host_particles_sorted, sizeof(PARTICLE_TYPES::PART_DOG)));
  checkCudaErrors(cudaMallocHost((void **)&host_grid, sizeof(GRID_TYPES::DOG)));

  cudaMemcpy(host_particles, d_particles, sizeof(PARTICLE_TYPES::PART_DOG), cudaMemcpyDeviceToHost);
  cudaMemcpy(
    host_particles_sorted, d_particles_sorted, sizeof(PARTICLE_TYPES::PART_DOG),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(host_grid, d_grid, sizeof(GRID_TYPES::DOG), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();  // Not needed really

  int num_part[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
    num_part_sorted[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  int i_x, i_y;
  memset(num_part, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int));
  for (int i_p = 0; i_p < PARTICLE_TYPES::NP_ACT; i_p++) {
    if (host_particles->valida[i_p]) {
      i_x = host_particles->indice_celda_x[i_p];
      i_y = host_particles->indice_celda_y[i_p];
      if (i_x != -1 && i_y != -1) {
        num_part[i_y][i_x]++;
      }
    }
  }

  memset(num_part_sorted, 0, GRID_TYPES::NC_X * GRID_TYPES::NC_Y * sizeof(int));
  for (int i_p = 0; i_p < PARTICLE_TYPES::NP_ACT; i_p++) {
    if (host_particles_sorted->valida[i_p]) {
      i_x = host_particles_sorted->indice_celda_x[i_p];
      i_y = host_particles_sorted->indice_celda_y[i_p];
      if (i_x != -1 && i_y != -1) {
        num_part_sorted[i_y][i_x]++;
      }
    }
  }

  int cont_printf = 0;
  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      if (abs(num_part[i_y][i_x] - num_part_sorted[i_y][i_x]) > 1e-9) {
        correcto = false;
        if (cont_printf < 20) {
          printf(
            "debug_fp_ordenar_particulas - celda [%d, %d]: resultado distinto  num part %d ?= %d\n",
            i_y, i_x, num_part[i_y][i_x], num_part_sorted[i_y][i_x]);
          cont_printf++;
        }
      }
    }
  }

  int aux_ix, aux_iy, aux_i_celda;
  bool out_bucle = false;
  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      for (int i_p = host_grid->indice_primera_particula[i_y][i_x];
           i_p < host_grid->indice_ultima_particula[i_y][i_x]; i_p++) {
        if (host_particles->valida[i_p]) {
          GRID_UTILS::calculo_indices_celda(
            &aux_ix, &aux_iy, host_particles->p_x[i_p], host_particles->p_y[i_p], GRID_TYPES::NC_X,
            GRID_TYPES::NC_Y, (double)host_grid_cart_data->MIN_X,
            (double)host_grid_cart_data->MIN_Y, host_grid_cart_data->RES);

          aux_i_celda = GRID_UTILS::sub2ind(aux_iy, aux_ix, GRID_TYPES::NC_X, GRID_TYPES::NC_Y);

          if (
            aux_ix != i_x || aux_iy != i_y || host_particles->indice_celda_x[i_p] != i_x ||
            host_particles->indice_celda_y[i_p] != i_y) {
            correcto = false;
            printf(
              "particle %d asociada a la celda [%d, %d] por ordenacion, pero "
              "es incorrecto.    Su posición [%f, %f], indices acordes [%d, %d] = %d; indices "
              "guardados "
              "[%d, %d] = %d\n",
              i_p, i_x, i_y, host_particles->p_x[i_p], host_particles->p_y[i_p], aux_ix, aux_iy,
              aux_i_celda, host_particles->indice_celda_x[i_p], host_particles->indice_celda_y[i_p],
              host_particles->indice_celda[i_p]);

            out_bucle = true;
          }
        }
        if (out_bucle) break;
      }
      if (out_bucle) break;
    }
    if (out_bucle) break;
  }

  cudaFreeHost(host_particles);
  cudaFreeHost(host_particles_sorted);
  cudaFreeHost(host_grid);

  cudaDeviceSynchronize();
  printf("DEBUG - ordenar_particulas hecho\n");

  return correcto;
}

// ------------------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------------------ //
// ---------------------------------------------- FICHEROS ---------------------------------------------- //
// ------------------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------------------ //
void DYN_CLASS_OG::write_files_occupancy(
  const double d_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X],
  const double d_mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const int i_iter,
  const DYN_CLASS_OG::config * config_DOG, const std::string nombre)
{
  cudaDeviceSynchronize();

  double h_mO[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];
  double h_mF[GRID_TYPES::NC_Y][GRID_TYPES::NC_X];

  cudaMemcpy(
    h_mO, d_mO, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(
    h_mF, d_mF, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(double), cudaMemcpyDeviceToHost);

  std::string folderName = std::filesystem::current_path().string() + "/output";

  char sO[200];
  snprintf(sO, sizeof(sO), "%s/%s_mO_%d.txt", folderName.c_str(), nombre.c_str(), i_iter);
  FILE * fO = fopen(sO, "w");
  char sF[200];
  snprintf(sF, sizeof(sF), "%s/%s_mF_%d.txt", folderName.c_str(), nombre.c_str(), i_iter);
  FILE * fF = fopen(sF, "w");

  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      fprintf(fO, "%f ", h_mO[i_y][i_x] / config_DOG->factor);
      fprintf(fF, "%f ", h_mF[i_y][i_x] / config_DOG->factor);
    }
    fprintf(fO, "\n");
    fprintf(fF, "\n");
  }

  cudaDeviceSynchronize();
}

void DYN_CLASS_OG::write_files_num_particles(
  const GRID_TYPES::DOG * d_dog, const int i_iter, const DYN_CLASS_OG::config * config_DOG)
{
  cudaDeviceSynchronize();

  GRID_TYPES::DOG * h_dog;

  cudaMallocHost((void **)&h_dog, sizeof(GRID_TYPES::DOG));

  cudaMemcpy(h_dog, d_dog, sizeof(GRID_TYPES::DOG), cudaMemcpyDeviceToHost);

  std::string folderName = std::filesystem::current_path().string() + "/output";

  char s[200];
  snprintf(s, sizeof(s), "%s/num_particles_%d.txt", folderName.c_str(), i_iter);
  FILE * f = fopen(s, "w");

  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      fprintf(f, "%d ", h_dog->numero_particulas[i_y][i_x]);
    }
    fprintf(f, "\n");
  }

  cudaFreeHost(h_dog);
  cudaDeviceSynchronize();
}

void DYN_CLASS_OG::write_files_DOG_color(
  const GRID_TYPES::DOG * d_grid, const int i_iter, const DYN_CLASS_OG::config * config_DOG)
{
  cudaDeviceSynchronize();

  GRID_TYPES::DOG * host_grid;
  cudaMallocHost((void **)&host_grid, sizeof(GRID_TYPES::DOG));
  cudaMemcpy(host_grid, d_grid, sizeof(GRID_TYPES::DOG), cudaMemcpyDeviceToHost);

  std::string folderName = std::filesystem::current_path().string() + "/output";

  char sgR[200];
  snprintf(sgR, sizeof(sgR), "%s/mapa_dinamica_color_R_%d.txt", folderName.c_str(), i_iter);
  FILE * fgR = fopen(sgR, "w");
  char sgG[200];
  snprintf(sgG, sizeof(sgG), "%s/mapa_dinamica_color_G_%d.txt", folderName.c_str(), i_iter);
  FILE * fgG = fopen(sgG, "w");
  char sgB[200];
  snprintf(sgB, sizeof(sgB), "%s/mapa_dinamica_color_B_%d.txt", folderName.c_str(), i_iter);
  FILE * fgB = fopen(sgB, "w");
  char sgM[200];
  snprintf(sgM, sizeof(sgM), "%s/mapa_mahalanobis_%d.txt", folderName.c_str(), i_iter);
  FILE * fgM = fopen(sgM, "w");

  if (fgR == NULL) {
    printf("NO SE PUDO CREAR EL FICHERO %s\n", sgR);
    exit(1);
  }

  double aux, R, G, B, modulo, angulo, mahal;
  double vel_full_bright = 1.0;
  if (vel_full_bright < 1e-5) {
    vel_full_bright = 1e-5;  // para evitar division entre 0
  }
  int32_t int_R, int_G, int_B;
  int32_t int_mahal;

  int cont_mahal = 0, cont_ocupadas = 0;
  for (int i_y = 0; i_y < GRID_TYPES::NC_Y; i_y++) {
    for (int i_x = 0; i_x < GRID_TYPES::NC_X; i_x++) {
      mahal = -1;
      if (
        host_grid->info_vel_valida[i_y][i_x] &&
        host_grid->masa_act_oc_factor[i_y][i_x] / config_DOG->factor >=
          config_DOG->threshold_celda_ocupada) {
        cont_ocupadas++;

        mahal = host_grid->vel_mahalanobis[i_y][i_x];

        if (host_grid->numero_particulas_velocidad[i_y][i_x] <= 0) {
          cont_mahal++;
          R = 0;
          G = 0;
          B = 0;
        } else {
          modulo = sqrt(
            host_grid->vel_media_x[i_y][i_x] * host_grid->vel_media_x[i_y][i_x] +
            host_grid->vel_media_y[i_y][i_x] * host_grid->vel_media_y[i_y][i_x]);
          angulo = atan2(host_grid->vel_media_y[i_y][i_x], host_grid->vel_media_x[i_y][i_x]);
          calculoRGB_angulo(angulo, &R, &G, &B);
          if (vel_full_bright >= 0) {
            aux = std::min(modulo / vel_full_bright, 1.0);
            R *= aux;
            G *= aux;
            B *= aux;
          }
        }
      } else {
        pignistic_transformation(
          &aux, host_grid->masa_act_oc_factor[i_y][i_x] / config_DOG->factor,
          host_grid->masa_act_libre_factor[i_y][i_x] / config_DOG->factor);
        //				aux = host_grid->masa_act_libre_factor[i_y][i_x] / config_DOG->factor * 0.5 + 0.5;
        R = 1 - aux;
        G = 1 - aux;
        B = 1 - aux;
      }

      int_R = (uint8_t)(std::round(R * 255));
      int_G = (uint8_t)(std::round(G * 255));
      int_B = (uint8_t)(std::round(B * 255));
      int_mahal = (int32_t)(std::round(mahal * config_DOG->factor));

      fprintf(fgR, "%d ", int_R);
      fprintf(fgG, "%d ", int_G);
      fprintf(fgB, "%d ", int_B);
      fprintf(fgM, "%d ", int_mahal);
    }
    fprintf(fgR, "\n");
    fprintf(fgG, "\n");
    fprintf(fgB, "\n");
    fprintf(fgM, "\n");
  }
  fclose(fgR);
  fclose(fgG);
  fclose(fgB);
  fclose(fgM);

  cudaFreeHost(host_grid);
  cudaDeviceSynchronize();
}
