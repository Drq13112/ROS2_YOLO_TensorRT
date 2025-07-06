#include "perception_obstacles/ego_vehicle/calculos_estado_coche.h"

void EGO_VEH::write_files_localization(const EGO_VEH::INFO_ego* info_coche, const bool start_file)
{
  char spt[200];
  std::string folderName = std::filesystem::current_path().string() + "/output";
  snprintf(spt, sizeof(spt), "%s/localization.txt", folderName.c_str());

  FILE* fpt;
  if (start_file)
  {
    if (std::filesystem::exists(folderName) == false)
    {
      if (std::filesystem::create_directory(folderName))
      {
        std::cout << "Folder '" << folderName << "' created successfully." << std::endl;
      }
      else
      {
        std::cerr << "Error: Could not create the folder '" << folderName << "'." << std::endl;
      }
    }

    fpt = fopen(spt, "w");

    if (fpt == NULL)
    {
      printf("NO SE PUDO CREAR EL FICHERO %s\n", spt);
      exit(1);
    }

    fprintf(fpt, "timestamp ");
    fprintf(fpt, "sec ");
    fprintf(fpt, "nanosec ");

    fprintf(fpt, "px_G ");
    fprintf(fpt, "py_G ");
    fprintf(fpt, "yaw_G ");

    fprintf(fpt, "vel  ");
    fprintf(fpt, "yaw_rate ");

    fprintf(fpt, "\n");

    fclose(fpt);
  }
  else
  {
    fpt = fopen(spt, "a");

    if (fpt == NULL)
    {
      printf("NO SE PUDO CREAR EL FICHERO %s\n", spt);
      exit(1);
    }

    fprintf(fpt, "%f ", info_coche->tiempo);
    fprintf(fpt, "%d ", info_coche->sec);
    fprintf(fpt, "%d ", info_coche->nanosec);
    fprintf(fpt, "%f ", info_coche->px_G);
    fprintf(fpt, "%f ", info_coche->py_G);
    fprintf(fpt, "%f ", info_coche->yaw_G);
    fprintf(fpt, "%f ", info_coche->vel);
    fprintf(fpt, "%f ", info_coche->yaw_rate);
    fprintf(fpt, "\n");
    fclose(fpt);
  }

  printf("file %s written\n", spt);
}