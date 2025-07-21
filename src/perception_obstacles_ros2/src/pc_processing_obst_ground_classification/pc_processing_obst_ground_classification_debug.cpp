#include "perception_obstacles/pc_processing_obst_ground_classification/pc_processing_obst_ground_classification.h"

void OBST_GROUND::write_file_pointcloud(const std::string name, const float x[], const float y[], const float z[],
                                        const float vert_ang[], const float intensity[], const int channel[],
                                        const int label[], const int channel_label[], const int label_reason[],
                                        const int n_points)
{
  FILE* fpt;

  std::string folderName = std::filesystem::current_path().string() + "/output";
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

  char spt[300];
  snprintf(spt, sizeof(spt), "%s/%s.txt", folderName.c_str(), name.c_str());
  fpt = fopen(spt, "w");
  if (fpt == NULL)
  {
    printf("NO SE PUDO CREAR EL FICHERO %s\n", spt);
  }

  fprintf(fpt, "x y z vert_ang intensity channel label channel_label label_reason\n");
  for (int i = 0; i < n_points; i++)
  {
    fprintf(fpt, "%f %f %f ", x[i], y[i], z[i]);
    fprintf(fpt, "%f ", vert_ang[i]);
    fprintf(fpt, "%f ", intensity[i]);
    fprintf(fpt, "%d ", channel[i]);
    fprintf(fpt, "%d ", label[i]);
    fprintf(fpt, "%d %d ", channel_label[i], label_reason[i]);
    fprintf(fpt, "\n");
    // printf("%f %f %f %d %d %d\n", x[i], y[i], z[i], intensity[i], channel[i], label[i]);
  }
  fclose(fpt);

  printf("file %s written\n", spt);
}