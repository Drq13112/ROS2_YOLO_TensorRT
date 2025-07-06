#include "perception_obstacles/perception_utilities/time_data.h"

void write_files_time(DATA_times* TIME_measurements, const bool first_time)
{
  char spt[200];
  std::string folderName = std::filesystem::current_path().string() + "/output";

  snprintf(spt, sizeof(spt), "%s/times.txt", folderName.c_str());

  FILE* fpt;
  if (first_time)
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

    fprintf(fpt, "RB_timestamp ");
    fprintf(fpt, "RB_n_callbacks ");
    fprintf(fpt, "RB_timestamps_diff ");
    fprintf(fpt, "RB_callback_frequency ");
    fprintf(fpt, "Time_RB_callback ");

    fprintf(fpt, "Hr_timestamp ");
    fprintf(fpt, "Hr_n_callbacks ");
    fprintf(fpt, "Hr_timestamps_diff ");
    fprintf(fpt, "Hr_callback_frequency ");
    fprintf(fpt, "Time_Hr_callback ");

    fprintf(fpt, "Hl_timestamp ");
    fprintf(fpt, "Hl_n_callbacks ");
    fprintf(fpt, "Hl_timestamps_diff ");
    fprintf(fpt, "Hl_callback_frequency ");
    fprintf(fpt, "Time_Hl_callback ");

    fprintf(fpt, "odom_timestamp ");
    fprintf(fpt, "odom_n_callbacks ");
    fprintf(fpt, "odom_timestamps_diff ");
    fprintf(fpt, "odom_callback_frequency ");
    fprintf(fpt, "Time_odom_callback ");

    // Code
    fprintf(fpt, "Time_total_iter ");

    fprintf(fpt, "Time_safety_data_copy ");

    fprintf(fpt, "Time_PCs ");
    fprintf(fpt, "Time_PCs_host2dev ");
    fprintf(fpt, "Time_PCs_translation ");
    fprintf(fpt, "Time_PCs_rotation ");
    fprintf(fpt, "Time_PCs_ego_motion ");
    fprintf(fpt, "Time_PCs_CB ");
    fprintf(fpt, "Time_PCs_FM ");

    fprintf(fpt, "Time_obs_OG ");
    fprintf(fpt, "Time_obs_OG_mallocs ");
    fprintf(fpt, "Time_obs_OG_frees ");
    fprintf(fpt, "Time_obs_OG_RB ");
    fprintf(fpt, "Time_obs_OG_Hr ");
    fprintf(fpt, "Time_obs_OG_Hl ");
    fprintf(fpt, "Time_obs_OG_fusion ");
    fprintf(fpt, "Time_obs_OG_formatting ");

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

    fprintf(fpt, "%f ", TIME_measurements->RubyPlus_timestamp);
    fprintf(fpt, "%d ", TIME_measurements->n_RubyPlus_callbacks);
    fprintf(fpt, "%f ", TIME_measurements->RubyPlus_timestamps_diff * 1000);
    fprintf(fpt, "%f ", TIME_measurements->time_RubyPlus_callback_frequency.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_RubyPlus_callback_duration.measured_time);

    fprintf(fpt, "%f ", TIME_measurements->HeliosRight_timestamp);
    fprintf(fpt, "%d ", TIME_measurements->n_HeliosRight_callbacks);
    fprintf(fpt, "%f ", TIME_measurements->HeliosRight_timestamps_diff * 1000);
    fprintf(fpt, "%f ", TIME_measurements->time_HeliosRight_callback_frequency.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_HeliosRight_callback_duration.measured_time);

    fprintf(fpt, "%f ", TIME_measurements->HeliosLeft_timestamp);
    fprintf(fpt, "%d ", TIME_measurements->n_HeliosLeft_callbacks);
    fprintf(fpt, "%f ", TIME_measurements->HeliosLeft_timestamps_diff * 1000);
    fprintf(fpt, "%f ", TIME_measurements->time_HeliosLeft_callback_frequency.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_HeliosLeft_callback_duration.measured_time);

    fprintf(fpt, "%f ", TIME_measurements->odom_timestamp);
    fprintf(fpt, "%d ", TIME_measurements->n_odom_callbacks);
    fprintf(fpt, "%f ", TIME_measurements->odom_timestamps_diff * 1000);
    fprintf(fpt, "%f ", TIME_measurements->time_odom_callback_frequency.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_odom_callback_duration.measured_time);

    // Code
    fprintf(fpt, "%f ", TIME_measurements->time_total_iteration.measured_time);

    fprintf(fpt, "%f ", TIME_measurements->time_safety_data_copy.measured_time);

    fprintf(fpt, "%f ", TIME_measurements->time_PCs_processing_total.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_PCs_processing_host2device.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_PCs_processing_translation.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_PCs_processing_rotation.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_PCs_correct_ego_motion.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_PCs_processing_CB.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_PCs_processing_FM.measured_time);

    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_total.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_mallocs.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_frees.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_RB.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_Hr.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_Hl.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_fusion.measured_time);
    fprintf(fpt, "%f ", TIME_measurements->time_obsOG_final_format.measured_time);

    ChronoTimer time_obsOG_total;

    fprintf(fpt, "\n");
    fclose(fpt);
  }

  printf("file %s written\n", spt);
}
