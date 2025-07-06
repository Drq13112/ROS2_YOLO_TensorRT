#pragma once

#include <iostream>
#include <unistd.h>
#include <filesystem>

#include "perception_obstacles/perception_utilities/ChronoTimer.hpp"

struct DATA_times
{
  // Callback
  double RubyPlus_timestamp = 0;
  double RubyPlus_timestamps_diff = 0;
  int n_RubyPlus_callbacks = 0;
  ChronoTimer time_RubyPlus_callback_frequency;
  ChronoTimer time_RubyPlus_callback_duration;

  double HeliosRight_timestamp = 0;
  double HeliosRight_timestamps_diff = 0;
  int n_HeliosRight_callbacks = 0;
  ChronoTimer time_HeliosRight_callback_frequency;
  ChronoTimer time_HeliosRight_callback_duration;

  double HeliosLeft_timestamp = 0;
  double HeliosLeft_timestamps_diff = 0;
  int n_HeliosLeft_callbacks = 0;
  ChronoTimer time_HeliosLeft_callback_frequency;
  ChronoTimer time_HeliosLeft_callback_duration;

  double odom_timestamp = 0;
  double odom_timestamps_diff = 0;
  int n_odom_callbacks = 0;
  ChronoTimer time_odom_callback_frequency;
  ChronoTimer time_odom_callback_duration;

  // Code
  ChronoTimer time_total_iteration;

  ChronoTimer time_safety_data_copy;

  ChronoTimer time_PCs_processing_total;
  ChronoTimer time_PCs_processing_host2device;
  ChronoTimer time_PCs_processing_translation;
  ChronoTimer time_PCs_processing_rotation;
  ChronoTimer time_PCs_correct_ego_motion;
  ChronoTimer time_PCs_processing_CB;
  ChronoTimer time_PCs_processing_FM;

  ChronoTimer time_obsOG_total;
  ChronoTimer time_obsOG_mallocs;
  ChronoTimer time_obsOG_RB;
  ChronoTimer time_obsOG_Hr;
  ChronoTimer time_obsOG_Hl;
  ChronoTimer time_obsOG_fusion;
  ChronoTimer time_obsOG_final_format;
  ChronoTimer time_obsOG_frees;

  ChronoTimer time_road_map_total;

  ChronoTimer time_DOG_total;
  ChronoTimer time_DOG_random_numbers;
  ChronoTimer time_DOG_prediction;
  ChronoTimer time_DOG_update;
  ChronoTimer time_DOG_velocity;
  ChronoTimer time_DOG_resampling_roughtening;
  ChronoTimer time_DOG_equalize_weights;
};

void write_files_time(DATA_times* TIME_measurements, const bool first_time);
