#include "perception_obstacles/perception_obstacles_node.hpp"

// ----------------------------------------------------------------------------------- //
// ------------------------------ CHECK PUBLISHERS Qos ------------------------------- //
// ----------------------------------------------------------------------------------- //
int PerceptionObstacles::check_QoS_publisher_s(const std::string topic_name)
{
  auto infos = this->get_publishers_info_by_topic(topic_name);

  printf("Found %zu publishers on topic '%s' \n", infos.size(), topic_name.c_str());

  for (int i = 0; i < (int)infos.size(); ++i)
  {
    printf("Publisher %d\n", i + 1);
    rclcpp::QoS qos = infos.at(i).qos_profile();

    printf("   Reliability: ");
    (qos.get_rmw_qos_profile().reliability == RMW_QOS_POLICY_RELIABILITY_RELIABLE) ? printf("Reliable\n") :
                                                                                     printf("Best effort\n");

    printf("   History: ");

    if (qos.get_rmw_qos_profile().history == RMW_QOS_POLICY_HISTORY_KEEP_LAST)
    {
      printf("Keep last (%zu)\n", qos.get_rmw_qos_profile().depth);
    }
    else if (qos.get_rmw_qos_profile().history == RMW_QOS_POLICY_HISTORY_KEEP_ALL)
    {
      printf("Keep all\n");
    }
    else
    {
      printf("  erroooorrr  value = %d   -> 2023 - 2025 seems to be a bug (https://github.com/ros2/ros2/issues/1451)\n",
             (int)qos.get_rmw_qos_profile().history);
    }

    printf("   Durability: ");
    (qos.get_rmw_qos_profile().durability == RMW_QOS_POLICY_DURABILITY_VOLATILE) ? printf("Volatile\n") :
                                                                                   printf("Transient local\n");
  }

  return (int)infos.size();
}

// ------------------------------------------------------------------------------- //
// ------------------------------ RubyPlus CALLBACK ------------------------------ //
// ------------------------------------------------------------------------------- //
void PerceptionObstacles::RubyPlusCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_pointcloud)
{
  if ((int)msg_pointcloud->height != RB_data_.rows)
  {
    RCLCPP_INFO(this->get_logger(), "Arrive Incomplete. PC height = %d, expected rows = %d",
                (int)msg_pointcloud->height, RB_data_.rows);
    return;
  }

  // -------------------- Synchronization -------------------- //

  {
    TIME_measurements_.time_RubyPlus_callback_duration.Reset();

    // Lock the mutex to safely write shared_data
    std::lock_guard<std::mutex> lock(mutex_);

    // Perform transformation to desired PC format
    AUTOPIA_RubyPlus::transform_PointCloud2_to_RubyPlusPointCloud(msg_pointcloud, &RB_pc_modificable_callback_,
                                                                  &RB_data_);

    // -------------------- Debug data -------------------- //

    // Increase callbacks counter
    TIME_measurements_.n_RubyPlus_callbacks++;

    // Get time elapsed since last callback
    if (TIME_measurements_.n_RubyPlus_callbacks > 1)
    {
      TIME_measurements_.time_RubyPlus_callback_frequency.GetElapsedTime();
      TIME_measurements_.time_RubyPlus_callback_frequency.ComputeStats();
    }
    TIME_measurements_.time_RubyPlus_callback_frequency.Reset();

    // Compute delta based on time stamp
    static u_int64_t sec, nsec, prev_sec, prev_nsec;
    sec = msg_pointcloud->header.stamp.sec;
    nsec = msg_pointcloud->header.stamp.nanosec;

    TIME_measurements_.RubyPlus_timestamp = (double)sec + ((double)nsec) * 1e-9;
    if (TIME_measurements_.n_RubyPlus_callbacks > 1)
    {
      TIME_measurements_.RubyPlus_timestamps_diff =
          (double)sec - (double)prev_sec + ((double)nsec - (double)prev_nsec) * 1e-9;
    }
    prev_sec = sec;
    prev_nsec = nsec;

    if (TIME_measurements_.RubyPlus_timestamps_diff > 0.12)
    {
      printf(
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---> RubyPlus_timestamps_diff = %fms = %fhz ---> Paquete potencialmente "
          "perdido\n\n",
          TIME_measurements_.RubyPlus_timestamps_diff, 1.0 / TIME_measurements_.RubyPlus_timestamps_diff);
    }
    TIME_measurements_.time_RubyPlus_callback_duration.GetElapsedTime();
    TIME_measurements_.time_RubyPlus_callback_duration.ComputeStats();

    if (false)
    {
      printf("CALLBACK %d - Point cloud recibido: Layers = %d; Rows = %d;\n", TIME_measurements_.n_RubyPlus_callbacks,
             msg_pointcloud->width, msg_pointcloud->height);

      printf("           Callback duration = %fms -> mean = %fms, max = %fms\n",
             TIME_measurements_.time_RubyPlus_callback_duration.measured_time,
             TIME_measurements_.time_RubyPlus_callback_duration.mean_time,
             TIME_measurements_.time_RubyPlus_callback_duration.max_time);

      if (TIME_measurements_.RubyPlus_timestamps_diff < 0.12)
      {
        printf(
            "           Time since last callback = %fms -> mean %fms, max = %fms;     RubyPlus_timestamps_diff = %f\n",
            TIME_measurements_.time_RubyPlus_callback_frequency.measured_time,
            TIME_measurements_.time_RubyPlus_callback_frequency.mean_time,
            TIME_measurements_.time_RubyPlus_callback_frequency.max_time, TIME_measurements_.RubyPlus_timestamps_diff);
      }
      else
      {
        printf(
            "           Time since last callback = %fms -> mean %fms, max = %fms;     RubyPlus_timestamps_diff = %f "
            "CUIDADO posible paquete perdido\n",
            TIME_measurements_.time_RubyPlus_callback_frequency.measured_time,
            TIME_measurements_.time_RubyPlus_callback_frequency.mean_time,
            TIME_measurements_.time_RubyPlus_callback_frequency.max_time, TIME_measurements_.RubyPlus_timestamps_diff);
      }
    }

    // Atomic flag to avoid data races when checking data_ready value outside the mutex
    flag_avoid_spurious_RubyPlus_callback_ = true;

    // Notify that data is accessible
    RubyPlus_condition_.notify_one();

  }  // Out of scope -> automatically unlocks
}

// ----------------------------------------------------------------------------------- //
// ------------------------------ HELIOS RIGHT CALLBACK ------------------------------ //
// ----------------------------------------------------------------------------------- //
void PerceptionObstacles::HeliosRightCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_pointcloud)
{
  if ((int)msg_pointcloud->height != Helios_data_.rows)
  {
    RCLCPP_INFO(this->get_logger(), "Arrive Incomplete. PC height = %d, expected rows = %d",
                (int)msg_pointcloud->height, Helios_data_.rows);
    return;
  }

  // -------------------- Synchronization -------------------- //
  {
    TIME_measurements_.time_HeliosRight_callback_duration.Reset();

    // Lock the mutex to safely write shared_data
    std::lock_guard<std::mutex> lock(mutex_);

    // Perform transformation to desired PC format
    AUTOPIA_Helios::transform_PointCloud2_to_HeliosPointCloud(msg_pointcloud, &Hr_pc_modificable_callback_,
                                                              &Helios_data_);

    // Notify that data is accessible
    // HeliosRight_condition_.notify_one();

    // -------------------- Debug data -------------------- //

    // Increase callbacks counter
    TIME_measurements_.n_HeliosRight_callbacks++;

    // Get time elapsed since last callback
    if (TIME_measurements_.n_HeliosRight_callbacks > 1)
    {
      TIME_measurements_.time_HeliosRight_callback_frequency.GetElapsedTime();
      TIME_measurements_.time_HeliosRight_callback_frequency.ComputeStats();
    }
    TIME_measurements_.time_HeliosRight_callback_frequency.Reset();

    // Compute delta based on time stamp
    static u_int64_t sec, nsec, prev_sec, prev_nsec;
    sec = msg_pointcloud->header.stamp.sec;
    nsec = msg_pointcloud->header.stamp.nanosec;

    TIME_measurements_.HeliosRight_timestamp = (double)sec + ((double)nsec) * 1e-9;
    if (TIME_measurements_.n_HeliosRight_callbacks > 1)
    {
      TIME_measurements_.HeliosRight_timestamps_diff =
          (double)sec - (double)prev_sec + ((double)nsec - (double)prev_nsec) * 1e-9;
    }
    prev_sec = sec;
    prev_nsec = nsec;

    if (TIME_measurements_.HeliosRight_timestamps_diff > 0.12)
    {
      printf(
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---> HeliosRight_timestamps_diff = %fms = %fhz ---> Paquete "
          "potencialmente perdido\n\n",
          TIME_measurements_.HeliosRight_timestamps_diff, 1.0 / TIME_measurements_.HeliosRight_timestamps_diff);
    }
    TIME_measurements_.time_HeliosRight_callback_duration.GetElapsedTime();
    TIME_measurements_.time_HeliosRight_callback_duration.ComputeStats();

    if (false)
    {
      printf("CALLBACK %d - Point cloud recibido: Layers = %d; Rows = %d;\n",
             TIME_measurements_.n_HeliosRight_callbacks, msg_pointcloud->width, msg_pointcloud->height);

      printf("           Callback duration = %fms -> mean = %fms, max = %fms\n",
             TIME_measurements_.time_HeliosRight_callback_duration.measured_time,
             TIME_measurements_.time_HeliosRight_callback_duration.mean_time,
             TIME_measurements_.time_HeliosRight_callback_duration.max_time);

      if (TIME_measurements_.HeliosRight_timestamps_diff < 0.12)
      {
        printf(
            "           Time since last callback = %fms -> mean %fms, max = %fms;     HeliosRight_timestamps_diff = "
            "%f\n",
            TIME_measurements_.time_HeliosRight_callback_frequency.measured_time,
            TIME_measurements_.time_HeliosRight_callback_frequency.mean_time,
            TIME_measurements_.time_HeliosRight_callback_frequency.max_time,
            TIME_measurements_.HeliosRight_timestamps_diff);
      }
      else
      {
        printf(
            "           Time since last callback = %fms -> mean %fms, max = %fms;     HeliosRight_timestamps_diff = %f "
            "CUIDADO posible paquete perdido\n",
            TIME_measurements_.time_HeliosRight_callback_frequency.measured_time,
            TIME_measurements_.time_HeliosRight_callback_frequency.mean_time,
            TIME_measurements_.time_HeliosRight_callback_frequency.max_time,
            TIME_measurements_.HeliosRight_timestamps_diff);
      }
    }
    // Atomic flag to avoid data races when checking data_ready value outside the mutex
    flag_HeliosRight_callback_available_ = true;

  }  // Out of scope -> automatically unlocks
}

void PerceptionObstacles::HeliosLeftCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_pointcloud)
{
  if ((int)msg_pointcloud->height != Helios_data_.rows)
  {
    RCLCPP_INFO(this->get_logger(), "Arrive Incomplete. PC height = %d, expected rows = %d",
                (int)msg_pointcloud->height, Helios_data_.rows);
    return;
  }

  // -------------------- Synchronization -------------------- //
  {
    TIME_measurements_.time_HeliosLeft_callback_duration.Reset();

    // Lock the mutex to safely write shared_data
    std::lock_guard<std::mutex> lock(mutex_);

    // Perform transformation to desired PC format
    AUTOPIA_Helios::transform_PointCloud2_to_HeliosPointCloud(msg_pointcloud, &Hl_pc_modificable_callback_,
                                                              &Helios_data_);

    // Increase callbacks counter
    TIME_measurements_.n_HeliosLeft_callbacks++;

    // Notify that data is accessible
    // HeliosLeft_condition_.notify_one();

    // -------------------- Debug data -------------------- //

    // Get time elapsed since last callback
    if (TIME_measurements_.n_HeliosLeft_callbacks > 1)
    {
      TIME_measurements_.time_HeliosLeft_callback_frequency.GetElapsedTime();
      TIME_measurements_.time_HeliosLeft_callback_frequency.ComputeStats();
    }
    TIME_measurements_.time_HeliosLeft_callback_frequency.Reset();

    // Compute delta based on time stamp
    static u_int64_t sec, nsec, prev_sec, prev_nsec;
    sec = msg_pointcloud->header.stamp.sec;
    nsec = msg_pointcloud->header.stamp.nanosec;

    TIME_measurements_.HeliosLeft_timestamp = (double)sec + ((double)nsec) * 1e-9;
    if (TIME_measurements_.n_HeliosLeft_callbacks > 1)
    {
      TIME_measurements_.HeliosLeft_timestamps_diff =
          (double)sec - (double)prev_sec + ((double)nsec - (double)prev_nsec) * 1e-9;
    }
    prev_sec = sec;
    prev_nsec = nsec;

    if (TIME_measurements_.HeliosLeft_timestamps_diff > 0.12)
    {
      printf(
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---> HeliosLeft_timestamps_diff = %fms = %fhz ---> Paquete "
          "potencialmente perdido\n\n",
          TIME_measurements_.HeliosLeft_timestamps_diff, 1.0 / TIME_measurements_.HeliosLeft_timestamps_diff);
    }
    TIME_measurements_.time_HeliosLeft_callback_duration.GetElapsedTime();
    TIME_measurements_.time_HeliosLeft_callback_duration.ComputeStats();

    if (false)
    {
      printf("CALLBACK %d - Point cloud recibido: Layers = %d; Rows = %d;\n", TIME_measurements_.n_HeliosLeft_callbacks,
             msg_pointcloud->width, msg_pointcloud->height);

      printf("           Callback duration = %fms -> mean = %fms, max = %fms\n",
             TIME_measurements_.time_HeliosLeft_callback_duration.measured_time,
             TIME_measurements_.time_HeliosLeft_callback_duration.mean_time,
             TIME_measurements_.time_HeliosLeft_callback_duration.max_time);

      if (TIME_measurements_.HeliosLeft_timestamps_diff < 0.12)
      {
        printf(
            "           Time since last callback = %fms -> mean %fms, max = %fms;     HeliosLeft_timestamps_diff = "
            "%f\n",
            TIME_measurements_.time_HeliosLeft_callback_frequency.measured_time,
            TIME_measurements_.time_HeliosLeft_callback_frequency.mean_time,
            TIME_measurements_.time_HeliosLeft_callback_frequency.max_time,
            TIME_measurements_.HeliosLeft_timestamps_diff);
      }
      else
      {
        printf(
            "           Time since last callback = %fms -> mean %fms, max = %fms;     HeliosLeft_timestamps_diff = %f "
            "CUIDADO posible paquete perdido\n",
            TIME_measurements_.time_HeliosLeft_callback_frequency.measured_time,
            TIME_measurements_.time_HeliosLeft_callback_frequency.mean_time,
            TIME_measurements_.time_HeliosLeft_callback_frequency.max_time,
            TIME_measurements_.HeliosLeft_timestamps_diff);
      }
    }

    // Atomic flag to avoid data races when checking data_ready value outside the mutex
    flag_HeliosLeft_callback_available_ = true;

  }  // Out of scope -> automatically unlocks
}

// ----------------------------------------------------------------------------------- //
// ------------------------------ LOCALIZATION CALLBACK ------------------------------ //
// ----------------------------------------------------------------------------------- //
void PerceptionObstacles::LocalizationCallback(const nav_msgs::msg::Odometry::SharedPtr msg_odom)
{
  // -------------------- Synchronization -------------------- //
  {
    // Lock the mutex to safely write shared_data
    std::lock_guard<std::mutex> lock(mutex_);

    TIME_measurements_.time_odom_callback_duration.Reset();

    // Position
    info_coche_modificable_callback_.px_G = msg_odom->pose.pose.position.x;
    info_coche_modificable_callback_.py_G = msg_odom->pose.pose.position.y;
    info_coche_modificable_callback_.yaw_G = msg_odom->pose.pose.orientation.z * M_PI / 180.0;

    // Velocity
    info_coche_modificable_callback_.vel = msg_odom->twist.twist.linear.x / 3.6;
    info_coche_modificable_callback_.yaw_rate = msg_odom->twist.twist.angular.z * M_PI / 180.0;

    info_coche_modificable_callback_.tiempo =
        (double)msg_odom->header.stamp.sec + ((double)msg_odom->header.stamp.nanosec) * 1e-9;

    info_coche_modificable_callback_.sec = msg_odom->header.stamp.sec;
    info_coche_modificable_callback_.nanosec = msg_odom->header.stamp.nanosec;

    // printf(
    //     "callback localization -------> [%f, %f]; yaw = %fº; vel = [%fkm/h, %frad/s]; tiempo = %f (sec = %d; nsec = "
    //     "%d)\n",
    //     info_coche_modificable_callback_.px_G, info_coche_modificable_callback_.py_G,
    //     info_coche_modificable_callback_.yaw_G * 180.0 / M_PI, info_coche_modificable_callback_.vel * 3.6,
    //     info_coche_modificable_callback_.yaw_rate * 180.0 / M_PI, info_coche_modificable_callback_.tiempo,
    //     msg_odom->header.stamp.sec, msg_odom->header.stamp.nanosec);

    flag_avoid_spurious_localization_callback_ = true;

    // Notify that data is accessible
    init_localization_condition_.notify_one();

    // -------------------- Debug data -------------------- //

    // Increase callbacks counter
    TIME_measurements_.n_odom_callbacks++;

    // Get time elapsed since last callback
    if (TIME_measurements_.n_odom_callbacks > 1)
    {
      TIME_measurements_.time_odom_callback_frequency.GetElapsedTime();
      TIME_measurements_.time_odom_callback_frequency.ComputeStats();
    }
    TIME_measurements_.time_odom_callback_frequency.Reset();

    // Compute delta based on time stamp
    static u_int64_t sec, nsec, prev_sec, prev_nsec;
    sec = msg_odom->header.stamp.sec;
    nsec = msg_odom->header.stamp.nanosec;

    TIME_measurements_.odom_timestamp = (double)sec + ((double)nsec) * 1e-9;
    if (TIME_measurements_.n_odom_callbacks > 1)
    {
      TIME_measurements_.odom_timestamps_diff =
          (double)sec - (double)prev_sec + ((double)nsec - (double)prev_nsec) * 1e-9;
    }
    prev_sec = sec;
    prev_nsec = nsec;

    if (TIME_measurements_.odom_timestamps_diff > 0.03)
    {
      printf(
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---> odom_timestamps_diff = %fms = %fhz ---> Paquete potencialmente "
          "perdido\n\n",
          TIME_measurements_.odom_timestamps_diff, 1.0 / TIME_measurements_.odom_timestamps_diff);
    }

    TIME_measurements_.time_odom_callback_duration.GetElapsedTime();
    TIME_measurements_.time_odom_callback_duration.ComputeStats();

    if (false)
    {
      printf("CALLBACK %d - Ego Localization recibido\n", TIME_measurements_.n_odom_callbacks);

      printf("           Callback duration = %fms -> mean = %fms, max = %fms\n",
             TIME_measurements_.time_odom_callback_duration.measured_time,
             TIME_measurements_.time_odom_callback_duration.mean_time,
             TIME_measurements_.time_odom_callback_duration.max_time);

      if (TIME_measurements_.odom_timestamps_diff < 0.03)
      {
        printf("           Time since last callback = %fms -> mean %fms, max = %fms;     odom_timestamps_diff = %f\n",
               TIME_measurements_.time_odom_callback_frequency.measured_time,
               TIME_measurements_.time_odom_callback_frequency.mean_time,
               TIME_measurements_.time_odom_callback_frequency.max_time, TIME_measurements_.odom_timestamps_diff);
      }
      else
      {
        printf(
            "           Time since last callback = %fms -> mean %fms, max = %fms;     odom_timestamps_diff = %f "
            "CUIDADO posible paquete perdido\n",
            TIME_measurements_.time_odom_callback_frequency.measured_time,
            TIME_measurements_.time_odom_callback_frequency.mean_time,
            TIME_measurements_.time_odom_callback_frequency.max_time, TIME_measurements_.odom_timestamps_diff);
      }
    }
  }  // Out of scope -> automatically unlocks
}