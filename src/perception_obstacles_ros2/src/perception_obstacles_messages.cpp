#include "perception_obstacles/perception_obstacles_node.hpp"

/** ----------------------------------------------------------------------------------------------------------------<
 * @brief Publish transform between odom and world (should be done by localization node...)
 *
 */
void PerceptionObstacles::PublishTransformOdomWorld(const double px_G, const double py_G, const double yaw_G,
                                                    const uint32_t sec, const uint32_t nsec)
{
  geometry_msgs::msg::TransformStamped transform;

  transform.header.stamp.sec = sec;
  transform.header.stamp.nanosec = nsec;

  transform.header.frame_id = "world";  // Parent frame
  transform.child_frame_id = "odom";    // own frame

  transform.transform.translation.x = px_G;
  transform.transform.translation.y = py_G;
  transform.transform.translation.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(0, 0, yaw_G);  // No rotation
  transform.transform.rotation.x = q.x();
  transform.transform.rotation.y = q.y();
  transform.transform.rotation.z = q.z();
  transform.transform.rotation.w = q.w();

  tf_odom_world_broadcaster_->sendTransform(transform);
}

/** ----------------------------------------------------------------------------------------------------------------
 * @brief Función que devuelve el mensaje de ROS2 en el formato necesario para la clasificación.
 *
 */
sensor_msgs::msg::PointCloud2 PerceptionObstacles::GeneratePointCloud2Message(
    const float x[], const float y[], const float z[], const float intensity[], const float vert_ang[],
    const int label[], const int channel_label[], const int metaChannel[], const int n_points,
    const std::string frame_id, const uint32_t sec, const uint32_t nanosec)
{
  // Create msg
  sensor_msgs::msg::PointCloud2 cloud_msg;
  cloud_msg.header.frame_id = frame_id;
  cloud_msg.header.stamp.sec = sec;
  cloud_msg.header.stamp.nanosec = nanosec;

  // Define dimensions
  cloud_msg.height = 1;  // Unodered
  cloud_msg.width = n_points;

  // Define fields
  sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
  modifier.setPointCloud2Fields(5, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
                                sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                "label", 1, sensor_msgs::msg::PointField::INT32, "channel_label", 1,
                                sensor_msgs::msg::PointField::INT32);
  //, "metaChannel", 1, sensor_msgs::msg::PointField::INT32, "intensity", 1,
  // sensor_msgs::msg::PointField::INT32, "vert_ang", 1, sensor_msgs::msg::PointField::FLOAT32);
  modifier.resize(cloud_msg.width);

  // Create iterators to explore the pc
  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<int> iter_label(cloud_msg, "label");
  sensor_msgs::PointCloud2Iterator<int> iter_channel_label(cloud_msg, "channel_label");
  // sensor_msgs::PointCloud2Iterator<float> iter_vert_ang(cloud_msg, "vert_ang");
  // sensor_msgs::PointCloud2Iterator<int> iter_metaChannel(cloud_msg, "metaChannel");
  // sensor_msgs::PointCloud2Iterator<int> iter_intensity(cloud_msg, "intensity");

  // Fill PC
  for (int i = 0; i < n_points; ++i)
  {
    *iter_x = x[i];
    *iter_y = y[i];
    *iter_z = z[i];
    *iter_label = label[i];
    *iter_channel_label = channel_label[i];
    // *iter_vert_ang = vert_ang[i];
    // *iter_metaChannel = metaChannel[i];
    // *iter_intensity = intensity[i];

    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_label;
    ++iter_channel_label;
    // ++iter_vert_ang;
    // ++iter_metaChannel;
    // ++iter_intensity;
  }

  return cloud_msg;
}

/** ----------------------------------------------------------------------------------------------------------------
 * @brief Función que devuelve el mensaje de ROS2 en el formato necesario para una OG
 */
nav_msgs::msg::OccupancyGrid PerceptionObstacles::GenerateOGMessage(
    const float array2D[GRID_TYPES::NC_Y][GRID_TYPES::NC_X], const int NC_X, const int NC_Y, const float MIN_X,
    const float MIN_Y, const float RES, const uint32_t sec, const uint32_t nsec)
{
  nav_msgs::msg::OccupancyGrid OG_msg;

  // Header
  // OG_msg.header.stamp = this->get_clock()->now();
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp.sec = sec;
  transform.header.stamp.nanosec = nsec;
  OG_msg.header.frame_id = "odom";

  // Info
  OG_msg.info.map_load_time.sec = sec;
  OG_msg.info.map_load_time.nanosec = nsec;

  OG_msg.info.resolution = RES;
  OG_msg.info.width = NC_X;
  OG_msg.info.height = NC_Y;
  OG_msg.info.origin.position.x = MIN_X;
  OG_msg.info.origin.position.y = MIN_Y;
  OG_msg.info.origin.position.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, 0.0);
  OG_msg.info.origin.orientation.x = q.x();
  OG_msg.info.origin.orientation.y = q.y();
  OG_msg.info.origin.orientation.z = q.z();
  OG_msg.info.origin.orientation.w = q.w();

  // Fill with dummy data (0 = free, 100 = occupied, -1 = unknown)
  OG_msg.data.resize(OG_msg.info.width * OG_msg.info.height, 0);

  // OccupancyGrid only works with int8 being probabilities in the range [0,100].  Unknown is -1.
  int idx_1D = -1;
  int aux_ih = -1;
  for (size_t i_h = 0; i_h < OG_msg.info.height; ++i_h)
  {
    for (size_t i_w = 0; i_w < OG_msg.info.width; ++i_w)
    {
      idx_1D = i_h * OG_msg.info.width + i_w;
      aux_ih = OG_msg.info.height - i_h - 1;

      OG_msg.data[idx_1D] = std::round(array2D[aux_ih][i_w] * 100);

#if DEBUG == TRUE
      if (isnan(array2D[aux_ih][i_w]))
      {
        printf("GenerateOGMessage error: %f\n", array2D[aux_ih][i_w]);
        exit(1);
      }
      if (OG_msg.data[idx_1D] > 100 || OG_msg.data[idx_1D] < 0)
      {
        printf("GenerateOGMessage error: [%zu, %zu] (aux_ih = %d)-> %d => value = %d (original data = %f) \n", i_w, i_h,
               aux_ih, idx_1D, OG_msg.data[idx_1D], array2D[aux_ih][i_w]);
        // exit(1);
      }
#endif
    }
  }

  return OG_msg;
}