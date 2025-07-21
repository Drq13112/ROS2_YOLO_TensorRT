#include "perception_obstacles/sensors_data/helios_data.hpp"

/** ------------------------------------------------------------------------------------------------
 * @brief function that initializes constant data of Helios: (i)~vec_channel,
 * (iv)~vec_metaChannel, (iii)~vec_order_metaChannel
 * In this case, no metachannel but subchannel is used
 *
 * @param Helios_basic_data: struct with the basic and constant data of the Helios
 */
void AUTOPIA_Helios::initialize_metachannels_data(AUTOPIA_Helios::MetaChannelData* helios_data)
{
  int n_rows = helios_data->rows;
  int n_cols = helios_data->cols;
  int n_groups = helios_data->groups;

  int total_channels = n_rows * n_groups;
  int total_points = n_rows * n_cols;

  // 1) Guardamos, para cada canal, la lista de índices de puntos que caen en él.
  std::vector<std::vector<int>> subgroup_indices(total_channels);
  helios_data->subchannel.resize(n_rows * n_cols);
  helios_data->subchannel_layer.resize(n_rows * n_cols);

  helios_data->idx1D.clear();  // Para cada punto i, a qué metacanal va
  helios_data->idx1D.reserve(total_points);
  std::fill(helios_data->idx1D.begin(), helios_data->idx1D.end(), -1);

  // Recorremos todos los puntos para asignarlos a subcanal
  for (int i = 0; i < total_points; ++i)
  {
    int original_channel = i / n_cols;
    int split_left_right = i % n_groups;  // channels are "split" in two, left and right

    // Índice del subcanal
    if (split_left_right == 0)
    {
      helios_data->subchannel[i] = original_channel * n_groups;  // Split-right -> assign to subchannel
    }
    else
    {
      helios_data->subchannel[i] = original_channel * n_groups + 1;  // Split-right -> assign to other subchannel
    }

    // To which layer corresponds (remember 32 is separated into 16x2)
    helios_data->subchannel_layer[i] = (n_cols - (i % n_cols) - 1) / n_groups;

    compute_1D_layers_angles(&helios_data->idx1D[i], helios_data->subchannel_layer[i], helios_data->subchannel[i],
                             AUTOPIA_Helios::N_LAYERS);

    // printf("%d -> original = %d -> [sub = %d, lay = %d]; idx1D = %d\n", i, original_channel,
    // helios_data->subchannel[i],
    //        helios_data->subchannel_layer[i], helios_data->idx1D[i]);
  }

  printf("Initialization Helios metaChannel done.\n");
}

/** ------------------------------------------------------------------------------------------------
 * @brief function that initializes constant data of Helios: (i)~vec_channel, (ii)~vec_subChannel,
 * (iii)~vec_metaChannel, (iv)~vec_order_metaChannel
 *
 * @param Helios_basic_data: struct with the basic and constant data of the Helios
 */
void AUTOPIA_Helios::initialize_pointcloud(AUTOPIA_Helios::PointCloud* pc, AUTOPIA_Helios::MetaChannelData* helios_data,
                                           const int LiDAR_ID, const float px, const float py, const float pz,
                                           const int n_points, const int n_layers,
                                           const geometry_msgs::msg::Transform& transform)
{
  pc->LiDAR_ID = LiDAR_ID;
  pc->LiDAR_px = px;
  pc->LiDAR_py = py;
  pc->LiDAR_pz = pz;
  pc->n_points = n_points;
  pc->n_layers = n_layers;

  std::memset(pc->rotation_matrix, 0, sizeof(pc->rotation_matrix));
  compute_rotation_matrix(transform, pc->rotation_matrix);

  int idx1D;
  for (int i = 0; i < pc->n_points; ++i)
  {
    idx1D = helios_data->idx1D[i];
    pc->metaChannel[idx1D] = helios_data->subchannel[i];
  }

  printf("Helios initialized: ID = %d; pos = [%f, %f, %f]m; n_points = %d\n", pc->LiDAR_ID, pc->LiDAR_px, pc->LiDAR_py,
         pc->LiDAR_pz, pc->n_points);
}

/** ------------------------------------------------------------------------------------------------
 * @brief Función que devuelve el mensaje de ROS2 en el formato necesario para la clasificación.
 *
 * @param pointcloud Mensaje de ROS2
 * @param desired point cloud structure
 * @param vector corresponding metachannel and position
 */
void AUTOPIA_Helios::transform_PointCloud2_to_HeliosPointCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr& msg_pointcloud, AUTOPIA_Helios::PointCloud* RB_pc,
    AUTOPIA_Helios::MetaChannelData* helios_data)
{
  // 1) Recorrer los puntos

  // Iterators for PointCloud msg
  sensor_msgs::PointCloud2Iterator<float> iterX(*msg_pointcloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iterY(*msg_pointcloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iterZ(*msg_pointcloud, "z");
  sensor_msgs::PointCloud2Iterator<float> iterIntensity(*msg_pointcloud, "intensity");
  // sensor_msgs::PointCloud2Iterator<uint16_t> iterRing(*msg_pointcloud, "ring");
  sensor_msgs::PointCloud2Iterator<double> iterTimestamp(*msg_pointcloud, "timestamp");
  // sensor_msgs::PointCloud2Iterator<double> itervert_angle(*msg_pointcloud, "vert_ang");
  // sensor_msgs::PointCloud2Iterator<double> iterHorAngle(*msg_pointcloud, "horAng");

  RB_pc->referenced_to_odom = false;
  RB_pc->end_sec = msg_pointcloud->header.stamp.sec;
  RB_pc->end_nsec = msg_pointcloud->header.stamp.nanosec;

  int idx1D = 0;
  int i = 0;
  while (iterX != iterX.end())
  {
    // Saber a qué metacanal va este punto, y en qué posición dentro del metacanal;
    idx1D = helios_data->idx1D[i];

    RB_pc->x[idx1D] = *iterX;
    RB_pc->y[idx1D] = *iterY;
    RB_pc->z[idx1D] = *iterZ;
    // RB_pc->dist3D[idx1D] = std::sqrt(RB_pc->x[idx1D] * RB_pc->x[idx1D] + RB_pc->y[idx1D] * RB_pc->y[idx1D] +
    //                                  RB_pc->z[idx1D] * RB_pc->z[idx1D]);
    RB_pc->intensity[idx1D] = *iterIntensity;
    // RB_pc->vert_ang[idx1D] =
    //     *itervert_angle;  // Queremos esta info porque cuando no hay return, aún así nos diga su azimuth
    // RB_pc->azimuth[idx1D] =
    //     *iterHorAngle;  // Queremos esta info porque cuando no hay return, aún así nos diga su inclinacion
    RB_pc->timestamp[idx1D] = *iterTimestamp;

    ++iterX;
    ++iterY;
    ++iterZ;
    ++iterIntensity;
    // ++itervert_angle;
    // ++iterHorAngle;
    ++iterTimestamp;
    ++i;
  }
}
