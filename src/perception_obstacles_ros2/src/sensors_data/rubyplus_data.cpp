#include "perception_obstacles/sensors_data/rubyplus_data.hpp"

/** ------------------------------------------------------------------------------------------------
 * @brief function that initializes constant data of RubyPlus: (i)~vec_channel, (ii)~vec_subChannel,
 * (iii)~vec_metaChannel, (iv)~vec_order_metaChannel
 *
 * @param RubyPlus_basic_data: struct with the basic and constant data of the RubyPlus
 */
void AUTOPIA_RubyPlus::initialize_metachannels_data(AUTOPIA_RubyPlus::MetaChannelData* rb_data)
{
  int n_rows = rb_data->rows;
  int n_cols = rb_data->cols;
  int n_groups = rb_data->groups;

  int total_channels = n_rows * n_groups;
  int total_points = n_rows * n_cols;

  // 1) Guardamos, para cada canal, la lista de índices de puntos que caen en él.
  std::vector<std::vector<int>> channel_indices(total_channels);

  // Recorremos todos los puntos para asignarlos a su canal "global_idx"
  for (int i = 0; i < total_points; ++i)
  {
    int main_group_idx = i / n_cols;              // "fila"
    int sub_group_idx = (i % n_cols) % n_groups;  // "columna" mod "groups"
    int global_idx = main_group_idx * n_groups + sub_group_idx;

    channel_indices[global_idx].push_back(i);
  }

  // 2) Vamos a generar un orden global de índices (index_order),
  //    Y AL MISMO TIEMPO mapearemos cada punto -> (metacanal, posiciónEnElMetacanal)
  rb_data->index_order.reserve(total_points);
  std::fill(rb_data->index_order.begin(), rb_data->index_order.end(), -1);

  std::vector<bool> processed_channels(total_channels, false);

  // point_metacanal = std::vector<int>(total_points, -1);  // Para cada punto i, a qué metacanal va
  // point_position = std::vector<int>(total_points, -1);   // Para cada punto i, en qué posición dentro del metacanal
  rb_data->point_metachannel.clear();  // Para cada punto i, a qué metacanal va
  rb_data->point_position.clear();     // Para cada punto i, en qué posición dentro del metacanal
  rb_data->point_metachannel.reserve(total_points);
  rb_data->point_position.reserve(total_points);
  std::fill(rb_data->point_metachannel.begin(), rb_data->point_metachannel.end(), -1);
  std::fill(rb_data->point_position.begin(), rb_data->point_position.end(), -1);

  int reduced_idx = 0;  // Contará los metacanales a medida que emparejamos canales
  const int limit = rb_data->n_points_subchannel;
  int shift = rb_data->shift_metachannel;

  // 3) Recorremos los canales y hacemos la misma lógica de "original" vs "displaced"
  for (int original_channel = 0; original_channel < total_channels; ++original_channel)
  {
    if (processed_channels[original_channel])
    {
      continue;
    }

    // Calcular el canal desplazado
    int displaced_channel;
    if ((original_channel / 4) % 2 == 0)
    {
      displaced_channel = (original_channel + shift) % total_channels;
    }
    else
    {
      displaced_channel =
          (original_channel >= shift) ? (original_channel - shift) : (total_channels + original_channel - shift);
    }

    if (processed_channels[displaced_channel])
    {
      continue;
    }

    // Marcar ambos como procesados
    processed_channels[original_channel] = true;
    processed_channels[displaced_channel] = true;

    // Queremos ir añadiendo los puntos de cada canal a "index_order" y,
    // al mismo tiempo, registrar en "point_metacanal" y "point_position"
    // qué metacanal es y en qué posición va cada punto.

    // offset llevará la cuenta de cuántos puntos van añadidos dentro de ESTE metacanal.
    int offset = 0;
    if ((original_channel / 4) % 2 == 0)
    {
      // "displaced_channel" es el primero (pero vamos a intercalar)
      for (int i = 0; i < limit; ++i)
      {
        // A) si hay punto en displaced_channel
        if (i < (int)channel_indices[displaced_channel].size())
        {
          int pt_idx = channel_indices[displaced_channel][i];
          rb_data->index_order.push_back(pt_idx);
          rb_data->point_metachannel[pt_idx] = reduced_idx;
          rb_data->point_position[pt_idx] = offset++;
        }

        // B) si hay punto en original_channel
        if (i < (int)channel_indices[original_channel].size())
        {
          int pt_idx = channel_indices[original_channel][i];
          rb_data->index_order.push_back(pt_idx);
          rb_data->point_metachannel[pt_idx] = reduced_idx;
          rb_data->point_position[pt_idx] = offset++;
        }
      }
    }
    else
    {
      // "original_channel" es el primero al intercalar
      for (int i = 0; i < limit; ++i)
      {
        // A) si hay punto en original_channel
        if (i < (int)channel_indices[original_channel].size())
        {
          int pt_idx = channel_indices[original_channel][i];
          rb_data->index_order.push_back(pt_idx);
          rb_data->point_metachannel[pt_idx] = reduced_idx;
          rb_data->point_position[pt_idx] = offset++;
        }

        // B) si hay punto en displaced_channel
        if (i < (int)channel_indices[displaced_channel].size())
        {
          int pt_idx = channel_indices[displaced_channel][i];
          rb_data->index_order.push_back(pt_idx);
          rb_data->point_metachannel[pt_idx] = reduced_idx;
          rb_data->point_position[pt_idx] = offset++;
        }
      }
    }

    ++reduced_idx;
  }

  // Compute 1D index
  rb_data->idx1D.clear();  // Para cada punto i, a qué metacanal va
  rb_data->idx1D.reserve(total_points);
  std::fill(rb_data->idx1D.begin(), rb_data->idx1D.end(), -1);
  for (int i = 0; i < total_points; ++i)
  {
    compute_1D_layers_angles(&rb_data->idx1D[i], rb_data->point_position[i], rb_data->point_metachannel[i],
                             AUTOPIA_RubyPlus::N_LAYERS);
  }

  // Al terminar, 'reduced_idx' es el número total de metachanneles (emparejamientos) que se han asignado.

  printf("Initialization RubyPlus metaChannel done -- (%d metaChannels).\n", reduced_idx);

  // 4) Ahora tenemos:
  //    - index_order: la lista global de puntos en el orden que definimos
  //    - point_metachannel[i]: metachannel al que pertenece el punto i
  //    - point_position[i]: posición (0,1,2,...) del punto i dentro de su metachannel
  //    - point_subcanal[i]: subcanal al que pertenece el punto i
  //    - point_canal[i]: canal al que pertenece el punto i
}

/** ------------------------------------------------------------------------------------------------
 * @brief function that initializes constant data of RubyPlus: (i)~vec_channel, (ii)~vec_subChannel,
 * (iii)~vec_metaChannel, (iv)~vec_order_metaChannel
 *
 * @param RubyPlus_basic_data: struct with the basic and constant data of the RubyPlus
 */
void AUTOPIA_RubyPlus::initialize_pointcloud(AUTOPIA_RubyPlus::PointCloud* pc,
                                             AUTOPIA_RubyPlus::MetaChannelData* RB_data, const int LiDAR_ID,
                                             const float px, const float py, const float pz, const int n_points,
                                             const int n_layers, const geometry_msgs::msg::Transform& transform)
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
    idx1D = RB_data->idx1D[i];
    pc->metaChannel[idx1D] = RB_data->point_metachannel[i];
  }

  printf("RubyPlus initialized: ID = %d; pos = [%f, %f, %f]m; n_points = %d\n", pc->LiDAR_ID, pc->LiDAR_px,
         pc->LiDAR_py, pc->LiDAR_pz, pc->n_points);
}

/** ------------------------------------------------------------------------------------------------
 * @brief Función que devuelve el mensaje de ROS2 en el formato necesario para la clasificación.
 *
 * @param pointcloud Mensaje de ROS2
 * @param desired point cloud structure
 * @param vector corresponding metachannel and position
 */
void AUTOPIA_RubyPlus::transform_PointCloud2_to_RubyPlusPointCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr& msg_pointcloud, AUTOPIA_RubyPlus::PointCloud* RB_pc,
    AUTOPIA_RubyPlus::MetaChannelData* RB_data)
{
  // 1) Recorrer los puntos

  // Iterators for PointCloud msg
  sensor_msgs::PointCloud2Iterator<float> iterX(*msg_pointcloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iterY(*msg_pointcloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iterZ(*msg_pointcloud, "z");
  sensor_msgs::PointCloud2Iterator<float> iterIntensity(*msg_pointcloud, "intensity");
  sensor_msgs::PointCloud2Iterator<uint16_t> iterRing(*msg_pointcloud, "ring");
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
    idx1D = RB_data->idx1D[i];

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
