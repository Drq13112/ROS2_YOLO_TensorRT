#include "perception_obstacles/perception_obstacles_node.hpp"

/** ----------------------------------------------------------------------------------------------------------------<
 * @brief Función para obtener las transformaciones de los distintos tipos de nubes de puntos
 *
 */
void PerceptionObstacles::GetTransforms()
{
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  if (!this->has_parameter("orig_frame"))
  {
    frames_.RubyPlus = this->declare_parameter("rs_top_frame", "rubyplus");
    frames_.HeliosLeft = this->declare_parameter("rs_left_frame", "helios_left");
    frames_.HeliosRight = this->declare_parameter("rs_right_frame", "helios_right");
    frames_.odom = this->declare_parameter("orig_frame", "odom");
  }

  frames_.world = this->declare_parameter("world_frame", "world");

  RubyPlus_transform_ = GetSensorTransform(frames_.odom, frames_.RubyPlus);
  HeliosLeft_transform_ = GetSensorTransform(frames_.odom, frames_.HeliosLeft);
  HeliosRight_transform_ = GetSensorTransform(frames_.odom, frames_.HeliosRight);

  printf("Waiting for transforms...\n");
  int cont = 1;
  int max_cont = 50;
  while (!RubyPlus_transform_ || !HeliosLeft_transform_ || !HeliosRight_transform_)
  {
    try
    {
      RubyPlus_transform_ = GetSensorTransform(frames_.odom, frames_.RubyPlus);
      HeliosLeft_transform_ = GetSensorTransform(frames_.odom, frames_.HeliosLeft);
      HeliosRight_transform_ = GetSensorTransform(frames_.odom, frames_.HeliosRight);
    }
    catch (tf2::TransformException& ex)
    {
      // rclcpp::sleep_for(std::chrono::milliseconds(1000));
    }
    sleep(1);
    // rclcpp::sleep_for(std::chrono::milliseconds(1000));
    printf("Intentos de conseguir transformadas: %d (max = %d)\n", cont, max_cont);
    cont++;
    if (cont >= max_cont)
    {
      printf("Maximum waiting time for transforms reached\n");
      exit(1);
      break;
    }
  }

  /*
  // TODO: hay que cambiar esto
  printf(" TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO esto tiene que ir en el driver\n");
  double Hr_roll = 0.5 * M_PI / 180.0;   // in radians
  double Hr_pitch = 8 * M_PI / 180.0;    // in radians
  double Hr_yaw = -86.7 * M_PI / 180.0;  // in radians
  tf2::Quaternion Hr_q;
  Hr_q.setRPY(Hr_roll, Hr_pitch, Hr_yaw);
  Hr_q = Hr_q.normalize();

  HeliosRight_transform_->transform.translation.x = 0.85;
  HeliosRight_transform_->transform.translation.y = -0.55;
  HeliosRight_transform_->transform.translation.z = 1.68;
  HeliosRight_transform_->transform.rotation.x = Hr_q.x();
  HeliosRight_transform_->transform.rotation.y = Hr_q.y();
  HeliosRight_transform_->transform.rotation.z = Hr_q.z();
  HeliosRight_transform_->transform.rotation.w = Hr_q.w();

  double Hl_roll = 0.35 * M_PI / 180.0;  // in radians
  double Hl_pitch = 8.5 * M_PI / 180.0;  // in radians
  double Hl_yaw = 94.2 * M_PI / 180.0;   // in radians
  tf2::Quaternion Hl_q;
  Hl_q.setRPY(Hl_roll, Hl_pitch, Hl_yaw);
  Hl_q = Hl_q.normalize();

  HeliosLeft_transform_->transform.translation.x = 0.81;
  HeliosLeft_transform_->transform.translation.y = 0.55;
  HeliosLeft_transform_->transform.translation.z = 1.70;
  HeliosLeft_transform_->transform.rotation.x = Hl_q.x();
  HeliosLeft_transform_->transform.rotation.y = Hl_q.y();
  HeliosLeft_transform_->transform.rotation.z = Hl_q.z();
  HeliosLeft_transform_->transform.rotation.w = Hl_q.w();
  printf(" TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO esto tiene que ir en el driver\n");
*/
  if (HeliosRight_transform_)
  {
    double roll, pitch, yaw;
    tf2::Quaternion q;
    tf2::fromMsg(RubyPlus_transform_->transform.rotation, q);
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    RCLCPP_INFO(rclcpp::get_logger("RubyPlus transform:"),
                "   Transform from '%s' to '%s':\n"
                "     Translation -> x: %.2f, y: %.2f, z: %.2f\n"
                "     Rotation    -> x: %.2f, y: %.2f, z: %.2f, w: %.2f [%f, %f, %f]\n",
                RubyPlus_transform_->header.frame_id.c_str(), RubyPlus_transform_->child_frame_id.c_str(),
                RubyPlus_transform_->transform.translation.x, RubyPlus_transform_->transform.translation.y,
                RubyPlus_transform_->transform.translation.z, RubyPlus_transform_->transform.rotation.x,
                RubyPlus_transform_->transform.rotation.y, RubyPlus_transform_->transform.rotation.z,
                RubyPlus_transform_->transform.rotation.w, roll * 180 / M_PI, pitch * 180 / M_PI, yaw * 180 / M_PI);
  }
  else
  {
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Ruby Plus transform could not be set");
    exit(1);
  }

  if (HeliosRight_transform_)
  {
    double roll, pitch, yaw;
    tf2::Quaternion q;
    tf2::fromMsg(HeliosRight_transform_->transform.rotation, q);
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    RCLCPP_INFO(rclcpp::get_logger("Helios Right transform:"),
                "   Transform from '%s' to '%s':\n"
                "     Translation -> x: %.2f, y: %.2f, z: %.2f\n"
                "     Rotation    -> x: %.2f, y: %.2f, z: %.2f, w: %.2f  [%f, %f, %f]\n",
                HeliosRight_transform_->header.frame_id.c_str(), HeliosRight_transform_->child_frame_id.c_str(),
                HeliosRight_transform_->transform.translation.x, HeliosRight_transform_->transform.translation.y,
                HeliosRight_transform_->transform.translation.z, HeliosRight_transform_->transform.rotation.x,
                HeliosRight_transform_->transform.rotation.y, HeliosRight_transform_->transform.rotation.z,
                HeliosRight_transform_->transform.rotation.w, roll * 180 / M_PI, pitch * 180 / M_PI, yaw * 180 / M_PI);
  }
  else
  {
    RCLCPP_INFO(rclcpp::get_logger("rclcpp\n"), "Helios Right transform could not be set");
  }

  if (HeliosLeft_transform_)
  {
    double roll, pitch, yaw;
    tf2::Quaternion q;
    tf2::fromMsg(HeliosLeft_transform_->transform.rotation, q);
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    RCLCPP_INFO(rclcpp::get_logger("Helios Left transform:"),
                "   Transform from '%s' to '%s':\n"
                "     Translation -> x: %.2f, y: %.2f, z: %.2f\n"
                "     Rotation    -> x: %.2f, y: %.2f, z: %.2f, w: %.2f [%f, %f, %f]\n",
                HeliosLeft_transform_->header.frame_id.c_str(), HeliosLeft_transform_->child_frame_id.c_str(),
                HeliosLeft_transform_->transform.translation.x, HeliosLeft_transform_->transform.translation.y,
                HeliosLeft_transform_->transform.translation.z, HeliosLeft_transform_->transform.rotation.x,
                HeliosLeft_transform_->transform.rotation.y, HeliosLeft_transform_->transform.rotation.z,
                HeliosLeft_transform_->transform.rotation.w, roll * 180 / M_PI, pitch * 180 / M_PI, yaw * 180 / M_PI);
  }
  else
  {
    RCLCPP_INFO(rclcpp::get_logger("rclcpp\n"), "Helios Left transform could not be set");
  }

  printf("... transforms arrived\n");
}

/**
 * @brief Función auxiliar para obtener la transformación de un tipo de nube de puntos
 *
 * @param toFrameRel Frame_id al que se transforma
 * @param fromFrameRel Frame_id original
 * @return std::unique_ptr<geometry_msgs::msg::TransformStamped> La transformación
 */
std::unique_ptr<geometry_msgs::msg::TransformStamped> PerceptionObstacles::GetSensorTransform(std::string toFrameRel,
                                                                                              std::string fromFrameRel)
{
  try
  {
    return std::make_unique<geometry_msgs::msg::TransformStamped>(
        tf_buffer_->lookupTransform(toFrameRel, fromFrameRel, tf2::TimePointZero));
  }
  catch (tf2::TransformException& ex)
  {
    // RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s", toFrameRel.c_str(), fromFrameRel.c_str(),
    //             ex.what());
    return nullptr;
  }
}