#pragma once

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

// AUTOPIA: Estructura que representa las etiquetas de los puntos
enum LabelType
{
  ground,
  obstacle,
  doubt,
  sidewalk,
  NaN
};

// AUTOPIA Estructura que representa el tipo de punto que tienen los puntos de la nube de puntos que se publica como
// resultado de la clasificación
struct PointChannelBasedClassification
{
  PCL_ADD_POINT4D;  // quad-word XYZ
  float intensity;  ///< laser intensity reading
  int is_ground;    // 0 Ground 1 Obstacle -1 Noise
  int ring;
  int channel;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
} EIGEN_ALIGN16;

// Estructura que representa los tipos de punto que tienen las nubes que llegan a la clasificación
struct PointXYZIRTVH
{
  PCL_ADD_POINT4D;
  float intensity;
  std::uint16_t ring;
  double timestamp;
  double vertAng;
  double horAng;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// Define common PointT
using PointT = PointChannelBasedClassification;

// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(PointChannelBasedClassification,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
                                      int, channel, channel)(int, is_ground, is_ground)(int, ring, ring))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRTVH,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t,
                                                                                                       ring, ring)(
                                      double, timestamp, timestamp)(double, vertAng, vertAng)(double, horAng, horAng))
