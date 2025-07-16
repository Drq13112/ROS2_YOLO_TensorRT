// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from yolo_custom_interfaces:msg/PidnetResult.idl
// generated code does not contain a copyright notice

#ifndef YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__STRUCT_H_
#define YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'segmentation_map'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'image_source_monotonic_capture_time'
// Member 'processing_node_monotonic_entry_time'
// Member 'processing_node_inference_start_time'
// Member 'processing_node_inference_end_time'
// Member 'processing_node_monotonic_publish_time'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/PidnetResult in the package yolo_custom_interfaces.
typedef struct yolo_custom_interfaces__msg__PidnetResult
{
  std_msgs__msg__Header header;
  /// El mapa de segmentación combinado.
  /// Encoding esperado: "8UC2"
  /// Canal 0: ID de la clase (0-19)
  /// Canal 1: Confianza del píxel (0-255)
  sensor_msgs__msg__Image segmentation_map;
  /// Número de secuencia para seguimiento de lotes y pérdidas
  uint64_t packet_sequence_number;
  /// Timestamps para análisis de latencia (usando el reloj MONOTONIC del nodo de procesamiento)
  /// T1: Entrada al callback de la imagen
  builtin_interfaces__msg__Time image_source_monotonic_capture_time;
  /// T2: Inicio del procesamiento del lote
  builtin_interfaces__msg__Time processing_node_monotonic_entry_time;
  /// T2a: Inicio de la inferencia
  builtin_interfaces__msg__Time processing_node_inference_start_time;
  /// T2b: Fin de la inferencia
  builtin_interfaces__msg__Time processing_node_inference_end_time;
  /// T3: Publicación del resultado
  builtin_interfaces__msg__Time processing_node_monotonic_publish_time;
} yolo_custom_interfaces__msg__PidnetResult;

// Struct for a sequence of yolo_custom_interfaces__msg__PidnetResult.
typedef struct yolo_custom_interfaces__msg__PidnetResult__Sequence
{
  yolo_custom_interfaces__msg__PidnetResult * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} yolo_custom_interfaces__msg__PidnetResult__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__STRUCT_H_
