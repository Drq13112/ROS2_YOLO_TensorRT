// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from docs_turtlesim:msg/KeyedPose.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__STRUCT_H_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/KeyedPose in the package docs_turtlesim.
typedef struct docs_turtlesim__msg__KeyedPose
{
  int32_t turtle_id;
  double x;
  double y;
  double theta;
  double linear_velocity;
  double angular_velocity;
} docs_turtlesim__msg__KeyedPose;

// Struct for a sequence of docs_turtlesim__msg__KeyedPose.
typedef struct docs_turtlesim__msg__KeyedPose__Sequence
{
  docs_turtlesim__msg__KeyedPose * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} docs_turtlesim__msg__KeyedPose__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__STRUCT_H_
