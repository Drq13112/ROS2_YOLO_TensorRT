// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from docs_turtlesim:msg/Vector3.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__VECTOR3__STRUCT_H_
#define DOCS_TURTLESIM__MSG__DETAIL__VECTOR3__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Vector3 in the package docs_turtlesim.
typedef struct docs_turtlesim__msg__Vector3
{
  double x;
  double y;
  double z;
} docs_turtlesim__msg__Vector3;

// Struct for a sequence of docs_turtlesim__msg__Vector3.
typedef struct docs_turtlesim__msg__Vector3__Sequence
{
  docs_turtlesim__msg__Vector3 * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} docs_turtlesim__msg__Vector3__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DOCS_TURTLESIM__MSG__DETAIL__VECTOR3__STRUCT_H_
