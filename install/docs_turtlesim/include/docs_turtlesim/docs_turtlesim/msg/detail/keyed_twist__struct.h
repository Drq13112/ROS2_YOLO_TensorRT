// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from docs_turtlesim:msg/KeyedTwist.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__STRUCT_H_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'linear'
// Member 'angular'
#include "docs_turtlesim/msg/detail/vector3__struct.h"

/// Struct defined in msg/KeyedTwist in the package docs_turtlesim.
typedef struct docs_turtlesim__msg__KeyedTwist
{
  int32_t turtle_id;
  docs_turtlesim__msg__Vector3 linear;
  docs_turtlesim__msg__Vector3 angular;
} docs_turtlesim__msg__KeyedTwist;

// Struct for a sequence of docs_turtlesim__msg__KeyedTwist.
typedef struct docs_turtlesim__msg__KeyedTwist__Sequence
{
  docs_turtlesim__msg__KeyedTwist * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} docs_turtlesim__msg__KeyedTwist__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__STRUCT_H_
