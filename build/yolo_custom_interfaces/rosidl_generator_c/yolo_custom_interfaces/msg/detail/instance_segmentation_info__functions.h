// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
// generated code does not contain a copyright notice

#ifndef YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__FUNCTIONS_H_
#define YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "yolo_custom_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__struct.h"

/// Initialize msg/InstanceSegmentationInfo message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo
 * )) before or use
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__init(yolo_custom_interfaces__msg__InstanceSegmentationInfo * msg);

/// Finalize msg/InstanceSegmentationInfo message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
void
yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(yolo_custom_interfaces__msg__InstanceSegmentationInfo * msg);

/// Create msg/InstanceSegmentationInfo message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
yolo_custom_interfaces__msg__InstanceSegmentationInfo *
yolo_custom_interfaces__msg__InstanceSegmentationInfo__create();

/// Destroy msg/InstanceSegmentationInfo message.
/**
 * It calls
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
void
yolo_custom_interfaces__msg__InstanceSegmentationInfo__destroy(yolo_custom_interfaces__msg__InstanceSegmentationInfo * msg);

/// Check for msg/InstanceSegmentationInfo message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__are_equal(const yolo_custom_interfaces__msg__InstanceSegmentationInfo * lhs, const yolo_custom_interfaces__msg__InstanceSegmentationInfo * rhs);

/// Copy a msg/InstanceSegmentationInfo message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__copy(
  const yolo_custom_interfaces__msg__InstanceSegmentationInfo * input,
  yolo_custom_interfaces__msg__InstanceSegmentationInfo * output);

/// Initialize array of msg/InstanceSegmentationInfo messages.
/**
 * It allocates the memory for the number of elements and calls
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__init(yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * array, size_t size);

/// Finalize array of msg/InstanceSegmentationInfo messages.
/**
 * It calls
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
void
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__fini(yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * array);

/// Create array of msg/InstanceSegmentationInfo messages.
/**
 * It allocates the memory for the array and calls
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence *
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__create(size_t size);

/// Destroy array of msg/InstanceSegmentationInfo messages.
/**
 * It calls
 * yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
void
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__destroy(yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * array);

/// Check for msg/InstanceSegmentationInfo message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__are_equal(const yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * lhs, const yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * rhs);

/// Copy an array of msg/InstanceSegmentationInfo messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_yolo_custom_interfaces
bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__copy(
  const yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * input,
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__FUNCTIONS_H_
