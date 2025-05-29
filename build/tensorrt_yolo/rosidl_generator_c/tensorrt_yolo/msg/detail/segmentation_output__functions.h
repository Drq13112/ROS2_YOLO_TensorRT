// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#ifndef TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__FUNCTIONS_H_
#define TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "tensorrt_yolo/msg/rosidl_generator_c__visibility_control.h"

#include "tensorrt_yolo/msg/detail/segmentation_output__struct.h"

/// Initialize msg/SegmentationOutput message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * tensorrt_yolo__msg__SegmentationOutput
 * )) before or use
 * tensorrt_yolo__msg__SegmentationOutput__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
bool
tensorrt_yolo__msg__SegmentationOutput__init(tensorrt_yolo__msg__SegmentationOutput * msg);

/// Finalize msg/SegmentationOutput message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
void
tensorrt_yolo__msg__SegmentationOutput__fini(tensorrt_yolo__msg__SegmentationOutput * msg);

/// Create msg/SegmentationOutput message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * tensorrt_yolo__msg__SegmentationOutput__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
tensorrt_yolo__msg__SegmentationOutput *
tensorrt_yolo__msg__SegmentationOutput__create();

/// Destroy msg/SegmentationOutput message.
/**
 * It calls
 * tensorrt_yolo__msg__SegmentationOutput__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
void
tensorrt_yolo__msg__SegmentationOutput__destroy(tensorrt_yolo__msg__SegmentationOutput * msg);

/// Check for msg/SegmentationOutput message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
bool
tensorrt_yolo__msg__SegmentationOutput__are_equal(const tensorrt_yolo__msg__SegmentationOutput * lhs, const tensorrt_yolo__msg__SegmentationOutput * rhs);

/// Copy a msg/SegmentationOutput message.
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
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
bool
tensorrt_yolo__msg__SegmentationOutput__copy(
  const tensorrt_yolo__msg__SegmentationOutput * input,
  tensorrt_yolo__msg__SegmentationOutput * output);

/// Initialize array of msg/SegmentationOutput messages.
/**
 * It allocates the memory for the number of elements and calls
 * tensorrt_yolo__msg__SegmentationOutput__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
bool
tensorrt_yolo__msg__SegmentationOutput__Sequence__init(tensorrt_yolo__msg__SegmentationOutput__Sequence * array, size_t size);

/// Finalize array of msg/SegmentationOutput messages.
/**
 * It calls
 * tensorrt_yolo__msg__SegmentationOutput__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
void
tensorrt_yolo__msg__SegmentationOutput__Sequence__fini(tensorrt_yolo__msg__SegmentationOutput__Sequence * array);

/// Create array of msg/SegmentationOutput messages.
/**
 * It allocates the memory for the array and calls
 * tensorrt_yolo__msg__SegmentationOutput__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
tensorrt_yolo__msg__SegmentationOutput__Sequence *
tensorrt_yolo__msg__SegmentationOutput__Sequence__create(size_t size);

/// Destroy array of msg/SegmentationOutput messages.
/**
 * It calls
 * tensorrt_yolo__msg__SegmentationOutput__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
void
tensorrt_yolo__msg__SegmentationOutput__Sequence__destroy(tensorrt_yolo__msg__SegmentationOutput__Sequence * array);

/// Check for msg/SegmentationOutput message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
bool
tensorrt_yolo__msg__SegmentationOutput__Sequence__are_equal(const tensorrt_yolo__msg__SegmentationOutput__Sequence * lhs, const tensorrt_yolo__msg__SegmentationOutput__Sequence * rhs);

/// Copy an array of msg/SegmentationOutput messages.
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
ROSIDL_GENERATOR_C_PUBLIC_tensorrt_yolo
bool
tensorrt_yolo__msg__SegmentationOutput__Sequence__copy(
  const tensorrt_yolo__msg__SegmentationOutput__Sequence * input,
  tensorrt_yolo__msg__SegmentationOutput__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__FUNCTIONS_H_
