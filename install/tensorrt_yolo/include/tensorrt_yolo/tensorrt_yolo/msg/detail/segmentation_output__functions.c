// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice
#include "tensorrt_yolo/msg/detail/segmentation_output__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `class_id_map`
// Member `instance_id_map`
// Member `instance_confidences`
// Member `instance_class_ids`
// Member `detected_instance_ids`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
tensorrt_yolo__msg__SegmentationOutput__init(tensorrt_yolo__msg__SegmentationOutput * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tensorrt_yolo__msg__SegmentationOutput__fini(msg);
    return false;
  }
  // image_height
  // image_width
  // class_id_map
  if (!rosidl_runtime_c__int32__Sequence__init(&msg->class_id_map, 0)) {
    tensorrt_yolo__msg__SegmentationOutput__fini(msg);
    return false;
  }
  // instance_id_map
  if (!rosidl_runtime_c__int32__Sequence__init(&msg->instance_id_map, 0)) {
    tensorrt_yolo__msg__SegmentationOutput__fini(msg);
    return false;
  }
  // instance_confidences
  if (!rosidl_runtime_c__float__Sequence__init(&msg->instance_confidences, 0)) {
    tensorrt_yolo__msg__SegmentationOutput__fini(msg);
    return false;
  }
  // instance_class_ids
  if (!rosidl_runtime_c__int32__Sequence__init(&msg->instance_class_ids, 0)) {
    tensorrt_yolo__msg__SegmentationOutput__fini(msg);
    return false;
  }
  // detected_instance_ids
  if (!rosidl_runtime_c__int32__Sequence__init(&msg->detected_instance_ids, 0)) {
    tensorrt_yolo__msg__SegmentationOutput__fini(msg);
    return false;
  }
  return true;
}

void
tensorrt_yolo__msg__SegmentationOutput__fini(tensorrt_yolo__msg__SegmentationOutput * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // image_height
  // image_width
  // class_id_map
  rosidl_runtime_c__int32__Sequence__fini(&msg->class_id_map);
  // instance_id_map
  rosidl_runtime_c__int32__Sequence__fini(&msg->instance_id_map);
  // instance_confidences
  rosidl_runtime_c__float__Sequence__fini(&msg->instance_confidences);
  // instance_class_ids
  rosidl_runtime_c__int32__Sequence__fini(&msg->instance_class_ids);
  // detected_instance_ids
  rosidl_runtime_c__int32__Sequence__fini(&msg->detected_instance_ids);
}

bool
tensorrt_yolo__msg__SegmentationOutput__are_equal(const tensorrt_yolo__msg__SegmentationOutput * lhs, const tensorrt_yolo__msg__SegmentationOutput * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // image_height
  if (lhs->image_height != rhs->image_height) {
    return false;
  }
  // image_width
  if (lhs->image_width != rhs->image_width) {
    return false;
  }
  // class_id_map
  if (!rosidl_runtime_c__int32__Sequence__are_equal(
      &(lhs->class_id_map), &(rhs->class_id_map)))
  {
    return false;
  }
  // instance_id_map
  if (!rosidl_runtime_c__int32__Sequence__are_equal(
      &(lhs->instance_id_map), &(rhs->instance_id_map)))
  {
    return false;
  }
  // instance_confidences
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->instance_confidences), &(rhs->instance_confidences)))
  {
    return false;
  }
  // instance_class_ids
  if (!rosidl_runtime_c__int32__Sequence__are_equal(
      &(lhs->instance_class_ids), &(rhs->instance_class_ids)))
  {
    return false;
  }
  // detected_instance_ids
  if (!rosidl_runtime_c__int32__Sequence__are_equal(
      &(lhs->detected_instance_ids), &(rhs->detected_instance_ids)))
  {
    return false;
  }
  return true;
}

bool
tensorrt_yolo__msg__SegmentationOutput__copy(
  const tensorrt_yolo__msg__SegmentationOutput * input,
  tensorrt_yolo__msg__SegmentationOutput * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // image_height
  output->image_height = input->image_height;
  // image_width
  output->image_width = input->image_width;
  // class_id_map
  if (!rosidl_runtime_c__int32__Sequence__copy(
      &(input->class_id_map), &(output->class_id_map)))
  {
    return false;
  }
  // instance_id_map
  if (!rosidl_runtime_c__int32__Sequence__copy(
      &(input->instance_id_map), &(output->instance_id_map)))
  {
    return false;
  }
  // instance_confidences
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->instance_confidences), &(output->instance_confidences)))
  {
    return false;
  }
  // instance_class_ids
  if (!rosidl_runtime_c__int32__Sequence__copy(
      &(input->instance_class_ids), &(output->instance_class_ids)))
  {
    return false;
  }
  // detected_instance_ids
  if (!rosidl_runtime_c__int32__Sequence__copy(
      &(input->detected_instance_ids), &(output->detected_instance_ids)))
  {
    return false;
  }
  return true;
}

tensorrt_yolo__msg__SegmentationOutput *
tensorrt_yolo__msg__SegmentationOutput__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tensorrt_yolo__msg__SegmentationOutput * msg = (tensorrt_yolo__msg__SegmentationOutput *)allocator.allocate(sizeof(tensorrt_yolo__msg__SegmentationOutput), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tensorrt_yolo__msg__SegmentationOutput));
  bool success = tensorrt_yolo__msg__SegmentationOutput__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tensorrt_yolo__msg__SegmentationOutput__destroy(tensorrt_yolo__msg__SegmentationOutput * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tensorrt_yolo__msg__SegmentationOutput__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tensorrt_yolo__msg__SegmentationOutput__Sequence__init(tensorrt_yolo__msg__SegmentationOutput__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tensorrt_yolo__msg__SegmentationOutput * data = NULL;

  if (size) {
    data = (tensorrt_yolo__msg__SegmentationOutput *)allocator.zero_allocate(size, sizeof(tensorrt_yolo__msg__SegmentationOutput), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tensorrt_yolo__msg__SegmentationOutput__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tensorrt_yolo__msg__SegmentationOutput__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
tensorrt_yolo__msg__SegmentationOutput__Sequence__fini(tensorrt_yolo__msg__SegmentationOutput__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      tensorrt_yolo__msg__SegmentationOutput__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

tensorrt_yolo__msg__SegmentationOutput__Sequence *
tensorrt_yolo__msg__SegmentationOutput__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tensorrt_yolo__msg__SegmentationOutput__Sequence * array = (tensorrt_yolo__msg__SegmentationOutput__Sequence *)allocator.allocate(sizeof(tensorrt_yolo__msg__SegmentationOutput__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tensorrt_yolo__msg__SegmentationOutput__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tensorrt_yolo__msg__SegmentationOutput__Sequence__destroy(tensorrt_yolo__msg__SegmentationOutput__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tensorrt_yolo__msg__SegmentationOutput__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tensorrt_yolo__msg__SegmentationOutput__Sequence__are_equal(const tensorrt_yolo__msg__SegmentationOutput__Sequence * lhs, const tensorrt_yolo__msg__SegmentationOutput__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tensorrt_yolo__msg__SegmentationOutput__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tensorrt_yolo__msg__SegmentationOutput__Sequence__copy(
  const tensorrt_yolo__msg__SegmentationOutput__Sequence * input,
  tensorrt_yolo__msg__SegmentationOutput__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tensorrt_yolo__msg__SegmentationOutput);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tensorrt_yolo__msg__SegmentationOutput * data =
      (tensorrt_yolo__msg__SegmentationOutput *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tensorrt_yolo__msg__SegmentationOutput__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tensorrt_yolo__msg__SegmentationOutput__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tensorrt_yolo__msg__SegmentationOutput__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
