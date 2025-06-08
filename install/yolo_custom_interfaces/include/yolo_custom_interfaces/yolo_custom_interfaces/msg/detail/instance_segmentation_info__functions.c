// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
// generated code does not contain a copyright notice
#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `mask`
#include "sensor_msgs/msg/detail/image__functions.h"
// Member `scores`
// Member `classes`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__init(yolo_custom_interfaces__msg__InstanceSegmentationInfo * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(msg);
    return false;
  }
  // mask
  if (!sensor_msgs__msg__Image__init(&msg->mask)) {
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(msg);
    return false;
  }
  // scores
  if (!rosidl_runtime_c__float__Sequence__init(&msg->scores, 0)) {
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(msg);
    return false;
  }
  // classes
  if (!rosidl_runtime_c__int32__Sequence__init(&msg->classes, 0)) {
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(msg);
    return false;
  }
  return true;
}

void
yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(yolo_custom_interfaces__msg__InstanceSegmentationInfo * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // mask
  sensor_msgs__msg__Image__fini(&msg->mask);
  // scores
  rosidl_runtime_c__float__Sequence__fini(&msg->scores);
  // classes
  rosidl_runtime_c__int32__Sequence__fini(&msg->classes);
}

bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__are_equal(const yolo_custom_interfaces__msg__InstanceSegmentationInfo * lhs, const yolo_custom_interfaces__msg__InstanceSegmentationInfo * rhs)
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
  // mask
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->mask), &(rhs->mask)))
  {
    return false;
  }
  // scores
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->scores), &(rhs->scores)))
  {
    return false;
  }
  // classes
  if (!rosidl_runtime_c__int32__Sequence__are_equal(
      &(lhs->classes), &(rhs->classes)))
  {
    return false;
  }
  return true;
}

bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__copy(
  const yolo_custom_interfaces__msg__InstanceSegmentationInfo * input,
  yolo_custom_interfaces__msg__InstanceSegmentationInfo * output)
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
  // mask
  if (!sensor_msgs__msg__Image__copy(
      &(input->mask), &(output->mask)))
  {
    return false;
  }
  // scores
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->scores), &(output->scores)))
  {
    return false;
  }
  // classes
  if (!rosidl_runtime_c__int32__Sequence__copy(
      &(input->classes), &(output->classes)))
  {
    return false;
  }
  return true;
}

yolo_custom_interfaces__msg__InstanceSegmentationInfo *
yolo_custom_interfaces__msg__InstanceSegmentationInfo__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  yolo_custom_interfaces__msg__InstanceSegmentationInfo * msg = (yolo_custom_interfaces__msg__InstanceSegmentationInfo *)allocator.allocate(sizeof(yolo_custom_interfaces__msg__InstanceSegmentationInfo), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(yolo_custom_interfaces__msg__InstanceSegmentationInfo));
  bool success = yolo_custom_interfaces__msg__InstanceSegmentationInfo__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
yolo_custom_interfaces__msg__InstanceSegmentationInfo__destroy(yolo_custom_interfaces__msg__InstanceSegmentationInfo * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__init(yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  yolo_custom_interfaces__msg__InstanceSegmentationInfo * data = NULL;

  if (size) {
    data = (yolo_custom_interfaces__msg__InstanceSegmentationInfo *)allocator.zero_allocate(size, sizeof(yolo_custom_interfaces__msg__InstanceSegmentationInfo), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = yolo_custom_interfaces__msg__InstanceSegmentationInfo__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(&data[i - 1]);
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
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__fini(yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * array)
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
      yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(&array->data[i]);
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

yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence *
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * array = (yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence *)allocator.allocate(sizeof(yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__destroy(yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__are_equal(const yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * lhs, const yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!yolo_custom_interfaces__msg__InstanceSegmentationInfo__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence__copy(
  const yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * input,
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(yolo_custom_interfaces__msg__InstanceSegmentationInfo);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    yolo_custom_interfaces__msg__InstanceSegmentationInfo * data =
      (yolo_custom_interfaces__msg__InstanceSegmentationInfo *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!yolo_custom_interfaces__msg__InstanceSegmentationInfo__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!yolo_custom_interfaces__msg__InstanceSegmentationInfo__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
