// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from yolo_custom_interfaces:msg/PidnetResult.idl
// generated code does not contain a copyright notice
#include "yolo_custom_interfaces/msg/detail/pidnet_result__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `segmentation_map`
#include "sensor_msgs/msg/detail/image__functions.h"
// Member `image_source_monotonic_capture_time`
// Member `processing_node_monotonic_entry_time`
// Member `processing_node_inference_start_time`
// Member `processing_node_inference_end_time`
// Member `processing_node_monotonic_publish_time`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
yolo_custom_interfaces__msg__PidnetResult__init(yolo_custom_interfaces__msg__PidnetResult * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
    return false;
  }
  // segmentation_map
  if (!sensor_msgs__msg__Image__init(&msg->segmentation_map)) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
    return false;
  }
  // packet_sequence_number
  // image_source_monotonic_capture_time
  if (!builtin_interfaces__msg__Time__init(&msg->image_source_monotonic_capture_time)) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
    return false;
  }
  // processing_node_monotonic_entry_time
  if (!builtin_interfaces__msg__Time__init(&msg->processing_node_monotonic_entry_time)) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
    return false;
  }
  // processing_node_inference_start_time
  if (!builtin_interfaces__msg__Time__init(&msg->processing_node_inference_start_time)) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
    return false;
  }
  // processing_node_inference_end_time
  if (!builtin_interfaces__msg__Time__init(&msg->processing_node_inference_end_time)) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
    return false;
  }
  // processing_node_monotonic_publish_time
  if (!builtin_interfaces__msg__Time__init(&msg->processing_node_monotonic_publish_time)) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
    return false;
  }
  return true;
}

void
yolo_custom_interfaces__msg__PidnetResult__fini(yolo_custom_interfaces__msg__PidnetResult * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // segmentation_map
  sensor_msgs__msg__Image__fini(&msg->segmentation_map);
  // packet_sequence_number
  // image_source_monotonic_capture_time
  builtin_interfaces__msg__Time__fini(&msg->image_source_monotonic_capture_time);
  // processing_node_monotonic_entry_time
  builtin_interfaces__msg__Time__fini(&msg->processing_node_monotonic_entry_time);
  // processing_node_inference_start_time
  builtin_interfaces__msg__Time__fini(&msg->processing_node_inference_start_time);
  // processing_node_inference_end_time
  builtin_interfaces__msg__Time__fini(&msg->processing_node_inference_end_time);
  // processing_node_monotonic_publish_time
  builtin_interfaces__msg__Time__fini(&msg->processing_node_monotonic_publish_time);
}

bool
yolo_custom_interfaces__msg__PidnetResult__are_equal(const yolo_custom_interfaces__msg__PidnetResult * lhs, const yolo_custom_interfaces__msg__PidnetResult * rhs)
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
  // segmentation_map
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->segmentation_map), &(rhs->segmentation_map)))
  {
    return false;
  }
  // packet_sequence_number
  if (lhs->packet_sequence_number != rhs->packet_sequence_number) {
    return false;
  }
  // image_source_monotonic_capture_time
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->image_source_monotonic_capture_time), &(rhs->image_source_monotonic_capture_time)))
  {
    return false;
  }
  // processing_node_monotonic_entry_time
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->processing_node_monotonic_entry_time), &(rhs->processing_node_monotonic_entry_time)))
  {
    return false;
  }
  // processing_node_inference_start_time
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->processing_node_inference_start_time), &(rhs->processing_node_inference_start_time)))
  {
    return false;
  }
  // processing_node_inference_end_time
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->processing_node_inference_end_time), &(rhs->processing_node_inference_end_time)))
  {
    return false;
  }
  // processing_node_monotonic_publish_time
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->processing_node_monotonic_publish_time), &(rhs->processing_node_monotonic_publish_time)))
  {
    return false;
  }
  return true;
}

bool
yolo_custom_interfaces__msg__PidnetResult__copy(
  const yolo_custom_interfaces__msg__PidnetResult * input,
  yolo_custom_interfaces__msg__PidnetResult * output)
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
  // segmentation_map
  if (!sensor_msgs__msg__Image__copy(
      &(input->segmentation_map), &(output->segmentation_map)))
  {
    return false;
  }
  // packet_sequence_number
  output->packet_sequence_number = input->packet_sequence_number;
  // image_source_monotonic_capture_time
  if (!builtin_interfaces__msg__Time__copy(
      &(input->image_source_monotonic_capture_time), &(output->image_source_monotonic_capture_time)))
  {
    return false;
  }
  // processing_node_monotonic_entry_time
  if (!builtin_interfaces__msg__Time__copy(
      &(input->processing_node_monotonic_entry_time), &(output->processing_node_monotonic_entry_time)))
  {
    return false;
  }
  // processing_node_inference_start_time
  if (!builtin_interfaces__msg__Time__copy(
      &(input->processing_node_inference_start_time), &(output->processing_node_inference_start_time)))
  {
    return false;
  }
  // processing_node_inference_end_time
  if (!builtin_interfaces__msg__Time__copy(
      &(input->processing_node_inference_end_time), &(output->processing_node_inference_end_time)))
  {
    return false;
  }
  // processing_node_monotonic_publish_time
  if (!builtin_interfaces__msg__Time__copy(
      &(input->processing_node_monotonic_publish_time), &(output->processing_node_monotonic_publish_time)))
  {
    return false;
  }
  return true;
}

yolo_custom_interfaces__msg__PidnetResult *
yolo_custom_interfaces__msg__PidnetResult__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  yolo_custom_interfaces__msg__PidnetResult * msg = (yolo_custom_interfaces__msg__PidnetResult *)allocator.allocate(sizeof(yolo_custom_interfaces__msg__PidnetResult), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(yolo_custom_interfaces__msg__PidnetResult));
  bool success = yolo_custom_interfaces__msg__PidnetResult__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
yolo_custom_interfaces__msg__PidnetResult__destroy(yolo_custom_interfaces__msg__PidnetResult * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    yolo_custom_interfaces__msg__PidnetResult__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
yolo_custom_interfaces__msg__PidnetResult__Sequence__init(yolo_custom_interfaces__msg__PidnetResult__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  yolo_custom_interfaces__msg__PidnetResult * data = NULL;

  if (size) {
    data = (yolo_custom_interfaces__msg__PidnetResult *)allocator.zero_allocate(size, sizeof(yolo_custom_interfaces__msg__PidnetResult), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = yolo_custom_interfaces__msg__PidnetResult__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        yolo_custom_interfaces__msg__PidnetResult__fini(&data[i - 1]);
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
yolo_custom_interfaces__msg__PidnetResult__Sequence__fini(yolo_custom_interfaces__msg__PidnetResult__Sequence * array)
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
      yolo_custom_interfaces__msg__PidnetResult__fini(&array->data[i]);
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

yolo_custom_interfaces__msg__PidnetResult__Sequence *
yolo_custom_interfaces__msg__PidnetResult__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  yolo_custom_interfaces__msg__PidnetResult__Sequence * array = (yolo_custom_interfaces__msg__PidnetResult__Sequence *)allocator.allocate(sizeof(yolo_custom_interfaces__msg__PidnetResult__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = yolo_custom_interfaces__msg__PidnetResult__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
yolo_custom_interfaces__msg__PidnetResult__Sequence__destroy(yolo_custom_interfaces__msg__PidnetResult__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    yolo_custom_interfaces__msg__PidnetResult__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
yolo_custom_interfaces__msg__PidnetResult__Sequence__are_equal(const yolo_custom_interfaces__msg__PidnetResult__Sequence * lhs, const yolo_custom_interfaces__msg__PidnetResult__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!yolo_custom_interfaces__msg__PidnetResult__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
yolo_custom_interfaces__msg__PidnetResult__Sequence__copy(
  const yolo_custom_interfaces__msg__PidnetResult__Sequence * input,
  yolo_custom_interfaces__msg__PidnetResult__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(yolo_custom_interfaces__msg__PidnetResult);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    yolo_custom_interfaces__msg__PidnetResult * data =
      (yolo_custom_interfaces__msg__PidnetResult *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!yolo_custom_interfaces__msg__PidnetResult__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          yolo_custom_interfaces__msg__PidnetResult__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!yolo_custom_interfaces__msg__PidnetResult__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
