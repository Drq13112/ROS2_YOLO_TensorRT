// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__rosidl_typesupport_introspection_c.h"
#include "yolo_custom_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__functions.h"
#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `mask`
#include "sensor_msgs/msg/image.h"
// Member `mask`
#include "sensor_msgs/msg/detail/image__rosidl_typesupport_introspection_c.h"
// Member `scores`
// Member `classes`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `image_source_monotonic_capture_time`
// Member `processing_node_monotonic_entry_time`
// Member `processing_node_monotonic_publish_time`
#include "builtin_interfaces/msg/time.h"
// Member `image_source_monotonic_capture_time`
// Member `processing_node_monotonic_entry_time`
// Member `processing_node_monotonic_publish_time`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__init(message_memory);
}

void yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_fini_function(void * message_memory)
{
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__fini(message_memory);
}

size_t yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__size_function__InstanceSegmentationInfo__scores(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_const_function__InstanceSegmentationInfo__scores(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_function__InstanceSegmentationInfo__scores(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__fetch_function__InstanceSegmentationInfo__scores(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_const_function__InstanceSegmentationInfo__scores(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__assign_function__InstanceSegmentationInfo__scores(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_function__InstanceSegmentationInfo__scores(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__resize_function__InstanceSegmentationInfo__scores(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__size_function__InstanceSegmentationInfo__classes(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_const_function__InstanceSegmentationInfo__classes(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_function__InstanceSegmentationInfo__classes(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__fetch_function__InstanceSegmentationInfo__classes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_const_function__InstanceSegmentationInfo__classes(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__assign_function__InstanceSegmentationInfo__classes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_function__InstanceSegmentationInfo__classes(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__resize_function__InstanceSegmentationInfo__classes(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_member_array[7] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__InstanceSegmentationInfo, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mask",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__InstanceSegmentationInfo, mask),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "scores",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__InstanceSegmentationInfo, scores),  // bytes offset in struct
    NULL,  // default value
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__size_function__InstanceSegmentationInfo__scores,  // size() function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_const_function__InstanceSegmentationInfo__scores,  // get_const(index) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_function__InstanceSegmentationInfo__scores,  // get(index) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__fetch_function__InstanceSegmentationInfo__scores,  // fetch(index, &value) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__assign_function__InstanceSegmentationInfo__scores,  // assign(index, value) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__resize_function__InstanceSegmentationInfo__scores  // resize(index) function pointer
  },
  {
    "classes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__InstanceSegmentationInfo, classes),  // bytes offset in struct
    NULL,  // default value
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__size_function__InstanceSegmentationInfo__classes,  // size() function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_const_function__InstanceSegmentationInfo__classes,  // get_const(index) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__get_function__InstanceSegmentationInfo__classes,  // get(index) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__fetch_function__InstanceSegmentationInfo__classes,  // fetch(index, &value) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__assign_function__InstanceSegmentationInfo__classes,  // assign(index, value) function pointer
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__resize_function__InstanceSegmentationInfo__classes  // resize(index) function pointer
  },
  {
    "image_source_monotonic_capture_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__InstanceSegmentationInfo, image_source_monotonic_capture_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "processing_node_monotonic_entry_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__InstanceSegmentationInfo, processing_node_monotonic_entry_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "processing_node_monotonic_publish_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__InstanceSegmentationInfo, processing_node_monotonic_publish_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_members = {
  "yolo_custom_interfaces__msg",  // message namespace
  "InstanceSegmentationInfo",  // message name
  7,  // number of fields
  sizeof(yolo_custom_interfaces__msg__InstanceSegmentationInfo),
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_member_array,  // message members
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_type_support_handle = {
  0,
  &yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_yolo_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, yolo_custom_interfaces, msg, InstanceSegmentationInfo)() {
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_member_array[5].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_member_array[6].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_type_support_handle.typesupport_identifier) {
    yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &yolo_custom_interfaces__msg__InstanceSegmentationInfo__rosidl_typesupport_introspection_c__InstanceSegmentationInfo_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
