// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from yolo_custom_interfaces:msg/PidnetResult.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "yolo_custom_interfaces/msg/detail/pidnet_result__rosidl_typesupport_introspection_c.h"
#include "yolo_custom_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "yolo_custom_interfaces/msg/detail/pidnet_result__functions.h"
#include "yolo_custom_interfaces/msg/detail/pidnet_result__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `segmentation_map`
#include "sensor_msgs/msg/image.h"
// Member `segmentation_map`
#include "sensor_msgs/msg/detail/image__rosidl_typesupport_introspection_c.h"
// Member `image_source_monotonic_capture_time`
// Member `processing_node_monotonic_entry_time`
// Member `processing_node_inference_start_time`
// Member `processing_node_inference_end_time`
// Member `processing_node_monotonic_publish_time`
#include "builtin_interfaces/msg/time.h"
// Member `image_source_monotonic_capture_time`
// Member `processing_node_monotonic_entry_time`
// Member `processing_node_inference_start_time`
// Member `processing_node_inference_end_time`
// Member `processing_node_monotonic_publish_time`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  yolo_custom_interfaces__msg__PidnetResult__init(message_memory);
}

void yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_fini_function(void * message_memory)
{
  yolo_custom_interfaces__msg__PidnetResult__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[8] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__PidnetResult, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "segmentation_map",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__PidnetResult, segmentation_map),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "packet_sequence_number",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__PidnetResult, packet_sequence_number),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "image_source_monotonic_capture_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__PidnetResult, image_source_monotonic_capture_time),  // bytes offset in struct
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
    offsetof(yolo_custom_interfaces__msg__PidnetResult, processing_node_monotonic_entry_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "processing_node_inference_start_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__PidnetResult, processing_node_inference_start_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "processing_node_inference_end_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces__msg__PidnetResult, processing_node_inference_end_time),  // bytes offset in struct
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
    offsetof(yolo_custom_interfaces__msg__PidnetResult, processing_node_monotonic_publish_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_members = {
  "yolo_custom_interfaces__msg",  // message namespace
  "PidnetResult",  // message name
  8,  // number of fields
  sizeof(yolo_custom_interfaces__msg__PidnetResult),
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array,  // message members
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_init_function,  // function to initialize message memory (memory has to be allocated)
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_type_support_handle = {
  0,
  &yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_yolo_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, yolo_custom_interfaces, msg, PidnetResult)() {
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[5].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[6].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_member_array[7].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_type_support_handle.typesupport_identifier) {
    yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &yolo_custom_interfaces__msg__PidnetResult__rosidl_typesupport_introspection_c__PidnetResult_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
