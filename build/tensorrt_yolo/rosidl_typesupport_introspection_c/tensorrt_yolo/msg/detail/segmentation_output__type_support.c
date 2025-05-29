// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tensorrt_yolo/msg/detail/segmentation_output__rosidl_typesupport_introspection_c.h"
#include "tensorrt_yolo/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tensorrt_yolo/msg/detail/segmentation_output__functions.h"
#include "tensorrt_yolo/msg/detail/segmentation_output__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `class_id_map`
// Member `instance_id_map`
// Member `instance_confidences`
// Member `instance_class_ids`
// Member `detected_instance_ids`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tensorrt_yolo__msg__SegmentationOutput__init(message_memory);
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_fini_function(void * message_memory)
{
  tensorrt_yolo__msg__SegmentationOutput__fini(message_memory);
}

size_t tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__class_id_map(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__class_id_map(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__class_id_map(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__class_id_map(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__class_id_map(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__class_id_map(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__class_id_map(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__class_id_map(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

size_t tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__instance_id_map(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_id_map(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_id_map(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__instance_id_map(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_id_map(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__instance_id_map(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_id_map(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__instance_id_map(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

size_t tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__instance_confidences(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_confidences(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_confidences(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__instance_confidences(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_confidences(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__instance_confidences(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_confidences(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__instance_confidences(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__instance_class_ids(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_class_ids(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_class_ids(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__instance_class_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_class_ids(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__instance_class_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_class_ids(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__instance_class_ids(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

size_t tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__detected_instance_ids(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__detected_instance_ids(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__detected_instance_ids(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__detected_instance_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__detected_instance_ids(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__detected_instance_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__detected_instance_ids(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__detected_instance_ids(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_member_array[8] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "image_height",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, image_height),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "image_width",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, image_width),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "class_id_map",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, class_id_map),  // bytes offset in struct
    NULL,  // default value
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__class_id_map,  // size() function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__class_id_map,  // get_const(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__class_id_map,  // get(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__class_id_map,  // fetch(index, &value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__class_id_map,  // assign(index, value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__class_id_map  // resize(index) function pointer
  },
  {
    "instance_id_map",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, instance_id_map),  // bytes offset in struct
    NULL,  // default value
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__instance_id_map,  // size() function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_id_map,  // get_const(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_id_map,  // get(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__instance_id_map,  // fetch(index, &value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__instance_id_map,  // assign(index, value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__instance_id_map  // resize(index) function pointer
  },
  {
    "instance_confidences",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, instance_confidences),  // bytes offset in struct
    NULL,  // default value
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__instance_confidences,  // size() function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_confidences,  // get_const(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_confidences,  // get(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__instance_confidences,  // fetch(index, &value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__instance_confidences,  // assign(index, value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__instance_confidences  // resize(index) function pointer
  },
  {
    "instance_class_ids",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, instance_class_ids),  // bytes offset in struct
    NULL,  // default value
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__instance_class_ids,  // size() function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__instance_class_ids,  // get_const(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__instance_class_ids,  // get(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__instance_class_ids,  // fetch(index, &value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__instance_class_ids,  // assign(index, value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__instance_class_ids  // resize(index) function pointer
  },
  {
    "detected_instance_ids",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo__msg__SegmentationOutput, detected_instance_ids),  // bytes offset in struct
    NULL,  // default value
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__size_function__SegmentationOutput__detected_instance_ids,  // size() function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_const_function__SegmentationOutput__detected_instance_ids,  // get_const(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__get_function__SegmentationOutput__detected_instance_ids,  // get(index) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__fetch_function__SegmentationOutput__detected_instance_ids,  // fetch(index, &value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__assign_function__SegmentationOutput__detected_instance_ids,  // assign(index, value) function pointer
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__resize_function__SegmentationOutput__detected_instance_ids  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_members = {
  "tensorrt_yolo__msg",  // message namespace
  "SegmentationOutput",  // message name
  8,  // number of fields
  sizeof(tensorrt_yolo__msg__SegmentationOutput),
  tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_member_array,  // message members
  tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_init_function,  // function to initialize message memory (memory has to be allocated)
  tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_type_support_handle = {
  0,
  &tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tensorrt_yolo
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tensorrt_yolo, msg, SegmentationOutput)() {
  tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_type_support_handle.typesupport_identifier) {
    tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tensorrt_yolo__msg__SegmentationOutput__rosidl_typesupport_introspection_c__SegmentationOutput_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
