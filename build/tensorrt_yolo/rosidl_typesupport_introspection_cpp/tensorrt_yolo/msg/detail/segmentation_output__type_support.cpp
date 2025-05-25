// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "tensorrt_yolo/msg/detail/segmentation_output__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace tensorrt_yolo
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void SegmentationOutput_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) tensorrt_yolo::msg::SegmentationOutput(_init);
}

void SegmentationOutput_fini_function(void * message_memory)
{
  auto typed_message = static_cast<tensorrt_yolo::msg::SegmentationOutput *>(message_memory);
  typed_message->~SegmentationOutput();
}

size_t size_function__SegmentationOutput__class_id_map(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SegmentationOutput__class_id_map(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__SegmentationOutput__class_id_map(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__SegmentationOutput__class_id_map(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__SegmentationOutput__class_id_map(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__SegmentationOutput__class_id_map(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__SegmentationOutput__class_id_map(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

void resize_function__SegmentationOutput__class_id_map(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

size_t size_function__SegmentationOutput__instance_id_map(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SegmentationOutput__instance_id_map(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__SegmentationOutput__instance_id_map(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__SegmentationOutput__instance_id_map(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__SegmentationOutput__instance_id_map(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__SegmentationOutput__instance_id_map(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__SegmentationOutput__instance_id_map(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

void resize_function__SegmentationOutput__instance_id_map(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

size_t size_function__SegmentationOutput__instance_confidences(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SegmentationOutput__instance_confidences(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__SegmentationOutput__instance_confidences(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__SegmentationOutput__instance_confidences(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__SegmentationOutput__instance_confidences(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__SegmentationOutput__instance_confidences(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__SegmentationOutput__instance_confidences(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__SegmentationOutput__instance_confidences(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__SegmentationOutput__instance_class_ids(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SegmentationOutput__instance_class_ids(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__SegmentationOutput__instance_class_ids(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__SegmentationOutput__instance_class_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__SegmentationOutput__instance_class_ids(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__SegmentationOutput__instance_class_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__SegmentationOutput__instance_class_ids(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

void resize_function__SegmentationOutput__instance_class_ids(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

size_t size_function__SegmentationOutput__detected_instance_ids(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SegmentationOutput__detected_instance_ids(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__SegmentationOutput__detected_instance_ids(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__SegmentationOutput__detected_instance_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__SegmentationOutput__detected_instance_ids(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__SegmentationOutput__detected_instance_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__SegmentationOutput__detected_instance_ids(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

void resize_function__SegmentationOutput__detected_instance_ids(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember SegmentationOutput_message_member_array[8] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "image_height",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, image_height),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "image_width",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, image_width),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "class_id_map",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, class_id_map),  // bytes offset in struct
    nullptr,  // default value
    size_function__SegmentationOutput__class_id_map,  // size() function pointer
    get_const_function__SegmentationOutput__class_id_map,  // get_const(index) function pointer
    get_function__SegmentationOutput__class_id_map,  // get(index) function pointer
    fetch_function__SegmentationOutput__class_id_map,  // fetch(index, &value) function pointer
    assign_function__SegmentationOutput__class_id_map,  // assign(index, value) function pointer
    resize_function__SegmentationOutput__class_id_map  // resize(index) function pointer
  },
  {
    "instance_id_map",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, instance_id_map),  // bytes offset in struct
    nullptr,  // default value
    size_function__SegmentationOutput__instance_id_map,  // size() function pointer
    get_const_function__SegmentationOutput__instance_id_map,  // get_const(index) function pointer
    get_function__SegmentationOutput__instance_id_map,  // get(index) function pointer
    fetch_function__SegmentationOutput__instance_id_map,  // fetch(index, &value) function pointer
    assign_function__SegmentationOutput__instance_id_map,  // assign(index, value) function pointer
    resize_function__SegmentationOutput__instance_id_map  // resize(index) function pointer
  },
  {
    "instance_confidences",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, instance_confidences),  // bytes offset in struct
    nullptr,  // default value
    size_function__SegmentationOutput__instance_confidences,  // size() function pointer
    get_const_function__SegmentationOutput__instance_confidences,  // get_const(index) function pointer
    get_function__SegmentationOutput__instance_confidences,  // get(index) function pointer
    fetch_function__SegmentationOutput__instance_confidences,  // fetch(index, &value) function pointer
    assign_function__SegmentationOutput__instance_confidences,  // assign(index, value) function pointer
    resize_function__SegmentationOutput__instance_confidences  // resize(index) function pointer
  },
  {
    "instance_class_ids",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, instance_class_ids),  // bytes offset in struct
    nullptr,  // default value
    size_function__SegmentationOutput__instance_class_ids,  // size() function pointer
    get_const_function__SegmentationOutput__instance_class_ids,  // get_const(index) function pointer
    get_function__SegmentationOutput__instance_class_ids,  // get(index) function pointer
    fetch_function__SegmentationOutput__instance_class_ids,  // fetch(index, &value) function pointer
    assign_function__SegmentationOutput__instance_class_ids,  // assign(index, value) function pointer
    resize_function__SegmentationOutput__instance_class_ids  // resize(index) function pointer
  },
  {
    "detected_instance_ids",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tensorrt_yolo::msg::SegmentationOutput, detected_instance_ids),  // bytes offset in struct
    nullptr,  // default value
    size_function__SegmentationOutput__detected_instance_ids,  // size() function pointer
    get_const_function__SegmentationOutput__detected_instance_ids,  // get_const(index) function pointer
    get_function__SegmentationOutput__detected_instance_ids,  // get(index) function pointer
    fetch_function__SegmentationOutput__detected_instance_ids,  // fetch(index, &value) function pointer
    assign_function__SegmentationOutput__detected_instance_ids,  // assign(index, value) function pointer
    resize_function__SegmentationOutput__detected_instance_ids  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers SegmentationOutput_message_members = {
  "tensorrt_yolo::msg",  // message namespace
  "SegmentationOutput",  // message name
  8,  // number of fields
  sizeof(tensorrt_yolo::msg::SegmentationOutput),
  SegmentationOutput_message_member_array,  // message members
  SegmentationOutput_init_function,  // function to initialize message memory (memory has to be allocated)
  SegmentationOutput_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t SegmentationOutput_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &SegmentationOutput_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace tensorrt_yolo


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<tensorrt_yolo::msg::SegmentationOutput>()
{
  return &::tensorrt_yolo::msg::rosidl_typesupport_introspection_cpp::SegmentationOutput_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, tensorrt_yolo, msg, SegmentationOutput)() {
  return &::tensorrt_yolo::msg::rosidl_typesupport_introspection_cpp::SegmentationOutput_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
