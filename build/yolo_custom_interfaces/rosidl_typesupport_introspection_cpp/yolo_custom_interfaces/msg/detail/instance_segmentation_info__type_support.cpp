// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace yolo_custom_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void InstanceSegmentationInfo_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) yolo_custom_interfaces::msg::InstanceSegmentationInfo(_init);
}

void InstanceSegmentationInfo_fini_function(void * message_memory)
{
  auto typed_message = static_cast<yolo_custom_interfaces::msg::InstanceSegmentationInfo *>(message_memory);
  typed_message->~InstanceSegmentationInfo();
}

size_t size_function__InstanceSegmentationInfo__scores(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__InstanceSegmentationInfo__scores(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__InstanceSegmentationInfo__scores(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__InstanceSegmentationInfo__scores(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__InstanceSegmentationInfo__scores(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__InstanceSegmentationInfo__scores(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__InstanceSegmentationInfo__scores(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__InstanceSegmentationInfo__scores(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__InstanceSegmentationInfo__classes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__InstanceSegmentationInfo__classes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__InstanceSegmentationInfo__classes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__InstanceSegmentationInfo__classes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__InstanceSegmentationInfo__classes(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__InstanceSegmentationInfo__classes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__InstanceSegmentationInfo__classes(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

void resize_function__InstanceSegmentationInfo__classes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember InstanceSegmentationInfo_message_member_array[7] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces::msg::InstanceSegmentationInfo, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mask",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<sensor_msgs::msg::Image>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces::msg::InstanceSegmentationInfo, mask),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "scores",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces::msg::InstanceSegmentationInfo, scores),  // bytes offset in struct
    nullptr,  // default value
    size_function__InstanceSegmentationInfo__scores,  // size() function pointer
    get_const_function__InstanceSegmentationInfo__scores,  // get_const(index) function pointer
    get_function__InstanceSegmentationInfo__scores,  // get(index) function pointer
    fetch_function__InstanceSegmentationInfo__scores,  // fetch(index, &value) function pointer
    assign_function__InstanceSegmentationInfo__scores,  // assign(index, value) function pointer
    resize_function__InstanceSegmentationInfo__scores  // resize(index) function pointer
  },
  {
    "classes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces::msg::InstanceSegmentationInfo, classes),  // bytes offset in struct
    nullptr,  // default value
    size_function__InstanceSegmentationInfo__classes,  // size() function pointer
    get_const_function__InstanceSegmentationInfo__classes,  // get_const(index) function pointer
    get_function__InstanceSegmentationInfo__classes,  // get(index) function pointer
    fetch_function__InstanceSegmentationInfo__classes,  // fetch(index, &value) function pointer
    assign_function__InstanceSegmentationInfo__classes,  // assign(index, value) function pointer
    resize_function__InstanceSegmentationInfo__classes  // resize(index) function pointer
  },
  {
    "image_source_monotonic_capture_time",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces::msg::InstanceSegmentationInfo, image_source_monotonic_capture_time),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "processing_node_monotonic_entry_time",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces::msg::InstanceSegmentationInfo, processing_node_monotonic_entry_time),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "processing_node_monotonic_publish_time",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(yolo_custom_interfaces::msg::InstanceSegmentationInfo, processing_node_monotonic_publish_time),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers InstanceSegmentationInfo_message_members = {
  "yolo_custom_interfaces::msg",  // message namespace
  "InstanceSegmentationInfo",  // message name
  7,  // number of fields
  sizeof(yolo_custom_interfaces::msg::InstanceSegmentationInfo),
  InstanceSegmentationInfo_message_member_array,  // message members
  InstanceSegmentationInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  InstanceSegmentationInfo_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t InstanceSegmentationInfo_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &InstanceSegmentationInfo_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace yolo_custom_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<yolo_custom_interfaces::msg::InstanceSegmentationInfo>()
{
  return &::yolo_custom_interfaces::msg::rosidl_typesupport_introspection_cpp::InstanceSegmentationInfo_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, yolo_custom_interfaces, msg, InstanceSegmentationInfo)() {
  return &::yolo_custom_interfaces::msg::rosidl_typesupport_introspection_cpp::InstanceSegmentationInfo_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
