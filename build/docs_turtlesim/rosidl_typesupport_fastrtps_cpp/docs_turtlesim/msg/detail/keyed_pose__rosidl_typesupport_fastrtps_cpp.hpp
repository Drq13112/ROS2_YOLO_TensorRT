// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from docs_turtlesim:msg/KeyedPose.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "docs_turtlesim/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "docs_turtlesim/msg/detail/keyed_pose__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"


namespace docs_turtlesim
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_serialize(
  const docs_turtlesim::msg::KeyedPose & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  docs_turtlesim::msg::KeyedPose & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
get_serialized_size(
  const docs_turtlesim::msg::KeyedPose & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
max_serialized_size_KeyedPose(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_serialize_key(
  const docs_turtlesim::msg::KeyedPose & ros_message,
  eprosima::fastcdr::Cdr &);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_deserialize_key(
  eprosima::fastcdr::Cdr & cdr,
  docs_turtlesim::msg::KeyedPose & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
get_serialized_size_key(
  const docs_turtlesim::msg::KeyedPose & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
max_serialized_size_key_KeyedPose(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace docs_turtlesim

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, docs_turtlesim, msg, KeyedPose)();

#ifdef __cplusplus
}
#endif

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
