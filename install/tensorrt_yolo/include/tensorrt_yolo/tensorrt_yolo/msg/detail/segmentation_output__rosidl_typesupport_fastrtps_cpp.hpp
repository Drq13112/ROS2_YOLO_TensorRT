// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#ifndef TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "tensorrt_yolo/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "tensorrt_yolo/msg/detail/segmentation_output__struct.hpp"

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

namespace tensorrt_yolo
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_tensorrt_yolo
cdr_serialize(
  const tensorrt_yolo::msg::SegmentationOutput & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_tensorrt_yolo
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  tensorrt_yolo::msg::SegmentationOutput & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_tensorrt_yolo
get_serialized_size(
  const tensorrt_yolo::msg::SegmentationOutput & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_tensorrt_yolo
max_serialized_size_SegmentationOutput(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace tensorrt_yolo

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_tensorrt_yolo
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, tensorrt_yolo, msg, SegmentationOutput)();

#ifdef __cplusplus
}
#endif

#endif  // TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
