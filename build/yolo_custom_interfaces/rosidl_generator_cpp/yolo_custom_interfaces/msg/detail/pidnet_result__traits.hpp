// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from yolo_custom_interfaces:msg/PidnetResult.idl
// generated code does not contain a copyright notice

#ifndef YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__TRAITS_HPP_
#define YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "yolo_custom_interfaces/msg/detail/pidnet_result__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'segmentation_map'
#include "sensor_msgs/msg/detail/image__traits.hpp"
// Member 'image_source_monotonic_capture_time'
// Member 'processing_node_monotonic_entry_time'
// Member 'processing_node_inference_start_time'
// Member 'processing_node_inference_end_time'
// Member 'processing_node_monotonic_publish_time'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace yolo_custom_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PidnetResult & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: segmentation_map
  {
    out << "segmentation_map: ";
    to_flow_style_yaml(msg.segmentation_map, out);
    out << ", ";
  }

  // member: packet_sequence_number
  {
    out << "packet_sequence_number: ";
    rosidl_generator_traits::value_to_yaml(msg.packet_sequence_number, out);
    out << ", ";
  }

  // member: image_source_monotonic_capture_time
  {
    out << "image_source_monotonic_capture_time: ";
    to_flow_style_yaml(msg.image_source_monotonic_capture_time, out);
    out << ", ";
  }

  // member: processing_node_monotonic_entry_time
  {
    out << "processing_node_monotonic_entry_time: ";
    to_flow_style_yaml(msg.processing_node_monotonic_entry_time, out);
    out << ", ";
  }

  // member: processing_node_inference_start_time
  {
    out << "processing_node_inference_start_time: ";
    to_flow_style_yaml(msg.processing_node_inference_start_time, out);
    out << ", ";
  }

  // member: processing_node_inference_end_time
  {
    out << "processing_node_inference_end_time: ";
    to_flow_style_yaml(msg.processing_node_inference_end_time, out);
    out << ", ";
  }

  // member: processing_node_monotonic_publish_time
  {
    out << "processing_node_monotonic_publish_time: ";
    to_flow_style_yaml(msg.processing_node_monotonic_publish_time, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PidnetResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: segmentation_map
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "segmentation_map:\n";
    to_block_style_yaml(msg.segmentation_map, out, indentation + 2);
  }

  // member: packet_sequence_number
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "packet_sequence_number: ";
    rosidl_generator_traits::value_to_yaml(msg.packet_sequence_number, out);
    out << "\n";
  }

  // member: image_source_monotonic_capture_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "image_source_monotonic_capture_time:\n";
    to_block_style_yaml(msg.image_source_monotonic_capture_time, out, indentation + 2);
  }

  // member: processing_node_monotonic_entry_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "processing_node_monotonic_entry_time:\n";
    to_block_style_yaml(msg.processing_node_monotonic_entry_time, out, indentation + 2);
  }

  // member: processing_node_inference_start_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "processing_node_inference_start_time:\n";
    to_block_style_yaml(msg.processing_node_inference_start_time, out, indentation + 2);
  }

  // member: processing_node_inference_end_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "processing_node_inference_end_time:\n";
    to_block_style_yaml(msg.processing_node_inference_end_time, out, indentation + 2);
  }

  // member: processing_node_monotonic_publish_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "processing_node_monotonic_publish_time:\n";
    to_block_style_yaml(msg.processing_node_monotonic_publish_time, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PidnetResult & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace yolo_custom_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use yolo_custom_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const yolo_custom_interfaces::msg::PidnetResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  yolo_custom_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use yolo_custom_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const yolo_custom_interfaces::msg::PidnetResult & msg)
{
  return yolo_custom_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<yolo_custom_interfaces::msg::PidnetResult>()
{
  return "yolo_custom_interfaces::msg::PidnetResult";
}

template<>
inline const char * name<yolo_custom_interfaces::msg::PidnetResult>()
{
  return "yolo_custom_interfaces/msg/PidnetResult";
}

template<>
struct has_fixed_size<yolo_custom_interfaces::msg::PidnetResult>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value && has_fixed_size<sensor_msgs::msg::Image>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<yolo_custom_interfaces::msg::PidnetResult>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value && has_bounded_size<sensor_msgs::msg::Image>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<yolo_custom_interfaces::msg::PidnetResult>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__TRAITS_HPP_
