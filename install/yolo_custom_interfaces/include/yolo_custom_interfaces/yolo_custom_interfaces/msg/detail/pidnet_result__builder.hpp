// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from yolo_custom_interfaces:msg/PidnetResult.idl
// generated code does not contain a copyright notice

#ifndef YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__BUILDER_HPP_
#define YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "yolo_custom_interfaces/msg/detail/pidnet_result__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace yolo_custom_interfaces
{

namespace msg
{

namespace builder
{

class Init_PidnetResult_processing_node_monotonic_publish_time
{
public:
  explicit Init_PidnetResult_processing_node_monotonic_publish_time(::yolo_custom_interfaces::msg::PidnetResult & msg)
  : msg_(msg)
  {}
  ::yolo_custom_interfaces::msg::PidnetResult processing_node_monotonic_publish_time(::yolo_custom_interfaces::msg::PidnetResult::_processing_node_monotonic_publish_time_type arg)
  {
    msg_.processing_node_monotonic_publish_time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

class Init_PidnetResult_processing_node_inference_end_time
{
public:
  explicit Init_PidnetResult_processing_node_inference_end_time(::yolo_custom_interfaces::msg::PidnetResult & msg)
  : msg_(msg)
  {}
  Init_PidnetResult_processing_node_monotonic_publish_time processing_node_inference_end_time(::yolo_custom_interfaces::msg::PidnetResult::_processing_node_inference_end_time_type arg)
  {
    msg_.processing_node_inference_end_time = std::move(arg);
    return Init_PidnetResult_processing_node_monotonic_publish_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

class Init_PidnetResult_processing_node_inference_start_time
{
public:
  explicit Init_PidnetResult_processing_node_inference_start_time(::yolo_custom_interfaces::msg::PidnetResult & msg)
  : msg_(msg)
  {}
  Init_PidnetResult_processing_node_inference_end_time processing_node_inference_start_time(::yolo_custom_interfaces::msg::PidnetResult::_processing_node_inference_start_time_type arg)
  {
    msg_.processing_node_inference_start_time = std::move(arg);
    return Init_PidnetResult_processing_node_inference_end_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

class Init_PidnetResult_processing_node_monotonic_entry_time
{
public:
  explicit Init_PidnetResult_processing_node_monotonic_entry_time(::yolo_custom_interfaces::msg::PidnetResult & msg)
  : msg_(msg)
  {}
  Init_PidnetResult_processing_node_inference_start_time processing_node_monotonic_entry_time(::yolo_custom_interfaces::msg::PidnetResult::_processing_node_monotonic_entry_time_type arg)
  {
    msg_.processing_node_monotonic_entry_time = std::move(arg);
    return Init_PidnetResult_processing_node_inference_start_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

class Init_PidnetResult_image_source_monotonic_capture_time
{
public:
  explicit Init_PidnetResult_image_source_monotonic_capture_time(::yolo_custom_interfaces::msg::PidnetResult & msg)
  : msg_(msg)
  {}
  Init_PidnetResult_processing_node_monotonic_entry_time image_source_monotonic_capture_time(::yolo_custom_interfaces::msg::PidnetResult::_image_source_monotonic_capture_time_type arg)
  {
    msg_.image_source_monotonic_capture_time = std::move(arg);
    return Init_PidnetResult_processing_node_monotonic_entry_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

class Init_PidnetResult_packet_sequence_number
{
public:
  explicit Init_PidnetResult_packet_sequence_number(::yolo_custom_interfaces::msg::PidnetResult & msg)
  : msg_(msg)
  {}
  Init_PidnetResult_image_source_monotonic_capture_time packet_sequence_number(::yolo_custom_interfaces::msg::PidnetResult::_packet_sequence_number_type arg)
  {
    msg_.packet_sequence_number = std::move(arg);
    return Init_PidnetResult_image_source_monotonic_capture_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

class Init_PidnetResult_segmentation_map
{
public:
  explicit Init_PidnetResult_segmentation_map(::yolo_custom_interfaces::msg::PidnetResult & msg)
  : msg_(msg)
  {}
  Init_PidnetResult_packet_sequence_number segmentation_map(::yolo_custom_interfaces::msg::PidnetResult::_segmentation_map_type arg)
  {
    msg_.segmentation_map = std::move(arg);
    return Init_PidnetResult_packet_sequence_number(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

class Init_PidnetResult_header
{
public:
  Init_PidnetResult_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PidnetResult_segmentation_map header(::yolo_custom_interfaces::msg::PidnetResult::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PidnetResult_segmentation_map(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::PidnetResult msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::yolo_custom_interfaces::msg::PidnetResult>()
{
  return yolo_custom_interfaces::msg::builder::Init_PidnetResult_header();
}

}  // namespace yolo_custom_interfaces

#endif  // YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__BUILDER_HPP_
