// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
// generated code does not contain a copyright notice

#ifndef YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__BUILDER_HPP_
#define YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace yolo_custom_interfaces
{

namespace msg
{

namespace builder
{

class Init_InstanceSegmentationInfo_packet_sequence_number
{
public:
  explicit Init_InstanceSegmentationInfo_packet_sequence_number(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo packet_sequence_number(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_packet_sequence_number_type arg)
  {
    msg_.packet_sequence_number = std::move(arg);
    return std::move(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time
{
public:
  explicit Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_packet_sequence_number processing_node_monotonic_publish_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_processing_node_monotonic_publish_time_type arg)
  {
    msg_.processing_node_monotonic_publish_time = std::move(arg);
    return Init_InstanceSegmentationInfo_packet_sequence_number(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_processing_node_inference_end_time
{
public:
  explicit Init_InstanceSegmentationInfo_processing_node_inference_end_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time processing_node_inference_end_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_processing_node_inference_end_time_type arg)
  {
    msg_.processing_node_inference_end_time = std::move(arg);
    return Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_processing_node_inference_start_time
{
public:
  explicit Init_InstanceSegmentationInfo_processing_node_inference_start_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_processing_node_inference_end_time processing_node_inference_start_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_processing_node_inference_start_time_type arg)
  {
    msg_.processing_node_inference_start_time = std::move(arg);
    return Init_InstanceSegmentationInfo_processing_node_inference_end_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_processing_node_monotonic_entry_time
{
public:
  explicit Init_InstanceSegmentationInfo_processing_node_monotonic_entry_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_processing_node_inference_start_time processing_node_monotonic_entry_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_processing_node_monotonic_entry_time_type arg)
  {
    msg_.processing_node_monotonic_entry_time = std::move(arg);
    return Init_InstanceSegmentationInfo_processing_node_inference_start_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_image_source_monotonic_capture_time
{
public:
  explicit Init_InstanceSegmentationInfo_image_source_monotonic_capture_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_processing_node_monotonic_entry_time image_source_monotonic_capture_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_image_source_monotonic_capture_time_type arg)
  {
    msg_.image_source_monotonic_capture_time = std::move(arg);
    return Init_InstanceSegmentationInfo_processing_node_monotonic_entry_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_classes
{
public:
  explicit Init_InstanceSegmentationInfo_classes(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_image_source_monotonic_capture_time classes(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_classes_type arg)
  {
    msg_.classes = std::move(arg);
    return Init_InstanceSegmentationInfo_image_source_monotonic_capture_time(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_scores
{
public:
  explicit Init_InstanceSegmentationInfo_scores(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_classes scores(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_scores_type arg)
  {
    msg_.scores = std::move(arg);
    return Init_InstanceSegmentationInfo_classes(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_mask_data
{
public:
  explicit Init_InstanceSegmentationInfo_mask_data(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_scores mask_data(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_mask_data_type arg)
  {
    msg_.mask_data = std::move(arg);
    return Init_InstanceSegmentationInfo_scores(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_mask_height
{
public:
  explicit Init_InstanceSegmentationInfo_mask_height(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_mask_data mask_height(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_mask_height_type arg)
  {
    msg_.mask_height = std::move(arg);
    return Init_InstanceSegmentationInfo_mask_data(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_mask_width
{
public:
  explicit Init_InstanceSegmentationInfo_mask_width(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_mask_height mask_width(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_mask_width_type arg)
  {
    msg_.mask_width = std::move(arg);
    return Init_InstanceSegmentationInfo_mask_height(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

class Init_InstanceSegmentationInfo_header
{
public:
  Init_InstanceSegmentationInfo_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_InstanceSegmentationInfo_mask_width header(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_InstanceSegmentationInfo_mask_width(msg_);
  }

private:
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::yolo_custom_interfaces::msg::InstanceSegmentationInfo>()
{
  return yolo_custom_interfaces::msg::builder::Init_InstanceSegmentationInfo_header();
}

}  // namespace yolo_custom_interfaces

#endif  // YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__BUILDER_HPP_
