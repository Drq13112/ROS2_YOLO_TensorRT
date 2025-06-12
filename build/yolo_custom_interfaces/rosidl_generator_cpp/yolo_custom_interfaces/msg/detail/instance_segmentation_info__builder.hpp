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

class Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time
{
public:
  explicit Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  ::yolo_custom_interfaces::msg::InstanceSegmentationInfo processing_node_monotonic_publish_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_processing_node_monotonic_publish_time_type arg)
  {
    msg_.processing_node_monotonic_publish_time = std::move(arg);
    return std::move(msg_);
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
  Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time processing_node_monotonic_entry_time(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_processing_node_monotonic_entry_time_type arg)
  {
    msg_.processing_node_monotonic_entry_time = std::move(arg);
    return Init_InstanceSegmentationInfo_processing_node_monotonic_publish_time(msg_);
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

class Init_InstanceSegmentationInfo_mask
{
public:
  explicit Init_InstanceSegmentationInfo_mask(::yolo_custom_interfaces::msg::InstanceSegmentationInfo & msg)
  : msg_(msg)
  {}
  Init_InstanceSegmentationInfo_scores mask(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_mask_type arg)
  {
    msg_.mask = std::move(arg);
    return Init_InstanceSegmentationInfo_scores(msg_);
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
  Init_InstanceSegmentationInfo_mask header(::yolo_custom_interfaces::msg::InstanceSegmentationInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_InstanceSegmentationInfo_mask(msg_);
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
