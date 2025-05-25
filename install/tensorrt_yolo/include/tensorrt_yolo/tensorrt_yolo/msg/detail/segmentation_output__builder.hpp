// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#ifndef TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__BUILDER_HPP_
#define TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tensorrt_yolo/msg/detail/segmentation_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tensorrt_yolo
{

namespace msg
{

namespace builder
{

class Init_SegmentationOutput_detected_instance_ids
{
public:
  explicit Init_SegmentationOutput_detected_instance_ids(::tensorrt_yolo::msg::SegmentationOutput & msg)
  : msg_(msg)
  {}
  ::tensorrt_yolo::msg::SegmentationOutput detected_instance_ids(::tensorrt_yolo::msg::SegmentationOutput::_detected_instance_ids_type arg)
  {
    msg_.detected_instance_ids = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

class Init_SegmentationOutput_instance_class_ids
{
public:
  explicit Init_SegmentationOutput_instance_class_ids(::tensorrt_yolo::msg::SegmentationOutput & msg)
  : msg_(msg)
  {}
  Init_SegmentationOutput_detected_instance_ids instance_class_ids(::tensorrt_yolo::msg::SegmentationOutput::_instance_class_ids_type arg)
  {
    msg_.instance_class_ids = std::move(arg);
    return Init_SegmentationOutput_detected_instance_ids(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

class Init_SegmentationOutput_instance_confidences
{
public:
  explicit Init_SegmentationOutput_instance_confidences(::tensorrt_yolo::msg::SegmentationOutput & msg)
  : msg_(msg)
  {}
  Init_SegmentationOutput_instance_class_ids instance_confidences(::tensorrt_yolo::msg::SegmentationOutput::_instance_confidences_type arg)
  {
    msg_.instance_confidences = std::move(arg);
    return Init_SegmentationOutput_instance_class_ids(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

class Init_SegmentationOutput_instance_id_map
{
public:
  explicit Init_SegmentationOutput_instance_id_map(::tensorrt_yolo::msg::SegmentationOutput & msg)
  : msg_(msg)
  {}
  Init_SegmentationOutput_instance_confidences instance_id_map(::tensorrt_yolo::msg::SegmentationOutput::_instance_id_map_type arg)
  {
    msg_.instance_id_map = std::move(arg);
    return Init_SegmentationOutput_instance_confidences(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

class Init_SegmentationOutput_class_id_map
{
public:
  explicit Init_SegmentationOutput_class_id_map(::tensorrt_yolo::msg::SegmentationOutput & msg)
  : msg_(msg)
  {}
  Init_SegmentationOutput_instance_id_map class_id_map(::tensorrt_yolo::msg::SegmentationOutput::_class_id_map_type arg)
  {
    msg_.class_id_map = std::move(arg);
    return Init_SegmentationOutput_instance_id_map(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

class Init_SegmentationOutput_image_width
{
public:
  explicit Init_SegmentationOutput_image_width(::tensorrt_yolo::msg::SegmentationOutput & msg)
  : msg_(msg)
  {}
  Init_SegmentationOutput_class_id_map image_width(::tensorrt_yolo::msg::SegmentationOutput::_image_width_type arg)
  {
    msg_.image_width = std::move(arg);
    return Init_SegmentationOutput_class_id_map(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

class Init_SegmentationOutput_image_height
{
public:
  explicit Init_SegmentationOutput_image_height(::tensorrt_yolo::msg::SegmentationOutput & msg)
  : msg_(msg)
  {}
  Init_SegmentationOutput_image_width image_height(::tensorrt_yolo::msg::SegmentationOutput::_image_height_type arg)
  {
    msg_.image_height = std::move(arg);
    return Init_SegmentationOutput_image_width(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

class Init_SegmentationOutput_header
{
public:
  Init_SegmentationOutput_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SegmentationOutput_image_height header(::tensorrt_yolo::msg::SegmentationOutput::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_SegmentationOutput_image_height(msg_);
  }

private:
  ::tensorrt_yolo::msg::SegmentationOutput msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tensorrt_yolo::msg::SegmentationOutput>()
{
  return tensorrt_yolo::msg::builder::Init_SegmentationOutput_header();
}

}  // namespace tensorrt_yolo

#endif  // TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__BUILDER_HPP_
