// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from yolo_custom_interfaces:msg/PidnetResult.idl
// generated code does not contain a copyright notice

#ifndef YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__STRUCT_HPP_
#define YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'segmentation_map'
#include "sensor_msgs/msg/detail/image__struct.hpp"
// Member 'image_source_monotonic_capture_time'
// Member 'processing_node_monotonic_entry_time'
// Member 'processing_node_inference_start_time'
// Member 'processing_node_inference_end_time'
// Member 'processing_node_monotonic_publish_time'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__yolo_custom_interfaces__msg__PidnetResult __attribute__((deprecated))
#else
# define DEPRECATED__yolo_custom_interfaces__msg__PidnetResult __declspec(deprecated)
#endif

namespace yolo_custom_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PidnetResult_
{
  using Type = PidnetResult_<ContainerAllocator>;

  explicit PidnetResult_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    segmentation_map(_init),
    image_source_monotonic_capture_time(_init),
    processing_node_monotonic_entry_time(_init),
    processing_node_inference_start_time(_init),
    processing_node_inference_end_time(_init),
    processing_node_monotonic_publish_time(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->packet_sequence_number = 0ull;
    }
  }

  explicit PidnetResult_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    segmentation_map(_alloc, _init),
    image_source_monotonic_capture_time(_alloc, _init),
    processing_node_monotonic_entry_time(_alloc, _init),
    processing_node_inference_start_time(_alloc, _init),
    processing_node_inference_end_time(_alloc, _init),
    processing_node_monotonic_publish_time(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->packet_sequence_number = 0ull;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _segmentation_map_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _segmentation_map_type segmentation_map;
  using _packet_sequence_number_type =
    uint64_t;
  _packet_sequence_number_type packet_sequence_number;
  using _image_source_monotonic_capture_time_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _image_source_monotonic_capture_time_type image_source_monotonic_capture_time;
  using _processing_node_monotonic_entry_time_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _processing_node_monotonic_entry_time_type processing_node_monotonic_entry_time;
  using _processing_node_inference_start_time_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _processing_node_inference_start_time_type processing_node_inference_start_time;
  using _processing_node_inference_end_time_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _processing_node_inference_end_time_type processing_node_inference_end_time;
  using _processing_node_monotonic_publish_time_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _processing_node_monotonic_publish_time_type processing_node_monotonic_publish_time;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__segmentation_map(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->segmentation_map = _arg;
    return *this;
  }
  Type & set__packet_sequence_number(
    const uint64_t & _arg)
  {
    this->packet_sequence_number = _arg;
    return *this;
  }
  Type & set__image_source_monotonic_capture_time(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->image_source_monotonic_capture_time = _arg;
    return *this;
  }
  Type & set__processing_node_monotonic_entry_time(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->processing_node_monotonic_entry_time = _arg;
    return *this;
  }
  Type & set__processing_node_inference_start_time(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->processing_node_inference_start_time = _arg;
    return *this;
  }
  Type & set__processing_node_inference_end_time(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->processing_node_inference_end_time = _arg;
    return *this;
  }
  Type & set__processing_node_monotonic_publish_time(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->processing_node_monotonic_publish_time = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator> *;
  using ConstRawPtr =
    const yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__yolo_custom_interfaces__msg__PidnetResult
    std::shared_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__yolo_custom_interfaces__msg__PidnetResult
    std::shared_ptr<yolo_custom_interfaces::msg::PidnetResult_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PidnetResult_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->segmentation_map != other.segmentation_map) {
      return false;
    }
    if (this->packet_sequence_number != other.packet_sequence_number) {
      return false;
    }
    if (this->image_source_monotonic_capture_time != other.image_source_monotonic_capture_time) {
      return false;
    }
    if (this->processing_node_monotonic_entry_time != other.processing_node_monotonic_entry_time) {
      return false;
    }
    if (this->processing_node_inference_start_time != other.processing_node_inference_start_time) {
      return false;
    }
    if (this->processing_node_inference_end_time != other.processing_node_inference_end_time) {
      return false;
    }
    if (this->processing_node_monotonic_publish_time != other.processing_node_monotonic_publish_time) {
      return false;
    }
    return true;
  }
  bool operator!=(const PidnetResult_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PidnetResult_

// alias to use template instance with default allocator
using PidnetResult =
  yolo_custom_interfaces::msg::PidnetResult_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace yolo_custom_interfaces

#endif  // YOLO_CUSTOM_INTERFACES__MSG__DETAIL__PIDNET_RESULT__STRUCT_HPP_
