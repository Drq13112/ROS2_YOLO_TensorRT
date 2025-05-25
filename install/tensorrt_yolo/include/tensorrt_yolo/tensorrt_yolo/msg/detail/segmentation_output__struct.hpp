// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#ifndef TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__STRUCT_HPP_
#define TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__STRUCT_HPP_

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

#ifndef _WIN32
# define DEPRECATED__tensorrt_yolo__msg__SegmentationOutput __attribute__((deprecated))
#else
# define DEPRECATED__tensorrt_yolo__msg__SegmentationOutput __declspec(deprecated)
#endif

namespace tensorrt_yolo
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SegmentationOutput_
{
  using Type = SegmentationOutput_<ContainerAllocator>;

  explicit SegmentationOutput_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->image_height = 0ul;
      this->image_width = 0ul;
    }
  }

  explicit SegmentationOutput_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->image_height = 0ul;
      this->image_width = 0ul;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _image_height_type =
    uint32_t;
  _image_height_type image_height;
  using _image_width_type =
    uint32_t;
  _image_width_type image_width;
  using _class_id_map_type =
    std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>>;
  _class_id_map_type class_id_map;
  using _instance_id_map_type =
    std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>>;
  _instance_id_map_type instance_id_map;
  using _instance_confidences_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _instance_confidences_type instance_confidences;
  using _instance_class_ids_type =
    std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>>;
  _instance_class_ids_type instance_class_ids;
  using _detected_instance_ids_type =
    std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>>;
  _detected_instance_ids_type detected_instance_ids;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__image_height(
    const uint32_t & _arg)
  {
    this->image_height = _arg;
    return *this;
  }
  Type & set__image_width(
    const uint32_t & _arg)
  {
    this->image_width = _arg;
    return *this;
  }
  Type & set__class_id_map(
    const std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>> & _arg)
  {
    this->class_id_map = _arg;
    return *this;
  }
  Type & set__instance_id_map(
    const std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>> & _arg)
  {
    this->instance_id_map = _arg;
    return *this;
  }
  Type & set__instance_confidences(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->instance_confidences = _arg;
    return *this;
  }
  Type & set__instance_class_ids(
    const std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>> & _arg)
  {
    this->instance_class_ids = _arg;
    return *this;
  }
  Type & set__detected_instance_ids(
    const std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>> & _arg)
  {
    this->detected_instance_ids = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator> *;
  using ConstRawPtr =
    const tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tensorrt_yolo__msg__SegmentationOutput
    std::shared_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tensorrt_yolo__msg__SegmentationOutput
    std::shared_ptr<tensorrt_yolo::msg::SegmentationOutput_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SegmentationOutput_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->image_height != other.image_height) {
      return false;
    }
    if (this->image_width != other.image_width) {
      return false;
    }
    if (this->class_id_map != other.class_id_map) {
      return false;
    }
    if (this->instance_id_map != other.instance_id_map) {
      return false;
    }
    if (this->instance_confidences != other.instance_confidences) {
      return false;
    }
    if (this->instance_class_ids != other.instance_class_ids) {
      return false;
    }
    if (this->detected_instance_ids != other.detected_instance_ids) {
      return false;
    }
    return true;
  }
  bool operator!=(const SegmentationOutput_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SegmentationOutput_

// alias to use template instance with default allocator
using SegmentationOutput =
  tensorrt_yolo::msg::SegmentationOutput_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tensorrt_yolo

#endif  // TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__STRUCT_HPP_
