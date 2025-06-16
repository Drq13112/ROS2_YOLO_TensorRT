// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from docs_turtlesim:msg/KeyedPose.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__STRUCT_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__docs_turtlesim__msg__KeyedPose __attribute__((deprecated))
#else
# define DEPRECATED__docs_turtlesim__msg__KeyedPose __declspec(deprecated)
#endif

namespace docs_turtlesim
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct KeyedPose_
{
  using Type = KeyedPose_<ContainerAllocator>;

  explicit KeyedPose_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->turtle_id = 0l;
      this->x = 0.0;
      this->y = 0.0;
      this->theta = 0.0;
      this->linear_velocity = 0.0;
      this->angular_velocity = 0.0;
    }
  }

  explicit KeyedPose_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->turtle_id = 0l;
      this->x = 0.0;
      this->y = 0.0;
      this->theta = 0.0;
      this->linear_velocity = 0.0;
      this->angular_velocity = 0.0;
    }
  }

  // field types and members
  using _turtle_id_type =
    int32_t;
  _turtle_id_type turtle_id;
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _theta_type =
    double;
  _theta_type theta;
  using _linear_velocity_type =
    double;
  _linear_velocity_type linear_velocity;
  using _angular_velocity_type =
    double;
  _angular_velocity_type angular_velocity;

  // setters for named parameter idiom
  Type & set__turtle_id(
    const int32_t & _arg)
  {
    this->turtle_id = _arg;
    return *this;
  }
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__theta(
    const double & _arg)
  {
    this->theta = _arg;
    return *this;
  }
  Type & set__linear_velocity(
    const double & _arg)
  {
    this->linear_velocity = _arg;
    return *this;
  }
  Type & set__angular_velocity(
    const double & _arg)
  {
    this->angular_velocity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    docs_turtlesim::msg::KeyedPose_<ContainerAllocator> *;
  using ConstRawPtr =
    const docs_turtlesim::msg::KeyedPose_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      docs_turtlesim::msg::KeyedPose_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      docs_turtlesim::msg::KeyedPose_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__docs_turtlesim__msg__KeyedPose
    std::shared_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__docs_turtlesim__msg__KeyedPose
    std::shared_ptr<docs_turtlesim::msg::KeyedPose_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const KeyedPose_ & other) const
  {
    if (this->turtle_id != other.turtle_id) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->theta != other.theta) {
      return false;
    }
    if (this->linear_velocity != other.linear_velocity) {
      return false;
    }
    if (this->angular_velocity != other.angular_velocity) {
      return false;
    }
    return true;
  }
  bool operator!=(const KeyedPose_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct KeyedPose_

// alias to use template instance with default allocator
using KeyedPose =
  docs_turtlesim::msg::KeyedPose_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace docs_turtlesim

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__STRUCT_HPP_
