// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from docs_turtlesim:msg/KeyedTwist.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__STRUCT_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'linear'
// Member 'angular'
#include "docs_turtlesim/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__docs_turtlesim__msg__KeyedTwist __attribute__((deprecated))
#else
# define DEPRECATED__docs_turtlesim__msg__KeyedTwist __declspec(deprecated)
#endif

namespace docs_turtlesim
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct KeyedTwist_
{
  using Type = KeyedTwist_<ContainerAllocator>;

  explicit KeyedTwist_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : linear(_init),
    angular(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->turtle_id = 0l;
    }
  }

  explicit KeyedTwist_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : linear(_alloc, _init),
    angular(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->turtle_id = 0l;
    }
  }

  // field types and members
  using _turtle_id_type =
    int32_t;
  _turtle_id_type turtle_id;
  using _linear_type =
    docs_turtlesim::msg::Vector3_<ContainerAllocator>;
  _linear_type linear;
  using _angular_type =
    docs_turtlesim::msg::Vector3_<ContainerAllocator>;
  _angular_type angular;

  // setters for named parameter idiom
  Type & set__turtle_id(
    const int32_t & _arg)
  {
    this->turtle_id = _arg;
    return *this;
  }
  Type & set__linear(
    const docs_turtlesim::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->linear = _arg;
    return *this;
  }
  Type & set__angular(
    const docs_turtlesim::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->angular = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    docs_turtlesim::msg::KeyedTwist_<ContainerAllocator> *;
  using ConstRawPtr =
    const docs_turtlesim::msg::KeyedTwist_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      docs_turtlesim::msg::KeyedTwist_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      docs_turtlesim::msg::KeyedTwist_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__docs_turtlesim__msg__KeyedTwist
    std::shared_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__docs_turtlesim__msg__KeyedTwist
    std::shared_ptr<docs_turtlesim::msg::KeyedTwist_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const KeyedTwist_ & other) const
  {
    if (this->turtle_id != other.turtle_id) {
      return false;
    }
    if (this->linear != other.linear) {
      return false;
    }
    if (this->angular != other.angular) {
      return false;
    }
    return true;
  }
  bool operator!=(const KeyedTwist_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct KeyedTwist_

// alias to use template instance with default allocator
using KeyedTwist =
  docs_turtlesim::msg::KeyedTwist_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace docs_turtlesim

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__STRUCT_HPP_
