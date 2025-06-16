// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from docs_turtlesim:msg/Vector3.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__VECTOR3__BUILDER_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__VECTOR3__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "docs_turtlesim/msg/detail/vector3__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace docs_turtlesim
{

namespace msg
{

namespace builder
{

class Init_Vector3_z
{
public:
  explicit Init_Vector3_z(::docs_turtlesim::msg::Vector3 & msg)
  : msg_(msg)
  {}
  ::docs_turtlesim::msg::Vector3 z(::docs_turtlesim::msg::Vector3::_z_type arg)
  {
    msg_.z = std::move(arg);
    return std::move(msg_);
  }

private:
  ::docs_turtlesim::msg::Vector3 msg_;
};

class Init_Vector3_y
{
public:
  explicit Init_Vector3_y(::docs_turtlesim::msg::Vector3 & msg)
  : msg_(msg)
  {}
  Init_Vector3_z y(::docs_turtlesim::msg::Vector3::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_Vector3_z(msg_);
  }

private:
  ::docs_turtlesim::msg::Vector3 msg_;
};

class Init_Vector3_x
{
public:
  Init_Vector3_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Vector3_y x(::docs_turtlesim::msg::Vector3::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_Vector3_y(msg_);
  }

private:
  ::docs_turtlesim::msg::Vector3 msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::docs_turtlesim::msg::Vector3>()
{
  return docs_turtlesim::msg::builder::Init_Vector3_x();
}

}  // namespace docs_turtlesim

#endif  // DOCS_TURTLESIM__MSG__DETAIL__VECTOR3__BUILDER_HPP_
