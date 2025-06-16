// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from docs_turtlesim:msg/KeyedPose.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__BUILDER_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "docs_turtlesim/msg/detail/keyed_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace docs_turtlesim
{

namespace msg
{

namespace builder
{

class Init_KeyedPose_angular_velocity
{
public:
  explicit Init_KeyedPose_angular_velocity(::docs_turtlesim::msg::KeyedPose & msg)
  : msg_(msg)
  {}
  ::docs_turtlesim::msg::KeyedPose angular_velocity(::docs_turtlesim::msg::KeyedPose::_angular_velocity_type arg)
  {
    msg_.angular_velocity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedPose msg_;
};

class Init_KeyedPose_linear_velocity
{
public:
  explicit Init_KeyedPose_linear_velocity(::docs_turtlesim::msg::KeyedPose & msg)
  : msg_(msg)
  {}
  Init_KeyedPose_angular_velocity linear_velocity(::docs_turtlesim::msg::KeyedPose::_linear_velocity_type arg)
  {
    msg_.linear_velocity = std::move(arg);
    return Init_KeyedPose_angular_velocity(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedPose msg_;
};

class Init_KeyedPose_theta
{
public:
  explicit Init_KeyedPose_theta(::docs_turtlesim::msg::KeyedPose & msg)
  : msg_(msg)
  {}
  Init_KeyedPose_linear_velocity theta(::docs_turtlesim::msg::KeyedPose::_theta_type arg)
  {
    msg_.theta = std::move(arg);
    return Init_KeyedPose_linear_velocity(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedPose msg_;
};

class Init_KeyedPose_y
{
public:
  explicit Init_KeyedPose_y(::docs_turtlesim::msg::KeyedPose & msg)
  : msg_(msg)
  {}
  Init_KeyedPose_theta y(::docs_turtlesim::msg::KeyedPose::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_KeyedPose_theta(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedPose msg_;
};

class Init_KeyedPose_x
{
public:
  explicit Init_KeyedPose_x(::docs_turtlesim::msg::KeyedPose & msg)
  : msg_(msg)
  {}
  Init_KeyedPose_y x(::docs_turtlesim::msg::KeyedPose::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_KeyedPose_y(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedPose msg_;
};

class Init_KeyedPose_turtle_id
{
public:
  Init_KeyedPose_turtle_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_KeyedPose_x turtle_id(::docs_turtlesim::msg::KeyedPose::_turtle_id_type arg)
  {
    msg_.turtle_id = std::move(arg);
    return Init_KeyedPose_x(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedPose msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::docs_turtlesim::msg::KeyedPose>()
{
  return docs_turtlesim::msg::builder::Init_KeyedPose_turtle_id();
}

}  // namespace docs_turtlesim

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__BUILDER_HPP_
