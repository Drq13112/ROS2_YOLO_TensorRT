// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from docs_turtlesim:msg/KeyedTwist.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__BUILDER_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "docs_turtlesim/msg/detail/keyed_twist__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace docs_turtlesim
{

namespace msg
{

namespace builder
{

class Init_KeyedTwist_angular
{
public:
  explicit Init_KeyedTwist_angular(::docs_turtlesim::msg::KeyedTwist & msg)
  : msg_(msg)
  {}
  ::docs_turtlesim::msg::KeyedTwist angular(::docs_turtlesim::msg::KeyedTwist::_angular_type arg)
  {
    msg_.angular = std::move(arg);
    return std::move(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedTwist msg_;
};

class Init_KeyedTwist_linear
{
public:
  explicit Init_KeyedTwist_linear(::docs_turtlesim::msg::KeyedTwist & msg)
  : msg_(msg)
  {}
  Init_KeyedTwist_angular linear(::docs_turtlesim::msg::KeyedTwist::_linear_type arg)
  {
    msg_.linear = std::move(arg);
    return Init_KeyedTwist_angular(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedTwist msg_;
};

class Init_KeyedTwist_turtle_id
{
public:
  Init_KeyedTwist_turtle_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_KeyedTwist_linear turtle_id(::docs_turtlesim::msg::KeyedTwist::_turtle_id_type arg)
  {
    msg_.turtle_id = std::move(arg);
    return Init_KeyedTwist_linear(msg_);
  }

private:
  ::docs_turtlesim::msg::KeyedTwist msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::docs_turtlesim::msg::KeyedTwist>()
{
  return docs_turtlesim::msg::builder::Init_KeyedTwist_turtle_id();
}

}  // namespace docs_turtlesim

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__BUILDER_HPP_
