// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from docs_turtlesim:msg/KeyedTwist.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__TRAITS_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "docs_turtlesim/msg/detail/keyed_twist__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'linear'
// Member 'angular'
#include "docs_turtlesim/msg/detail/vector3__traits.hpp"

namespace docs_turtlesim
{

namespace msg
{

inline void to_flow_style_yaml(
  const KeyedTwist & msg,
  std::ostream & out)
{
  out << "{";
  // member: turtle_id
  {
    out << "turtle_id: ";
    rosidl_generator_traits::value_to_yaml(msg.turtle_id, out);
    out << ", ";
  }

  // member: linear
  {
    out << "linear: ";
    to_flow_style_yaml(msg.linear, out);
    out << ", ";
  }

  // member: angular
  {
    out << "angular: ";
    to_flow_style_yaml(msg.angular, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const KeyedTwist & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: turtle_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "turtle_id: ";
    rosidl_generator_traits::value_to_yaml(msg.turtle_id, out);
    out << "\n";
  }

  // member: linear
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear:\n";
    to_block_style_yaml(msg.linear, out, indentation + 2);
  }

  // member: angular
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angular:\n";
    to_block_style_yaml(msg.angular, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const KeyedTwist & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace docs_turtlesim

namespace rosidl_generator_traits
{

[[deprecated("use docs_turtlesim::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const docs_turtlesim::msg::KeyedTwist & msg,
  std::ostream & out, size_t indentation = 0)
{
  docs_turtlesim::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use docs_turtlesim::msg::to_yaml() instead")]]
inline std::string to_yaml(const docs_turtlesim::msg::KeyedTwist & msg)
{
  return docs_turtlesim::msg::to_yaml(msg);
}

template<>
inline const char * data_type<docs_turtlesim::msg::KeyedTwist>()
{
  return "docs_turtlesim::msg::KeyedTwist";
}

template<>
inline const char * name<docs_turtlesim::msg::KeyedTwist>()
{
  return "docs_turtlesim/msg/KeyedTwist";
}

template<>
struct has_fixed_size<docs_turtlesim::msg::KeyedTwist>
  : std::integral_constant<bool, has_fixed_size<docs_turtlesim::msg::Vector3>::value> {};

template<>
struct has_bounded_size<docs_turtlesim::msg::KeyedTwist>
  : std::integral_constant<bool, has_bounded_size<docs_turtlesim::msg::Vector3>::value> {};

template<>
struct is_message<docs_turtlesim::msg::KeyedTwist>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_TWIST__TRAITS_HPP_
