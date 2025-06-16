// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from docs_turtlesim:msg/KeyedPose.idl
// generated code does not contain a copyright notice

#ifndef DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__TRAITS_HPP_
#define DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "docs_turtlesim/msg/detail/keyed_pose__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace docs_turtlesim
{

namespace msg
{

inline void to_flow_style_yaml(
  const KeyedPose & msg,
  std::ostream & out)
{
  out << "{";
  // member: turtle_id
  {
    out << "turtle_id: ";
    rosidl_generator_traits::value_to_yaml(msg.turtle_id, out);
    out << ", ";
  }

  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: theta
  {
    out << "theta: ";
    rosidl_generator_traits::value_to_yaml(msg.theta, out);
    out << ", ";
  }

  // member: linear_velocity
  {
    out << "linear_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.linear_velocity, out);
    out << ", ";
  }

  // member: angular_velocity
  {
    out << "angular_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.angular_velocity, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const KeyedPose & msg,
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

  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: theta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "theta: ";
    rosidl_generator_traits::value_to_yaml(msg.theta, out);
    out << "\n";
  }

  // member: linear_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.linear_velocity, out);
    out << "\n";
  }

  // member: angular_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angular_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.angular_velocity, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const KeyedPose & msg, bool use_flow_style = false)
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
  const docs_turtlesim::msg::KeyedPose & msg,
  std::ostream & out, size_t indentation = 0)
{
  docs_turtlesim::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use docs_turtlesim::msg::to_yaml() instead")]]
inline std::string to_yaml(const docs_turtlesim::msg::KeyedPose & msg)
{
  return docs_turtlesim::msg::to_yaml(msg);
}

template<>
inline const char * data_type<docs_turtlesim::msg::KeyedPose>()
{
  return "docs_turtlesim::msg::KeyedPose";
}

template<>
inline const char * name<docs_turtlesim::msg::KeyedPose>()
{
  return "docs_turtlesim/msg/KeyedPose";
}

template<>
struct has_fixed_size<docs_turtlesim::msg::KeyedPose>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<docs_turtlesim::msg::KeyedPose>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<docs_turtlesim::msg::KeyedPose>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DOCS_TURTLESIM__MSG__DETAIL__KEYED_POSE__TRAITS_HPP_
