// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from docs_turtlesim:msg/Vector3.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "docs_turtlesim/msg/detail/vector3__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace docs_turtlesim
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Vector3_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) docs_turtlesim::msg::Vector3(_init);
}

void Vector3_fini_function(void * message_memory)
{
  auto typed_message = static_cast<docs_turtlesim::msg::Vector3 *>(message_memory);
  typed_message->~Vector3();
}


static const bool Vector3_key_members_array[3] = {
  false,
  false,
  false
};

static const ::rosidl_typesupport_introspection_cpp::MessageMember Vector3_message_member_array[3] = {
  {
    "x",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim::msg::Vector3, x),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "y",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim::msg::Vector3, y),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "z",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim::msg::Vector3, z),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Vector3_message_members = {
  "docs_turtlesim::msg",  // message namespace
  "Vector3",  // message name
  3,  // number of fields
  sizeof(docs_turtlesim::msg::Vector3),
  Vector3_message_member_array,  // message members
  Vector3_init_function,  // function to initialize message memory (memory has to be allocated)
  Vector3_fini_function,  // function to terminate message instance (will not free memory)
  Vector3_key_members_array // mapping to each field to know whether it is keyed or not
};

static const rosidl_message_type_support_t Vector3_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier_v2,
  &Vector3_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace docs_turtlesim


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<docs_turtlesim::msg::Vector3>()
{
  return &::docs_turtlesim::msg::rosidl_typesupport_introspection_cpp::Vector3_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, docs_turtlesim, msg, Vector3)() {
  return &::docs_turtlesim::msg::rosidl_typesupport_introspection_cpp::Vector3_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
