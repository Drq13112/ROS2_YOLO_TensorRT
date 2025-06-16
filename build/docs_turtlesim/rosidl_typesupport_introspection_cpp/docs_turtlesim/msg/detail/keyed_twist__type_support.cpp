// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from docs_turtlesim:msg/KeyedTwist.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "docs_turtlesim/msg/detail/keyed_twist__struct.hpp"
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

void KeyedTwist_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) docs_turtlesim::msg::KeyedTwist(_init);
}

void KeyedTwist_fini_function(void * message_memory)
{
  auto typed_message = static_cast<docs_turtlesim::msg::KeyedTwist *>(message_memory);
  typed_message->~KeyedTwist();
}


static const bool KeyedTwist_key_members_array[3] = {
  true,
  false,
  false
};

static const ::rosidl_typesupport_introspection_cpp::MessageMember KeyedTwist_message_member_array[3] = {
  {
    "turtle_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim::msg::KeyedTwist, turtle_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "linear",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<docs_turtlesim::msg::Vector3>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim::msg::KeyedTwist, linear),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "angular",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<docs_turtlesim::msg::Vector3>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim::msg::KeyedTwist, angular),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers KeyedTwist_message_members = {
  "docs_turtlesim::msg",  // message namespace
  "KeyedTwist",  // message name
  3,  // number of fields
  sizeof(docs_turtlesim::msg::KeyedTwist),
  KeyedTwist_message_member_array,  // message members
  KeyedTwist_init_function,  // function to initialize message memory (memory has to be allocated)
  KeyedTwist_fini_function,  // function to terminate message instance (will not free memory)
  KeyedTwist_key_members_array // mapping to each field to know whether it is keyed or not
};

static const rosidl_message_type_support_t KeyedTwist_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier_v2,
  &KeyedTwist_message_members,
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
get_message_type_support_handle<docs_turtlesim::msg::KeyedTwist>()
{
  return &::docs_turtlesim::msg::rosidl_typesupport_introspection_cpp::KeyedTwist_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, docs_turtlesim, msg, KeyedTwist)() {
  return &::docs_turtlesim::msg::rosidl_typesupport_introspection_cpp::KeyedTwist_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
