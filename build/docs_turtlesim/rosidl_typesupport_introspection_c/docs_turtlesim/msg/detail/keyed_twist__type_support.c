// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from docs_turtlesim:msg/KeyedTwist.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "docs_turtlesim/msg/detail/keyed_twist__rosidl_typesupport_introspection_c.h"
#include "docs_turtlesim/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "docs_turtlesim/msg/detail/keyed_twist__functions.h"
#include "docs_turtlesim/msg/detail/keyed_twist__struct.h"


// Include directives for member types
// Member `linear`
// Member `angular`
#include "docs_turtlesim/msg/vector3.h"
// Member `linear`
// Member `angular`
#include "docs_turtlesim/msg/detail/vector3__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  docs_turtlesim__msg__KeyedTwist__init(message_memory);
}

void docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_fini_function(void * message_memory)
{
  docs_turtlesim__msg__KeyedTwist__fini(message_memory);
}


static bool docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_key_members_array[3] = {
  true,
  false,
  false
};

static rosidl_typesupport_introspection_c__MessageMember docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_member_array[3] = {
  {
    "turtle_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedTwist, turtle_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "linear",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedTwist, linear),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "angular",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedTwist, angular),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_members = {
  "docs_turtlesim__msg",  // message namespace
  "KeyedTwist",  // message name
  3,  // number of fields
  sizeof(docs_turtlesim__msg__KeyedTwist),
  docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_member_array,  // message members
  docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_init_function,  // function to initialize message memory (memory has to be allocated)
  docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_fini_function,  // function to terminate message instance (will not free memory)
  docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_key_members_array // mapping to each field to know whether it is keyed or not
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_type_support_handle = {
  0,
  &docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_docs_turtlesim
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, docs_turtlesim, msg, KeyedTwist)() {
  docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, docs_turtlesim, msg, Vector3)();
  docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, docs_turtlesim, msg, Vector3)();
  if (!docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_type_support_handle.typesupport_identifier) {
    docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier_v2;
  }
  return &docs_turtlesim__msg__KeyedTwist__rosidl_typesupport_introspection_c__KeyedTwist_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
