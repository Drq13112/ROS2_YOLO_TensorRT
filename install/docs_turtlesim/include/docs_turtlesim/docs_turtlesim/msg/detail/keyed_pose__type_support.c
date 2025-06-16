// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from docs_turtlesim:msg/KeyedPose.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "docs_turtlesim/msg/detail/keyed_pose__rosidl_typesupport_introspection_c.h"
#include "docs_turtlesim/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "docs_turtlesim/msg/detail/keyed_pose__functions.h"
#include "docs_turtlesim/msg/detail/keyed_pose__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  docs_turtlesim__msg__KeyedPose__init(message_memory);
}

void docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_fini_function(void * message_memory)
{
  docs_turtlesim__msg__KeyedPose__fini(message_memory);
}


static bool docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_key_members_array[6] = {
  true,
  false,
  false,
  false,
  false,
  false
};

static rosidl_typesupport_introspection_c__MessageMember docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_member_array[6] = {
  {
    "turtle_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedPose, turtle_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedPose, x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedPose, y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "theta",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedPose, theta),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "linear_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedPose, linear_velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "angular_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(docs_turtlesim__msg__KeyedPose, angular_velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_members = {
  "docs_turtlesim__msg",  // message namespace
  "KeyedPose",  // message name
  6,  // number of fields
  sizeof(docs_turtlesim__msg__KeyedPose),
  docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_member_array,  // message members
  docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_init_function,  // function to initialize message memory (memory has to be allocated)
  docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_fini_function,  // function to terminate message instance (will not free memory)
  docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_key_members_array // mapping to each field to know whether it is keyed or not
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_type_support_handle = {
  0,
  &docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_docs_turtlesim
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, docs_turtlesim, msg, KeyedPose)() {
  if (!docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_type_support_handle.typesupport_identifier) {
    docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier_v2;
  }
  return &docs_turtlesim__msg__KeyedPose__rosidl_typesupport_introspection_c__KeyedPose_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
