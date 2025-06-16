// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from docs_turtlesim:msg/KeyedPose.idl
// generated code does not contain a copyright notice
#include "docs_turtlesim/msg/detail/keyed_pose__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "docs_turtlesim/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "docs_turtlesim/msg/detail/keyed_pose__struct.h"
#include "docs_turtlesim/msg/detail/keyed_pose__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _KeyedPose__ros_msg_type = docs_turtlesim__msg__KeyedPose;


static bool _KeyedPose__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _KeyedPose__ros_msg_type * ros_message = static_cast<const _KeyedPose__ros_msg_type *>(untyped_ros_message);
  // Field name: turtle_id
  {
    cdr << ros_message->turtle_id;
  }

  // Field name: x
  {
    cdr << ros_message->x;
  }

  // Field name: y
  {
    cdr << ros_message->y;
  }

  // Field name: theta
  {
    cdr << ros_message->theta;
  }

  // Field name: linear_velocity
  {
    cdr << ros_message->linear_velocity;
  }

  // Field name: angular_velocity
  {
    cdr << ros_message->angular_velocity;
  }

  return true;
}

static bool _KeyedPose__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _KeyedPose__ros_msg_type * ros_message = static_cast<_KeyedPose__ros_msg_type *>(untyped_ros_message);
  // Field name: turtle_id
  {
    cdr >> ros_message->turtle_id;
  }

  // Field name: x
  {
    cdr >> ros_message->x;
  }

  // Field name: y
  {
    cdr >> ros_message->y;
  }

  // Field name: theta
  {
    cdr >> ros_message->theta;
  }

  // Field name: linear_velocity
  {
    cdr >> ros_message->linear_velocity;
  }

  // Field name: angular_velocity
  {
    cdr >> ros_message->angular_velocity;
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_docs_turtlesim
size_t get_serialized_size_docs_turtlesim__msg__KeyedPose(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _KeyedPose__ros_msg_type * ros_message = static_cast<const _KeyedPose__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: turtle_id
  {
    size_t item_size = sizeof(ros_message->turtle_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: x
  {
    size_t item_size = sizeof(ros_message->x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: y
  {
    size_t item_size = sizeof(ros_message->y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: theta
  {
    size_t item_size = sizeof(ros_message->theta);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: linear_velocity
  {
    size_t item_size = sizeof(ros_message->linear_velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: angular_velocity
  {
    size_t item_size = sizeof(ros_message->angular_velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_docs_turtlesim
size_t max_serialized_size_docs_turtlesim__msg__KeyedPose(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: turtle_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: theta
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: linear_velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: angular_velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = docs_turtlesim__msg__KeyedPose;
    is_plain =
      (
      offsetof(DataType, angular_velocity) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_docs_turtlesim
bool cdr_serialize_key_docs_turtlesim__msg__KeyedPose(
  const docs_turtlesim__msg__KeyedPose * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: turtle_id
  {
    cdr << ros_message->turtle_id;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_docs_turtlesim
bool cdr_deserialize_key_docs_turtlesim__msg__KeyedPose(
  eprosima::fastcdr::Cdr &cdr,
  docs_turtlesim__msg__KeyedPose * ros_message)
{
  (void)ros_message;
  (void)cdr;
  // TODO
  return false;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_docs_turtlesim
size_t get_serialized_size_key_docs_turtlesim__msg__KeyedPose(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _KeyedPose__ros_msg_type * ros_message = static_cast<const _KeyedPose__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: turtle_id
  {
    size_t item_size = sizeof(ros_message->turtle_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_docs_turtlesim
size_t max_serialized_size_key_docs_turtlesim__msg__KeyedPose(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: turtle_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = docs_turtlesim__msg__KeyedPose;
    is_plain =
      (
      offsetof(DataType, angular_velocity) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

static bool _KeyedPose__cdr_serialize_key(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const docs_turtlesim__msg__KeyedPose * ros_message = static_cast<const docs_turtlesim__msg__KeyedPose *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_key_docs_turtlesim__msg__KeyedPose(ros_message, cdr);
}

static bool _KeyedPose__cdr_deserialize_key(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  docs_turtlesim__msg__KeyedPose * ros_message = static_cast<docs_turtlesim__msg__KeyedPose *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_key_docs_turtlesim__msg__KeyedPose(cdr, ros_message);
}

static size_t _KeyedPose__get_serialized_size_key(
  const void * untyped_ros_message)
{
  return get_serialized_size_key_docs_turtlesim__msg__KeyedPose(
      untyped_ros_message, 0);
}

static
size_t
_KeyedPose__max_serialized_size_key(
  bool & is_unbounded)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_key_docs_turtlesim__msg__KeyedPose(
    full_bounded, is_plain, 0);

  is_unbounded = !full_bounded;
  return ret_val;
}

static message_type_support_key_callbacks_t __key_callbacks_KeyedPose = {
  _KeyedPose__cdr_serialize_key,
  _KeyedPose__cdr_deserialize_key,
  _KeyedPose__get_serialized_size_key,
  _KeyedPose__max_serialized_size_key,
};


static uint32_t _KeyedPose__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_docs_turtlesim__msg__KeyedPose(
      untyped_ros_message, 0));
}

static size_t _KeyedPose__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_docs_turtlesim__msg__KeyedPose(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t __callbacks_KeyedPose = {
  "docs_turtlesim::msg",
  "KeyedPose",
  _KeyedPose__cdr_serialize,
  _KeyedPose__cdr_deserialize,
  _KeyedPose__get_serialized_size,
  _KeyedPose__max_serialized_size,
  &__key_callbacks_KeyedPose
};

static rosidl_message_type_support_t _KeyedPose__type_support = {
  rosidl_typesupport_fastrtps_c__identifier_v2,
  &__callbacks_KeyedPose,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, docs_turtlesim, msg, KeyedPose)() {
  return &_KeyedPose__type_support;
}

#if defined(__cplusplus)
}
#endif
