// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from docs_turtlesim:msg/KeyedTwist.idl
// generated code does not contain a copyright notice
#include "docs_turtlesim/msg/detail/keyed_twist__rosidl_typesupport_fastrtps_cpp.hpp"
#include "docs_turtlesim/msg/detail/keyed_twist__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions
namespace docs_turtlesim
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const docs_turtlesim::msg::Vector3 &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  docs_turtlesim::msg::Vector3 &);
size_t get_serialized_size(
  const docs_turtlesim::msg::Vector3 &,
  size_t current_alignment);
size_t
max_serialized_size_Vector3(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
bool cdr_serialize_key(
  const docs_turtlesim::msg::Vector3 &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize_key(
  eprosima::fastcdr::Cdr &,
  docs_turtlesim::msg::Vector3 &);
size_t get_serialized_size_key(
  const docs_turtlesim::msg::Vector3 &,
  size_t current_alignment);
size_t
max_serialized_size_key_Vector3(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace docs_turtlesim

// functions for docs_turtlesim::msg::Vector3 already declared above


namespace docs_turtlesim
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_serialize(
  const docs_turtlesim::msg::KeyedTwist & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: turtle_id
  cdr << ros_message.turtle_id;

  // Member: linear
  docs_turtlesim::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.linear,
    cdr);

  // Member: angular
  docs_turtlesim::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.angular,
    cdr);

  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  docs_turtlesim::msg::KeyedTwist & ros_message)
{
  // Member: turtle_id
  cdr >> ros_message.turtle_id;

  // Member: linear
  docs_turtlesim::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.linear);

  // Member: angular
  docs_turtlesim::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.angular);

  return true;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
get_serialized_size(
  const docs_turtlesim::msg::KeyedTwist & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: turtle_id
  {
    size_t item_size = sizeof(ros_message.turtle_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Member: linear
  current_alignment +=
    docs_turtlesim::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.linear, current_alignment);

  // Member: angular
  current_alignment +=
    docs_turtlesim::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.angular, current_alignment);

  return current_alignment - initial_alignment;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
max_serialized_size_KeyedTwist(
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

  // Member: turtle_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: linear
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        docs_turtlesim::msg::typesupport_fastrtps_cpp::max_serialized_size_Vector3(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: angular
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        docs_turtlesim::msg::typesupport_fastrtps_cpp::max_serialized_size_Vector3(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = docs_turtlesim::msg::KeyedTwist;
    is_plain =
      (
      offsetof(DataType, angular) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_serialize_key(
  const docs_turtlesim::msg::KeyedTwist & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: turtle_id
  cdr << ros_message.turtle_id;

  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
cdr_deserialize_key(
  const docs_turtlesim::msg::KeyedTwist & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  (void)ros_message;
  (void)cdr;
  // TODO
  return false;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
get_serialized_size_key(
  const docs_turtlesim::msg::KeyedTwist & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: turtle_id
  {
    size_t item_size = sizeof(ros_message.turtle_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_docs_turtlesim
max_serialized_size_key_KeyedTwist(
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

  // Member: turtle_id
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
    using DataType = docs_turtlesim::msg::KeyedTwist;
    is_plain =
      (
      offsetof(DataType, angular) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

static bool _KeyedTwist__cdr_serialize_key(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const docs_turtlesim::msg::KeyedTwist *>(
    untyped_ros_message);

  return cdr_serialize_key(*typed_message, cdr);
}

static
bool
_KeyedTwist__cdr_deserialize_key(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<docs_turtlesim::msg::KeyedTwist *>(
    untyped_ros_message);

  return cdr_deserialize_key(cdr, *typed_message);
}

static
size_t
_KeyedTwist__get_serialized_size_key(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const docs_turtlesim::msg::KeyedTwist *>(
    untyped_ros_message);

  return get_serialized_size_key(*typed_message, 0);
}

static size_t _KeyedTwist__max_serialized_size_key(
  bool & is_unbounded)
{
  bool full_bounded = true;
  bool is_plain = true;

  size_t ret_val = max_serialized_size_key_KeyedTwist(
    full_bounded,
    is_plain,
    0);

  is_unbounded = !full_bounded;
  return ret_val;
}

static message_type_support_key_callbacks_t _KeyedTwist__key_callbacks = {
  _KeyedTwist__cdr_serialize_key,
  _KeyedTwist__cdr_deserialize_key,
  _KeyedTwist__get_serialized_size_key,
  _KeyedTwist__max_serialized_size_key,
};

static bool _KeyedTwist__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const docs_turtlesim::msg::KeyedTwist *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _KeyedTwist__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<docs_turtlesim::msg::KeyedTwist *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _KeyedTwist__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const docs_turtlesim::msg::KeyedTwist *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _KeyedTwist__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_KeyedTwist(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _KeyedTwist__callbacks = {
  "docs_turtlesim::msg",
  "KeyedTwist",
  _KeyedTwist__cdr_serialize,
  _KeyedTwist__cdr_deserialize,
  _KeyedTwist__get_serialized_size,
  _KeyedTwist__max_serialized_size,
  &_KeyedTwist__key_callbacks
};

static rosidl_message_type_support_t _KeyedTwist__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier_v2,
  &_KeyedTwist__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace docs_turtlesim

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_docs_turtlesim
const rosidl_message_type_support_t *
get_message_type_support_handle<docs_turtlesim::msg::KeyedTwist>()
{
  return &docs_turtlesim::msg::typesupport_fastrtps_cpp::_KeyedTwist__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, docs_turtlesim, msg, KeyedTwist)() {
  return &docs_turtlesim::msg::typesupport_fastrtps_cpp::_KeyedTwist__handle;
}

#ifdef __cplusplus
}
#endif
