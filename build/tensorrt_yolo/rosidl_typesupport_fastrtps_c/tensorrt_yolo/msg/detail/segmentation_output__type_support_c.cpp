// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice
#include "tensorrt_yolo/msg/detail/segmentation_output__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "tensorrt_yolo/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "tensorrt_yolo/msg/detail/segmentation_output__struct.h"
#include "tensorrt_yolo/msg/detail/segmentation_output__functions.h"
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

#include "rosidl_runtime_c/primitives_sequence.h"  // class_id_map, detected_instance_ids, instance_class_ids, instance_confidences, instance_id_map
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // class_id_map, detected_instance_ids, instance_class_ids, instance_confidences, instance_id_map
#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tensorrt_yolo
size_t get_serialized_size_std_msgs__msg__Header(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tensorrt_yolo
size_t max_serialized_size_std_msgs__msg__Header(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tensorrt_yolo
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, std_msgs, msg, Header)();


using _SegmentationOutput__ros_msg_type = tensorrt_yolo__msg__SegmentationOutput;

static bool _SegmentationOutput__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _SegmentationOutput__ros_msg_type * ros_message = static_cast<const _SegmentationOutput__ros_msg_type *>(untyped_ros_message);
  // Field name: header
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Header
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->header, cdr))
    {
      return false;
    }
  }

  // Field name: image_height
  {
    cdr << ros_message->image_height;
  }

  // Field name: image_width
  {
    cdr << ros_message->image_width;
  }

  // Field name: class_id_map
  {
    size_t size = ros_message->class_id_map.size;
    auto array_ptr = ros_message->class_id_map.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: instance_id_map
  {
    size_t size = ros_message->instance_id_map.size;
    auto array_ptr = ros_message->instance_id_map.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: instance_confidences
  {
    size_t size = ros_message->instance_confidences.size;
    auto array_ptr = ros_message->instance_confidences.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: instance_class_ids
  {
    size_t size = ros_message->instance_class_ids.size;
    auto array_ptr = ros_message->instance_class_ids.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: detected_instance_ids
  {
    size_t size = ros_message->detected_instance_ids.size;
    auto array_ptr = ros_message->detected_instance_ids.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _SegmentationOutput__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _SegmentationOutput__ros_msg_type * ros_message = static_cast<_SegmentationOutput__ros_msg_type *>(untyped_ros_message);
  // Field name: header
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Header
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->header))
    {
      return false;
    }
  }

  // Field name: image_height
  {
    cdr >> ros_message->image_height;
  }

  // Field name: image_width
  {
    cdr >> ros_message->image_width;
  }

  // Field name: class_id_map
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->class_id_map.data) {
      rosidl_runtime_c__int32__Sequence__fini(&ros_message->class_id_map);
    }
    if (!rosidl_runtime_c__int32__Sequence__init(&ros_message->class_id_map, size)) {
      fprintf(stderr, "failed to create array for field 'class_id_map'");
      return false;
    }
    auto array_ptr = ros_message->class_id_map.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: instance_id_map
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->instance_id_map.data) {
      rosidl_runtime_c__int32__Sequence__fini(&ros_message->instance_id_map);
    }
    if (!rosidl_runtime_c__int32__Sequence__init(&ros_message->instance_id_map, size)) {
      fprintf(stderr, "failed to create array for field 'instance_id_map'");
      return false;
    }
    auto array_ptr = ros_message->instance_id_map.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: instance_confidences
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->instance_confidences.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->instance_confidences);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->instance_confidences, size)) {
      fprintf(stderr, "failed to create array for field 'instance_confidences'");
      return false;
    }
    auto array_ptr = ros_message->instance_confidences.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: instance_class_ids
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->instance_class_ids.data) {
      rosidl_runtime_c__int32__Sequence__fini(&ros_message->instance_class_ids);
    }
    if (!rosidl_runtime_c__int32__Sequence__init(&ros_message->instance_class_ids, size)) {
      fprintf(stderr, "failed to create array for field 'instance_class_ids'");
      return false;
    }
    auto array_ptr = ros_message->instance_class_ids.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: detected_instance_ids
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->detected_instance_ids.data) {
      rosidl_runtime_c__int32__Sequence__fini(&ros_message->detected_instance_ids);
    }
    if (!rosidl_runtime_c__int32__Sequence__init(&ros_message->detected_instance_ids, size)) {
      fprintf(stderr, "failed to create array for field 'detected_instance_ids'");
      return false;
    }
    auto array_ptr = ros_message->detected_instance_ids.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tensorrt_yolo
size_t get_serialized_size_tensorrt_yolo__msg__SegmentationOutput(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SegmentationOutput__ros_msg_type * ros_message = static_cast<const _SegmentationOutput__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name image_height
  {
    size_t item_size = sizeof(ros_message->image_height);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name image_width
  {
    size_t item_size = sizeof(ros_message->image_width);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name class_id_map
  {
    size_t array_size = ros_message->class_id_map.size;
    auto array_ptr = ros_message->class_id_map.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name instance_id_map
  {
    size_t array_size = ros_message->instance_id_map.size;
    auto array_ptr = ros_message->instance_id_map.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name instance_confidences
  {
    size_t array_size = ros_message->instance_confidences.size;
    auto array_ptr = ros_message->instance_confidences.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name instance_class_ids
  {
    size_t array_size = ros_message->instance_class_ids.size;
    auto array_ptr = ros_message->instance_class_ids.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name detected_instance_ids
  {
    size_t array_size = ros_message->detected_instance_ids.size;
    auto array_ptr = ros_message->detected_instance_ids.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _SegmentationOutput__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_tensorrt_yolo__msg__SegmentationOutput(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tensorrt_yolo
size_t max_serialized_size_tensorrt_yolo__msg__SegmentationOutput(
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

  // member: header
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_std_msgs__msg__Header(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: image_height
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: image_width
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: class_id_map
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: instance_id_map
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: instance_confidences
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: instance_class_ids
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: detected_instance_ids
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = tensorrt_yolo__msg__SegmentationOutput;
    is_plain =
      (
      offsetof(DataType, detected_instance_ids) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _SegmentationOutput__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_tensorrt_yolo__msg__SegmentationOutput(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_SegmentationOutput = {
  "tensorrt_yolo::msg",
  "SegmentationOutput",
  _SegmentationOutput__cdr_serialize,
  _SegmentationOutput__cdr_deserialize,
  _SegmentationOutput__get_serialized_size,
  _SegmentationOutput__max_serialized_size
};

static rosidl_message_type_support_t _SegmentationOutput__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_SegmentationOutput,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, tensorrt_yolo, msg, SegmentationOutput)() {
  return &_SegmentationOutput__type_support;
}

#if defined(__cplusplus)
}
#endif
