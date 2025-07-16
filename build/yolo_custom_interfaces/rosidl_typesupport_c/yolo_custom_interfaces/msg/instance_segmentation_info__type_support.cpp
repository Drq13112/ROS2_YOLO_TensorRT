// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__struct.h"
#include "yolo_custom_interfaces/msg/detail/instance_segmentation_info__type_support.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace yolo_custom_interfaces
{

namespace msg
{

namespace rosidl_typesupport_c
{

typedef struct _InstanceSegmentationInfo_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _InstanceSegmentationInfo_type_support_ids_t;

static const _InstanceSegmentationInfo_type_support_ids_t _InstanceSegmentationInfo_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _InstanceSegmentationInfo_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _InstanceSegmentationInfo_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _InstanceSegmentationInfo_type_support_symbol_names_t _InstanceSegmentationInfo_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, yolo_custom_interfaces, msg, InstanceSegmentationInfo)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, yolo_custom_interfaces, msg, InstanceSegmentationInfo)),
  }
};

typedef struct _InstanceSegmentationInfo_type_support_data_t
{
  void * data[2];
} _InstanceSegmentationInfo_type_support_data_t;

static _InstanceSegmentationInfo_type_support_data_t _InstanceSegmentationInfo_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _InstanceSegmentationInfo_message_typesupport_map = {
  2,
  "yolo_custom_interfaces",
  &_InstanceSegmentationInfo_message_typesupport_ids.typesupport_identifier[0],
  &_InstanceSegmentationInfo_message_typesupport_symbol_names.symbol_name[0],
  &_InstanceSegmentationInfo_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t InstanceSegmentationInfo_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_InstanceSegmentationInfo_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace msg

}  // namespace yolo_custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, yolo_custom_interfaces, msg, InstanceSegmentationInfo)() {
  return &::yolo_custom_interfaces::msg::rosidl_typesupport_c::InstanceSegmentationInfo_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
