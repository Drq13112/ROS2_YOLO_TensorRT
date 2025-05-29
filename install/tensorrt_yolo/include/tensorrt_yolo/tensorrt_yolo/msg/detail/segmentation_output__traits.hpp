// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#ifndef TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__TRAITS_HPP_
#define TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tensorrt_yolo/msg/detail/segmentation_output__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace tensorrt_yolo
{

namespace msg
{

inline void to_flow_style_yaml(
  const SegmentationOutput & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: image_height
  {
    out << "image_height: ";
    rosidl_generator_traits::value_to_yaml(msg.image_height, out);
    out << ", ";
  }

  // member: image_width
  {
    out << "image_width: ";
    rosidl_generator_traits::value_to_yaml(msg.image_width, out);
    out << ", ";
  }

  // member: class_id_map
  {
    if (msg.class_id_map.size() == 0) {
      out << "class_id_map: []";
    } else {
      out << "class_id_map: [";
      size_t pending_items = msg.class_id_map.size();
      for (auto item : msg.class_id_map) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: instance_id_map
  {
    if (msg.instance_id_map.size() == 0) {
      out << "instance_id_map: []";
    } else {
      out << "instance_id_map: [";
      size_t pending_items = msg.instance_id_map.size();
      for (auto item : msg.instance_id_map) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: instance_confidences
  {
    if (msg.instance_confidences.size() == 0) {
      out << "instance_confidences: []";
    } else {
      out << "instance_confidences: [";
      size_t pending_items = msg.instance_confidences.size();
      for (auto item : msg.instance_confidences) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: instance_class_ids
  {
    if (msg.instance_class_ids.size() == 0) {
      out << "instance_class_ids: []";
    } else {
      out << "instance_class_ids: [";
      size_t pending_items = msg.instance_class_ids.size();
      for (auto item : msg.instance_class_ids) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: detected_instance_ids
  {
    if (msg.detected_instance_ids.size() == 0) {
      out << "detected_instance_ids: []";
    } else {
      out << "detected_instance_ids: [";
      size_t pending_items = msg.detected_instance_ids.size();
      for (auto item : msg.detected_instance_ids) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SegmentationOutput & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: image_height
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "image_height: ";
    rosidl_generator_traits::value_to_yaml(msg.image_height, out);
    out << "\n";
  }

  // member: image_width
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "image_width: ";
    rosidl_generator_traits::value_to_yaml(msg.image_width, out);
    out << "\n";
  }

  // member: class_id_map
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.class_id_map.size() == 0) {
      out << "class_id_map: []\n";
    } else {
      out << "class_id_map:\n";
      for (auto item : msg.class_id_map) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: instance_id_map
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.instance_id_map.size() == 0) {
      out << "instance_id_map: []\n";
    } else {
      out << "instance_id_map:\n";
      for (auto item : msg.instance_id_map) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: instance_confidences
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.instance_confidences.size() == 0) {
      out << "instance_confidences: []\n";
    } else {
      out << "instance_confidences:\n";
      for (auto item : msg.instance_confidences) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: instance_class_ids
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.instance_class_ids.size() == 0) {
      out << "instance_class_ids: []\n";
    } else {
      out << "instance_class_ids:\n";
      for (auto item : msg.instance_class_ids) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: detected_instance_ids
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.detected_instance_ids.size() == 0) {
      out << "detected_instance_ids: []\n";
    } else {
      out << "detected_instance_ids:\n";
      for (auto item : msg.detected_instance_ids) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SegmentationOutput & msg, bool use_flow_style = false)
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

}  // namespace tensorrt_yolo

namespace rosidl_generator_traits
{

[[deprecated("use tensorrt_yolo::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tensorrt_yolo::msg::SegmentationOutput & msg,
  std::ostream & out, size_t indentation = 0)
{
  tensorrt_yolo::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tensorrt_yolo::msg::to_yaml() instead")]]
inline std::string to_yaml(const tensorrt_yolo::msg::SegmentationOutput & msg)
{
  return tensorrt_yolo::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tensorrt_yolo::msg::SegmentationOutput>()
{
  return "tensorrt_yolo::msg::SegmentationOutput";
}

template<>
inline const char * name<tensorrt_yolo::msg::SegmentationOutput>()
{
  return "tensorrt_yolo/msg/SegmentationOutput";
}

template<>
struct has_fixed_size<tensorrt_yolo::msg::SegmentationOutput>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tensorrt_yolo::msg::SegmentationOutput>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tensorrt_yolo::msg::SegmentationOutput>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__TRAITS_HPP_
