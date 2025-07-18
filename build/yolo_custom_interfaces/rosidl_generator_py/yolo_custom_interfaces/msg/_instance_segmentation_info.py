# generated from rosidl_generator_py/resource/_idl.py.em
# with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'mask_data'
# Member 'scores'
# Member 'classes'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_InstanceSegmentationInfo(type):
    """Metaclass of message 'InstanceSegmentationInfo'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('yolo_custom_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'yolo_custom_interfaces.msg.InstanceSegmentationInfo')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__instance_segmentation_info
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__instance_segmentation_info
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__instance_segmentation_info
            cls._TYPE_SUPPORT = module.type_support_msg__msg__instance_segmentation_info
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__instance_segmentation_info

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class InstanceSegmentationInfo(metaclass=Metaclass_InstanceSegmentationInfo):
    """Message class 'InstanceSegmentationInfo'."""

    __slots__ = [
        '_header',
        '_mask_width',
        '_mask_height',
        '_mask_data',
        '_scores',
        '_classes',
        '_image_source_monotonic_capture_time',
        '_processing_node_monotonic_entry_time',
        '_processing_node_inference_start_time',
        '_processing_node_inference_end_time',
        '_processing_node_monotonic_publish_time',
        '_packet_sequence_number',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'mask_width': 'uint16',
        'mask_height': 'uint16',
        'mask_data': 'sequence<uint8>',
        'scores': 'sequence<float>',
        'classes': 'sequence<uint8>',
        'image_source_monotonic_capture_time': 'builtin_interfaces/Time',
        'processing_node_monotonic_entry_time': 'builtin_interfaces/Time',
        'processing_node_inference_start_time': 'builtin_interfaces/Time',
        'processing_node_inference_end_time': 'builtin_interfaces/Time',
        'processing_node_monotonic_publish_time': 'builtin_interfaces/Time',
        'packet_sequence_number': 'uint64',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.mask_width = kwargs.get('mask_width', int())
        self.mask_height = kwargs.get('mask_height', int())
        self.mask_data = array.array('B', kwargs.get('mask_data', []))
        self.scores = array.array('f', kwargs.get('scores', []))
        self.classes = array.array('B', kwargs.get('classes', []))
        from builtin_interfaces.msg import Time
        self.image_source_monotonic_capture_time = kwargs.get('image_source_monotonic_capture_time', Time())
        from builtin_interfaces.msg import Time
        self.processing_node_monotonic_entry_time = kwargs.get('processing_node_monotonic_entry_time', Time())
        from builtin_interfaces.msg import Time
        self.processing_node_inference_start_time = kwargs.get('processing_node_inference_start_time', Time())
        from builtin_interfaces.msg import Time
        self.processing_node_inference_end_time = kwargs.get('processing_node_inference_end_time', Time())
        from builtin_interfaces.msg import Time
        self.processing_node_monotonic_publish_time = kwargs.get('processing_node_monotonic_publish_time', Time())
        self.packet_sequence_number = kwargs.get('packet_sequence_number', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.mask_width != other.mask_width:
            return False
        if self.mask_height != other.mask_height:
            return False
        if self.mask_data != other.mask_data:
            return False
        if self.scores != other.scores:
            return False
        if self.classes != other.classes:
            return False
        if self.image_source_monotonic_capture_time != other.image_source_monotonic_capture_time:
            return False
        if self.processing_node_monotonic_entry_time != other.processing_node_monotonic_entry_time:
            return False
        if self.processing_node_inference_start_time != other.processing_node_inference_start_time:
            return False
        if self.processing_node_inference_end_time != other.processing_node_inference_end_time:
            return False
        if self.processing_node_monotonic_publish_time != other.processing_node_monotonic_publish_time:
            return False
        if self.packet_sequence_number != other.packet_sequence_number:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def mask_width(self):
        """Message field 'mask_width'."""
        return self._mask_width

    @mask_width.setter
    def mask_width(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mask_width' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'mask_width' field must be an unsigned integer in [0, 65535]"
        self._mask_width = value

    @builtins.property
    def mask_height(self):
        """Message field 'mask_height'."""
        return self._mask_height

    @mask_height.setter
    def mask_height(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mask_height' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'mask_height' field must be an unsigned integer in [0, 65535]"
        self._mask_height = value

    @builtins.property
    def mask_data(self):
        """Message field 'mask_data'."""
        return self._mask_data

    @mask_data.setter
    def mask_data(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'mask_data' array.array() must have the type code of 'B'"
            self._mask_data = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'mask_data' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._mask_data = array.array('B', value)

    @builtins.property
    def scores(self):
        """Message field 'scores'."""
        return self._scores

    @scores.setter
    def scores(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'scores' array.array() must have the type code of 'f'"
            self._scores = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'scores' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._scores = array.array('f', value)

    @builtins.property
    def classes(self):
        """Message field 'classes'."""
        return self._classes

    @classes.setter
    def classes(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'classes' array.array() must have the type code of 'B'"
            self._classes = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'classes' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._classes = array.array('B', value)

    @builtins.property
    def image_source_monotonic_capture_time(self):
        """Message field 'image_source_monotonic_capture_time'."""
        return self._image_source_monotonic_capture_time

    @image_source_monotonic_capture_time.setter
    def image_source_monotonic_capture_time(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'image_source_monotonic_capture_time' field must be a sub message of type 'Time'"
        self._image_source_monotonic_capture_time = value

    @builtins.property
    def processing_node_monotonic_entry_time(self):
        """Message field 'processing_node_monotonic_entry_time'."""
        return self._processing_node_monotonic_entry_time

    @processing_node_monotonic_entry_time.setter
    def processing_node_monotonic_entry_time(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'processing_node_monotonic_entry_time' field must be a sub message of type 'Time'"
        self._processing_node_monotonic_entry_time = value

    @builtins.property
    def processing_node_inference_start_time(self):
        """Message field 'processing_node_inference_start_time'."""
        return self._processing_node_inference_start_time

    @processing_node_inference_start_time.setter
    def processing_node_inference_start_time(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'processing_node_inference_start_time' field must be a sub message of type 'Time'"
        self._processing_node_inference_start_time = value

    @builtins.property
    def processing_node_inference_end_time(self):
        """Message field 'processing_node_inference_end_time'."""
        return self._processing_node_inference_end_time

    @processing_node_inference_end_time.setter
    def processing_node_inference_end_time(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'processing_node_inference_end_time' field must be a sub message of type 'Time'"
        self._processing_node_inference_end_time = value

    @builtins.property
    def processing_node_monotonic_publish_time(self):
        """Message field 'processing_node_monotonic_publish_time'."""
        return self._processing_node_monotonic_publish_time

    @processing_node_monotonic_publish_time.setter
    def processing_node_monotonic_publish_time(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'processing_node_monotonic_publish_time' field must be a sub message of type 'Time'"
        self._processing_node_monotonic_publish_time = value

    @builtins.property
    def packet_sequence_number(self):
        """Message field 'packet_sequence_number'."""
        return self._packet_sequence_number

    @packet_sequence_number.setter
    def packet_sequence_number(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'packet_sequence_number' field must be of type 'int'"
            assert value >= 0 and value < 18446744073709551616, \
                "The 'packet_sequence_number' field must be an unsigned integer in [0, 18446744073709551615]"
        self._packet_sequence_number = value
