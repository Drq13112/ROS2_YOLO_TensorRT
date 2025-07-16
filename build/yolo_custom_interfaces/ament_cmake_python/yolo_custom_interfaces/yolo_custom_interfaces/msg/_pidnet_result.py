# generated from rosidl_generator_py/resource/_idl.py.em
# with input from yolo_custom_interfaces:msg/PidnetResult.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PidnetResult(type):
    """Metaclass of message 'PidnetResult'."""

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
                'yolo_custom_interfaces.msg.PidnetResult')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__pidnet_result
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__pidnet_result
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__pidnet_result
            cls._TYPE_SUPPORT = module.type_support_msg__msg__pidnet_result
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__pidnet_result

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

            from sensor_msgs.msg import Image
            if Image.__class__._TYPE_SUPPORT is None:
                Image.__class__.__import_type_support__()

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


class PidnetResult(metaclass=Metaclass_PidnetResult):
    """Message class 'PidnetResult'."""

    __slots__ = [
        '_header',
        '_segmentation_map',
        '_packet_sequence_number',
        '_image_source_monotonic_capture_time',
        '_processing_node_monotonic_entry_time',
        '_processing_node_inference_start_time',
        '_processing_node_inference_end_time',
        '_processing_node_monotonic_publish_time',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'segmentation_map': 'sensor_msgs/Image',
        'packet_sequence_number': 'uint64',
        'image_source_monotonic_capture_time': 'builtin_interfaces/Time',
        'processing_node_monotonic_entry_time': 'builtin_interfaces/Time',
        'processing_node_inference_start_time': 'builtin_interfaces/Time',
        'processing_node_inference_end_time': 'builtin_interfaces/Time',
        'processing_node_monotonic_publish_time': 'builtin_interfaces/Time',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'Image'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        from sensor_msgs.msg import Image
        self.segmentation_map = kwargs.get('segmentation_map', Image())
        self.packet_sequence_number = kwargs.get('packet_sequence_number', int())
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
        if self.segmentation_map != other.segmentation_map:
            return False
        if self.packet_sequence_number != other.packet_sequence_number:
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
    def segmentation_map(self):
        """Message field 'segmentation_map'."""
        return self._segmentation_map

    @segmentation_map.setter
    def segmentation_map(self, value):
        if __debug__:
            from sensor_msgs.msg import Image
            assert \
                isinstance(value, Image), \
                "The 'segmentation_map' field must be a sub message of type 'Image'"
        self._segmentation_map = value

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
