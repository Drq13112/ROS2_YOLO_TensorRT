# generated from rosidl_generator_py/resource/_idl.py.em
# with input from tensorrt_yolo:msg/SegmentationOutput.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'class_id_map'
# Member 'instance_id_map'
# Member 'instance_confidences'
# Member 'instance_class_ids'
# Member 'detected_instance_ids'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SegmentationOutput(type):
    """Metaclass of message 'SegmentationOutput'."""

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
            module = import_type_support('tensorrt_yolo')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'tensorrt_yolo.msg.SegmentationOutput')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__segmentation_output
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__segmentation_output
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__segmentation_output
            cls._TYPE_SUPPORT = module.type_support_msg__msg__segmentation_output
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__segmentation_output

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


class SegmentationOutput(metaclass=Metaclass_SegmentationOutput):
    """Message class 'SegmentationOutput'."""

    __slots__ = [
        '_header',
        '_image_height',
        '_image_width',
        '_class_id_map',
        '_instance_id_map',
        '_instance_confidences',
        '_instance_class_ids',
        '_detected_instance_ids',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'image_height': 'uint32',
        'image_width': 'uint32',
        'class_id_map': 'sequence<int32>',
        'instance_id_map': 'sequence<int32>',
        'instance_confidences': 'sequence<float>',
        'instance_class_ids': 'sequence<int32>',
        'detected_instance_ids': 'sequence<int32>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int32')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int32')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int32')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int32')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.image_height = kwargs.get('image_height', int())
        self.image_width = kwargs.get('image_width', int())
        self.class_id_map = array.array('i', kwargs.get('class_id_map', []))
        self.instance_id_map = array.array('i', kwargs.get('instance_id_map', []))
        self.instance_confidences = array.array('f', kwargs.get('instance_confidences', []))
        self.instance_class_ids = array.array('i', kwargs.get('instance_class_ids', []))
        self.detected_instance_ids = array.array('i', kwargs.get('detected_instance_ids', []))

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
        if self.image_height != other.image_height:
            return False
        if self.image_width != other.image_width:
            return False
        if self.class_id_map != other.class_id_map:
            return False
        if self.instance_id_map != other.instance_id_map:
            return False
        if self.instance_confidences != other.instance_confidences:
            return False
        if self.instance_class_ids != other.instance_class_ids:
            return False
        if self.detected_instance_ids != other.detected_instance_ids:
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
    def image_height(self):
        """Message field 'image_height'."""
        return self._image_height

    @image_height.setter
    def image_height(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'image_height' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'image_height' field must be an unsigned integer in [0, 4294967295]"
        self._image_height = value

    @builtins.property
    def image_width(self):
        """Message field 'image_width'."""
        return self._image_width

    @image_width.setter
    def image_width(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'image_width' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'image_width' field must be an unsigned integer in [0, 4294967295]"
        self._image_width = value

    @builtins.property
    def class_id_map(self):
        """Message field 'class_id_map'."""
        return self._class_id_map

    @class_id_map.setter
    def class_id_map(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'i', \
                "The 'class_id_map' array.array() must have the type code of 'i'"
            self._class_id_map = value
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
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'class_id_map' field must be a set or sequence and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._class_id_map = array.array('i', value)

    @builtins.property
    def instance_id_map(self):
        """Message field 'instance_id_map'."""
        return self._instance_id_map

    @instance_id_map.setter
    def instance_id_map(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'i', \
                "The 'instance_id_map' array.array() must have the type code of 'i'"
            self._instance_id_map = value
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
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'instance_id_map' field must be a set or sequence and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._instance_id_map = array.array('i', value)

    @builtins.property
    def instance_confidences(self):
        """Message field 'instance_confidences'."""
        return self._instance_confidences

    @instance_confidences.setter
    def instance_confidences(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'instance_confidences' array.array() must have the type code of 'f'"
            self._instance_confidences = value
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
                "The 'instance_confidences' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._instance_confidences = array.array('f', value)

    @builtins.property
    def instance_class_ids(self):
        """Message field 'instance_class_ids'."""
        return self._instance_class_ids

    @instance_class_ids.setter
    def instance_class_ids(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'i', \
                "The 'instance_class_ids' array.array() must have the type code of 'i'"
            self._instance_class_ids = value
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
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'instance_class_ids' field must be a set or sequence and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._instance_class_ids = array.array('i', value)

    @builtins.property
    def detected_instance_ids(self):
        """Message field 'detected_instance_ids'."""
        return self._detected_instance_ids

    @detected_instance_ids.setter
    def detected_instance_ids(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'i', \
                "The 'detected_instance_ids' array.array() must have the type code of 'i'"
            self._detected_instance_ids = value
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
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'detected_instance_ids' field must be a set or sequence and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._detected_instance_ids = array.array('i', value)
