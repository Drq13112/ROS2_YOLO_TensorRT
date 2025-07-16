from setuptools import find_packages
from setuptools import setup

setup(
    name='yolo_custom_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('yolo_custom_interfaces', 'yolo_custom_interfaces.*')),
)
