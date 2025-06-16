from setuptools import find_packages
from setuptools import setup

setup(
    name='docs_turtlesim',
    version='1.4.2',
    packages=find_packages(
        include=('docs_turtlesim', 'docs_turtlesim.*')),
)
