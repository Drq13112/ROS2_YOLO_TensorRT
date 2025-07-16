# Sensor Synchronization Package

## Overview
The Sensor Synchronization Package is designed to synchronize data from lidar and camera sensors in real-time. This package ensures that the timestamps of the messages from both sensors are aligned, enabling accurate data fusion for applications such as robotics and autonomous vehicles.

## Installation
To install the Sensor Synchronization Package, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the package directory:
   ```
   cd sensor_sync_pkg
   ```

3. Install the package using the following command:
   ```
   python setup.py install
   ```

## Usage
To launch the sensor synchronization node, use the following command:
```
ros2 launch sensor_sync_pkg sync.launch.py
```

This will start the synchronization node, which will subscribe to the lidar and camera topics and begin processing the incoming data.

## Directory Structure
- `src/sync_node.py`: Contains the main logic for the sensor synchronization node.
- `launch/sync.launch.py`: Defines the launch configuration for the package.
- `resource/sensor_sync_pkg`: Contains additional resources or configuration files.
- `test/test_copyright.py`: Contains unit tests related to copyright compliance.
- `test/test_flake8.py`: Contains tests for code style and formatting.
- `package.xml`: The package manifest defining metadata and dependencies.
- `setup.cfg`: Configuration settings for testing and packaging.
- `setup.py`: The setup script for installing the package.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.