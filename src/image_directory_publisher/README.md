# Image Directory Publisher

This ROS 2 package, `image_directory_publisher`, is designed to read images from a specified directory and publish them as `sensor_msgs::msg::Image` messages. The package specifically handles images named "left", "front", and "right", which should have the same timestamp, allowing for synchronized image publishing.

## Features

- Reads images from a specified directory.
- Publishes images as ROS 2 messages.
- Supports camera calibration information for each camera.
- Configurable parameters for directory path and frame ID.

## Directory Structure

```
image_directory_publisher
├── src
│   └── directory_publisher_node.cpp        # Implementation of the DirectoryPublisherNode class
├── include
│   └── image_directory_publisher
│       └── directory_publisher_node.hpp    # Header file for the DirectoryPublisherNode class
├── launch
│   └── directory_publisher.launch.py        # Launch file to start the DirectoryPublisherNode
├── config
│   ├── params.yaml                          # Parameters for the DirectoryPublisherNode
│   ├── left_camera_info.yaml                # Camera calibration info for the left camera
│   ├── front_camera_info.yaml               # Camera calibration info for the front camera
│   └── right_camera_info.yaml               # Camera calibration info for the right camera
├── CMakeLists.txt                           # Build configuration for the package
├── package.xml                               # Metadata about the ROS 2 package
└── README.md                                 # Documentation for the project
```

## Installation

To build and install the package, follow these steps:

1. Clone the repository:
   ```
   git clone <repository_url>
   cd image_directory_publisher
   ```

2. Install dependencies:
   ```
   rosdep install -i --from-path src --rosdistro humble -y
   ```

3. Build the package:
   ```
   colcon build
   ```

4. Source the setup file:
   ```
   source install/setup.bash
   ```

## Usage

To run the `DirectoryPublisherNode`, use the provided launch file:

```
ros2 launch image_directory_publisher directory_publisher.launch.py
```

Make sure to configure the `params.yaml` file with the correct directory path where the images are stored and set the appropriate frame ID.

## Configuration

The package includes several configuration files located in the `config` directory:

- `params.yaml`: Contains parameters for the node, including the image directory and frame ID.
- `left_camera_info.yaml`, `front_camera_info.yaml`, `right_camera_info.yaml`: Contain camera calibration information for each respective camera.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.