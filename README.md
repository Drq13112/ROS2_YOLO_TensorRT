# YOLOv11 ROS 2 Node (Humble)

This repository provides a ROS 2 Humble integration of YOLOv11, based on the original project [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO). It wraps the detector in a ROS 2 node, enabling real-time inference through TensorRT acceleration.

## 📌 Origin

This work is derived from the original repository by `laugh12321`:  
➡️ https://github.com/laugh12321/TensorRT-YOLO/tree/main

The original code was adapted to work as a ROS 2 node, allowing easy integration into robotic systems and ROS 2-based pipelines.

## ⚙️ Functionality

This package includes:

- 📥 Subscription to image streams (`sensor_msgs/msg/Image`)
- 🧠 Inference using TensorRT-optimized YOLOv11
- 📤 Publication of detections as `vision_msgs/msg/Detection2DArray`
- 🧪 Optional debug visualization via RViz or image topics

## 🧱 Project Structure

yolo11_ros2/
├── launch/
│ └── yolo_inference.launch.py
├── config/
│ └── params.yaml
├── src/
│ └── yolo_node.cpp
├── models/
│ └── <TensorRT optimized models>
├── CMakeLists.txt
├── package.xml
└── README.md

bash
Copy
Edit

## 🚀 Usage

1. **Clone the repository into your ROS 2 workspace**

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/your_username/yolo11_ros2.git
   cd ..
   colcon build
Launch the YOLOv11 node

bash
Copy
Edit
source install/setup.bash
ros2 launch yolo11_ros2 yolo_inference.launch.py
Relevant Topics

/image_raw → Input image stream

/detections → Output 2D object detections

📂 Model Requirements
You will need a TensorRT-optimized YOLOv11 model (.engine file). You can generate one by following the instructions in the original repository, or use a precompiled one if available.

🧾 Credits
This project is built on top of the excellent work by laugh12321. The ROS 2 adaptation was developed independently to facilitate YOLOv11 deployment in ROS 2 Humble-based environments.

📃 License
This repository inherits the license from the original project. See the corresponding LICENSE file or refer to the license used in the source repository.

yaml
Copy
Edit

---

Let me know if you want the launch file and node skeleton too (`yolo_node.cpp`, `launch.py`, etc.), so you have the base of the ROS 2 package ready to go.
