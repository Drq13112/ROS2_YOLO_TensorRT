# Base CUDA 12.9 con Ubuntu 22.04
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'
RUN apt-get update

# Actualizar e instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    build-essential \
    cmake \
    git \
    locales \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# Configurar locale
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# ROS 2 Humble Setup
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-rclcpp \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-rosidl-default-generators \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*


# Instala TensorRT y dependencias
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel
RUN python3 -m pip install --upgrade tensorrt
RUN python3 -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12


# RUN apt-get update && apt-get install -y \
#     libnvinfer-dev=10.12.0.36-1+cuda12.9 \
#     libnvinfer-plugin-dev=10.12.0.36-1+cuda12.9 \
#     libnvonnxparsers-dev=10.12.0.36-1+cuda12.9 \
#     libnvparsers-dev=10.12.0.36-1+cuda12.9 \
#     && rm -rf /var/lib/apt/lists/*

    
#Instala dependencias adicionales de ROS 2
RUN apt-get install -y ros-humble-image-transport-plugins

RUN apt-get update && apt-get install -y \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-camera-calibration-parsers \
    && rm -rf /var/lib/apt/lists/*



# Copia tu código y compila
WORKDIR /workspace
COPY . /workspace

# Instala dependencias Python si tienes
RUN pip3 install -r requirements.txt || true

# Build workspace
# RUN /bin/bash -c ". /opt/ros/humble/setup.bash && colcon build --cmake-args -DTRT_PATH=/usr -DCMAKE_CUDA_ARCHITECTURES=\\\"89\\\" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc"

# Build ROS 2 workspace
# RUN . /opt/ros/humble/setup.bash && colcon build --cmake-args -DTRT_PATH=/usr

# Define entrypoint para ROS 2 (opcional)
# ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source install/setup.bash && exec \"$@\"", "--"]

# CMD ["ros2", "run", "tensorrt_yolo", "segment_node"]
