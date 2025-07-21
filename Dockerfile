# Base CUDA 12.2 con Ubuntu 22.04
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04


ARG USER_ID=1000
ARG GROUP_ID=1000

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


RUN groupadd --gid $GROUP_ID david && \
    useradd --uid $USER_ID --gid $GROUP_ID -m -s /bin/bash david && \
    echo "david ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

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

#Instala dependencias adicionales de ROS 2
RUN apt-get install -y ros-humble-image-transport-plugins

RUN apt-get update && apt-get install -y \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-camera-calibration-parsers \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*


# 3. Copia e instala TensorRT desde el archivo .tar.gz
COPY TensorRT-10.12.0.36.Linux.x86_64-gnu.cuda-12.9.tar.gz /tmp/
RUN tar -xzf /tmp/TensorRT-10.12.0.36.Linux.x86_64-gnu.cuda-12.9.tar.gz -C /usr/local/ && \
    mv /usr/local/TensorRT-10.12.0.36 /usr/local/tensorrt && \
    rm /tmp/TensorRT-10.12.0.36.Linux.x86_64-gnu.cuda-12.9.tar.gz

# 4. Configura las variables de entorno para que el compilador y el enlazador encuentren TensorRT
ENV LD_LIBRARY_PATH="/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="/usr/local/tensorrt/include:${CPLUS_INCLUDE_PATH}"
ENV PATH="/usr/local/tensorrt/bin:${PATH}"


RUN apt-get update && apt-get -y install vim

USER david

WORKDIR /home/david/yolocpp_ws

# Instala TensorRT y dependencias
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel
RUN python3 -m pip install --upgrade tensorrt
RUN python3 -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12
RUN pip install onnx
RUN pip install onnxruntime-gpu
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122
RUN pip install ultralytics
RUN pip install opencv-python


# RUN git clone https://github.com/laugh12321/TensorRT-YOLO.git src/TensorRT-YOLO
COPY --chown=david:david TensorRT-YOLO ./TensorRT-YOLO/
RUN cd TensorRT-YOLO && \
    pip install "pybind11[global]" && \
    cmake -S . -B build -DTRT_PATH=/usr/local/tensorrt -DBUILD_PYTHON=ON && \
    cmake --build build -j$(nproc) --config Release && \
    cd python && \
    pip install --upgrade build && \
    python3 -m build --wheel && \
    # Install only inference-related dependencies
    pip install dist/tensorrt_yolo-6.2.0-py3-none-any.whl && \
    pip install dist/tensorrt_yolo-6.2.0-py3-none-any.whl[export]


# Define entrypoint para ROS 2
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && if [ -f install/setup.bash ]; then source install/setup.bash; fi && exec \"$@\"", "--"]

