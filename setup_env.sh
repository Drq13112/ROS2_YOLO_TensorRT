#!/bin/bash
# filepath: /home/david/yolocpp_ws/setup_env.sh

# Source ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash

# Apply system optimizations

echo "Optimizing system for high-performance ROS2 with large messages..."

# Network buffer optimizations
sudo sysctl -w net.core.rmem_max=536870912      # 512MB
sudo sysctl -w net.core.wmem_max=536870912      # 512MB
sudo sysctl -w net.core.rmem_default=33554432   # 32MB
sudo sysctl -w net.core.wmem_default=33554432   # 32MB
sudo sysctl -w net.core.netdev_max_backlog=10000
sudo sysctl net.ipv4.ipfrag_time=3
sudo sysctl net.ipv4.ipfrag_high_thresh=134217728     # (128 MB)


# Make changes persistent
echo "# ROS2 High Performance Settings" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_max=536870912" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max=536870912" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_default=33554432" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_default=33554432" | sudo tee -a /etc/sysctl.conf
echo "net.core.netdev_max_backlog=10000" | sudo tee -a /etc/sysctl.conf

echo "System optimizations applied!"


# FastDDS configuration for maximum performance
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
#export FASTDDS_DEFAULT_PROFILES_FILE="/home/david/yolocpp_ws/fastdds_config.xml"
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4


ulimit -l unlimited 2>/dev/null || echo "Warning: Could not set unlimited memory lock"

echo "========================================="
echo "FastDDS High Performance Configuration"
echo "========================================="
echo "Environment configured for maximum performance"
echo "FastDDS profile: $FASTDDS_DEFAULT_PROFILES_FILE"
echo "Transport: $FASTDDS_BUILTIN_TRANSPORTS"
echo "CPU cores: $(nproc)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="






