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

# TCP optimizations
sudo sysctl -w net.ipv4.tcp_rmem="4096 33554432 536870912"
sudo sysctl -w net.ipv4.tcp_wmem="4096 33554432 536870912"
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# Memory optimizations
sudo sysctl -w vm.swappiness=10                 # Reduce swapping
sudo sysctl -w vm.dirty_ratio=15                # Reduce dirty page ratio
sudo sysctl -w vm.dirty_background_ratio=5      # Background writeback

# CPU optimizations
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make changes persistent
echo "# ROS2 High Performance Settings" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_max=536870912" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max=536870912" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_default=33554432" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_default=33554432" | sudo tee -a /etc/sysctl.conf
echo "net.core.netdev_max_backlog=10000" | sudo tee -a /etc/sysctl.conf

echo "System optimizations applied!"




# FastDDS configuration for maximum performance
#export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
#export FASTDDS_DEFAULT_PROFILES_FILE="/home/david/yolocpp_ws/fastdds_config.xml"
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# High performance settings
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export FASTDDS_STATISTICS=OFF
export ROS_DOMAIN_ID=56

# Memory and CPU optimizations
export OMP_NUM_THREADS=$(nproc)
export CUDA_VISIBLE_DEVICES=0

# Set process priority for real-time performance (optional)
# sudo renice -10 $$ 2>/dev/null || echo "Warning: Could not set process priority"

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






