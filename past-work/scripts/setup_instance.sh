#!/bin/bash

sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tree \
    unzip

sudo apt install -y libjsoncpp-dev pkg-config

sudo apt install -y \
    libopencv-dev \
    libopencv-contrib-dev \
    python3-opencv

if pkg-config --exists opencv4; then
    echo "OpenCV4 found: $(pkg-config --modversion opencv4)"
elif pkg-config --exists opencv; then
    echo "OpenCV found: $(pkg-config --modversion opencv)"
else
    echo "OpenCV not found via pkg-config, but may still be available"
    find /usr -name "opencv*" -type d 2>/dev/null | head -3
fi

if [ -f "/usr/include/nvml.h" ] || [ -f "/usr/local/cuda/include/nvml.h" ]; then
    echo "NVML headers found"
else
    sudo apt install -y nvidia-ml-dev
fi

if [ -f "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so" ] || [ -f "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1" ]; then
    echo "NVML library found"
fi

cat >> ~/.bashrc << 'EOF'
# CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Project environment
export PROJECT_HOME=~/gpu-scheduler
EOF

source ~/.bashrc

mkdir -p ~/gpu-scheduler/{include,src,config,scripts,results/{baseline,scheduling}}

# Clone or create your project files
cd ~/gpu-scheduler

# Create CMakeLists.txt for EC2
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(VideoFeatureExtraction CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Find jsoncpp for config files
pkg_check_modules(JSONCPP jsoncpp)
if(NOT JSONCPP_FOUND)
    find_package(jsoncpp REQUIRED)
    set(JSONCPP_LIBRARIES jsoncpp_lib)
endif()

# Find NVML with comprehensive search
set(NVML_SEARCH_PATHS
    /usr/local/cuda
    /usr/local/cuda-12
    /usr/local/cuda-11
    /opt/cuda
    /usr
)

find_path(NVML_INCLUDE_DIR 
    NAMES nvml.h
    PATHS ${NVML_SEARCH_PATHS}
    PATH_SUFFIXES include
)

find_library(NVML_LIBRARY 
    NAMES nvidia-ml
    PATHS ${NVML_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

# Also search standard system paths
if(NOT NVML_LIBRARY)
    find_library(NVML_LIBRARY 
        NAMES nvidia-ml
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib64
    )
endif()

if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
    message(STATUS "NVML found - include: ${NVML_INCLUDE_DIR}, lib: ${NVML_LIBRARY}")
    set(HAS_NVML TRUE)
else()
    message(STATUS "NVML not found - using fallback monitoring")
    set(HAS_NVML FALSE)
endif()

# Set CUDA architectures for T4 (7.5) and potential future GPUs
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
if(HAS_NVML)
    include_directories(${NVML_INCLUDE_DIR})
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/helpers.cpp
    src/kernels.cu
)

if(HAS_NVML)
    list(APPEND SOURCES src/gpu_monitor.cpp)
    list(APPEND SOURCES src/scheduler.cpp)
    list(APPEND SOURCES src/execution_manager.cpp)
else()
    list(APPEND SOURCES src/simple_scheduler.cpp)
endif()

# Main executable
add_executable(feature_extraction ${SOURCES})

# Set CUDA properties
set_property(TARGET feature_extraction PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link libraries
set(LINK_LIBRARIES ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${JSONCPP_LIBRARIES})
if(HAS_NVML)
    list(APPEND LINK_LIBRARIES ${NVML_LIBRARY})
endif()

target_link_libraries(feature_extraction ${LINK_LIBRARIES})

# Compiler features and definitions
target_compile_features(feature_extraction PUBLIC cxx_std_17)
if(HAS_NVML)
    target_compile_definitions(feature_extraction PRIVATE HAS_NVML=1)
endif()

# Test executable for GPU monitoring
if(HAS_NVML)
    add_executable(test_gpu_monitor
        src/test_gpu_monitor.cpp
        src/gpu_monitor.cpp
    )
    
    target_link_libraries(test_gpu_monitor ${CUDA_LIBRARIES} ${NVML_LIBRARY})
    target_compile_features(test_gpu_monitor PUBLIC cxx_std_17)
    target_compile_definitions(test_gpu_monitor PRIVATE HAS_NVML=1)
    if(HAS_NVML)
        include_directories(${NVML_INCLUDE_DIR})
    endif()
endif()

# Compiler flags
target_compile_options(feature_extraction PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)
EOF

# Create the T4 configuration file
cat > config/ec2_t4.json << 'EOF'
{
  "gpu_name": "Tesla T4",
  "architecture": "Turing",
  "compute_capability": 7.5,
  
  "hardware_specs": {
    "sm_count": 40,
    "cuda_cores": 2560,
    "memory_gb": 16,
    "memory_bandwidth_gb_s": 320,
    "base_clock_mhz": 585,
    "boost_clock_mhz": 1590
  },
  
  "scheduling_thresholds": {
    "memory_bandwidth_threshold_gb_s": 200.0,
    "sm_utilization_threshold_percent": 70.0,
    "memory_utilization_threshold_percent": 85.0,
    "min_concurrent_image_pixels": 1382400,
    "max_blocks_per_sm": 6,
    "thermal_throttle_temp_c": 89
  },
  
  "workload_characteristics": {
    "canny_kernel": {
      "type": "memory_bound",
      "estimated_bandwidth_per_pixel_gb": 0.000000015,
      "blocks_per_sm_optimal": 3,
      "register_pressure": "medium"
    },
    "harris_kernel": {
      "type": "compute_bound",
      "estimated_bandwidth_per_pixel_gb": 0.000000006,
      "blocks_per_sm_optimal": 6,
      "register_pressure": "high"
    }
  },
  
  "heuristic_weights": {
    "memory_pressure_weight": 0.5,
    "sm_occupancy_weight": 0.3,
    "thermal_weight": 0.15,
    "workload_complementarity_weight": 0.05
  }
}
EOF

# Create basic build script
cat > scripts/build.sh << 'EOF'
#!/bin/bash
echo "Building GPU Scheduler Project..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)
echo "Build complete!"
EOF

chmod +x scripts/build.sh

# Test the setup
echo "10. Testing setup..."
echo "CUDA Version:"
nvcc --version

echo "OpenCV Version:"
pkg-config --modversion opencv4

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Create a simple system info script
cat > scripts/system_info.sh << 'EOF'
#!/bin/bash
echo "=== System Information ==="
echo "Date: $(date)"
echo "Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Availability Zone: $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)"
echo ""
echo "=== GPU Information ==="
nvidia-smi
echo ""
echo "=== CUDA Information ==="
nvcc --version
echo ""
echo "=== OpenCV Information ==="
pkg-config --modversion opencv4
echo ""
echo "=== Build Tools ==="
gcc --version | head -1
cmake --version | head -1
EOF

chmod +x scripts/system_info.sh

echo ""
echo "=== Setup Complete! ==="
echo "Project directory: ~/gpu-scheduler"
echo "To get started:"
echo "  cd ~/gpu-scheduler"
echo "  ./scripts/system_info.sh  # Check setup"
echo "  ./scripts/build.sh        # Build project"
echo ""
echo "Next steps:"
echo "1. Copy your existing source files to src/"
echo "2. Copy your header files to include/"
echo "3. Run ./scripts/build.sh to build"
echo "4. Run ./build/test_gpu_monitor to test GPU monitoring"