cmake_minimum_required(VERSION 3.18)
project(GPUSchedulingCapstone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# find nvml
find_path(NVML_INCLUDE_DIR nvml.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATHS
        /usr/local/cuda/include
        /usr/local/cuda-13.0/include
        /usr/local/cuda-13.0/targets/x86_64-linux/include
)

find_library(NVML_LIBRARY nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/cuda/lib64
        /usr/local/cuda-13.0/lib64
)

if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
    message(STATUS "nvml include: ${NVML_INCLUDE_DIR}")
    message(STATUS "nvml library: ${NVML_LIBRARY}")
    set(HAS_NVML TRUE)
else()
    message(FATAL_ERROR "no nvml :(, install first")
endif()

# 75 = turing (T4), 86 = ampere (3060), 89 = ada (L4)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${NVML_INCLUDE_DIR})

set(CORE_SOURCES
    src/gpu_monitor.cpp
    src/scheduler.cpp
    src/workload_manager.cpp
)

set(KERNEL_SOURCES
    kernels/canny.cu
    kernels/harris.cu
    kernels/matmul.cu
    kernels/conv2d.cu
    kernels/histogram.cu
    kernels/gaussian_blur.cu
    kernels/synthetic_bandwidth.cu
    kernels/synthetic_compute.cu
    kernels/kernel_helpers.cu
)

add_executable(characterize_kernels
    testing/characterize_kernels.cpp
    src/gpu_monitor.cpp
    ${KERNEL_SOURCES}
)

target_link_libraries(characterize_kernels
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(characterize_kernels PUBLIC cxx_std_17)
target_compile_definitions(characterize_kernels PRIVATE HAS_NVML=1)

add_executable(test_workload_pairs
    testing/test_workload_pairs.cpp
    ${CORE_SOURCES}
    ${KERNEL_SOURCES}
)

target_link_libraries(test_workload_pairs
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(test_workload_pairs PUBLIC cxx_std_17)
target_compile_definitions(test_workload_pairs PRIVATE HAS_NVML=1)