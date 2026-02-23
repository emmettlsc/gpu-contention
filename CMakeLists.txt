cmake_minimum_required(VERSION 3.18)
project(GPUSchedulingCapstone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find required packages
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
    message(STATUS "nvml found - gpu monitoring enabled")
    message(STATUS "nvml include: ${NVML_INCLUDE_DIR}")
    message(STATUS "nvml library: ${NVML_LIBRARY}")
    set(HAS_NVML TRUE)
else()
    message(FATAL_ERROR "nvml not found. please ensure cuda toolkit with nvml is installed.")
endif()

# set cuda architectures (t4, l4, a10g support)
# 70 = turing (t4), 75 = turing (t4), 86 = ampere (a10g), 89 = ada (l4)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# include directories
include_directories(include)
include_directories(past-work/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${NVML_INCLUDE_DIR})

# reusable sources from past-work (no modifications)
set(REUSED_SOURCES
    past-work/src/gpu_monitor.cpp
    past-work/src/scheduler.cpp
    past-work/src/execution_manager.cpp
    past-work/src/kernels.cu
)

# new workload kernels (each in own file)
set(WORKLOAD_SOURCES
    workloads/canny.cu
    workloads/harris.cu
    workloads/matmul.cu
    workloads/conv2d.cu
    workloads/histogram.cu
    workloads/gaussian_blur.cu
    workloads/synthetic_bandwidth.cu
    workloads/synthetic_compute.cu
    workloads/kernel_test_helpers.cu
)

# new capstone source files
set(CAPSTONE_SOURCES
    src/workload_manager.cpp
)

# ===== STANDALONE KERNEL TESTING =====

add_executable(test_kernels_solo
    testing/test_kernels_solo.cpp
    ${WORKLOAD_SOURCES}
)

target_link_libraries(test_kernels_solo
    ${OpenCV_LIBS}
    CUDA::cudart
)

target_compile_features(test_kernels_solo PUBLIC cxx_std_17)

# ===== KERNEL CHARACTERIZATION =====

add_executable(characterize_kernels
    testing/characterize_kernels.cpp
    past-work/src/gpu_monitor.cpp
    ${WORKLOAD_SOURCES}
)

target_link_libraries(characterize_kernels
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(characterize_kernels PUBLIC cxx_std_17)
target_compile_definitions(characterize_kernels PRIVATE HAS_NVML=1)

# ===== WORKLOAD PAIR TESTING =====
# TODO: implement testing/test_workload_pairs.cpp

# add_executable(test_workload_pairs
#     testing/test_workload_pairs.cpp
#     ${REUSED_SOURCES}
#     ${WORKLOAD_SOURCES}
#     ${CAPSTONE_SOURCES}
# )
#
# target_link_libraries(test_workload_pairs
#     ${OpenCV_LIBS}
#     CUDA::cudart
#     ${NVML_LIBRARY}
# )
#
# target_compile_features(test_workload_pairs PUBLIC cxx_std_17)
# target_compile_definitions(test_workload_pairs PRIVATE HAS_NVML=1)

# ===== MAIN EXPERIMENT RUNNER =====
# TODO: implement src/run_experiments.cpp

# add_executable(run_experiments
#     src/run_experiments.cpp
#     ${REUSED_SOURCES}
#     ${WORKLOAD_SOURCES}
#     ${CAPSTONE_SOURCES}
# )
#
# target_link_libraries(run_experiments
#     ${OpenCV_LIBS}
#     CUDA::cudart
#     ${NVML_LIBRARY}
# )
#
# target_compile_features(run_experiments PUBLIC cxx_std_17)
# target_compile_definitions(run_experiments PRIVATE HAS_NVML=1)

message(STATUS "capstone extension build configured")
message(STATUS "target gpus: t4 (turing), l4 (ada), a10g (ampere)")
message(STATUS "workload kernels: ${WORKLOAD_SOURCES}")
