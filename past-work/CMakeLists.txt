cmake_minimum_required(VERSION 3.18)
project(VideoFeatureExtraction CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find NVML
find_path(NVML_INCLUDE_DIR nvml.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATHS /usr/local/cuda/include /usr/local/cuda-12.8/include
)

find_library(NVML_LIBRARY nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib64
)

# NVML??
if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
    message(STATUS "NVML found - GPU monitoring enabled")
    message(STATUS "NVML include: ${NVML_INCLUDE_DIR}")
    message(STATUS "NVML library: ${NVML_LIBRARY}")
    set(HAS_NVML TRUE)
else()
    message(FATAL_ERROR "NVML not found. Please ensure CUDA toolkit with NVML is installed.")
endif()

# Set CUDA architecture for T4 
set(CMAKE_CUDA_ARCHITECTURES "70;75")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include dirs
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${NVML_INCLUDE_DIR})

set(BASE_SOURCES
    src/gpu_monitor.cpp
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    list(APPEND BASE_SOURCES src/main.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/helpers.cpp")
    list(APPEND BASE_SOURCES src/helpers.cpp)

endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu")
    list(APPEND BASE_SOURCES src/kernels.cu)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu")
    list(APPEND BASE_SOURCES src/kernels.cu)
endif()

# make executables for GPU monitor + scheduler + execution manager
add_executable(test_gpu_monitor
    testing/test_gpu_monitor.cpp
    src/gpu_monitor.cpp
)

target_link_libraries(test_gpu_monitor
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(test_gpu_monitor PUBLIC cxx_std_17)
target_compile_definitions(test_gpu_monitor PRIVATE HAS_NVML=1)

add_executable(test_scheduler
    testing/test_scheduler.cpp
    src/gpu_monitor.cpp
    src/scheduler.cpp
)

target_link_libraries(test_scheduler 
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(test_scheduler PUBLIC cxx_std_17)
target_compile_definitions(test_scheduler PRIVATE HAS_NVML=1)

add_executable(test_execution_manager
    testing/test_execution_manager.cpp
    src/gpu_monitor.cpp
    src/scheduler.cpp
    src/execution_manager.cpp
    src/kernels.cu
)

target_link_libraries(test_execution_manager 
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(test_execution_manager PUBLIC cxx_std_17)
target_compile_definitions(test_execution_manager PRIVATE HAS_NVML=1)

add_executable(test_full_system
    testing/test_full_system.cpp
    src/gpu_monitor.cpp
    src/scheduler.cpp
    src/execution_manager.cpp
    src/kernels.cu
)

target_link_libraries(test_full_system 
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(test_full_system PUBLIC cxx_std_17)
target_compile_definitions(test_full_system PRIVATE HAS_NVML=1)

add_executable(quick_video_test
    testing/quick_video_test.cpp
    src/gpu_monitor.cpp
    src/scheduler.cpp
    src/execution_manager.cpp
    src/video_processor.cpp
    src/kernels.cu
)

target_link_libraries(quick_video_test 
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(quick_video_test PUBLIC cxx_std_17)
target_compile_definitions(quick_video_test PRIVATE HAS_NVML=1)

add_executable(run_video_experiments
    src/run_video_experiments.cpp
    src/gpu_monitor.cpp
    src/scheduler.cpp
    src/execution_manager.cpp
    src/video_processor.cpp
    src/kernels.cu
)

target_link_libraries(run_video_experiments 
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_features(run_video_experiments PUBLIC cxx_std_17)
target_compile_definitions(run_video_experiments PRIVATE HAS_NVML=1)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(feature_extraction ${BASE_SOURCES})

    set_property(TARGET feature_extraction PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_link_libraries(feature_extraction
        ${OpenCV_LIBS}
        CUDA::cudart
        CUDA::cufft
        ${NVML_LIBRARY}
    )

    target_compile_features(feature_extraction PUBLIC cxx_std_17)

    target_compile_definitions(feature_extraction PRIVATE
        OPENCV_VERSION_MAJOR=${OpenCV_VERSION_MAJOR}
        HAS_NVML=1
    )
    target_compile_options(feature_extraction PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --ptxas-options=-v>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    )
endif()

message(STATUS "Found source files: ${BASE_SOURCES}")
message(STATUS "Target GPU: Tesla T4 (Compute Capability 7.5)")