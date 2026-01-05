# cmake for just testing one kernel at a time with their harnesses

cmake_minimum_required(VERSION 3.18)
project(KernelTests CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;86;89") # this right??? TODO: change for other gpu archs... not finalized on what to test on yet

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../past-work/include)

# all the individual kernel test executables r below w/ their .cu and .cpp files
add_executable(test_canny
    examples/test_canny.cpp
    canny.cu
)
target_link_libraries(test_canny ${OpenCV_LIBS} CUDA::cudart)

add_executable(test_harris
    examples/test_harris.cpp
    harris.cu
)
target_link_libraries(test_harris ${OpenCV_LIBS} CUDA::cudart)

add_executable(test_matmul
    examples/test_matmul.cpp
    matmul.cu
)
target_link_libraries(test_matmul CUDA::cudart)

add_executable(test_conv2d
    examples/test_conv2d.cpp
    conv2d.cu
)
target_link_libraries(test_conv2d ${OpenCV_LIBS} CUDA::cudart)

add_executable(test_histogram
    examples/test_histogram.cpp
    histogram.cu
)
target_link_libraries(test_histogram ${OpenCV_LIBS} CUDA::cudart)

add_executable(test_gaussian
    examples/test_gaussian.cpp
    gaussian_blur.cu
)
target_link_libraries(test_gaussian ${OpenCV_LIBS} CUDA::cudart)

add_executable(test_bandwidth
    examples/test_bandwidth.cpp
    synthetic_bandwidth.cu
)
target_link_libraries(test_bandwidth CUDA::cudart)

add_executable(test_compute
    examples/test_compute.cpp
    synthetic_compute.cu
)
target_link_libraries(test_compute CUDA::cudart)

