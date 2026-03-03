#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cuda_runtime.h>

extern "C" void launch_canny_full_pipeline(
    unsigned char* d_input,
    unsigned char* d_output,
    float* d_temp_buffers,
    int width, int height,
    float low_threshold,
    float high_threshold,
    cudaStream_t stream
);

extern "C" void launch_harris_full_pipeline(
    unsigned char* d_input,
    unsigned char* d_output,
    float* d_temp_response,
    int width, int height,
    int block_size,
    float k,
    float threshold,
    cudaStream_t stream
);

extern "C" void launch_histogram_kernel(
    const unsigned char* d_input,
    unsigned int* d_hist,
    int width, int height,
    cudaStream_t stream
);

extern "C" void launch_conv2d_kernel(
    const float* d_input,
    float* d_output,
    const float* d_kernel,
    int width, int height,
    int kernel_size,
    cudaStream_t stream
);

extern "C" void launch_gaussian_blur_float_kernel(
    const float* d_input,
    float* d_output,
    int width, int height,
    cudaStream_t stream
);

extern "C" void launch_gaussian_blur_uchar_kernel(
    unsigned char* d_input,
    float* d_output,
    int width, int height,
    cudaStream_t stream
);

extern "C" void launch_matmul_kernel(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream
);

extern "C" void launch_synthetic_bandwidth_kernel(
    const float* d_input,
    float* d_output,
    int N,
    cudaStream_t stream
);

extern "C" void launch_synthetic_compute_kernel(
    const float* d_input,
    float* d_output,
    int N,
    int iterations,
    cudaStream_t stream
);

void test_matmul_solo(int N, float* time_ms);
void test_conv2d_solo(int width, int height, float* time_ms);
void test_histogram_solo(int width, int height, float* time_ms);
void test_synthetic_bandwidth_solo(int N, float* time_ms);
void test_synthetic_compute_solo(int N, int iterations, float* time_ms);

#endif // KERNELS_HPP
