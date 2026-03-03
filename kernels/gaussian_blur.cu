/*
 * gaussian blur kernel (5x5)
 *
 * input: image (float*), width x height
 * output: blurred image (float*), same dimensions
 * characteristics: memory-bound, coalesced memory access, moderate arithmetic intensity
 *
 * applies 5x5 gaussian blur for noise reduction
 * simpler than conv2d (bc no shared memory optimization)
 *
 * to run solo: look at instructions in workloads/examples/test_gaussian.cpp
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===== simple gaussian blur (w/o shared memory) =====

__global__ void gaussian_blur_float_kernel(const float* input, float* output,
                                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 5x5 gaussian kernel
    float kernel[25] = {
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1
    };
    float kernel_sum = 256.0f;

    float sum = 0.0f;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            int kernel_idx = (ky + 2) * 5 + (kx + 2);
            sum += input[py * width + px] * kernel[kernel_idx];
        }
    }

    output[y * width + x] = sum / kernel_sum;
}

// version for unsigned char input (used in canny)
__global__ void gaussian_blur_uchar_kernel(unsigned char* input, float* output,
                                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float kernel[25] = {
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1
    };
    float kernel_sum = 256.0f;

    float sum = 0.0f;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            int kernel_idx = (ky + 2) * 5 + (kx + 2);
            sum += input[py * width + px] * kernel[kernel_idx];
        }
    }

    output[y * width + x] = sum / kernel_sum;
}

// ===== launch wrappers =====

extern "C" void launch_gaussian_blur_float_kernel(const float* d_input, float* d_output,
                                                  int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    gaussian_blur_float_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, width, height);
}

extern "C" void launch_gaussian_blur_uchar_kernel(unsigned char* d_input, float* d_output,
                                                  int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    gaussian_blur_uchar_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, width, height);
}
