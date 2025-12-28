/*
 * 2d convolution kernel (5x5 gaussian)
 *
 * input: image (float*), width x height
 * output: filtered image (float*), same dimensions
 * characteristics: compute-bound, moderate arithmetic intensity (~8 FLOPs/byte)
 *
 * applies 5x5 gaussian convolution filter using shared memory tiling
 * reduces global memory accesses by loading tile with halo region
 *
 * to run solo: see workloads/examples/test_conv2d.cpp
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CONV_TILE_SIZE 16
#define CONV_KERNEL_RADIUS 2
#define CONV_KERNEL_SIZE 5

// ===== 2d convolution with shared memory tiling =====

__global__ void conv2d_kernel(const float* input, float* output, int width, int height) {
    // shared memory for tile with halo
    __shared__ float kernel[CONV_KERNEL_SIZE][CONV_KERNEL_SIZE];
    __shared__ float tile[CONV_TILE_SIZE + 2*CONV_KERNEL_RADIUS][CONV_TILE_SIZE + 2*CONV_KERNEL_RADIUS];

    // initialize gaussian kernel (5x5)
    if (threadIdx.x < CONV_KERNEL_SIZE && threadIdx.y < CONV_KERNEL_SIZE) {
        float k[5][5] = {
            {1, 4, 7, 4, 1},
            {4, 16, 26, 16, 4},
            {7, 26, 41, 26, 7},
            {4, 16, 26, 16, 4},
            {1, 4, 7, 4, 1}
        };
        kernel[threadIdx.y][threadIdx.x] = k[threadIdx.y][threadIdx.x] / 273.0f;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * CONV_TILE_SIZE + tx;
    int row = blockIdx.y * CONV_TILE_SIZE + ty;

    // load tile with halo into shared memory
    for (int dy = 0; dy < CONV_TILE_SIZE + 2*CONV_KERNEL_RADIUS; dy += CONV_TILE_SIZE) {
        for (int dx = 0; dx < CONV_TILE_SIZE + 2*CONV_KERNEL_RADIUS; dx += CONV_TILE_SIZE) {
            int tile_y = ty + dy;
            int tile_x = tx + dx;

            if (tile_y < CONV_TILE_SIZE + 2*CONV_KERNEL_RADIUS &&
                tile_x < CONV_TILE_SIZE + 2*CONV_KERNEL_RADIUS) {
                int img_y = blockIdx.y * CONV_TILE_SIZE + tile_y - CONV_KERNEL_RADIUS;
                int img_x = blockIdx.x * CONV_TILE_SIZE + tile_x - CONV_KERNEL_RADIUS;

                // clamp to boundaries
                img_y = max(0, min(img_y, height - 1));
                img_x = max(0, min(img_x, width - 1));

                tile[tile_y][tile_x] = input[img_y * width + img_x];
            }
        }
    }

    __syncthreads();

    // compute convolution
    if (row < height && col < width) {
        float sum = 0.0f;

        #pragma unroll
        for (int ky = 0; ky < CONV_KERNEL_SIZE; ky++) {
            #pragma unroll
            for (int kx = 0; kx < CONV_KERNEL_SIZE; kx++) {
                int tile_y = ty + ky;
                int tile_x = tx + kx;
                sum += tile[tile_y][tile_x] * kernel[ky][kx];
            }
        }

        output[row * width + col] = sum;
    }
}

// ===== launch wrapper =====

extern "C" void launch_conv2d_kernel(const float* d_input, float* d_output,
                                     int width, int height, cudaStream_t stream) {
    dim3 blockSize(CONV_TILE_SIZE, CONV_TILE_SIZE);
    dim3 gridSize((width + CONV_TILE_SIZE - 1) / CONV_TILE_SIZE,
                  (height + CONV_TILE_SIZE - 1) / CONV_TILE_SIZE);
    conv2d_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, width, height);
}
