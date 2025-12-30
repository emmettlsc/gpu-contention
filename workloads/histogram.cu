/*
 * histogram kernel (256-bin grayscale)
 *
 * input: grayscale image (unsigned char*), width x height
 * output: 256-element histogram (unsigned int*)
 * characteristics: atomic-bound, irregular memory access, serializes at l2 cache
 *
 * computes frequency distribution of pixel intensities (0-255)
 * uses atomicAdd which causes serialization when bins contend
 *
 * to run solo: see workloads/examples/test_histogram.cpp
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===== histogram using global atomics =====

__global__ void histogram_kernel(const unsigned char* input, unsigned int* hist,
                                int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    // each thread processes multiple pixels
    for (int i = idx; i < total_pixels; i += blockDim.x * gridDim.x) {
        unsigned char value = input[i];
        atomicAdd(&hist[value], 1);
    }
}

// ===== launch wrapper =====

extern "C" void launch_histogram_kernel(const unsigned char* d_input, unsigned int* d_hist,
                                       int width, int height, cudaStream_t stream) {
    // clear histogram first
    cudaMemsetAsync(d_hist, 0, 256 * sizeof(unsigned int), stream);

    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 1024);

    histogram_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_hist, width, height);
}
