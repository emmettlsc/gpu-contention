/*
 * synthetic bandwidth kernel (pure memory test)
 *
 * input: array (float*), size N
 * output: copied array (float*), size N
 * characteristics: pure memory bandwidth, minimal compute, tests memory subsystem limits
 *
 * performs simple memory copy operation with strided access
 * designed to saturate memory bandwidth without compute overhead
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===== simple bandwidth test (memory copy) =====

__global__ void synthetic_bandwidth_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // simple copy with strided access
    for (int i = idx; i < N; i += stride) {
        output[i] = input[i];
    }
}

// ===== launch wrapper =====

extern "C" void launch_synthetic_bandwidth_kernel(const float* d_input, float* d_output,
                                                  int N, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = min((N + blockSize - 1) / blockSize, 2048);
    synthetic_bandwidth_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, N);
}
