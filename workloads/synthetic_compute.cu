/*
 * synthetic compute kernel (pure compute test)
 *
 * input: array (float*), size N
 * output: computed array (float*), size N
 * parameters: iterations (int) - controls compute intensity
 * characteristics: pure compute, minimal memory access, tests sm utilization
 *
 * performs many fused multiply-add operations per element
 * designed to saturate compute resources without memory bottleneck
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===== ure compute-intensive kernel =====

__global__ void synthetic_compute_kernel(const float* input, float* output, int N,
                                        int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    float value = input[idx];

    // perform many fused multiply-add operations
    // where each iteration does 8 FLOPs (4 fma instructions)
    #pragma unroll 8
    for (int i = 0; i < iterations; i++) {
        value = value * 1.0001f + 0.0001f;
        value = value * 0.9999f + 0.0001f;
        value = value * 1.0001f - 0.0001f;
        value = value * 0.9999f - 0.0001f;
    }

    output[idx] = value;
}

// ===== launch wrapper =====

extern "C" void launch_synthetic_compute_kernel(const float* d_input, float* d_output,
                                               int N, int iterations, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    synthetic_compute_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, N, iterations);
}
