/*
 * matrix multiply kernel (tiled implementation)
 *
 * input: two square matrices A and B (float*), size N x N each
 * output: matrix C = A × B (float*), size N x N
 * characteristics: compute-bound, high arithmetic intensity (~2N FLOPs per element)
 *
 * shared memory tiling to reduce global memory accesses
 * each tile is 16x16, loaded collaboratively by thread block
 *
 * to run solo: look at instructions in harness
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 16

// ===== tiled mat mul=====

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // load tile from A
        int tileRow = row;
        int tileCol = t * TILE_SIZE + threadIdx.x;
        if (tileRow < N && tileCol < N) {
            As[threadIdx.y][threadIdx.x] = A[tileRow * N + tileCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load tile from B
        tileRow = t * TILE_SIZE + threadIdx.y;
        tileCol = col;
        if (tileRow < N && tileCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tileRow * N + tileCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ===== launch wrapper =====

extern "C" void launch_matmul_kernel(const float* d_A, const float* d_B, float* d_C,
                                     int N, cudaStream_t stream) {
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, N);
}
