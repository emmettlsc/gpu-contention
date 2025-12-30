/*
 * standalone test harness for matrix multiply
 *
 * run: ./test_matmul [matrix_size]
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

extern "C" void launch_matmul_kernel(const float* d_A, const float* d_B, float* d_C,
                                     int N, cudaStream_t stream);

// simple cpu matmul for verification
void cpu_matmul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 256;

    std::cout << "matrix multiply: " << N << " x " << N << "\n";

    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C, *h_C_ref;

    // allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_C_ref = (float*)malloc(size);

    // initialize with simple pattern
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // run gpu matmul
    std::cout << "running gpu matrix multiply...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    launch_matmul_kernel(d_A, d_B, d_C, N, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "execution time: " << elapsed_ms << " ms\n";

    // compute gflops
    float gflops = (2.0f * N * N * N) / (elapsed_ms / 1000.0f) / 1e9;
    std::cout << "performance: " << gflops << " GFLOPS\n";

    // verify correctness (only for small matrices)
    if (N <= 512) {
        std::cout << "verifying correctness...\n";
        cpu_matmul(h_A, h_B, h_C_ref, N);

        float max_error = 0.0f;
        for (int i = 0; i < N*N; i++) {
            float error = fabs(h_C[i] - h_C_ref[i]);
            max_error = fmax(max_error, error);
        }

        std::cout << "max error: " << max_error << "\n";
        if (max_error < 1e-3) {
            std::cout << "✓ result correct!\n";
        } else {
            std::cout << "✗ result incorrect!\n";
        }
    }

    // print sample values
    std::cout << "\nsample values (top-left 3x3 of result):\n";
    for (int i = 0; i < 3 && i < N; i++) {
        for (int j = 0; j < 3 && j < N; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
