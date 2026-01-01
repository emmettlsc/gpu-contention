/*
 * standalone test harness for synthetic bandwidth kernel
 *
 * run: ./test_bandwidth [num_elements]
 * example: ./test_bandwidth 10000000
 *
 * tests pure memory bandwidth by performing simple memory copy
 * useful for measuring gpu memory subsystem performance
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

extern "C" void launch_synthetic_bandwidth_kernel(const float* d_input, float* d_output,
                                                  int N, cudaStream_t stream);

int main(int argc, char** argv) {
    // default: 1080p image worth of data
    int N = argc > 1 ? atoi(argv[1]) : 1920 * 1080;

    std::cout << "synthetic bandwidth test\n";
    std::cout << "elements: " << N << " (" << (N * sizeof(float) / 1024 / 1024) << " MB)\n";

    size_t size = N * sizeof(float);
    float *h_input, *h_output;
    float *d_input, *d_output;

    // allocate memory
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    // initialize with pattern
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i / N;
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // copy to device (not timed)
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // warmup
    launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    cudaDeviceSynchronize();

    // benchmark
    std::cout << "running bandwidth test...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 10;
    cudaEventRecord(start);

    for (int i = 0; i < num_iterations; i++) {
        launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_ms = elapsed_ms / num_iterations;

    std::cout << "execution time: " << avg_time_ms << " ms (average over " << num_iterations << " runs)\n";

    // calculate bandwidth
    // we read N floats and write N floats = 2N * sizeof(float) bytes
    float bytes_transferred = 2.0f * N * sizeof(float);
    float bandwidth_gbs = (bytes_transferred / (avg_time_ms / 1000.0f)) / 1e9;

    std::cout << "achieved bandwidth: " << bandwidth_gbs << " GB/s\n";

    // verify correctness
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N && i < 1000; i++) {
        if (h_output[i] != h_input[i]) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "✓ data copied correctly\n";
    } else {
        std::cout << "✗ data copy error!\n";
    }

    // cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
