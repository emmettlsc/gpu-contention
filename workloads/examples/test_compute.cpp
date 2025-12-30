/*
 * standalone test harness for synthetic compute kernel
 *
 * run: ./test_compute [num_elements] [iterations]
 * so for example you can do ./test_compute 2073600 100
 *
 * pure compute performance w/ configurable intensity (higher iterations = more compute per element)
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

extern "C" void launch_synthetic_compute_kernel(const float* d_input, float* d_output,
                                               int N, int iterations, cudaStream_t stream);

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1920 * 1080;
    int iterations = argc > 2 ? atoi(argv[2]) : 100;

    std::cout << "synthetic compute test\n";
    std::cout << "elements: " << N << "\n";
    std::cout << "iterations per element: " << iterations << "\n";

    size_t size = N * sizeof(float);
    float *h_input, *h_output;
    float *d_input, *d_output;

    // allocate memory
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    // initialize with pattern
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f + (float)i / N;
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // warmup
    launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    cudaDeviceSynchronize();

    // benchmark
    std::cout << "running compute test...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_runs = 10;
    cudaEventRecord(start);

    for (int i = 0; i < num_runs; i++) {
        launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_ms = elapsed_ms / num_runs;

    std::cout << "execution time: " << avg_time_ms << " ms (average over " << num_runs << " runs)\n";

    // calculate gflops
    // each iteration does 4 fma operations = 8 flops
    // total flops = N * iterations * 8
    float total_flops = (float)N * iterations * 8.0f;
    float gflops = (total_flops / (avg_time_ms / 1000.0f)) / 1e9;

    std::cout << "performance: " << gflops << " GFLOPS\n";

    // arithmetic intensity
    // bytes transferred: N floats in + N floats out = 2N * 4 bytes
    float bytes_transferred = 2.0f * N * sizeof(float);
    float arithmetic_intensity = total_flops / bytes_transferred;

    std::cout << "arithmetic intensity: " << arithmetic_intensity << " FLOPs/byte\n";

    // copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // print sample values
    std::cout << "\nsample values:\n";
    std::cout << "input[0] = " << h_input[0] << " -> output[0] = " << h_output[0] << "\n";
    std::cout << "input[100] = " << h_input[100] << " -> output[100] = " << h_output[100] << "\n";

    // cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
