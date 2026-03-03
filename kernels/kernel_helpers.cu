// helpers JSUT FOR standalone kernel testing 
#include "kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

void test_matmul_solo(int N, float* time_ms) {
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    launch_matmul_kernel(d_A, d_B, d_C, N, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_matmul_kernel(d_A, d_B, d_C, N, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

void test_conv2d_solo(int width, int height, float* time_ms) {
    int img_size = width * height;

    float* h_input = new float[img_size];
    float* h_output = new float[img_size];

    for (int i = 0; i < img_size; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    int kernel_size = 3;
    float h_kernel[9] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    cudaMemcpy(d_input, h_input, img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    launch_conv2d_kernel(d_input, d_output, d_kernel, width, height, kernel_size, 0);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_conv2d_kernel(d_input, d_output, d_kernel, width, height, kernel_size, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;
}


void test_histogram_solo(int width, int height, float* time_ms) {
    int img_size = width * height;
    unsigned char* h_input = new unsigned char[img_size];
    unsigned int* h_hist = new unsigned int[256];

    for (int i = 0; i < img_size; i++) {
        h_input[i] = rand() % 256;
    }

    unsigned char* d_input;
    unsigned int* d_hist;
    cudaMalloc(&d_input, img_size * sizeof(unsigned char));
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));

    cudaMemcpy(d_input, h_input, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    launch_histogram_kernel(d_input, d_hist, width, height, 0);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_histogram_kernel(d_input, d_hist, width, height, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_hist;
}


void test_synthetic_bandwidth_solo(int N, float* time_ms) {
    float* h_input = new float[N];
    float* h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;
}

// ===== synthetic compute test =====

void test_synthetic_compute_solo(int N, int iterations, float* time_ms) {
    float* h_input = new float[N];
    float* h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i * 0.001f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;
}
