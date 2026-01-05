// to ensure the kernel is actually stressing what i think its stressing 
// nvml monitoring to measure sm util, mem bw, and arithmetic intensity 
#include "../workloads/kernels.hpp"
#include "../past-work/include/gpu_monitor.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>

// ===== kernel characterization data =====

struct KernelCharacteristics {
    std::string name;
    float avg_time_ms;
    float sm_utilization_pct;
    float memory_bandwidth_gbs;
    size_t bytes_transferred;
    size_t flops;
    float arithmetic_intensity;  // flops/byte
    std::string bottleneck_type;
};

// ===== nvml sampling during kernel execution =====

struct NVMLSamples {
    std::vector<float> sm_util;
    std::vector<float> mem_bw;
    std::vector<float> mem_util;
};

// sample nvml metrics during kernel execution
void sample_nvml_during_execution(GPUMonitor& monitor, std::atomic<bool>& running,
                                  NVMLSamples& samples) {
    while (running) {
        GPUResourceState state = monitor.get_current_state();
        if (state.is_valid) {
            samples.sm_util.push_back(state.sm_utilization);
            samples.mem_bw.push_back(state.memory_bandwidth_used_gb_s);
            samples.mem_util.push_back(state.memory_utilization);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500)); // sample every 0.5ms
    }
}

// calculate average from samples
float calculate_average(const std::vector<float>& samples) {
    if (samples.empty()) return 0.0f;
    float sum = 0.0f;
    for (float val : samples) sum += val;
    return sum / samples.size();
}

// ===== characterize individual kernels =====

KernelCharacteristics characterize_matmul(GPUMonitor& monitor, int N) {
    KernelCharacteristics result;
    result.name = "matmul_" + std::to_string(N) + "x" + std::to_string(N);

    // allocate memory
    float *d_A, *d_B, *d_C;
    size_t matrix_bytes = N * N * sizeof(float);
    cudaMalloc(&d_A, matrix_bytes);
    cudaMalloc(&d_B, matrix_bytes);
    cudaMalloc(&d_C, matrix_bytes);

    // initialize with random data
    std::vector<float> h_data(N * N);
    for (auto& v : h_data) v = (float)rand() / RAND_MAX;
    cudaMemcpy(d_A, h_data.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_data.data(), matrix_bytes, cudaMemcpyHostToDevice);

    // warmup
    launch_matmul_kernel(d_A, d_B, d_C, N, 0);
    cudaDeviceSynchronize();

    // run with nvml sampling
    NVMLSamples samples;
    std::atomic<bool> running{true};
    std::thread sampler(sample_nvml_during_execution, std::ref(monitor),
                       std::ref(running), std::ref(samples));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run multiple times for stable measurements
    const int num_runs = 10;
    float total_time = 0.0f;

    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        launch_matmul_kernel(d_A, d_B, d_C, N, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        total_time += time_ms;
    }

    running = false;
    sampler.join();

    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util);
    result.memory_bandwidth_gbs = calculate_average(samples.mem_bw);

    // arithmetic intensity: matmul does 2N^3 flops (N^3 multiply-adds)
    // reads 2*N^2 elements (A and B), writes N^2 elements (C)
    result.flops = 2ULL * N * N * N;
    result.bytes_transferred = 3ULL * N * N * sizeof(float);
    result.arithmetic_intensity = (float)result.flops / result.bytes_transferred;

    // classify bottleneck
    if (result.arithmetic_intensity > 5.0f) {
        result.bottleneck_type = "compute-bound";
    } else {
        result.bottleneck_type = "memory-bound";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_conv2d(GPUMonitor& monitor, int width, int height) {
    KernelCharacteristics result;
    result.name = "conv2d_" + std::to_string(width) + "x" + std::to_string(height);

    size_t img_bytes = width * height * sizeof(float);
    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_output, img_bytes);
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    // initialize
    std::vector<float> h_data(width * height, 0.5f);
    cudaMemcpy(d_input, h_data.data(), img_bytes, cudaMemcpyHostToDevice);

    float h_kernel[9] = {1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9};
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // warmup
    launch_conv2d_kernel(d_input, d_output, d_kernel, width, height, 3, 0);
    cudaDeviceSynchronize();

    // run with sampling
    NVMLSamples samples;
    std::atomic<bool> running{true};
    std::thread sampler(sample_nvml_during_execution, std::ref(monitor),
                       std::ref(running), std::ref(samples));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_runs = 10;
    float total_time = 0.0f;

    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        launch_conv2d_kernel(d_input, d_output, d_kernel, width, height, 3, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        total_time += time_ms;
    }

    running = false;
    sampler.join();

    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util);
    result.memory_bandwidth_gbs = calculate_average(samples.mem_bw);

    // 3x3 conv: 9 multiply-adds per output pixel
    result.flops = 2ULL * 9 * width * height;
    result.bytes_transferred = 2ULL * width * height * sizeof(float); // read input + write output
    result.arithmetic_intensity = (float)result.flops / result.bytes_transferred;

    if (result.arithmetic_intensity > 5.0f) {
        result.bottleneck_type = "compute-bound";
    } else {
        result.bottleneck_type = "memory-bound";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_histogram(GPUMonitor& monitor, int width, int height) {
    KernelCharacteristics result;
    result.name = "histogram_" + std::to_string(width) + "x" + std::to_string(height);

    size_t img_bytes = width * height * sizeof(unsigned char);
    unsigned char* d_input;
    unsigned int* d_hist;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));

    std::vector<unsigned char> h_data(width * height);
    for (auto& v : h_data) v = rand() % 256;
    cudaMemcpy(d_input, h_data.data(), img_bytes, cudaMemcpyHostToDevice);

    launch_histogram_kernel(d_input, d_hist, width, height, 0);
    cudaDeviceSynchronize();

    NVMLSamples samples;
    std::atomic<bool> running{true};
    std::thread sampler(sample_nvml_during_execution, std::ref(monitor),
                       std::ref(running), std::ref(samples));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_runs = 10;
    float total_time = 0.0f;

    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        launch_histogram_kernel(d_input, d_hist, width, height, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        total_time += time_ms;
    }

    running = false;
    sampler.join();

    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util);
    result.memory_bandwidth_gbs = calculate_average(samples.mem_bw);

    // histogram reads each pixel once, writes to histogram (atomics)
    result.flops = 0; // minimal compute
    result.bytes_transferred = width * height * sizeof(unsigned char);
    result.arithmetic_intensity = 0.0f;
    result.bottleneck_type = "atomic-bound";

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_synthetic_bandwidth(GPUMonitor& monitor, int N) {
    KernelCharacteristics result;
    result.name = "synthetic_bandwidth_" + std::to_string(N) + "_elements";

    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    std::vector<float> h_data(N, 1.0f);
    cudaMemcpy(d_input, h_data.data(), bytes, cudaMemcpyHostToDevice);

    launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    cudaDeviceSynchronize();

    NVMLSamples samples;
    std::atomic<bool> running{true};
    std::thread sampler(sample_nvml_during_execution, std::ref(monitor),
                       std::ref(running), std::ref(samples));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_runs = 10;
    float total_time = 0.0f;

    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        total_time += time_ms;
    }

    running = false;
    sampler.join();

    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util);
    result.memory_bandwidth_gbs = calculate_average(samples.mem_bw);

    // reads N floats, writes N floats
    result.flops = 0;
    result.bytes_transferred = 2ULL * N * sizeof(float);
    result.arithmetic_intensity = 0.0f;
    result.bottleneck_type = "memory-bound";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_synthetic_compute(GPUMonitor& monitor, int N, int iterations) {
    KernelCharacteristics result;
    result.name = "synthetic_compute_" + std::to_string(N) + "_elements_" +
                  std::to_string(iterations) + "_iters";

    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    std::vector<float> h_data(N, 1.0f);
    cudaMemcpy(d_input, h_data.data(), bytes, cudaMemcpyHostToDevice);

    launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    cudaDeviceSynchronize();

    NVMLSamples samples;
    std::atomic<bool> running{true};
    std::thread sampler(sample_nvml_during_execution, std::ref(monitor),
                       std::ref(running), std::ref(samples));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_runs = 10;
    float total_time = 0.0f;

    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        total_time += time_ms;
    }

    running = false;
    sampler.join();

    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util);
    result.memory_bandwidth_gbs = calculate_average(samples.mem_bw);

    // each iteration does ~8 fma ops per element (see synthetic_compute.cu)
    result.flops = (size_t)N * iterations * 8 * 2; // 8 fma = 16 flops
    result.bytes_transferred = 2ULL * N * sizeof(float); // read input + write output
    result.arithmetic_intensity = (float)result.flops / result.bytes_transferred;
    result.bottleneck_type = "compute-bound";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

// ===== main characterization =====

void print_characteristics(const KernelCharacteristics& k) {
    printf("%-40s | %8.2f ms | %6.1f%% | %8.1f GB/s | %10.2f | %-15s\n",
           k.name.c_str(),
           k.avg_time_ms,
           k.sm_utilization_pct,
           k.memory_bandwidth_gbs,
           k.arithmetic_intensity,
           k.bottleneck_type.c_str());
}

int main() {
    printf("=== kernel characterization test ===\n\n");

    // initialize gpu monitor
    GPUMonitor monitor;
    if (!monitor.initialize()) {
        fprintf(stderr, "failed to initialize gpu monitor - nvml required\n");
        return 1;
    }

    GPUHardwareInfo hw = monitor.get_hardware_info();
    printf("gpu: %s\n", hw.name.c_str());
    printf("compute capability: %d.%d\n", hw.major_compute_capability, hw.minor_compute_capability);
    printf("sm count: %d\n", hw.sm_count);
    printf("max bandwidth: %.1f GB/s\n\n", hw.memory_bandwidth_gb_s);

    printf("%-40s | %10s | %8s | %12s | %12s | %-15s\n",
           "kernel", "time", "sm util", "bandwidth", "arith int", "bottleneck");
    printf("----------------------------------------+------------+----------+--------------+--------------+-----------------\n");

    // characterize all kernels
    std::vector<KernelCharacteristics> results;

    // matmul at different sizes
    results.push_back(characterize_matmul(monitor, 256));
    print_characteristics(results.back());

    results.push_back(characterize_matmul(monitor, 512));
    print_characteristics(results.back());

    results.push_back(characterize_matmul(monitor, 1024));
    print_characteristics(results.back());

    // conv2d
    results.push_back(characterize_conv2d(monitor, 1280, 720));
    print_characteristics(results.back());

    results.push_back(characterize_conv2d(monitor, 1920, 1080));
    print_characteristics(results.back());

    // histogram
    results.push_back(characterize_histogram(monitor, 1280, 720));
    print_characteristics(results.back());

    results.push_back(characterize_histogram(monitor, 1920, 1080));
    print_characteristics(results.back());

    // synthetic kernels
    results.push_back(characterize_synthetic_bandwidth(monitor, 1920*1080));
    print_characteristics(results.back());

    results.push_back(characterize_synthetic_compute(monitor, 1920*1080, 50));
    print_characteristics(results.back());

    results.push_back(characterize_synthetic_compute(monitor, 1920*1080, 100));
    print_characteristics(results.back());

    results.push_back(characterize_synthetic_compute(monitor, 1920*1080, 200));
    print_characteristics(results.back());

    printf("\n=== characterization complete ===\n");
    printf("use this data to design workload pairs in workloads/kernel_specs.md\n");

    return 0;
}
