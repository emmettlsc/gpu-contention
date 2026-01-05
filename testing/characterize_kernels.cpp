// characterize all kernels to verify they stress the gpu the way i expect uses cuda event timing (accurate) and nvml polling (coarse but useful for relative comparison)

// things get messed up using just nvml sampling below:
// ./characterize_kernels test_image.jpg
// runs each kernel for ~3 seconds with nvml sampling

//  for wayyy more realistic profiling run with ncu (no nvml sampling)
// ./characterize_kernels --ncu test_image.jpg
// ->runs each kernel 3 times, no nvml (designed to be wrapped by ncu)
// use with: experiments/scripts/run_ncu_characterization.sh test_image.jpg

#include "../workloads/kernels.hpp"
#include "../past-work/include/gpu_monitor.hpp"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <cstring>

// set w/ --ncu arg if wanted
static bool g_ncu_mode = false;
#define LOG(...) fprintf(g_ncu_mode ? stderr : stdout, __VA_ARGS__)

// ===== kernel characterization data =====
struct KernelCharacteristics {
    std::string name;
    float avg_time_ms;
    float sm_utilization_pct;      // nvml: % of time SMs were active
    float mem_utilization_pct;  // nvml: % of time memory controllers were active
    size_t theoretical_bytes;
    size_t theoretical_flops;
    float theoretical_arith_intensity;  // flops/byte (theoretical, not measured)
    std::string bottleneck_type;
    int num_runs;
    size_t num_nvml_samples;
};

// ===== nvml sampling during kernel execution =====

struct NVMLSamples {
    std::vector<float> sm_util;
    std::vector<float> mem_util;
};

// sample nvml metrics during kernel execution, store in struct
void sample_nvml_during_execution(GPUMonitor& monitor, std::atomic<bool>& running,
                                  NVMLSamples& samples) {
    while (running) {
        GPUResourceState state = monitor.get_current_state();
        if (state.is_valid) {
            samples.sm_util.push_back(state.sm_utilization);
            samples.mem_util.push_back(state.memory_utilization);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// calculate average from samples
// NOTE: better to skip first N (notice warmup contamination???? w/o skip)
float calculate_average(const std::vector<float>& samples, size_t skip = 0) {
    if (samples.size() <= skip) return 0.0f;
    float sum = 0.0f;
    for (size_t i = skip; i < samples.size(); i++) sum += samples[i];
    return sum / (samples.size() - skip);
}

// ===== characterize individual kernels =====
// 1. allocate + copy data
// 2. dummy run to clear lazy init
// 3. measure single run time for num_runs calculation (only for nvml as else runtims will be too short for many samples)
// 4. enqueue all runs async (no per-run sync), time the whole batch <-- llm says this is needed
// 5. sample nvml in background thread during the batch

KernelCharacteristics characterize_matmul(GPUMonitor& monitor, int N) {
    KernelCharacteristics result;
    result.name = "matmul_" + std::to_string(N) + "x" + std::to_string(N);

    float *d_A, *d_B, *d_C;
    size_t matrix_bytes = N * N * sizeof(float);
    cudaMalloc(&d_A, matrix_bytes);
    cudaMalloc(&d_B, matrix_bytes);
    cudaMalloc(&d_C, matrix_bytes);

    std::vector<float> h_data(N * N);
    for (auto& v : h_data) v = (float)rand() / RAND_MAX;
    cudaMemcpy(d_A, h_data.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_data.data(), matrix_bytes, cudaMemcpyHostToDevice);

    // dummy run to clear lazy init
    launch_matmul_kernel(d_A, d_B, d_C, N, 0);
    cudaDeviceSynchronize();

    // measure single run time (post-warmup)
    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_matmul_kernel(d_A, d_B, d_C, N, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    // target 3 seconds of gpu-saturated runtime
    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    // enqueue all kernels async, time the whole batch
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_matmul_kernel(d_A, d_B, d_C, N, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    result.theoretical_flops = 2ULL * N * N * N;
    result.theoretical_bytes = 3ULL * N * N * sizeof(float);
    result.theoretical_arith_intensity = (float)result.theoretical_flops / result.theoretical_bytes;
    result.bottleneck_type = (result.theoretical_arith_intensity > 5.0f) ? "compute-bound" : "memory-bound";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_conv2d(GPUMonitor& monitor, int width, int height,
                                          const float* h_image_data_float) {
    KernelCharacteristics result;
    result.name = "conv2d_" + std::to_string(width) + "x" + std::to_string(height);

    size_t img_bytes = width * height * sizeof(float);
    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_output, img_bytes);
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    cudaMemcpy(d_input, h_image_data_float, img_bytes, cudaMemcpyHostToDevice);
    float h_kernel[9] = {1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9};
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // dummy + measure
    launch_conv2d_kernel(d_input, d_output, d_kernel, width, height, 3, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_conv2d_kernel(d_input, d_output, d_kernel, width, height, 3, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_conv2d_kernel(d_input, d_output, d_kernel, width, height, 3, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    result.theoretical_flops = 2ULL * 9 * width * height;
    result.theoretical_bytes = 2ULL * width * height * sizeof(float);
    result.theoretical_arith_intensity = (float)result.theoretical_flops / result.theoretical_bytes;
    result.bottleneck_type = (result.theoretical_arith_intensity > 5.0f) ? "compute-bound" : "memory-bound";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_histogram(GPUMonitor& monitor, int width, int height,
                                             const unsigned char* h_image_data) {
    KernelCharacteristics result;
    result.name = "histogram_" + std::to_string(width) + "x" + std::to_string(height);

    size_t img_bytes = width * height * sizeof(unsigned char);
    unsigned char* d_input;
    unsigned int* d_hist;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));

    cudaMemcpy(d_input, h_image_data, img_bytes, cudaMemcpyHostToDevice);

    // dummy + measure
    launch_histogram_kernel(d_input, d_hist, width, height, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_histogram_kernel(d_input, d_hist, width, height, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_histogram_kernel(d_input, d_hist, width, height, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    result.theoretical_flops = 0;
    result.theoretical_bytes = width * height * sizeof(unsigned char);
    result.theoretical_arith_intensity = 0.0f;
    result.bottleneck_type = "atomic-bound";

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_synthetic_bandwidth(GPUMonitor& monitor, int N) {
    KernelCharacteristics result;
    result.name = "synthetic_bw_" + std::to_string(N) + "_elems";

    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    std::vector<float> h_data(N, 1.0f);
    cudaMemcpy(d_input, h_data.data(), bytes, cudaMemcpyHostToDevice);

    // dummy + measure
    launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_synthetic_bandwidth_kernel(d_input, d_output, N, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    result.theoretical_flops = 0;
    result.theoretical_bytes = 2ULL * N * sizeof(float);
    result.theoretical_arith_intensity = 0.0f;
    result.bottleneck_type = "memory-bound";

    // also compute achieved bandwidth from timing (this IS accurate unlike nvml)
    float achieved_bw_gbs = (result.theoretical_bytes / (result.avg_time_ms / 1000.0f)) / 1e9;
    LOG("  [achieved bandwidth: %.1f GB/s]\n", achieved_bw_gbs);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_synthetic_compute(GPUMonitor& monitor, int N, int iterations) {
    KernelCharacteristics result;
    result.name = "synthetic_compute_" + std::to_string(N) + "_" +
                  std::to_string(iterations) + "iter";

    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    std::vector<float> h_data(N, 1.0f);
    cudaMemcpy(d_input, h_data.data(), bytes, cudaMemcpyHostToDevice);

    // dummy + measure
    launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_synthetic_compute_kernel(d_input, d_output, N, iterations, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    result.theoretical_flops = (size_t)N * iterations * 8 * 2; // 8 fma = 16 flops
    result.theoretical_bytes = 2ULL * N * sizeof(float);
    result.theoretical_arith_intensity = (float)result.theoretical_flops / result.theoretical_bytes;
    result.bottleneck_type = "compute-bound";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_canny(GPUMonitor& monitor, int width, int height,
                                         const unsigned char* h_image_data) {
    KernelCharacteristics result;
    result.name = "canny_" + std::to_string(width) + "x" + std::to_string(height);

    size_t img_bytes = width * height * sizeof(unsigned char);
    size_t temp_bytes = width * height * sizeof(float) * 5;

    unsigned char *d_input, *d_output;
    float* d_temp_buffers;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_output, img_bytes);
    cudaMalloc(&d_temp_buffers, temp_bytes);

    cudaMemcpy(d_input, h_image_data, img_bytes, cudaMemcpyHostToDevice);

    // dummy + measure
    launch_canny_full_pipeline(d_input, d_output, d_temp_buffers, width, height, 50.0f, 150.0f, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_canny_full_pipeline(d_input, d_output, d_temp_buffers, width, height, 50.0f, 150.0f, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_canny_full_pipeline(d_input, d_output, d_temp_buffers, width, height, 50.0f, 150.0f, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    // canny: 5-stage pipeline, primarily memory-bound with many passes over data
    result.theoretical_flops = (size_t)width * height * 20;
    result.theoretical_bytes = (size_t)width * height * 10;
    result.theoretical_arith_intensity = (float)result.theoretical_flops / result.theoretical_bytes;
    result.bottleneck_type = "memory-bound";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_buffers);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_harris(GPUMonitor& monitor, int width, int height,
                                          const unsigned char* h_image_data) {
    KernelCharacteristics result;
    result.name = "harris_" + std::to_string(width) + "x" + std::to_string(height);

    size_t img_bytes = width * height * sizeof(unsigned char);
    size_t response_bytes = width * height * sizeof(float);

    unsigned char *d_input, *d_output;
    float* d_temp_response;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_output, img_bytes);
    cudaMalloc(&d_temp_response, response_bytes);

    cudaMemcpy(d_input, h_image_data, img_bytes, cudaMemcpyHostToDevice);

    // dummy + measure
    launch_harris_full_pipeline(d_input, d_output, d_temp_response, width, height, 3, 0.04f, 1000.0f, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_harris_full_pipeline(d_input, d_output, d_temp_response, width, height, 3, 0.04f, 1000.0f, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_harris_full_pipeline(d_input, d_output, d_temp_response, width, height, 3, 0.04f, 1000.0f, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    // harris: gradients + structure tensor + response
    result.theoretical_flops = (size_t)width * height * 30;
    result.theoretical_bytes = (size_t)width * height * 8;
    result.theoretical_arith_intensity = (float)result.theoretical_flops / result.theoretical_bytes;
    result.bottleneck_type = "memory-bound";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_response);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

KernelCharacteristics characterize_gaussian_blur(GPUMonitor& monitor, int width, int height,
                                                 const unsigned char* h_image_data) {
    KernelCharacteristics result;
    result.name = "gaussian_blur_" + std::to_string(width) + "x" + std::to_string(height);

    size_t img_bytes = width * height * sizeof(unsigned char);
    size_t float_bytes = width * height * sizeof(float);

    unsigned char* d_input;
    float* d_output;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_output, float_bytes);

    cudaMemcpy(d_input, h_image_data, img_bytes, cudaMemcpyHostToDevice);

    // dummy + measure
    launch_gaussian_blur_uchar_kernel(d_input, d_output, width, height, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ws, we;
    cudaEventCreate(&ws);
    cudaEventCreate(&we);
    cudaEventRecord(ws);
    launch_gaussian_blur_uchar_kernel(d_input, d_output, width, height, 0);
    cudaEventRecord(we);
    cudaEventSynchronize(we);
    float single_ms;
    cudaEventElapsedTime(&single_ms, ws, we);
    cudaEventDestroy(ws);
    cudaEventDestroy(we);

    int num_runs;
    if (g_ncu_mode) {
        num_runs = 3;
    } else {
        const float target_ms = 3000.0f;
        num_runs = std::max(100, (int)(target_ms / std::max(single_ms, 0.001f)));
    }
    result.num_runs = num_runs;

    NVMLSamples samples;
    std::atomic<bool> running{!g_ncu_mode};
    std::thread sampler;
    if (!g_ncu_mode) {
        sampler = std::thread(sample_nvml_during_execution, std::ref(monitor),
                             std::ref(running), std::ref(samples));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launch_gaussian_blur_uchar_kernel(d_input, d_output, width, height, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    running = false;
    if (sampler.joinable()) sampler.join();

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    result.avg_time_ms = total_time / num_runs;
    result.sm_utilization_pct = calculate_average(samples.sm_util, 1);
    result.mem_utilization_pct = calculate_average(samples.mem_util, 1);
    result.num_nvml_samples = samples.sm_util.size();

    LOG("  [%zu nvml samples over %.1f ms, %d runs]\n",
        result.num_nvml_samples, total_time, num_runs);

    // gaussian blur: 5x5 kernel = 25 multiply-adds per pixel
    result.theoretical_flops = (size_t)width * height * 50;
    result.theoretical_bytes = (size_t)width * height * (sizeof(unsigned char) + sizeof(float));
    result.theoretical_arith_intensity = (float)result.theoretical_flops / result.theoretical_bytes;
    result.bottleneck_type = (result.theoretical_arith_intensity > 5.0f) ? "compute-bound" : "memory-bound";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

// ===== output =====

void print_characteristics(const KernelCharacteristics& k) {
    LOG("%-40s | %8.3f ms | %6.1f%% | %6.1f%% | %10.2f | %-15s\n",
        k.name.c_str(),
        k.avg_time_ms,
        k.sm_utilization_pct,
        k.mem_utilization_pct,
        k.theoretical_arith_intensity,
        k.bottleneck_type.c_str());
}

int main(int argc, char** argv) {
    // parse args: [--ncu] <image_path>
    const char* image_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ncu") == 0) {
            g_ncu_mode = true;
        } else {
            image_path = argv[i];
        }
    }

    if (!image_path) {
        fprintf(stderr, "usage: %s [--ncu] <image_path>\n", argv[0]);
        fprintf(stderr, "  normal mode:  %s test_image.jpg\n", argv[0]);
        fprintf(stderr, "  ncu mode:     ncu --csv ... %s --ncu test_image.jpg\n", argv[0]);
        return 1;
    }

    LOG("=== kernel characterization%s ===\n\n", g_ncu_mode ? " (ncu mode: 3 runs per kernel)" : "");

    // load real image
    cv::Mat input_img = cv::imread(image_path);
    if (input_img.empty()) {
        fprintf(stderr, "failed to load image: %s\n", image_path);
        return 1;
    }

    cv::Mat gray_img;
    if (input_img.channels() == 3) {
        cv::cvtColor(input_img, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = input_img.clone();
    }

    LOG("loaded image: %s (%dx%d)\n\n", image_path, gray_img.cols, gray_img.rows);

    // prepare 720p and 1080p versions
    cv::Mat gray_720p, gray_1080p;
    cv::resize(gray_img, gray_720p, cv::Size(1280, 720));
    cv::resize(gray_img, gray_1080p, cv::Size(1920, 1080));

    // convert to float for conv2d
    std::vector<float> float_720p(1280 * 720);
    std::vector<float> float_1080p(1920 * 1080);
    for (int i = 0; i < 1280 * 720; i++) float_720p[i] = gray_720p.data[i] / 255.0f;
    for (int i = 0; i < 1920 * 1080; i++) float_1080p[i] = gray_1080p.data[i] / 255.0f;

    // init gpu monitor (needed even in ncu mode for function signatures, but won't sample)
    GPUMonitor monitor;
    if (!monitor.initialize()) {
        if (!g_ncu_mode) {
            fprintf(stderr, "failed to initialize gpu monitor - nvml required\n");
            return 1;
        }
        // in ncu mode, monitor failure is ok - we don't use nvml
        fprintf(stderr, "warning: gpu monitor init failed (ok in ncu mode)\n");
    }

    GPUHardwareInfo hw = monitor.get_hardware_info();
    LOG("gpu: %s\n", hw.name.c_str());
    LOG("compute capability: %d.%d\n", hw.major_compute_capability, hw.minor_compute_capability);
    LOG("sm count: %d\n", hw.sm_count);
    LOG("max theoretical bandwidth: %.1f GB/s\n\n", hw.memory_bandwidth_gb_s);

    if (g_ncu_mode) {
        LOG("ncu mode: 3 runs per kernel, no nvml sampling\n");
        LOG("ncu will profile each kernel launch with hardware counters\n\n");
    } else {
        LOG("note: sm_util and mem_util are from nvml (coarse, ~16ms update rate)\n");
        LOG("      for accurate per-kernel metrics, use --ncu mode with ncu\n");
        LOG("      arith_int is theoretical (flops/byte), not measured\n\n");
    }

    LOG("%-40s | %10s | %8s | %8s | %12s | %-15s\n",
           "kernel", "time", "sm_util", "mem_util", "arith_int", "bottleneck");
    LOG("----------------------------------------+------------+----------+----------+--------------+-----------------\n");

    std::vector<KernelCharacteristics> results;

    results.push_back(characterize_canny(monitor, 1280, 720, gray_720p.data));
    print_characteristics(results.back());

    results.push_back(characterize_canny(monitor, 1920, 1080, gray_1080p.data));
    print_characteristics(results.back());

    results.push_back(characterize_harris(monitor, 1280, 720, gray_720p.data));
    print_characteristics(results.back());

    results.push_back(characterize_harris(monitor, 1920, 1080, gray_1080p.data));
    print_characteristics(results.back());

    results.push_back(characterize_gaussian_blur(monitor, 1280, 720, gray_720p.data));
    print_characteristics(results.back());

    results.push_back(characterize_gaussian_blur(monitor, 1920, 1080, gray_1080p.data));
    print_characteristics(results.back());

    results.push_back(characterize_matmul(monitor, 256));
    print_characteristics(results.back());

    results.push_back(characterize_matmul(monitor, 512));
    print_characteristics(results.back());

    results.push_back(characterize_matmul(monitor, 1024));
    print_characteristics(results.back());

    results.push_back(characterize_conv2d(monitor, 1280, 720, float_720p.data()));
    print_characteristics(results.back());

    results.push_back(characterize_conv2d(monitor, 1920, 1080, float_1080p.data()));
    print_characteristics(results.back());

    results.push_back(characterize_histogram(monitor, 1280, 720, gray_720p.data));
    print_characteristics(results.back());

    results.push_back(characterize_histogram(monitor, 1920, 1080, gray_1080p.data));
    print_characteristics(results.back());

    results.push_back(characterize_synthetic_bandwidth(monitor, 1920*1080));
    print_characteristics(results.back());

    results.push_back(characterize_synthetic_compute(monitor, 1920*1080, 50));
    print_characteristics(results.back());

    results.push_back(characterize_synthetic_compute(monitor, 1920*1080, 100));
    print_characteristics(results.back());

    results.push_back(characterize_synthetic_compute(monitor, 1920*1080, 200));
    print_characteristics(results.back());

    return 0;
}
