#include "execution_manager.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>

ExecutionManager::ExecutionManager() : initialized_(false) {
    stream_sequential_ = 0;
    stream_canny_ = 0;
    stream_harris_ = 0;
    
    event_start_ = 0;
    event_canny_start_ = 0;
    event_canny_end_ = 0;
    event_harris_start_ = 0;
    event_harris_end_ = 0;
    event_end_ = 0;
}

ExecutionManager::~ExecutionManager() {
    cleanup();
}

bool ExecutionManager::initialize() {
    if (initialized_) {
        return true;
    }
    
    // init CUDA streams
    if (!initialize_cuda_streams()) {
        std::cerr << "Failed to initialize CUDA streams" << std::endl;
        return false;
    }
    
    // init CUDA events
    if (!initialize_cuda_events()) {
        std::cerr << "Failed to initialize CUDA events" << std::endl;
        cleanup();
        return false;
    }
    
    initialized_ = true;
    std::cout << "Execution Manager initialized successfully" << std::endl;
    
    return true;
}

void ExecutionManager::cleanup() {
    if (!initialized_) {
        return;
    }
    
    free_gpu_memory();
    
    // Destroy CUDA events
    if (event_start_) cudaEventDestroy(event_start_);
    if (event_canny_start_) cudaEventDestroy(event_canny_start_);
    if (event_canny_end_) cudaEventDestroy(event_canny_end_);
    if (event_harris_start_) cudaEventDestroy(event_harris_start_);
    if (event_harris_end_) cudaEventDestroy(event_harris_end_);
    if (event_end_) cudaEventDestroy(event_end_);
    
    // Destroy CUDA streams
    if (stream_sequential_) cudaStreamDestroy(stream_sequential_);
    if (stream_canny_) cudaStreamDestroy(stream_canny_);
    if (stream_harris_) cudaStreamDestroy(stream_harris_);
    
    initialized_ = false;
}

bool ExecutionManager::initialize_cuda_streams() {
    cudaError_t err;
    
    err = cudaStreamCreate(&stream_sequential_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create sequential stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaStreamCreate(&stream_canny_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create canny stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaStreamCreate(&stream_harris_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create harris stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool ExecutionManager::initialize_cuda_events() {
    cudaError_t err;
    
    err = cudaEventCreate(&event_start_);
    if (err != cudaSuccess) return false;
    
    err = cudaEventCreate(&event_canny_start_);
    if (err != cudaSuccess) return false;
    
    err = cudaEventCreate(&event_canny_end_);
    if (err != cudaSuccess) return false;
    
    err = cudaEventCreate(&event_harris_start_);
    if (err != cudaSuccess) return false;
    
    err = cudaEventCreate(&event_harris_end_);
    if (err != cudaSuccess) return false;
    
    err = cudaEventCreate(&event_end_);
    if (err != cudaSuccess) return false;
    
    return true;
}

ExecutionResult ExecutionManager::execute_pipeline(
    const cv::Mat& input_image,
    const SchedulingDecision& decision,
    cv::Mat& canny_output,
    cv::Mat& harris_output) {
    
    ExecutionResult result;
    result.strategy_used = decision.strategy;
    
    if (!initialized_) {
        set_error_result(result, "Execution Manager not initialized");
        return result;
    }
    
    // alloc GPU memory if needed
    size_t required_memory = calculate_required_memory(
        input_image.cols, input_image.rows, input_image.channels());
    
    if (!allocate_gpu_memory(required_memory)) {
        set_error_result(result, "Failed to allocate GPU memory");
        return result;
    }
    
    switch (decision.strategy) {
        case ExecutionStrategy::SEQUENTIAL:
            result = execute_sequential_impl(input_image, canny_output, harris_output);
            break;
            
        case ExecutionStrategy::CONCURRENT:
            result = execute_concurrent_impl(input_image, canny_output, harris_output);
            break;
    }
    
    return result;
}

ExecutionResult ExecutionManager::execute_with_comparison(
    const cv::Mat& input_image,
    const SchedulingDecision& decision,
    cv::Mat& canny_output,
    cv::Mat& harris_output) {
    
    ExecutionResult chosen_result = execute_pipeline(input_image, decision, canny_output, harris_output);
    
    if (!chosen_result.success) {
        return chosen_result;
    }
    
    cv::Mat alt_canny, alt_harris;
    ExecutionResult alt_result;
    
    if (decision.strategy == ExecutionStrategy::SEQUENTIAL) {
        alt_result = execute_concurrent_impl(input_image, alt_canny, alt_harris);
        chosen_result.estimated_concurrent_time_ms = alt_result.total_time_ms;
        chosen_result.estimated_sequential_time_ms = chosen_result.total_time_ms;
    } else {
        alt_result = execute_sequential_impl(input_image, alt_canny, alt_harris);
        chosen_result.estimated_sequential_time_ms = alt_result.total_time_ms;
        chosen_result.estimated_concurrent_time_ms = chosen_result.total_time_ms;
    }
    
    if (alt_result.success) {
        analyze_performance(chosen_result, 
                          chosen_result.estimated_sequential_time_ms,
                          chosen_result.estimated_concurrent_time_ms);
    }
    
    return chosen_result;
}

ExecutionResult ExecutionManager::execute_sequential_impl(
    const cv::Mat& input_image,
    cv::Mat& canny_output,
    cv::Mat& harris_output) {
    
    ExecutionResult result;
    result.strategy_used = ExecutionStrategy::SEQUENTIAL;
    
    // Upload image to GPU
    if (!upload_image_to_gpu(input_image)) {
        set_error_result(result, "Failed to upload image to GPU");
        return result;
    }
    
    // Start overall timing
    cudaEventRecord(event_start_, stream_sequential_);
    
    // Execute Canny kernel
    cudaEventRecord(event_canny_start_, stream_sequential_);
    
    // Call your existing Canny kernel
    if (!launch_canny_kernel(buffers_.d_input, buffers_.d_canny_output, 
                           input_image.cols, input_image.rows, stream_sequential_)) {
        set_error_result(result, "Canny kernel execution failed");
        return result;
    }
    
    cudaEventRecord(event_canny_end_, stream_sequential_);
    
    //Harris kernel
    cudaEventRecord(event_harris_start_, stream_sequential_);
    
    // call harris kernel
    if (!launch_harris_kernel(buffers_.d_input, buffers_.d_harris_output,
                            input_image.cols, input_image.rows, stream_sequential_)) {
        set_error_result(result, "Harris kernel execution failed");
        return result;
    }
    
    cudaEventRecord(event_harris_end_, stream_sequential_);
    cudaEventRecord(event_end_, stream_sequential_);
    
    cudaStreamSynchronize(stream_sequential_);
    
    if (!download_results_from_gpu(canny_output, harris_output, 
                                 input_image.cols, input_image.rows)) {
        set_error_result(result, "Failed to download results from GPU");
        return result;
    }
    
    result.total_time_ms = get_elapsed_time(event_start_, event_end_);
    result.canny_time_ms = get_elapsed_time(event_canny_start_, event_canny_end_);
    result.harris_time_ms = get_elapsed_time(event_harris_start_, event_harris_end_);
    result.overhead_time_ms = result.total_time_ms - result.canny_time_ms - result.harris_time_ms;
    result.success = true;
    
    return result;
}

ExecutionResult ExecutionManager::execute_concurrent_impl(
    const cv::Mat& input_image,
    cv::Mat& canny_output,
    cv::Mat& harris_output) {
    
    ExecutionResult result;
    result.strategy_used = ExecutionStrategy::CONCURRENT;
    
    // Upload image to GPU
    if (!upload_image_to_gpu(input_image)) {
        set_error_result(result, "Failed to upload image to GPU");
        return result;
    }
    
    cudaEventRecord(event_start_, stream_sequential_);
    
    cudaEventRecord(event_canny_start_, stream_canny_);
    if (!launch_canny_kernel(buffers_.d_input, buffers_.d_canny_output,
                           input_image.cols, input_image.rows, stream_canny_)) {
        set_error_result(result, "Canny kernel execution failed");
        return result;
    }
    cudaEventRecord(event_canny_end_, stream_canny_);
    
    cudaEventRecord(event_harris_start_, stream_harris_);
    if (!launch_harris_kernel(buffers_.d_input, buffers_.d_harris_output,
                            input_image.cols, input_image.rows, stream_harris_)) {
        set_error_result(result, "Harris kernel execution failed");
        return result;
    }
    cudaEventRecord(event_harris_end_, stream_harris_);
    
    // Synchronize both streams
    cudaStreamSynchronize(stream_canny_);
    cudaStreamSynchronize(stream_harris_);
    
    cudaEventRecord(event_end_, stream_sequential_);
    cudaStreamSynchronize(stream_sequential_);
    
    if (!download_results_from_gpu(canny_output, harris_output,
                                 input_image.cols, input_image.rows)) {
        set_error_result(result, "Failed to download results from GPU");
        return result;
    }
    
    result.total_time_ms = get_elapsed_time(event_start_, event_end_);
    result.canny_time_ms = get_elapsed_time(event_canny_start_, event_canny_end_);
    result.harris_time_ms = get_elapsed_time(event_harris_start_, event_harris_end_);
    
    float kernel_overlap_time = std::max(result.canny_time_ms, result.harris_time_ms);
    result.overhead_time_ms = result.total_time_ms - kernel_overlap_time;
    
    result.success = true;
    
    return result;
}

bool ExecutionManager::allocate_gpu_memory(size_t required_size) {
    if (buffers_.is_allocated && buffers_.allocated_size >= required_size) {
        return true;
    }
    
    if (buffers_.is_allocated) {
        free_gpu_memory();
    }
    
    cudaError_t err;
    
    err = cudaMalloc(&buffers_.d_input, required_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&buffers_.d_canny_output, required_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate Canny output buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(buffers_.d_input);
        return false;
    }
    
    err = cudaMalloc(&buffers_.d_harris_output, required_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate Harris output buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(buffers_.d_input);
        cudaFree(buffers_.d_canny_output);
        return false;
    }
    

    int max_width = 4096, max_height = 2160; // 4k max dims
    size_t canny_temp_size = TempBufferLayout::get_canny_temp_size(max_width, max_height);
    size_t harris_temp_size = TempBufferLayout::get_harris_temp_size(max_width, max_height);
    err = cudaMalloc(&buffers_.d_canny_temp, canny_temp_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate Canny temp buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(buffers_.d_input);
        cudaFree(buffers_.d_canny_output);
        cudaFree(buffers_.d_harris_output);
        return false;
    }
    
    err = cudaMalloc(&buffers_.d_harris_temp, harris_temp_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate Harris temp buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(buffers_.d_input);
        cudaFree(buffers_.d_canny_output);
        cudaFree(buffers_.d_harris_output);
        cudaFree(buffers_.d_canny_temp);
        return false;
    }
    
    buffers_.allocated_size = required_size;
    buffers_.is_allocated = true;

    return true;
}

void ExecutionManager::free_gpu_memory() {
    if (!buffers_.is_allocated) {
        return;
    }
    
    if (buffers_.d_input) cudaFree(buffers_.d_input);
    if (buffers_.d_canny_output) cudaFree(buffers_.d_canny_output);
    if (buffers_.d_harris_output) cudaFree(buffers_.d_harris_output);
    if (buffers_.d_canny_temp) cudaFree(buffers_.d_canny_temp);
    if (buffers_.d_harris_temp) cudaFree(buffers_.d_harris_temp);
    
    buffers_.d_input = nullptr;
    buffers_.d_canny_output = nullptr;
    buffers_.d_harris_output = nullptr;
    buffers_.d_canny_temp = nullptr;
    buffers_.d_harris_temp = nullptr;
    
    buffers_.allocated_size = 0;
    buffers_.is_allocated = false;
}

bool ExecutionManager::upload_image_to_gpu(const cv::Mat& image) {
    if (!buffers_.is_allocated || !buffers_.d_input) {
        return false;
    }
    
    size_t image_size = image.rows * image.cols * image.channels();
    cudaError_t err = cudaMemcpy(buffers_.d_input, image.data, image_size, cudaMemcpyHostToDevice);
    
    return (err == cudaSuccess);
}

bool ExecutionManager::download_results_from_gpu(cv::Mat& canny_output, cv::Mat& harris_output,
                                                int width, int height) {
    if (!buffers_.is_allocated) {
        return false;
    }
    
    canny_output = cv::Mat::zeros(height, width, CV_8UC1);
    harris_output = cv::Mat::zeros(height, width, CV_8UC1);
    
    //  Canny result
    cudaError_t err = cudaMemcpy(canny_output.data, buffers_.d_canny_output,
                               width * height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }
    
    //  Harris result
    err = cudaMemcpy(harris_output.data, buffers_.d_harris_output,
                    width * height, cudaMemcpyDeviceToHost);
    
    return (err == cudaSuccess);
}

float ExecutionManager::get_elapsed_time(cudaEvent_t start, cudaEvent_t end) {
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    return elapsed_ms;
}

size_t ExecutionManager::calculate_required_memory(int width, int height, int channels) {
    return width * height * channels * sizeof(unsigned char);
}

void ExecutionManager::analyze_performance(ExecutionResult& result,
                                         float sequential_time, float concurrent_time) {
    float faster_time = std::min(sequential_time, concurrent_time);
    float chosen_time = result.total_time_ms;
    
    result.was_optimal_choice = (chosen_time == faster_time);
    
    if (faster_time > 0.0f) {
        result.performance_gain_percent = ((faster_time - chosen_time) / faster_time) * 100.0f;
    }
}

void ExecutionManager::print_execution_summary(const ExecutionResult& result) const {
    std::cout << "=== Execution Summary ===" << std::endl;
    std::cout << "Strategy: " << Scheduler::strategy_to_string(result.strategy_used) << std::endl;
    std::cout << "Success: " << (result.success ? "Yes" : "No") << std::endl;
    
    if (!result.success) {
        std::cout << "Error: " << result.error_message << std::endl;
        return;
    }
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Time: " << result.total_time_ms << " ms" << std::endl;
    std::cout << "  Canny: " << result.canny_time_ms << " ms" << std::endl;
    std::cout << "  Harris: " << result.harris_time_ms << " ms" << std::endl;
    std::cout << "  Overhead: " << result.overhead_time_ms << " ms" << std::endl;
    
    if (result.estimated_sequential_time_ms > 0.0f && result.estimated_concurrent_time_ms > 0.0f) {
        std::cout << "Performance Comparison:" << std::endl;
        std::cout << "  Sequential would take: " << result.estimated_sequential_time_ms << " ms" << std::endl;
        std::cout << "  Concurrent would take: " << result.estimated_concurrent_time_ms << " ms" << std::endl;
        std::cout << "  Optimal choice: " << (result.was_optimal_choice ? "Yes" : "No") << std::endl;
        std::cout << "  Performance gain: " << result.performance_gain_percent << "%" << std::endl;
    }
}

size_t ExecutionManager::get_allocated_memory_mb() const {
    return buffers_.allocated_size / (1024 * 1024);
}

void ExecutionManager::set_error_result(ExecutionResult& result, const std::string& error) {
    result.success = false;
    result.error_message = error;
}

bool ExecutionManager::check_cuda_error(const std::string& operation) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

// wrapper from day 1
bool ExecutionManager::launch_canny_kernel(unsigned char* d_input, unsigned char* d_output,
                                          int width, int height, cudaStream_t stream) {
    size_t temp_size = TempBufferLayout::get_canny_temp_size(width, height);
    
    float* temp_buffer = static_cast<float*>(buffers_.d_canny_temp);
    
    KernelResult result = launch_custom_canny_stream(
        d_input, d_output, temp_buffer, width, height,
        50.0f, 150.0f, stream
    );
    
    return result.success;
}

bool ExecutionManager::launch_harris_kernel(unsigned char* d_input, unsigned char* d_output,
                                           int width, int height, cudaStream_t stream) {
    float* temp_buffer = static_cast<float*>(buffers_.d_harris_temp);
    
    KernelResult result = launch_custom_harris_stream(
        d_input, d_output, temp_buffer, width, height,
        2, 3, 0.04f, 0.01f, stream
    );
    
    return result.success;
}