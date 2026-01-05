#pragma once
#include "scheduler.hpp"
#include "kernels.hpp"  
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <memory>

struct ExecutionResult {
    ExecutionStrategy strategy_used;
    float total_time_ms;
    float canny_time_ms;
    float harris_time_ms;
    float overhead_time_ms;
    
    bool success;
    std::string error_message;
    
    float estimated_sequential_time_ms;
    float estimated_concurrent_time_ms; 
    bool was_optimal_choice;
    float performance_gain_percent;
    
    ExecutionResult() : 
        strategy_used(ExecutionStrategy::SEQUENTIAL),
        total_time_ms(0.0f), canny_time_ms(0.0f), harris_time_ms(0.0f), overhead_time_ms(0.0f),
        success(false), estimated_sequential_time_ms(0.0f), estimated_concurrent_time_ms(0.0f),
        was_optimal_choice(false), performance_gain_percent(0.0f) {}
};

struct GPUMemoryBuffers {
    unsigned char* d_input;
    unsigned char* d_canny_output;
    unsigned char* d_harris_output;
    
    float* d_canny_temp;
    float* d_harris_temp;
    
    size_t allocated_size;
    bool is_allocated;
    
    GPUMemoryBuffers() : 
        d_input(nullptr), d_canny_output(nullptr), d_harris_output(nullptr),
        d_canny_temp(nullptr), d_harris_temp(nullptr),
        allocated_size(0), is_allocated(false) {}
};

class ExecutionManager {
public:
    ExecutionManager();
    ~ExecutionManager();
    
    bool initialize();
    void cleanup();
    
    ExecutionResult execute_pipeline(
        const cv::Mat& input_image,
        const SchedulingDecision& decision,
        cv::Mat& canny_output,
        cv::Mat& harris_output
    );
    
    ExecutionResult execute_with_comparison(
        const cv::Mat& input_image,
        const SchedulingDecision& decision,
        cv::Mat& canny_output,
        cv::Mat& harris_output
    );
    
    ExecutionResult execute_sequential(
        const cv::Mat& input_image,
        cv::Mat& canny_output,
        cv::Mat& harris_output
    );
    
    ExecutionResult execute_concurrent(
        const cv::Mat& input_image,
        cv::Mat& canny_output,
        cv::Mat& harris_output
    );

    bool is_initialized() const { return initialized_; }
    void print_execution_summary(const ExecutionResult& result) const;
    
    size_t get_allocated_memory_mb() const;
    void force_memory_cleanup();
    
private:
    bool initialized_;
    
    cudaStream_t stream_sequential_;
    cudaStream_t stream_canny_;
    cudaStream_t stream_harris_;
    cudaEvent_t event_start_;
    cudaEvent_t event_canny_start_;
    cudaEvent_t event_canny_end_;
    cudaEvent_t event_harris_start_;
    cudaEvent_t event_harris_end_;
    cudaEvent_t event_end_;
    GPUMemoryBuffers buffers_;
    
    bool initialize_cuda_streams();
    bool initialize_cuda_events();
    bool allocate_gpu_memory(size_t required_size);
    void free_gpu_memory();
    
    bool upload_image_to_gpu(const cv::Mat& image);
    bool download_results_from_gpu(cv::Mat& canny_output, cv::Mat& harris_output, 
                                  int width, int height);
    
    void start_timing();
    void mark_canny_start();
    void mark_canny_end();
    void mark_harris_start();
    void mark_harris_end();
    void end_timing();
    float get_elapsed_time(cudaEvent_t start, cudaEvent_t end);
    
    ExecutionResult execute_sequential_impl(const cv::Mat& input_image,
                                           cv::Mat& canny_output, cv::Mat& harris_output);
    ExecutionResult execute_concurrent_impl(const cv::Mat& input_image,
                                           cv::Mat& canny_output, cv::Mat& harris_output);
    
    void set_error_result(ExecutionResult& result, const std::string& error);
    bool check_cuda_error(const std::string& operation);
    
    size_t calculate_required_memory(int width, int height, int channels);
    
    void analyze_performance(ExecutionResult& result, 
                           float sequential_time, float concurrent_time);

    bool launch_canny_kernel(unsigned char* d_input, unsigned char* d_output,
                           int width, int height, cudaStream_t stream);
    bool launch_harris_kernel(unsigned char* d_input, unsigned char* d_output,
                            int width, int height, cudaStream_t stream);
};