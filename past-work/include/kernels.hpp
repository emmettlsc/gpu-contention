#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cuda_runtime.h>

struct KernelResult {
    float execution_time_ms;
    bool success;
    size_t memory_used_bytes;
};

KernelResult launch_custom_canny(
    unsigned char* input, 
    unsigned char* output, 
    int width, 
    int height,
    float low_threshold = 50.0f,
    float high_threshold = 150.0f
);

KernelResult launch_custom_harris(
    unsigned char* input,
    float* output,  
    unsigned char* corner_output,
    int width,
    int height,
    int block_size = 2,
    int ksize = 3,
    float k = 0.04f,
    float threshold = 0.01f
);

extern "C" {
    void launch_gaussian_blur_kernel(unsigned char* input, float* output, 
                                   int width, int height, cudaStream_t stream = 0);
    void launch_sobel_kernel(float* input, float* grad_x, float* grad_y, 
                           int width, int height, cudaStream_t stream = 0);
    void launch_gradient_magnitude_kernel(float* grad_x, float* grad_y, float* magnitude, 
                                        float* direction, int width, int height, cudaStream_t stream = 0);
    void launch_non_maximum_suppression_kernel(float* magnitude, float* direction, 
                                             float* suppressed, int width, int height, cudaStream_t stream = 0);
    void launch_double_threshold_kernel(float* input, unsigned char* output, 
                                      int width, int height, float low_thresh, float high_thresh, cudaStream_t stream = 0);
    
    void launch_harris_response_kernel(unsigned char* input, float* response, 
                                     int width, int height, int block_size, int ksize, float k, cudaStream_t stream = 0);
    void launch_harris_threshold_kernel(float* response, unsigned char* output, 
                                      int width, int height, float threshold, cudaStream_t stream = 0);
}

struct GPUMemoryManager {
    void* d_input;
    void* d_output;
    void* d_temp1;
    void* d_temp2;
    void* d_temp3;
    void* d_temp4;
    size_t allocated_bytes;
    
    GPUMemoryManager();
    ~GPUMemoryManager();
    
    bool allocate_for_image(int width, int height, int channels = 1);
    void deallocate();
    void* get_temp_buffer(int index, size_t min_size);
};

struct CannyParams {
    float low_threshold;
    float high_threshold;
    float gaussian_sigma;
    int gaussian_kernel_size;
};

struct HarrisParams {
    int block_size;
    int ksize;
    float k;
    float threshold;
    bool use_non_max_suppression;
};

struct KernelProfiler {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool events_created;
    
    KernelProfiler();
    ~KernelProfiler();
    
    void start_timing();
    float stop_timing();
    void reset();
};


KernelResult launch_custom_canny_stream(
    unsigned char* d_input,
    unsigned char* d_output,
    float* d_temp_buffers,
    int width, 
    int height,
    float low_threshold = 50.0f,
    float high_threshold = 150.0f,
    cudaStream_t stream = 0
);

KernelResult launch_custom_harris_stream(
    unsigned char* d_input, 
    unsigned char* d_output,
    float* d_temp_buffers,
    int width,
    int height,
    int block_size = 2,
    int ksize = 3,
    float k = 0.04f,
    float threshold = 0.01f,
    cudaStream_t stream = 0
);

struct TempBufferLayout {
    static size_t get_canny_temp_size(int width, int height) {
        return width * height * sizeof(float) * 4; // blurred, grad_x, grad_y, magnitude/direction
    }

    static size_t get_harris_temp_size(int width, int height) {
        return width * height * sizeof(float) * 1; // harris response
    }

    static float* get_canny_buffer_ptr(float* base, int buffer_index, int width, int height) {
        return base + (buffer_index * width * height);
    }

    static float* get_harris_buffer_ptr(float* base, int buffer_index, int width, int height) {
        return base + (buffer_index * width * height);
    }
};

// ===== ADDITIONAL KERNELS (workloads/additional_kernels.cu) =====

extern "C" {
    // matrix multiply - compute-bound kernel
    void launch_matmul_kernel(const float* d_A, const float* d_B, float* d_C,
                             int N, cudaStream_t stream = 0);

    // 2d convolution - compute-bound kernel
    void launch_conv2d_kernel(const float* d_input, float* d_output,
                             int width, int height, cudaStream_t stream = 0);

    // histogram - atomic-bound kernel
    void launch_histogram_kernel(const unsigned char* d_input, unsigned int* d_hist,
                               int width, int height, cudaStream_t stream = 0);

    // gaussian blur (float version) - memory-bound kernel
    void launch_gaussian_blur_float_kernel(const float* d_input, float* d_output,
                                          int width, int height, cudaStream_t stream = 0);

    // synthetic bandwidth - pure memory bandwidth test
    void launch_synthetic_bandwidth_kernel(const float* d_input, float* d_output,
                                          int N, cudaStream_t stream = 0);

    // synthetic compute - pure compute test
    void launch_synthetic_compute_kernel(const float* d_input, float* d_output,
                                        int N, int iterations, cudaStream_t stream = 0);

    // memory allocation helpers
    void allocate_matmul_buffers(float** d_A, float** d_B, float** d_C, int N);
    void free_matmul_buffers(float* d_A, float* d_B, float* d_C);

    void allocate_conv2d_buffers(float** d_input, float** d_output, int width, int height);
    void free_conv2d_buffers(float* d_input, float* d_output);

    void allocate_histogram_buffers(unsigned char** d_input, unsigned int** d_hist,
                                   int width, int height);
    void free_histogram_buffers(unsigned char* d_input, unsigned int* d_hist);

    void allocate_synthetic_buffers(float** d_input, float** d_output, int N);
    void free_synthetic_buffers(float* d_input, float* d_output);

    // standalone test functions for solo kernel runs
    void test_matmul_solo(int N, float* h_result_time);
    void test_conv2d_solo(int width, int height, float* h_result_time);
    void test_histogram_solo(int width, int height, float* h_result_time);
    void test_synthetic_bandwidth_solo(int N, float* h_result_time);
    void test_synthetic_compute_solo(int N, int iterations, float* h_result_time);
}

#endif