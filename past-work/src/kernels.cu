#include "kernels.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// ===== CANNY EDGE DETECTION KERNELS =====

__global__ void gaussian_blur_kernel(unsigned char* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // 5x5 Gaussian kernel (sigma ≈ 1.0)
    float kernel[25] = {
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1
    };
    float kernel_sum = 256.0f;
    
    float sum = 0.0f;
    
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            int kernel_idx = (ky + 2) * 5 + (kx + 2);
            sum += input[py * width + px] * kernel[kernel_idx];
        }
    }
    
    output[y * width + x] = sum / kernel_sum;
}

__global__ void sobel_kernel(float* input, float* grad_x, float* grad_y, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float gx = 0.0f, gy = 0.0f;
    
    // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
    // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        // Sobel X (vertical edges)
        gx = -1.0f * input[(y-1)*width + (x-1)] + 1.0f * input[(y-1)*width + (x+1)] +
             -2.0f * input[y*width + (x-1)]     + 2.0f * input[y*width + (x+1)] +
             -1.0f * input[(y+1)*width + (x-1)] + 1.0f * input[(y+1)*width + (x+1)];
        
        // Sobel Y (horizontal edges)
        gy = -1.0f * input[(y-1)*width + (x-1)] + -2.0f * input[(y-1)*width + x] + -1.0f * input[(y-1)*width + (x+1)] +
             1.0f * input[(y+1)*width + (x-1)]  + 2.0f * input[(y+1)*width + x]  + 1.0f * input[(y+1)*width + (x+1)];
    }
    
    grad_x[y * width + x] = gx;
    grad_y[y * width + x] = gy;
}

__global__ void gradient_magnitude_kernel(float* grad_x, float* grad_y, float* magnitude, 
                                         float* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float gx = grad_x[idx];
    float gy = grad_y[idx];
    
    magnitude[idx] = sqrtf(gx * gx + gy * gy);
    direction[idx] = atan2f(gy, gx);
}

__global__ void non_maximum_suppression_kernel(float* magnitude, float* direction, 
                                             float* suppressed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height || x == 0 || y == 0 || x == width-1 || y == height-1) {
        if (x < width && y < height) {
            suppressed[y * width + x] = 0.0f;
        }
        return;
    }
    
    int idx = y * width + x;
    float angle = direction[idx];
    float mag = magnitude[idx];
    angle = angle * 180.0f / M_PI;
    if (angle < 0) angle += 180.0f;
    
    float neighbor1, neighbor2;
    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        neighbor1 = magnitude[idx - 1];
        neighbor2 = magnitude[idx + 1];
    } else if (angle >= 22.5 && angle < 67.5) {
        neighbor1 = magnitude[(y-1)*width + (x+1)];
        neighbor2 = magnitude[(y+1)*width + (x-1)];
    } else if (angle >= 67.5 && angle < 112.5) {
        neighbor1 = magnitude[(y-1)*width + x];
        neighbor2 = magnitude[(y+1)*width + x];
    } else {
        neighbor1 = magnitude[(y-1)*width + (x-1)];
        neighbor2 = magnitude[(y+1)*width + (x+1)];
    }
    if (mag >= neighbor1 && mag >= neighbor2) {
        suppressed[idx] = mag;
    } else {
        suppressed[idx] = 0.0f;
    }
}

__global__ void double_threshold_kernel(float* input, unsigned char* output, 
                                      int width, int height, float low_thresh, float high_thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float val = input[idx];
    
    if (val >= high_thresh) {
        output[idx] = 255; // Strong edge
    } else if (val >= low_thresh) {
        output[idx] = 127; // Weak edge (to be processed in hysteresis later)
    } else {
        output[idx] = 0;   // No an edge
    }
}

// ===== HARRIS CORNER DETECTION KERNELS =====

__global__ void harris_response_kernel(unsigned char* input, float* response, 
                                      int width, int height, int block_size, int ksize, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate gradients using Sobel operator
    float Ix = 0.0f, Iy = 0.0f;
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        // Sobel X
        Ix = -1.0f * input[(y-1)*width + (x-1)] + 1.0f * input[(y-1)*width + (x+1)] +
             -2.0f * input[y*width + (x-1)]     + 2.0f * input[y*width + (x+1)] +
             -1.0f * input[(y+1)*width + (x-1)] + 1.0f * input[(y+1)*width + (x+1)];
        
        // Sobel Y
        Iy = -1.0f * input[(y-1)*width + (x-1)] + -2.0f * input[(y-1)*width + x] + -1.0f * input[(y-1)*width + (x+1)] +
             1.0f * input[(y+1)*width + (x-1)]  + 2.0f * input[(y+1)*width + x]  + 1.0f * input[(y+1)*width + (x+1)];
    }
    
    // Products of derivatives
    float Ixx = Ix * Ix;
    float Ixy = Ix * Iy;
    float Iyy = Iy * Iy;
    
    // Sum over the window
    float Sxx = 0.0f, Sxy = 0.0f, Syy = 0.0f;
    int half_block = block_size / 2;
    
    for (int dy = -half_block; dy <= half_block; dy++) {
        for (int dx = -half_block; dx <= half_block; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            
            if (nx > 0 && nx < width-1 && ny > 0 && ny < height-1) {
                // Recalculate gradients for this pixel
                float local_Ix = -1.0f * input[(ny-1)*width + (nx-1)] + 1.0f * input[(ny-1)*width + (nx+1)] +
                                -2.0f * input[ny*width + (nx-1)]     + 2.0f * input[ny*width + (nx+1)] +
                                -1.0f * input[(ny+1)*width + (nx-1)] + 1.0f * input[(ny+1)*width + (nx+1)];
                
                float local_Iy = -1.0f * input[(ny-1)*width + (nx-1)] + -2.0f * input[(ny-1)*width + nx] + -1.0f * input[(ny-1)*width + (nx+1)] +
                                 1.0f * input[(ny+1)*width + (nx-1)]  + 2.0f * input[(ny+1)*width + nx]  + 1.0f * input[(ny+1)*width + (nx+1)];
                
                Sxx += local_Ix * local_Ix;
                Sxy += local_Ix * local_Iy;
                Syy += local_Iy * local_Iy;
            }
        }
    }
    
    // Harris response: det(M) - k * trace(M)^2
    // & M = [Sxx Sxy; Sxy Syy]
    float det = Sxx * Syy - Sxy * Sxy;
    float trace = Sxx + Syy;
    float harris_resp = det - k * trace * trace;
    
    response[y * width + x] = harris_resp;
}

__global__ void harris_threshold_kernel(float* response, unsigned char* output, 
                                       int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float resp = response[idx];
    
    // Find max response in neighborhood for non-max suppression
    float max_resp = resp;
    bool is_local_max = true;
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                float neighbor_resp = response[(y+dy)*width + (x+dx)];
                if (neighbor_resp > resp) {
                    is_local_max = false;
                    break;
                }
            }
            if (!is_local_max) break;
        }
    }
    
    if (resp > threshold && is_local_max) {
        output[idx] = 255;
    } else {
        output[idx] = 0;
    }
}

// ===== KERNEL LAUNCH FUNCTIONS =====

extern "C" {
    void launch_gaussian_blur_kernel(unsigned char* input, float* output, 
                                   int width, int height, cudaStream_t stream) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        gaussian_blur_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
    }
    
    void launch_sobel_kernel(float* input, float* grad_x, float* grad_y, 
                           int width, int height, cudaStream_t stream) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        sobel_kernel<<<gridSize, blockSize, 0, stream>>>(input, grad_x, grad_y, width, height);
    }
    
    void launch_gradient_magnitude_kernel(float* grad_x, float* grad_y, float* magnitude, 
                                        float* direction, int width, int height, cudaStream_t stream) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        gradient_magnitude_kernel<<<gridSize, blockSize, 0, stream>>>(grad_x, grad_y, magnitude, direction, width, height);
    }
    
    void launch_non_maximum_suppression_kernel(float* magnitude, float* direction, 
                                             float* suppressed, int width, int height, cudaStream_t stream) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        non_maximum_suppression_kernel<<<gridSize, blockSize, 0, stream>>>(magnitude, direction, suppressed, width, height);
    }
    
    void launch_double_threshold_kernel(float* input, unsigned char* output, 
                                      int width, int height, float low_thresh, float high_thresh, cudaStream_t stream) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        double_threshold_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height, low_thresh, high_thresh);
    }
    
    void launch_harris_response_kernel(unsigned char* input, float* response, 
                                     int width, int height, int block_size, int ksize, float k, cudaStream_t stream) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        harris_response_kernel<<<gridSize, blockSize, 0, stream>>>(input, response, width, height, block_size, ksize, k);
    }
    
    void launch_harris_threshold_kernel(float* response, unsigned char* output, 
                                      int width, int height, float threshold, cudaStream_t stream) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        harris_threshold_kernel<<<gridSize, blockSize, 0, stream>>>(response, output, width, height, threshold);
    }
}

// ===== MEMORY MANAGER IMPLEMENTATION =====

GPUMemoryManager::GPUMemoryManager() 
    : d_input(nullptr), d_output(nullptr), d_temp1(nullptr), 
      d_temp2(nullptr), d_temp3(nullptr), d_temp4(nullptr), allocated_bytes(0) {}

GPUMemoryManager::~GPUMemoryManager() {
    deallocate();
}

bool GPUMemoryManager::allocate_for_image(int width, int height, int channels) {
    deallocate(); // Clean up any existing allocation
    
    size_t image_size = width * height * sizeof(unsigned char) * channels;
    size_t float_size = width * height * sizeof(float);
    
    cudaError_t error;
    error = cudaMalloc(&d_input, image_size);
    if (error != cudaSuccess) { deallocate(); return false; }
    error = cudaMalloc(&d_output, image_size);
    if (error != cudaSuccess) { deallocate(); return false; }
    error = cudaMalloc(&d_temp1, float_size);
    if (error != cudaSuccess) { deallocate(); return false; }
    error = cudaMalloc(&d_temp2, float_size);
    if (error != cudaSuccess) { deallocate(); return false; }
    error = cudaMalloc(&d_temp3, float_size);
    if (error != cudaSuccess) { deallocate(); return false; }
    error = cudaMalloc(&d_temp4, float_size);
    if (error != cudaSuccess) { deallocate(); return false; }
    
    if (error != cudaSuccess) {
        deallocate();
        return false;
    }
    
    allocated_bytes = image_size * 2 + float_size * 4;
    return true;
}

void GPUMemoryManager::deallocate() {
    if (d_input) { cudaFree(d_input); d_input = nullptr; }
    if (d_output) { cudaFree(d_output); d_output = nullptr; }
    if (d_temp1) { cudaFree(d_temp1); d_temp1 = nullptr; }
    if (d_temp2) { cudaFree(d_temp2); d_temp2 = nullptr; }
    if (d_temp3) { cudaFree(d_temp3); d_temp3 = nullptr; }
    if (d_temp4) { cudaFree(d_temp4); d_temp4 = nullptr; }
    allocated_bytes = 0;
}

void* GPUMemoryManager::get_temp_buffer(int index, size_t min_size) {
    switch (index) {
        case 0: return d_temp1;
        case 1: return d_temp2;
        case 2: return d_temp3;
        case 3: return d_temp4;
        default: return nullptr;
    }
}

// ===== KERNEL PROFILER IMPLEMENTATION =====

KernelProfiler::KernelProfiler() : events_created(false) {
    reset();
}

KernelProfiler::~KernelProfiler() {
    if (events_created) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
}

void KernelProfiler::start_timing() {
    if (!events_created) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        events_created = true;
    }
    cudaEventRecord(start_event);
}

float KernelProfiler::stop_timing() {
    if (!events_created) return 0.0f;
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    return elapsed_time;
}

void KernelProfiler::reset() {
    if (events_created) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        events_created = false;
    }
}

// ===== HIGH-LEVEL KERNEL FUNCTIONS =====

KernelResult launch_custom_canny(unsigned char* input, unsigned char* output, 
                                int width, int height, float low_threshold, float high_threshold) {
    KernelResult result;
    result.success = false;
    result.execution_time_ms = 0.0f;
    result.memory_used_bytes = 0;
    
    GPUMemoryManager memory;
    KernelProfiler profiler;
    
    if (!memory.allocate_for_image(width, height, 1)) {
        return result;
    }
    
    result.memory_used_bytes = memory.allocated_bytes;
    
    cudaError_t error = cudaMemcpy(memory.d_input, input, 
                                  width * height * sizeof(unsigned char), 
                                  cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return result;
    }
    
    float* d_blurred = static_cast<float*>(memory.get_temp_buffer(0, width * height * sizeof(float)));
    float* d_grad_x = static_cast<float*>(memory.get_temp_buffer(1, width * height * sizeof(float)));
    float* d_grad_y = static_cast<float*>(memory.get_temp_buffer(2, width * height * sizeof(float)));
    float* d_magnitude = static_cast<float*>(memory.get_temp_buffer(3, width * height * sizeof(float)));
    
    float* d_direction = d_grad_x;
    float* d_suppressed = d_grad_y;
    
    profiler.start_timing();
    
    // Step 1: Gaussian blur
    launch_gaussian_blur_kernel(static_cast<unsigned char*>(memory.d_input), d_blurred, width, height);
    
    // Step 2: Compute gradients
    launch_sobel_kernel(d_blurred, d_grad_x, d_grad_y, width, height);
    
    // Step 3: Compute gradient magnitude and direction
    launch_gradient_magnitude_kernel(d_grad_x, d_grad_y, d_magnitude, d_direction, width, height);
    
    // Step 4: Non-maximum suppression
    launch_non_maximum_suppression_kernel(d_magnitude, d_direction, d_suppressed, width, height);
    
    // Step 5: Double thresholding
    launch_double_threshold_kernel(d_suppressed, static_cast<unsigned char*>(memory.d_output), 
                                 width, height, low_threshold, high_threshold);
    
    // Synchronize and measure time
    cudaDeviceSynchronize();
    result.execution_time_ms = profiler.stop_timing();
    
    // Copy result back to host
    error = cudaMemcpy(output, memory.d_output, 
                      width * height * sizeof(unsigned char), 
                      cudaMemcpyDeviceToHost);
    
    if (error == cudaSuccess) {
        result.success = true;
    }
    
    return result;
}

KernelResult launch_custom_harris(unsigned char* input, float* output, unsigned char* corner_output,
                                 int width, int height, int block_size, int ksize, float k, float threshold) {
    KernelResult result;
    result.success = false;
    result.execution_time_ms = 0.0f;
    result.memory_used_bytes = 0;
    
    GPUMemoryManager memory;
    KernelProfiler profiler;
    
    if (!memory.allocate_for_image(width, height, 1)) {
        return result;
    }
    
    result.memory_used_bytes = memory.allocated_bytes;
    
    cudaError_t error = cudaMemcpy(memory.d_input, input, 
                                  width * height * sizeof(unsigned char), 
                                  cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return result;
    }
    
    float* d_response = static_cast<float*>(memory.get_temp_buffer(0, width * height * sizeof(float)));
    
    profiler.start_timing();
    
    // Step 1: Compute Harris response
    launch_harris_response_kernel(static_cast<unsigned char*>(memory.d_input), d_response, 
                                width, height, block_size, ksize, k);
    
    // Step 2: Threshold and non-maximum suppression
    launch_harris_threshold_kernel(d_response, static_cast<unsigned char*>(memory.d_output), 
                                 width, height, threshold);
    
    // Synchronize and measure time
    cudaDeviceSynchronize();
    result.execution_time_ms = profiler.stop_timing();
    
    // Copy results back to host
    error = cudaMemcpy(output, d_response, 
                      width * height * sizeof(float), 
                      cudaMemcpyDeviceToHost);
    
    if (error == cudaSuccess && corner_output != nullptr) {
        error = cudaMemcpy(corner_output, memory.d_output, 
                          width * height * sizeof(unsigned char), 
                          cudaMemcpyDeviceToHost);
    }
    
    if (error == cudaSuccess) {
        result.success = true;
    }
    
    return result;
}

KernelResult launch_custom_canny_stream(
    unsigned char* d_input,
    unsigned char* d_output,
    float* d_temp_buffers,
    int width, 
    int height,
    float low_threshold,
    float high_threshold,
    cudaStream_t stream) {
    
    KernelResult result;
    result.success = false;
    result.execution_time_ms = 0.0f;
    result.memory_used_bytes = width * height * sizeof(float) * 4; // temp buffers
    
    float* d_blurred = TempBufferLayout::get_canny_buffer_ptr(d_temp_buffers, 0, width, height);
    float* d_grad_x = TempBufferLayout::get_canny_buffer_ptr(d_temp_buffers, 1, width, height);
    float* d_grad_y = TempBufferLayout::get_canny_buffer_ptr(d_temp_buffers, 2, width, height);
    float* d_magnitude = TempBufferLayout::get_canny_buffer_ptr(d_temp_buffers, 3, width, height);
    
    float* d_direction = d_grad_x;
    float* d_suppressed = d_grad_y;
    
    try {
        // Step 1: Gaussian blur
        launch_gaussian_blur_kernel(d_input, d_blurred, width, height, stream);
        
        // Step 2: Compute gradients
        launch_sobel_kernel(d_blurred, d_grad_x, d_grad_y, width, height, stream);
        
        // Step 3: Compute gradient magnitude and direction
        launch_gradient_magnitude_kernel(d_grad_x, d_grad_y, d_magnitude, d_direction, width, height, stream);
        
        // Step 4: Non-maximum suppression
        launch_non_maximum_suppression_kernel(d_magnitude, d_direction, d_suppressed, width, height, stream);
        
        // Step 5: Double thresholding
        launch_double_threshold_kernel(d_suppressed, d_output, width, height, low_threshold, high_threshold, stream);
        
        // Check for any CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error in Canny pipeline: %s\n", cudaGetErrorString(error));
            return result;
        }
        
        result.success = true;
    } catch (...) {
        result.success = false;
    }
    
    return result;
}

KernelResult launch_custom_harris_stream(
    unsigned char* d_input,
    unsigned char* d_output,
    float* d_temp_buffers,
    int width,
    int height,
    int block_size,
    int ksize,
    float k,
    float threshold,
    cudaStream_t stream) {
    
    KernelResult result;
    result.success = false;
    result.execution_time_ms = 0.0f;
    result.memory_used_bytes = width * height * sizeof(float) * 1;
    
    float* d_response = TempBufferLayout::get_harris_buffer_ptr(d_temp_buffers, 0, width, height);
    
    try {
        // Step 1: Compute Harris response
        launch_harris_response_kernel(d_input, d_response, width, height, block_size, ksize, k, stream);
        
        // Step 2: Threshold and non-maximum suppression
        launch_harris_threshold_kernel(d_response, d_output, width, height, threshold, stream);
        
        // Check for any CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error in Harris pipeline: %s\n", cudaGetErrorString(error));
            return result;
        }
        
        result.success = true;
    } catch (...) {
        result.success = false;
    }
    
    return result;
}