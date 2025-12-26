/*
 * canny edge detection kernel
 *
 * input: grayscale image (unsigned char*), width x height
 * output: edge-detected image (unsigned char*), same dimensions
 * characteristics: memory-bound, high bandwidth usage (~180 GB/s)
 *
 * this is a multi-stage pipeline:
 * 1. gaussian blur to reduce noise
 * 2. sobel gradients to find edge directions
 * 3. non-maximum suppression to thin edges
 * 4. double threshold to classify strong/weak edges
 *
 * to run solo: see workloads/examples/test_canny.cpp
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ===== stage 1: gaussian blur =====

__global__ void gaussian_blur_kernel(unsigned char* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 5x5 gaussian kernel (sigma ≈ 1.0)
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

// ===== stage 2: sobel gradients =====

__global__ void sobel_kernel(float* input, float* grad_x, float* grad_y, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float gx = 0.0f, gy = 0.0f;

    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        // sobel x (vertical edges)
        gx = -1.0f * input[(y-1)*width + (x-1)] + 1.0f * input[(y-1)*width + (x+1)] +
             -2.0f * input[y*width + (x-1)]     + 2.0f * input[y*width + (x+1)] +
             -1.0f * input[(y+1)*width + (x-1)] + 1.0f * input[(y+1)*width + (x+1)];

        // sobel y (horizontal edges)
        gy = -1.0f * input[(y-1)*width + (x-1)] + -2.0f * input[(y-1)*width + x] + -1.0f * input[(y-1)*width + (x+1)] +
             1.0f * input[(y+1)*width + (x-1)]  + 2.0f * input[(y+1)*width + x]  + 1.0f * input[(y+1)*width + (x+1)];
    }

    grad_x[y * width + x] = gx;
    grad_y[y * width + x] = gy;
}

// ===== stage 3: gradient magnitude and direction =====

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

// ===== stage 4: non-maximum suppression =====

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

// ===== stage 5: double threshold =====

__global__ void double_threshold_kernel(float* input, unsigned char* output,
                                      int width, int height, float low_thresh, float high_thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float val = input[idx];

    if (val >= high_thresh) {
        output[idx] = 255; // strong edge
    } else if (val >= low_thresh) {
        output[idx] = 127; // weak edge
    } else {
        output[idx] = 0;   // not an edge
    }
}

// ===== launch wrappers =====

extern "C" void launch_gaussian_blur_kernel(unsigned char* input, float* output,
                                           int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    gaussian_blur_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
}

extern "C" void launch_sobel_kernel(float* input, float* grad_x, float* grad_y,
                                   int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    sobel_kernel<<<gridSize, blockSize, 0, stream>>>(input, grad_x, grad_y, width, height);
}

extern "C" void launch_gradient_magnitude_kernel(float* grad_x, float* grad_y, float* magnitude,
                                                float* direction, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    gradient_magnitude_kernel<<<gridSize, blockSize, 0, stream>>>(grad_x, grad_y, magnitude, direction, width, height);
}

extern "C" void launch_non_maximum_suppression_kernel(float* magnitude, float* direction,
                                                     float* suppressed, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    non_maximum_suppression_kernel<<<gridSize, blockSize, 0, stream>>>(magnitude, direction, suppressed, width, height);
}

extern "C" void launch_double_threshold_kernel(float* input, unsigned char* output,
                                              int width, int height, float low_thresh, float high_thresh, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    double_threshold_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height, low_thresh, high_thresh);
}

// ===== full pipeline wrapper =====

extern "C" void launch_canny_full_pipeline(unsigned char* d_input, unsigned char* d_output,
                                          float* d_temp_buffers, int width, int height,
                                          float low_threshold, float high_threshold,
                                          cudaStream_t stream) {
    // temp buffer layout: blurred, grad_x, grad_y, magnitude
    float* d_blurred = d_temp_buffers;
    float* d_grad_x = d_temp_buffers + width * height;
    float* d_grad_y = d_temp_buffers + 2 * width * height;
    float* d_magnitude = d_temp_buffers + 3 * width * height;
    float* d_direction = d_grad_x;  // reuse
    float* d_suppressed = d_grad_y; // reuse

    launch_gaussian_blur_kernel(d_input, d_blurred, width, height, stream);
    launch_sobel_kernel(d_blurred, d_grad_x, d_grad_y, width, height, stream);
    launch_gradient_magnitude_kernel(d_grad_x, d_grad_y, d_magnitude, d_direction, width, height, stream);
    launch_non_maximum_suppression_kernel(d_magnitude, d_direction, d_suppressed, width, height, stream);
    launch_double_threshold_kernel(d_suppressed, d_output, width, height, low_threshold, high_threshold, stream);
}
