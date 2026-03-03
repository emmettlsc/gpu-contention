/*
 * harris corner detection kernel
 *
 * input: grayscale image (unsigned char*), width x height
 * output: corner map (unsigned char*), 255 = corner, 0 = not corner
 * characteristics: memory-bound, high bandwidth usage (~160 GB/s)
 *
 * harris corner detector finds points where image gradients change in multiple directions
 * uses structure tensor (matrix of gradient products) and computes corner response
 *
 * to run solo: see workloads/examples/test_harris.cpp
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===== compute harris response =====

__global__ void harris_response_kernel(unsigned char* input, float* response,
                                      int width, int height, int block_size, int ksize, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // calculate gradients using sobel
    float Ix = 0.0f, Iy = 0.0f;

    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        // sobel x
        Ix = -1.0f * input[(y-1)*width + (x-1)] + 1.0f * input[(y-1)*width + (x+1)] +
             -2.0f * input[y*width + (x-1)]     + 2.0f * input[y*width + (x+1)] +
             -1.0f * input[(y+1)*width + (x-1)] + 1.0f * input[(y+1)*width + (x+1)];

        // sobel y
        Iy = -1.0f * input[(y-1)*width + (x-1)] + -2.0f * input[(y-1)*width + x] + -1.0f * input[(y-1)*width + (x+1)] +
             1.0f * input[(y+1)*width + (x-1)]  + 2.0f * input[(y+1)*width + x]  + 1.0f * input[(y+1)*width + (x+1)];
    }

    // products of derivatives
    float Ixx = Ix * Ix;
    float Ixy = Ix * Iy;
    float Iyy = Iy * Iy;

    // sum over window
    float Sxx = 0.0f, Sxy = 0.0f, Syy = 0.0f;
    int half_block = block_size / 2;

    for (int dy = -half_block; dy <= half_block; dy++) {
        for (int dx = -half_block; dx <= half_block; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);

            if (nx > 0 && nx < width-1 && ny > 0 && ny < height-1) {
                // recalculate gradients for this pixel
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

    // harris response: det(M) - k * trace(M)^2
    // M = [Sxx Sxy; Sxy Syy]
    float det = Sxx * Syy - Sxy * Sxy;
    float trace = Sxx + Syy;
    float harris_resp = det - k * trace * trace;

    response[y * width + x] = harris_resp;
}

// ===== threshold and non-max suppression =====

__global__ void harris_threshold_kernel(float* response, unsigned char* output,
                                       int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float resp = response[idx];

    // non-max suppression in 3x3 neighborhood
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

// ===== launch wrappers =====

extern "C" void launch_harris_response_kernel(unsigned char* input, float* response,
                                             int width, int height, int block_size, int ksize, float k, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    harris_response_kernel<<<gridSize, blockSize, 0, stream>>>(input, response, width, height, block_size, ksize, k);
}

extern "C" void launch_harris_threshold_kernel(float* response, unsigned char* output,
                                              int width, int height, float threshold, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    harris_threshold_kernel<<<gridSize, blockSize, 0, stream>>>(response, output, width, height, threshold);
}

// ===== full pipeline wrapper =====

extern "C" void launch_harris_full_pipeline(unsigned char* d_input, unsigned char* d_output,
                                           float* d_temp_response, int width, int height,
                                           int block_size, float k, float threshold,
                                           cudaStream_t stream) {
    launch_harris_response_kernel(d_input, d_temp_response, width, height, block_size, 3, k, stream);
    launch_harris_threshold_kernel(d_temp_response, d_output, width, height, threshold, stream);
}
