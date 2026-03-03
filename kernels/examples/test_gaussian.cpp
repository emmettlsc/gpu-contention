/*
 * run: ./test_gaussian <input_image.png> [output_image.png]
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

extern "C" void launch_gaussian_blur_float_kernel(const float* d_input, float* d_output,
                                                  int width, int height, cudaStream_t stream);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <input_image> [output_image]\n";
        std::cout << "example: " << argv[0] << " ../test_images/noise_1080p.png blurred.png\n";
        std::cout << "\ngaussian blur is good for denoising\n";
        return 0;
    }

    std::string input_path = argv[1];
    std::string output_path = argc > 2 ? argv[2] : "gaussian_output.png";

    // load input
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "error: failed to load image: " << input_path << "\n";
        return 1;
    }

    cv::Mat img_float;
    img.convertTo(img_float, CV_32F);

    int width = img.cols;
    int height = img.rows;
    std::cout << "processing " << width << "x" << height << " image\n";

    // allocate device memory
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, img_float.data, size, cudaMemcpyHostToDevice);

    // run gaussian blur
    std::cout << "running gaussian blur (5x5)...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    launch_gaussian_blur_float_kernel(d_input, d_output, width, height, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "execution time: " << elapsed_ms << " ms\n";

    // copy result back
    cv::Mat output_float(height, width, CV_32F);
    cudaMemcpy(output_float.data, d_output, size, cudaMemcpyDeviceToHost);

    cv::Mat output;
    output_float.convertTo(output, CV_8U);

    // save output
    cv::imwrite(output_path, output);
    std::cout << "blurred image saved to: " << output_path << "\n";

    // cleannnn
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
