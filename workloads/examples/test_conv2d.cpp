/*
 * standalone test harness for 2d convolution
 *
 * run: ./test_conv2d <input_image.png> [output_image.png]
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

extern "C" void launch_conv2d_kernel(const float* d_input, float* d_output,
                                     int width, int height, cudaStream_t stream);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <input_image> [output_image]\n";
        std::cout << "example: " << argv[0] << " ../test_images/noise_1080p.png smoothed.png\n";
        return 0;
    }

    std::string input_path = argv[1];
    std::string output_path = argc > 2 ? argv[2] : "conv2d_output.png";

    // load input
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "error: failed to load image: " << input_path << "\n";
        return 1;
    }

    // convert to float
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

    // run convolution
    std::cout << "running 2d convolution (5x5 gaussian)...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    launch_conv2d_kernel(d_input, d_output, width, height, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "execution time: " << elapsed_ms << " ms\n";

    // copy result back
    cv::Mat output_float(height, width, CV_32F);
    cudaMemcpy(output_float.data, d_output, size, cudaMemcpyDeviceToHost);

    // convert back to uchar
    cv::Mat output;
    output_float.convertTo(output, CV_8U);

    // save output
    cv::imwrite(output_path, output);
    std::cout << "filtered image saved to: " << output_path << "\n";

    // compute difference for visualization
    cv::Mat diff;
    cv::absdiff(img, output, diff);
    cv::imwrite("conv2d_difference.png", diff * 5); // amplify for visibility

    std::cout << "difference map saved to: conv2d_difference.png\n";

    // statistics
    cv::Scalar mean_before = cv::mean(img);
    cv::Scalar mean_after = cv::mean(output);
    std::cout << "\nmean intensity:\n";
    std::cout << "  before: " << mean_before[0] << "\n";
    std::cout << "  after: " << mean_after[0] << "\n";

    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
