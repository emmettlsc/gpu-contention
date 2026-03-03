/*
 * run: ./test_harris <input_image.png> <output_corners.png>
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

extern "C" void launch_harris_full_pipeline(unsigned char* d_input, unsigned char* d_output,
                                           float* d_temp_response, int width, int height,
                                           int block_size, float k, float threshold,
                                           cudaStream_t stream);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <input_image> [output_image]\n";
        std::cout << "example: " << argv[0] << " ../test_images/circles_1080p.png corners.png\n";
        return 0;
    }

    std::string input_path = argv[1];
    std::string output_path = argc > 2 ? argv[2] : "harris_output.png";

    // load input
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "error: failed to load image: " << input_path << "\n";
        return 1;
    }

    int width = img.cols;
    int height = img.rows;
    std::cout << "processing " << width << "x" << height << " image\n";

    // allocate device memory
    unsigned char *d_input, *d_output;
    float *d_response;
    size_t img_size = width * height * sizeof(unsigned char);
    size_t response_size = width * height * sizeof(float);

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_response, response_size);

    cudaMemcpy(d_input, img.data, img_size, cudaMemcpyHostToDevice);

    // run harris corner detection
    std::cout << "running harris corner detection...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int block_size = 2;
    float k = 0.04f;
    float threshold = 1000000.0f;  // adjust based on image
    launch_harris_full_pipeline(d_input, d_output, d_response, width, height,
                                block_size, k, threshold, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "execution time: " << elapsed_ms << " ms\n";

    // copy result back
    cv::Mat output(height, width, CV_8UC1);
    cudaMemcpy(output.data, d_output, img_size, cudaMemcpyDeviceToHost);

    // visualize corners on original image (for better visibility)
    cv::Mat vis;
    cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (output.at<unsigned char>(y, x) == 255) {
                cv::circle(vis, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    // save outputs
    cv::imwrite(output_path, output);
    cv::imwrite("harris_visualized.png", vis);
    std::cout << "corner map saved to: " << output_path << "\n";
    std::cout << "visualization saved to: harris_visualized.png\n";

    // statistics
    int corner_count = cv::countNonZero(output);
    std::cout << "corners detected: " << corner_count << "\n";

    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_response);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
