/*
 * run: ./test_canny <input_image.png> <output_edges.png>
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// kernel declarations
extern "C" void launch_canny_full_pipeline(unsigned char* d_input, unsigned char* d_output,
                                          float* d_temp_buffers, int width, int height,
                                          float low_threshold, float high_threshold,
                                          cudaStream_t stream);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <input_image> [output_image]\n";
        std::cout << "example: " << argv[0] << " ../test_images/lena.png edges.png\n";
        std::cout << "\nif no input image, will generate test pattern\n";
        return 0;
    }

    std::string input_path = argv[1];
    std::string output_path = argc > 2 ? argv[2] : "canny_output.png";

    // load or generate input image
    cv::Mat img;
    if (input_path == "generate") {
        std::cout << "generating test image (checkerboard pattern)...\n";
        img = cv::Mat(1080, 1920, CV_8UC1);
        int square_size = 64;
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                bool white = ((x / square_size) + (y / square_size)) % 2 == 0;
                img.at<unsigned char>(y, x) = white ? 255 : 0;
            }
        }
    } else {
        img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "error: failed to load image: " << input_path << "\n";
            return 1;
        }
    }

    int width = img.cols;
    int height = img.rows;
    std::cout << "processing " << width << "x" << height << " image\n";

    // allocate device memory
    unsigned char *d_input, *d_output;
    float *d_temp;
    size_t img_size = width * height * sizeof(unsigned char);
    size_t temp_size = width * height * sizeof(float) * 4;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_temp, temp_size);

    // copy input to device
    cudaMemcpy(d_input, img.data, img_size, cudaMemcpyHostToDevice);

    // run canny edge detection
    std::cout << "running canny edge detection...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float low_threshold = 50.0f;
    float high_threshold = 150.0f;
    launch_canny_full_pipeline(d_input, d_output, d_temp, width, height,
                               low_threshold, high_threshold, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "execution time: " << elapsed_ms << " ms\n";

    // copy result back
    cv::Mat output(height, width, CV_8UC1);
    cudaMemcpy(output.data, d_output, img_size, cudaMemcpyDeviceToHost);

    // save output
    cv::imwrite(output_path, output);
    std::cout << "edges saved to: " << output_path << "\n";

    // show statistics
    int edge_pixels = cv::countNonZero(output);
    float edge_percentage = (edge_pixels * 100.0f) / (width * height);
    std::cout << "edge pixels: " << edge_pixels << " (" << edge_percentage << "%)\n";

    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
