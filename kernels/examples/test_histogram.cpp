/* 
 * run: ./test_histogram <input_image.png>
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

extern "C" void launch_histogram_kernel(const unsigned char* d_input, unsigned int* d_hist,
                                       int width, int height, cudaStream_t stream);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <input_image>\n";
        std::cout << "example: " << argv[0] << " ../test_images/gradient_1080p.png\n";
        return 0;
    }

    std::string input_path = argv[1];

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
    unsigned char *d_input;
    unsigned int *d_hist;
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));

    cudaMemcpy(d_input, img.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // run histogram
    std::cout << "computing histogram...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    launch_histogram_kernel(d_input, d_hist, width, height, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "execution time: " << elapsed_ms << " ms\n";

    // copy result back
    unsigned int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // print histogram summary
    std::cout << "\nhistogram summary:\n";
    unsigned int total_pixels = width * height;
    unsigned int cumulative = 0;

    std::cout << "intensity | count | percentage | cumulative\n";
    std::cout << "----------------------------------------------\n";

    for (int i = 0; i < 256; i++) {
        if (h_hist[i] > 0) {
            cumulative += h_hist[i];
            float pct = (h_hist[i] * 100.0f) / total_pixels;
            float cum_pct = (cumulative * 100.0f) / total_pixels;
            printf("%3d       | %6u | %6.2f%%   | %6.2f%%\n", i, h_hist[i], pct, cum_pct);
        }
    }

    // statistics
    float mean = 0.0f;
    for (int i = 0; i < 256; i++) {
        mean += i * h_hist[i];
    }
    mean /= total_pixels;

    float variance = 0.0f;
    for (int i = 0; i < 256; i++) {
        float diff = i - mean;
        variance += diff * diff * h_hist[i];
    }
    variance /= total_pixels;
    float stddev = sqrt(variance);

    std::cout << "\nstatistics:\n";
    std::cout << "  mean intensity: " << mean << "\n";
    std::cout << "  std deviation: " << stddev << "\n";

    // visualize histogram
    int hist_height = 200;
    int bin_width = 2;
    cv::Mat hist_img(hist_height, 256 * bin_width, CV_8UC3, cv::Scalar(255, 255, 255));

    unsigned int max_count = *std::max_element(h_hist, h_hist + 256);
    for (int i = 0; i < 256; i++) {
        int bar_height = (h_hist[i] * hist_height) / max_count;
        cv::rectangle(hist_img,
                     cv::Point(i * bin_width, hist_height - bar_height),
                     cv::Point((i + 1) * bin_width, hist_height),
                     cv::Scalar(0, 0, 0), -1);
    }

    cv::imwrite("histogram_visualization.png", hist_img);
    std::cout << "\nhistogram visualization saved to: histogram_visualization.png\n";

    // cleanup
    cudaFree(d_input);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
