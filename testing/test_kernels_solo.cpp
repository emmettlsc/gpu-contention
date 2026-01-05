// sanity test for each solo kernel + metrics (execution time + validates functionality)
#include "../workloads/kernels.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    printf("=== standalone kernel testing ===\n");

    float time_ms;
    // matmul tests
    printf("matrix multiply:\n");
    test_matmul_solo(256, &time_ms);
    printf("  256x256: %.2f ms\n", time_ms);

    test_matmul_solo(512, &time_ms);
    printf("  512x512: %.2f ms\n", time_ms);

    test_matmul_solo(1024, &time_ms);
    printf("  1024x1024: %.2f ms\n", time_ms);
    printf("\n");

    // conv tests
    printf("2d convolution:\n");
    test_conv2d_solo(1280, 720, &time_ms);
    printf("  720p: %.2f ms\n", time_ms);

    test_conv2d_solo(1920, 1080, &time_ms);
    printf("  1080p: %.2f ms\n", time_ms);
    printf("\n");

    // hist tests
    printf("histogram:\n");
    test_histogram_solo(1280, 720, &time_ms);
    printf("  720p: %.2f ms\n", time_ms);

    test_histogram_solo(1920, 1080, &time_ms);
    printf("  1080p: %.2f ms\n", time_ms);
    printf("\n");

    // bandwidth test
    printf("synthetic bandwidth:\n");
    test_synthetic_bandwidth_solo(1920*1080, &time_ms);
    printf("  1080p elements: %.2f ms\n", time_ms);
    printf("\n");

    //compute tests
    printf("synthetic compute:\n");
    test_synthetic_compute_solo(1920*1080, 50, &time_ms);
    printf("  50 iterations: %.2f ms\n", time_ms);

    test_synthetic_compute_solo(1920*1080, 100, &time_ms);
    printf("  100 iterations: %.2f ms\n", time_ms);

    test_synthetic_compute_solo(1920*1080, 200, &time_ms);
    printf("  200 iterations: %.2f ms\n", time_ms);
    printf("\n");

    printf("=== done!!!! ===\n");

    return 0;
}
