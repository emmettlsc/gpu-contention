#include "workload_manager.hpp"
#include "kernels.hpp"
#include <stdexcept>

// singleton instance
WorkloadRegistry& WorkloadRegistry::instance() {
    static WorkloadRegistry registry;
    return registry;
}

void WorkloadRegistry::register_workload(const std::string& name, const WorkloadPair& workload) {
    workloads_[name] = workload;
}

const WorkloadPair& WorkloadRegistry::get_workload(const std::string& name) const {
    auto it = workloads_.find(name);
    if (it == workloads_.end()) {
        throw std::runtime_error("workload not found: " + name);
    }
    return it->second;
}

bool WorkloadRegistry::has_workload(const std::string& name) const {
    return workloads_.find(name) != workloads_.end();
}

std::vector<std::string> WorkloadRegistry::list_workloads() const {
    std::vector<std::string> names;
    for (const auto& pair : workloads_) {
        names.push_back(pair.first);
    }
    return names;
}

// ===== KERNEL SPEC CREATORS =====

KernelSpec create_canny_spec(int width, int height) {
    int total_pixels = width * height;
    int estimated_blocks = (total_pixels + 255) / 256;

    KernelSpec spec(KernelType::MEMORY_BOUND, "canny_edge_detection",
                    estimated_blocks, 180.0f, 0.3f, 45.0f);
    spec.input_size_bytes = width * height * sizeof(unsigned char);
    spec.output_size_bytes = width * height * sizeof(unsigned char);
    spec.temp_size_bytes = width * height * sizeof(float) * 4;
    return spec;
}

KernelSpec create_harris_spec(int width, int height) {
    int total_pixels = width * height;
    int estimated_blocks = (total_pixels + 255) / 256;

    KernelSpec spec(KernelType::MEMORY_BOUND, "harris_corner_detection",
                    estimated_blocks, 160.0f, 0.4f, 40.0f);
    spec.input_size_bytes = width * height * sizeof(unsigned char);
    spec.output_size_bytes = width * height * sizeof(unsigned char);
    spec.temp_size_bytes = width * height * sizeof(float);
    return spec;
}

KernelSpec create_matmul_spec(int N, bool small) {
    int estimated_blocks = ((N + 15) / 16) * ((N + 15) / 16);  // 16x16 tiles

    KernelSpec spec(KernelType::COMPUTE_BOUND, small ? "matmul_small" : "matmul",
                    estimated_blocks, 80.0f, 10.0f, 85.0f);
    spec.input_size_bytes = N * N * sizeof(float) * 2;  // two matrices
    spec.output_size_bytes = N * N * sizeof(float);
    spec.temp_size_bytes = 0;
    return spec;
}

KernelSpec create_conv2d_spec(int width, int height) {
    int total_pixels = width * height;
    int estimated_blocks = (total_pixels + 255) / 256;

    KernelSpec spec(KernelType::COMPUTE_BOUND, "conv2d",
                    estimated_blocks, 100.0f, 8.0f, 75.0f);
    spec.input_size_bytes = width * height * sizeof(float);
    spec.output_size_bytes = width * height * sizeof(float);
    spec.temp_size_bytes = 0;
    return spec;
}

KernelSpec create_histogram_spec(int width, int height) {
    int total_pixels = width * height;
    int estimated_blocks = (total_pixels + 255) / 256;

    KernelSpec spec(KernelType::ATOMIC_BOUND, "histogram",
                    estimated_blocks, 120.0f, 0.1f, 30.0f);
    spec.input_size_bytes = width * height * sizeof(unsigned char);
    spec.output_size_bytes = 256 * sizeof(unsigned int);
    spec.temp_size_bytes = 0;
    return spec;
}

KernelSpec create_gaussian_spec(int width, int height) {
    int total_pixels = width * height;
    int estimated_blocks = (total_pixels + 255) / 256;

    KernelSpec spec(KernelType::MEMORY_BOUND, "gaussian_blur",
                    estimated_blocks, 140.0f, 0.5f, 50.0f);
    spec.input_size_bytes = width * height * sizeof(float);
    spec.output_size_bytes = width * height * sizeof(float);
    spec.temp_size_bytes = 0;
    return spec;
}

KernelSpec create_synthetic_bandwidth_spec(int N) {
    int estimated_blocks = (N + 255) / 256;

    KernelSpec spec(KernelType::MEMORY_BOUND, "synthetic_bandwidth",
                    estimated_blocks, 250.0f, 0.01f, 15.0f);
    spec.input_size_bytes = N * sizeof(float);
    spec.output_size_bytes = N * sizeof(float);
    spec.temp_size_bytes = 0;
    return spec;
}

KernelSpec create_synthetic_compute_spec(int N, int iterations) {
    int estimated_blocks = (N + 255) / 256;

    KernelSpec spec(KernelType::COMPUTE_BOUND, "synthetic_compute",
                    estimated_blocks, 20.0f, 100.0f, 95.0f);
    spec.input_size_bytes = N * sizeof(float);
    spec.output_size_bytes = N * sizeof(float);
    spec.temp_size_bytes = 0;
    return spec;
}

// ===== WORKLOAD REGISTRATION =====

void register_all_workloads() {
    auto& registry = WorkloadRegistry::instance();

    // dimensions for testing
    int width = 1920;
    int height = 1080;
    int matmul_size = 1024;
    int small_matmul_size = 256;
    int synthetic_size = width * height;

    // workload 1: memory-saturating pair (baseline)
    WorkloadPair workload1;
    workload1.name = "canny-harris";
    workload1.description = "memory-bound pair: canny edge detection + harris corner detection";
    workload1.hypothesis = "sequential should win due to memory saturation";
    workload1.kernel1 = create_canny_spec(width, height);
    workload1.kernel2 = create_harris_spec(width, height);
    // launchers will be set in actual execution code
    registry.register_workload("canny-harris", workload1);

    // workload 2: compute-heavy pair
    WorkloadPair workload2;
    workload2.name = "matmul-conv";
    workload2.description = "compute-bound pair: matrix multiply + 2d convolution";
    workload2.hypothesis = "concurrent might help since both use compute without saturating memory";
    workload2.kernel1 = create_matmul_spec(matmul_size, false);
    workload2.kernel2 = create_conv2d_spec(width, height);
    registry.register_workload("matmul-conv", workload2);

    // workload 3: asymmetric pair
    WorkloadPair workload3;
    workload3.name = "small-matmul-canny";
    workload3.description = "asymmetric pair: small matrix multiply + canny edge detection";
    workload3.hypothesis = "concurrent might help if small kernel hides in large kernel execution";
    workload3.kernel1 = create_matmul_spec(small_matmul_size, true);
    workload3.kernel2 = create_canny_spec(width, height);
    registry.register_workload("small-matmul-canny", workload3);

    // workload 4: different memory patterns
    WorkloadPair workload4;
    workload4.name = "histogram-gaussian";
    workload4.description = "different memory patterns: histogram (atomics) + gaussian blur (coalesced)";
    workload4.hypothesis = "sequential probably wins but different memory patterns might enable some concurrency";
    workload4.kernel1 = create_histogram_spec(width, height);
    workload4.kernel2 = create_gaussian_spec(width, height);
    registry.register_workload("histogram-gaussian", workload4);

    // workload 5: synthetic stress test
    WorkloadPair workload5;
    workload5.name = "bandwidth-compute";
    workload5.description = "synthetic stress test: pure bandwidth + pure compute";
    workload5.hypothesis = "best case for concurrent - if sequential wins here, thesis is proven";
    workload5.kernel1 = create_synthetic_bandwidth_spec(synthetic_size);
    workload5.kernel2 = create_synthetic_compute_spec(synthetic_size, 100);
    registry.register_workload("bandwidth-compute", workload5);
}
