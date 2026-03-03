#pragma once
#include "scheduler.hpp"
#include <string>
#include <functional>
#include <map>
#include <cuda_runtime.h>

using KernelLauncher = std::function<void(void* input, void* output, void* temp, cudaStream_t)>;

struct WorkloadPair {
    std::string name;
    std::string description;
    std::string hypothesis;
    KernelSpec kernel1;
    KernelSpec kernel2;

    KernelLauncher launcher1;
    KernelLauncher launcher2;
};

class WorkloadRegistry {
public:
    static WorkloadRegistry& instance();

    void register_workload(const std::string& name, const WorkloadPair& workload);
    const WorkloadPair& get_workload(const std::string& name) const;
    bool has_workload(const std::string& name) const;

    std::vector<std::string> list_workloads() const;

private:
    WorkloadRegistry() = default;
    std::map<std::string, WorkloadPair> workloads_;
};

void register_all_workloads();

KernelSpec create_canny_spec(int width, int height);
KernelSpec create_harris_spec(int width, int height);
KernelSpec create_matmul_spec(int N, bool small = false);
KernelSpec create_conv2d_spec(int width, int height);
KernelSpec create_histogram_spec(int width, int height);
KernelSpec create_gaussian_spec(int width, int height);
KernelSpec create_synthetic_bandwidth_spec(int N);
KernelSpec create_synthetic_compute_spec(int N, int iterations);
