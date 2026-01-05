// GPU resource monitoring

#pragma once
#include <nvml.h>
#include <cuda_runtime.h>
#include <string>
#include <memory>

struct GPUResourceState {
    float sm_utilization;              // 0-100% SM utilization
    float memory_utilization;          // 0-100% memory utilization
    size_t free_memory_mb;             // available GPU memory in MB
    size_t total_memory_mb;            // total GPU mem in MB
    
    float memory_bandwidth_used_gb_s;  // est bandwidth usage
    float temperature_c;               // GPU temp
    int active_compute_processes;      // # of active compute processes
    
    bool is_valid;                     // reading valid?
    std::string error_message;         // Error details if invalid
    
    float memory_pressure_factor;      // 0-1, calculated pressure
    bool is_thermally_throttled;       // Temperature-based throttling
    
    GPUResourceState() : 
        sm_utilization(0.0f), memory_utilization(0.0f),
        free_memory_mb(0), total_memory_mb(0),
        memory_bandwidth_used_gb_s(0.0f), temperature_c(0.0f),
        active_compute_processes(0), is_valid(false),
        memory_pressure_factor(0.0f), is_thermally_throttled(false) {}
};

struct GPUHardwareInfo {
    std::string name;
    std::string architecture;
    int major_compute_capability;
    int minor_compute_capability;
    int sm_count;
    size_t total_memory_mb;
    int max_threads_per_block;
    int max_blocks_per_sm;
    float memory_bandwidth_gb_s;
    
    GPUHardwareInfo() :
        major_compute_capability(0), minor_compute_capability(0),
        sm_count(0), total_memory_mb(0), max_threads_per_block(0),
        max_blocks_per_sm(0), memory_bandwidth_gb_s(0.0f) {}
};

class GPUMonitor {
public:
    GPUMonitor();
    ~GPUMonitor();
    
    bool initialize();
    void cleanup();
    
    GPUResourceState get_current_state();
    GPUHardwareInfo get_hardware_info();
    
    bool is_initialized() const { return initialized_; }
    std::string get_gpu_name() const;
    float get_max_memory_bandwidth() const;
    
    static std::string normalize_gpu_name(const std::string& raw_name);
    static bool is_nvml_available();
    
private:
    bool initialized_;
    nvmlDevice_t nvml_device_;
    int cuda_device_id_;
    GPUHardwareInfo hardware_info_;
    
    bool initialize_nvml();
    bool initialize_cuda_device();
    bool query_hardware_info();
    
    bool query_utilization(GPUResourceState& state);
    bool query_memory_info(GPUResourceState& state);
    bool query_temperature(GPUResourceState& state);
    bool query_processes(GPUResourceState& state);
    void calculate_derived_metrics(GPUResourceState& state);
    float estimate_memory_bandwidth_usage(const GPUResourceState& state);
    void set_error_state(GPUResourceState& state, const std::string& error);
    std::string nvml_error_string(nvmlReturn_t result);
};