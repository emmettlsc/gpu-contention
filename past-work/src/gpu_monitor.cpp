#include "gpu_monitor.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

GPUMonitor::GPUMonitor() : initialized_(false), cuda_device_id_(0) {
    memset(&nvml_device_, 0, sizeof(nvml_device_));
}

GPUMonitor::~GPUMonitor() {
    cleanup();
}

bool GPUMonitor::initialize() {
    if (initialized_) {
        return true;
    }
    
    if (!initialize_nvml()) {
        std::cerr << "Failed to initialize NVML" << std::endl;
        return false;
    }
    
    if (!initialize_cuda_device()) {
        std::cerr << "Failed to initialize CUDA device" << std::endl;
        cleanup();
        return false;
    }
    
    if (!query_hardware_info()) {
        std::cerr << "Failed to query GPU hardware info" << std::endl;
        cleanup();
        return false;
    }
    
    initialized_ = true;
    std::cout << "GPU Monitor initialized successfully for: " 
              << hardware_info_.name << std::endl;
    
    return true;
}

void GPUMonitor::cleanup() {
    if (initialized_) {
        nvmlShutdown();
        initialized_ = false;
    }
}

bool GPUMonitor::initialize_nvml() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "NVML Init failed: " << nvml_error_string(result) << std::endl;
        return false;
    }
    
    // Get device handle for GPU 0
    result = nvmlDeviceGetHandleByIndex(0, &nvml_device_);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get NVML device handle: " 
                  << nvml_error_string(result) << std::endl;
        nvmlShutdown();
        return false;
    }
    
    return true;
}

bool GPUMonitor::initialize_cuda_device() {
    cudaError_t cuda_result = cudaSetDevice(0);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " 
                  << cudaGetErrorString(cuda_result) << std::endl;
        return false;
    }
    
    cuda_device_id_ = 0;
    return true;
}

bool GPUMonitor::query_hardware_info() {
    nvmlReturn_t result;
    
    //  GPU name
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetName(nvml_device_, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU name: " << nvml_error_string(result) << std::endl;
        return false;
    }
    hardware_info_.name = normalize_gpu_name(std::string(name));
    
    //  compute capability
    int major, minor;
    cudaError_t cuda_result = cudaDeviceGetAttribute(&major, 
        cudaDevAttrComputeCapabilityMajor, cuda_device_id_);
    if (cuda_result == cudaSuccess) {
        hardware_info_.major_compute_capability = major;
        cuda_result = cudaDeviceGetAttribute(&minor, 
            cudaDevAttrComputeCapabilityMinor, cuda_device_id_);
        if (cuda_result == cudaSuccess) {
            hardware_info_.minor_compute_capability = minor;
        }
    }
    
    //  SM count
    int sm_count;
    cuda_result = cudaDeviceGetAttribute(&sm_count, 
        cudaDevAttrMultiProcessorCount, cuda_device_id_);
    if (cuda_result == cudaSuccess) {
        hardware_info_.sm_count = sm_count;
    }
    
    //  memory info
    size_t free_mem, total_mem;
    cuda_result = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_result == cudaSuccess) {
        hardware_info_.total_memory_mb = total_mem / (1024 * 1024);
    }
    
    //  max threads per block
    int max_threads;
    cuda_result = cudaDeviceGetAttribute(&max_threads, 
        cudaDevAttrMaxThreadsPerBlock, cuda_device_id_);
    if (cuda_result == cudaSuccess) {
        hardware_info_.max_threads_per_block = max_threads;
    }
    
    // est memory bandwidth based on known GPU types
    if (hardware_info_.name.find("TITAN V") != std::string::npos) {
        hardware_info_.memory_bandwidth_gb_s = 650.0f;
        hardware_info_.architecture = "Volta";
        hardware_info_.max_blocks_per_sm = 32;
    } else if (hardware_info_.name.find("Tesla T4") != std::string::npos) {
        hardware_info_.memory_bandwidth_gb_s = 320.0f;
        hardware_info_.architecture = "Turing";
        hardware_info_.max_blocks_per_sm = 16;
    } else if (hardware_info_.name.find("A10G") != std::string::npos) {
        hardware_info_.memory_bandwidth_gb_s = 600.0f;
        hardware_info_.architecture = "Ampere";
        hardware_info_.max_blocks_per_sm = 16;
    } else {
        // Default fallback based on compute capability
        if (hardware_info_.major_compute_capability >= 8) {
            hardware_info_.memory_bandwidth_gb_s = 500.0f;
            hardware_info_.architecture = "Ampere";
            hardware_info_.max_blocks_per_sm = 16;
        } else if (hardware_info_.major_compute_capability >= 7) {
            hardware_info_.memory_bandwidth_gb_s = 400.0f;
            hardware_info_.architecture = "Volta/Turing";
            hardware_info_.max_blocks_per_sm = 24;
        } else {
            hardware_info_.memory_bandwidth_gb_s = 200.0f;
            hardware_info_.architecture = "Pascal";
            hardware_info_.max_blocks_per_sm = 32;
        }
    }
    
    return true;
}

GPUResourceState GPUMonitor::get_current_state() {
    GPUResourceState state;
    
    if (!initialized_) {
        set_error_state(state, "GPU Monitor not initialized");
        return state;
    }
    
    bool success = true;
    success &= query_utilization(state);
    success &= query_memory_info(state);
    success &= query_temperature(state);
    success &= query_processes(state);
    
    if (success) {
        calculate_derived_metrics(state);
        state.is_valid = true;
    }
    
    return state;
}

bool GPUMonitor::query_utilization(GPUResourceState& state) {
    nvmlUtilization_t utilization;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(nvml_device_, &utilization);
    
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get utilization: " << nvml_error_string(result) << std::endl;
        return false;
    }
    
    state.sm_utilization = static_cast<float>(utilization.gpu);    
    return true;
}

bool GPUMonitor::query_memory_info(GPUResourceState& state) {
    nvmlMemory_t memory;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(nvml_device_, &memory);
    
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get memory info: " << nvml_error_string(result) << std::endl;
        return false;
    }
    
    state.total_memory_mb = memory.total / (1024 * 1024);
    state.free_memory_mb = memory.free / (1024 * 1024);
    state.memory_utilization = (static_cast<float>(memory.used) / memory.total) * 100.0f;
    
    return true;
}

bool GPUMonitor::query_temperature(GPUResourceState& state) {
    unsigned int temp;
    nvmlReturn_t result = nvmlDeviceGetTemperature(nvml_device_, 
        NVML_TEMPERATURE_GPU, &temp);
    
    if (result != NVML_SUCCESS) {
        // Temperature query failure is not critical
        state.temperature_c = 0.0f;
        return true;
    }
    
    state.temperature_c = static_cast<float>(temp);
    return true;
}

bool GPUMonitor::query_processes(GPUResourceState& state) {
    unsigned int process_count = 0;
    
    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses(nvml_device_, 
        &process_count, nullptr);
    
    if (result == NVML_ERROR_INSUFFICIENT_SIZE || result == NVML_SUCCESS) {
        state.active_compute_processes = static_cast<int>(process_count);
        return true;
    }
    
    state.active_compute_processes = 0;
    return true;
}

void GPUMonitor::calculate_derived_metrics(GPUResourceState& state) {
    state.memory_pressure_factor = std::min(1.0f, state.memory_utilization / 100.0f);
    
    state.memory_bandwidth_used_gb_s = estimate_memory_bandwidth_usage(state);
    
    float thermal_threshold = 80.0f;
    if (hardware_info_.name.find("TITAN V") != std::string::npos) {
        thermal_threshold = 83.0f;
    } else if (hardware_info_.name.find("Tesla T4") != std::string::npos) {
        thermal_threshold = 89.0f;
    }
    
    state.is_thermally_throttled = (state.temperature_c > thermal_threshold);
}

float GPUMonitor::estimate_memory_bandwidth_usage(const GPUResourceState& state) {
    float base_usage = hardware_info_.memory_bandwidth_gb_s * 0.1f;
    
    float utilization_factor = (state.sm_utilization + state.memory_utilization) / 200.0f;
    utilization_factor = std::min(1.0f, utilization_factor);
    
    return base_usage + (hardware_info_.memory_bandwidth_gb_s * 0.8f * utilization_factor);
}

GPUHardwareInfo GPUMonitor::get_hardware_info() {
    return hardware_info_;
}

std::string GPUMonitor::get_gpu_name() const {
    return hardware_info_.name;
}

float GPUMonitor::get_max_memory_bandwidth() const {
    return hardware_info_.memory_bandwidth_gb_s;
}

std::string GPUMonitor::normalize_gpu_name(const std::string& raw_name) {
    std::string normalized = raw_name;
    
    size_t pos = normalized.find("NVIDIA ");
    if (pos != std::string::npos) {
        normalized.erase(pos, 7);
    }
    
    pos = normalized.find("GeForce ");
    if (pos != std::string::npos) {
        normalized.erase(pos, 8);
    }
    
    return normalized;
}

bool GPUMonitor::is_nvml_available() {
    nvmlReturn_t result = nvmlInit();
    if (result == NVML_SUCCESS) {
        nvmlShutdown();
        return true;
    }
    return false;
}

void GPUMonitor::set_error_state(GPUResourceState& state, const std::string& error) {
    state.is_valid = false;
    state.error_message = error;
}

std::string GPUMonitor::nvml_error_string(nvmlReturn_t result) {
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_UNINITIALIZED: return "NVML not initialized";
        case NVML_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case NVML_ERROR_NOT_SUPPORTED: return "Not supported";
        case NVML_ERROR_NO_PERMISSION: return "No permission";
        case NVML_ERROR_ALREADY_INITIALIZED: return "Already initialized";
        case NVML_ERROR_NOT_FOUND: return "Not found";
        case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient size";
        case NVML_ERROR_INSUFFICIENT_POWER: return "Insufficient power";
        case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver not loaded";
        case NVML_ERROR_TIMEOUT: return "Timeout";
        case NVML_ERROR_IRQ_ISSUE: return "IRQ issue";
        case NVML_ERROR_LIBRARY_NOT_FOUND: return "Library not found";
        case NVML_ERROR_FUNCTION_NOT_FOUND: return "Function not found";
        case NVML_ERROR_CORRUPTED_INFOROM: return "Corrupted inforom";
        case NVML_ERROR_GPU_IS_LOST: return "GPU is lost";
        case NVML_ERROR_RESET_REQUIRED: return "Reset required";
        case NVML_ERROR_OPERATING_SYSTEM: return "Operating system error";
        case NVML_ERROR_LIB_RM_VERSION_MISMATCH: return "Library/RM version mismatch";
        case NVML_ERROR_IN_USE: return "In use";
        case NVML_ERROR_MEMORY: return "Memory error";
        case NVML_ERROR_NO_DATA: return "No data";
        case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED: return "vGPU ECC not supported";
        case NVML_ERROR_UNKNOWN: 
        default: return "Unknown error";
    }
}