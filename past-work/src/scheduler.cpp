#include "scheduler.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

Scheduler::Scheduler() : initialized_(false), total_decisions_(0), correct_predictions_(0) {
}

bool Scheduler::initialize(const GPUHardwareInfo& gpu_info) {
    config_ = create_config_for_gpu(gpu_info);
    initialized_ = true;
    
    std::cout << "Scheduler initialized for " << config_.gpu_name 
              << " (" << config_.architecture << ")" << std::endl;
    std::cout << "Config: BW threshold=" << config_.memory_bandwidth_threshold_gb_s 
              << " GB/s, SM threshold=" << config_.sm_utilization_threshold_percent << "%" << std::endl;
    
    return true;
}

bool Scheduler::initialize_with_config(const SchedulerConfig& config) {
    config_ = config;
    initialized_ = true;
    
    std::cout << "Scheduler initialized with custom config for " << config_.gpu_name << std::endl;
    return true;
}

SchedulingDecision Scheduler::decide_execution_strategy(
    const ImageSpec& image,
    const GPUResourceState& gpu_state,
    const std::vector<KernelSpec>& kernels) {
    
    SchedulingDecision decision;
    
    if (!initialized_) {
        decision.reasoning = "Scheduler not initialized";
        decision.confidence = 0.0f;
        return decision;
    }
    
    if (!gpu_state.is_valid) {
        decision.reasoning = "Invalid GPU state: " + gpu_state.error_message;
        decision.confidence = 0.0f;
        return decision;
    }
    
    // Individual decision factors
    decision.memory_pressure_score = calculate_memory_pressure_score(gpu_state, kernels, image);
    decision.sm_occupancy_score = calculate_sm_occupancy_score(image, kernels, gpu_state);
    decision.thermal_score = calculate_thermal_score(gpu_state);
    decision.workload_complementarity_score = calculate_workload_complementarity_score(kernels);
    
    // Weighted final score
    decision.final_score = 
        (decision.memory_pressure_score * config_.memory_pressure_weight) +
        (decision.sm_occupancy_score * config_.sm_occupancy_weight) +
        (decision.thermal_score * config_.thermal_weight) +
        (decision.workload_complementarity_score * config_.workload_complementarity_weight);
    
    // Final decision
    decision.strategy = make_final_decision(
        decision.memory_pressure_score,
        decision.sm_occupancy_score,
        decision.thermal_score,
        decision.workload_complementarity_score
    );
    
    decision.reasoning = generate_reasoning(
        decision.strategy,
        decision.memory_pressure_score,
        decision.sm_occupancy_score,
        decision.thermal_score,
        decision.workload_complementarity_score
    );
    
    float score_magnitude = std::abs(decision.final_score - 0.5f);
    decision.confidence = std::min(1.0f, score_magnitude * 2.0f);
    
    total_decisions_++;
    
    return decision;
}

float Scheduler::calculate_memory_pressure_score(
    const GPUResourceState& state,
    const std::vector<KernelSpec>& kernels,
    const ImageSpec& image) {
    
    float estimated_bandwidth = estimate_total_bandwidth_usage(kernels, image);
    
    float current_bandwidth_pressure = state.memory_bandwidth_used_gb_s / config_.memory_bandwidth_gb_s;
    
    float combined_bandwidth = (state.memory_bandwidth_used_gb_s + estimated_bandwidth) / 
                              config_.memory_bandwidth_gb_s;
    
    float memory_pressure = state.memory_utilization / 100.0f;
    
    float bandwidth_score = std::min(1.0f, combined_bandwidth);
    float memory_score = std::min(1.0f, memory_pressure);
    
    return (bandwidth_score * 0.7f) + (memory_score * 0.3f);
}

float Scheduler::calculate_sm_occupancy_score(
    const ImageSpec& image,
    const std::vector<KernelSpec>& kernels,
    const GPUResourceState& state) {
    
    int total_blocks = estimate_total_blocks(image, kernels);
    
    float blocks_per_sm = static_cast<float>(total_blocks) / config_.sm_count;
    
    float current_sm_utilization = state.sm_utilization / 100.0f;
    
    float occupancy_pressure = blocks_per_sm / config_.max_blocks_per_sm;
    float utilization_pressure = current_sm_utilization;
    
    return std::min(1.0f, (occupancy_pressure * 0.6f) + (utilization_pressure * 0.4f));
}

float Scheduler::calculate_thermal_score(const GPUResourceState& state) {
    if (state.temperature_c <= 0.0f) {
        return 0.0f;
    }
    
    float thermal_pressure = state.temperature_c / config_.thermal_threshold_c;
    
    return std::min(1.0f, thermal_pressure);
}

float Scheduler::calculate_workload_complementarity_score(const std::vector<KernelSpec>& kernels) {
    if (kernels.size() < 2) {
        return 0.0f;
    }
    
    bool has_memory_bound = false;
    bool has_compute_bound = false;
    
    for (const auto& kernel : kernels) {
        if (kernel.type == KernelType::MEMORY_BOUND) has_memory_bound = true;
        if (kernel.type == KernelType::COMPUTE_BOUND) has_compute_bound = true;
    }
    

    if (has_memory_bound && has_compute_bound) {
        return 1.0f;  // Perfect complementarity
    } else {
        return 0.0f;  // Same workload type
    }
}

ExecutionStrategy Scheduler::make_final_decision(
    float memory_score, float sm_score, float thermal_score, float complementarity_score) {
    
    // Simple threshold-based decision
    float pressure_score = (memory_score + sm_score + thermal_score) / 3.0f;
    
    // High pressure or no complementarity = Sequential
    if (pressure_score > 0.8f) {
        return ExecutionStrategy::SEQUENTIAL;
    }
    
    // Low pressure with good complementarity = Concurrent
    if (pressure_score < 0.3f && complementarity_score > 0.7f) {
        return ExecutionStrategy::CONCURRENT;
    }

    if (complementarity_score > 0.7f) {
        return ExecutionStrategy::CONCURRENT;  // Favor concurrent if workloads complement
    }
    
    // Medium pressure = Sequential (conservative choice)
    return ExecutionStrategy::SEQUENTIAL;
}

std::string Scheduler::generate_reasoning(
    ExecutionStrategy strategy,
    float memory_score, float sm_score, float thermal_score, float complementarity_score) {
    
    std::ostringstream oss;
    oss << strategy_to_string(strategy) << " chosen: ";
    
    // Identify dominant factors
    std::vector<std::pair<std::string, float>> factors = {
        {"Memory pressure", memory_score},
        {"SM occupancy", sm_score},
        {"Thermal", thermal_score},
        {"Complementarity", complementarity_score}
    };
    
    std::sort(factors.begin(), factors.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (size_t i = 0; i < std::min(size_t(2), factors.size()); i++) {
        if (i > 0) oss << ", ";
        oss << factors[i].first << "=" << std::fixed << std::setprecision(2) << factors[i].second;
    }
    
    return oss.str();
}

int Scheduler::estimate_total_blocks(const ImageSpec& image, const std::vector<KernelSpec>& kernels) {

    int blocks_per_kernel = (image.total_pixels() + 255) / 256;
    
    return blocks_per_kernel * kernels.size();
}

float Scheduler::estimate_total_bandwidth_usage(const std::vector<KernelSpec>& kernels, const ImageSpec& image) {
    float total_bandwidth = 0.0f;
    
    for (const auto& kernel : kernels) {
        float pixel_factor = static_cast<float>(image.total_pixels()) / (1280.0f * 720.0f);  // Relative to 720p
        total_bandwidth += kernel.estimated_bandwidth_gb_s * pixel_factor;
    }
    
    return total_bandwidth;
}

bool Scheduler::has_mixed_workload_types(const std::vector<KernelSpec>& kernels) {
    bool has_memory = false, has_compute = false;
    
    for (const auto& kernel : kernels) {
        if (kernel.type == KernelType::MEMORY_BOUND) has_memory = true;
        if (kernel.type == KernelType::COMPUTE_BOUND) has_compute = true;
    }
    
    return has_memory && has_compute;
}

void Scheduler::update_performance_feedback(
    const SchedulingDecision& decision,
    float actual_time_ms,
    bool was_optimal) {
    
    if (was_optimal) {
        correct_predictions_++;
    }
}

void Scheduler::print_decision_breakdown(const SchedulingDecision& decision) const {
    std::cout << "=== Scheduling Decision Breakdown ===" << std::endl;
    std::cout << "Strategy: " << strategy_to_string(decision.strategy) << std::endl;
    std::cout << "Confidence: " << std::fixed << std::setprecision(2) << decision.confidence << std::endl;
    std::cout << "Reasoning: " << decision.reasoning << std::endl;
    std::cout << "Scores:" << std::endl;
    std::cout << "  Memory Pressure: " << decision.memory_pressure_score << std::endl;
    std::cout << "  SM Occupancy: " << decision.sm_occupancy_score << std::endl;
    std::cout << "  Thermal: " << decision.thermal_score << std::endl;
    std::cout << "  Complementarity: " << decision.workload_complementarity_score << std::endl;
    std::cout << "  Final Score: " << decision.final_score << std::endl;
    
    if (total_decisions_ > 0) {
        float accuracy = static_cast<float>(correct_predictions_) / total_decisions_ * 100.0f;
        std::cout << "Scheduler Accuracy: " << accuracy << "% (" 
                  << correct_predictions_ << "/" << total_decisions_ << ")" << std::endl;
    }
}

SchedulerConfig Scheduler::create_config_for_gpu(const GPUHardwareInfo& gpu_info) {
    SchedulerConfig config;
    
    config.gpu_name = gpu_info.name;
    config.architecture = gpu_info.architecture;
    config.sm_count = gpu_info.sm_count;
    config.memory_bandwidth_gb_s = gpu_info.memory_bandwidth_gb_s;
    
    // GPU-specific tuning
    if (gpu_info.name.find("TITAN V") != std::string::npos) {
        // TITAN V configuration
        config.memory_bandwidth_threshold_gb_s = 400.0f;
        config.sm_utilization_threshold_percent = 75.0f;
        config.memory_utilization_threshold_percent = 80.0f;
        config.min_concurrent_image_pixels = 921600;  // 720p
        config.max_blocks_per_sm = 4;
        config.thermal_threshold_c = 83.0f;
        config.memory_pressure_weight = 0.4f;
        config.sm_occupancy_weight = 0.3f;
        config.thermal_weight = 0.2f;
        config.workload_complementarity_weight = 0.1f;
    } else if (gpu_info.name.find("Tesla T4") != std::string::npos) {
        // Tesla T4 configuration
        config.memory_bandwidth_threshold_gb_s = 200.0f;
        config.sm_utilization_threshold_percent = 70.0f;
        config.memory_utilization_threshold_percent = 85.0f;
        config.min_concurrent_image_pixels = 1382400;  // 1080p
        config.max_blocks_per_sm = 6;
        config.thermal_threshold_c = 89.0f;
        config.memory_pressure_weight = 0.5f;
        config.sm_occupancy_weight = 0.3f;
        config.thermal_weight = 0.15f;
        config.workload_complementarity_weight = 0.05f;
    }
    return config;
}

std::string Scheduler::strategy_to_string(ExecutionStrategy strategy) {
    switch (strategy) {
        case ExecutionStrategy::SEQUENTIAL: return "SEQUENTIAL";
        case ExecutionStrategy::CONCURRENT: return "CONCURRENT";
        default: return "UNKNOWN";
    }
}