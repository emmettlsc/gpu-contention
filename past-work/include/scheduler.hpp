#pragma once
#include "gpu_monitor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>

enum class ExecutionStrategy {
    SEQUENTIAL,
    CONCURRENT
};

enum class KernelType {
    MEMORY_BOUND,    // canny edge detection
    COMPUTE_BOUND,   // harris corner detection
    MIXED            // oflow????? hell no
};

struct KernelSpec {
    KernelType type;
    std::string name;
    int estimated_blocks;
    float estimated_bandwidth_gb_s;
    
    KernelSpec(KernelType t, const std::string& n, int blocks, float bandwidth) 
        : type(t), name(n), estimated_blocks(blocks), estimated_bandwidth_gb_s(bandwidth) {}
};

struct ImageSpec {
    int width;
    int height;
    int channels;
    
    ImageSpec(int w, int h, int c = 3) : width(w), height(h), channels(c) {}
    
    int total_pixels() const { return width * height; }
    size_t memory_size_bytes() const { return width * height * channels; }
};

struct SchedulingDecision {
    ExecutionStrategy strategy;
    std::string reasoning;
    float confidence;          // 0-1, how confident scheduler is
    
    float memory_pressure_score;
    float sm_occupancy_score;
    float thermal_score;
    float workload_complementarity_score;
    float final_score;         // combined weighted score
    
    SchedulingDecision() : 
        strategy(ExecutionStrategy::SEQUENTIAL), confidence(0.0f),
        memory_pressure_score(0.0f), sm_occupancy_score(0.0f),
        thermal_score(0.0f), workload_complementarity_score(0.0f),
        final_score(0.0f) {}
};

// TODO: add JSON dependency (sike not doing that but know that it could be make into loading from JSON)
struct SchedulerConfig {
    std::string gpu_name;
    std::string architecture;
    
    // Hardware specs
    int sm_count;
    float memory_bandwidth_gb_s;
    
    // Scheduling thresholds
    float memory_bandwidth_threshold_gb_s;
    float sm_utilization_threshold_percent;
    float memory_utilization_threshold_percent;
    int min_concurrent_image_pixels;
    int max_blocks_per_sm;
    float thermal_threshold_c;
    
    // Heuristic weights
    float memory_pressure_weight;
    float sm_occupancy_weight;
    float thermal_weight;
    float workload_complementarity_weight;
    
    // stuff i found through trial and error
    SchedulerConfig() :
        memory_bandwidth_threshold_gb_s(200.0f),
        sm_utilization_threshold_percent(70.0f),
        memory_utilization_threshold_percent(80.0f),
        min_concurrent_image_pixels(921600),  // 720p
        max_blocks_per_sm(4),
        thermal_threshold_c(80.0f),
        memory_pressure_weight(0.4f),
        sm_occupancy_weight(0.3f),
        thermal_weight(0.2f),
        workload_complementarity_weight(0.1f) {}
};

class Scheduler {
public:
    Scheduler();
    ~Scheduler() = default;
    
    bool initialize(const GPUHardwareInfo& gpu_info);
    bool initialize_with_config(const SchedulerConfig& config);
    
    SchedulingDecision decide_execution_strategy(
        const ImageSpec& image,
        const GPUResourceState& gpu_state,
        const std::vector<KernelSpec>& kernels
    );
    
    void update_performance_feedback(
        const SchedulingDecision& decision,
        float actual_time_ms,
        bool was_optimal
    );
    
    const SchedulerConfig& get_config() const { return config_; }
    void print_decision_breakdown(const SchedulingDecision& decision) const;
    
    static SchedulerConfig create_config_for_gpu(const GPUHardwareInfo& gpu_info);
    static std::string strategy_to_string(ExecutionStrategy strategy);
    
private:
    SchedulerConfig config_;
    bool initialized_;
    
    int total_decisions_;
    int correct_predictions_;
    
    float calculate_memory_pressure_score(
        const GPUResourceState& state,
        const std::vector<KernelSpec>& kernels,
        const ImageSpec& image
    );
    
    float calculate_sm_occupancy_score(
        const ImageSpec& image,
        const std::vector<KernelSpec>& kernels,
        const GPUResourceState& state
    );
    
    float calculate_thermal_score(const GPUResourceState& state);
    
    float calculate_workload_complementarity_score(const std::vector<KernelSpec>& kernels);
    
    int estimate_total_blocks(const ImageSpec& image, const std::vector<KernelSpec>& kernels);
    float estimate_total_bandwidth_usage(const std::vector<KernelSpec>& kernels, const ImageSpec& image);
    bool has_mixed_workload_types(const std::vector<KernelSpec>& kernels);
    
    ExecutionStrategy make_final_decision(
        float memory_score, float sm_score, float thermal_score, float complementarity_score
    );
    
    std::string generate_reasoning(
        ExecutionStrategy strategy,
        float memory_score, float sm_score, float thermal_score, float complementarity_score
    );
};