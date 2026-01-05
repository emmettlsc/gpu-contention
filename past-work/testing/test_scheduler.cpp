// fully llm generated for sanity check
#include "gpu_monitor.hpp"
#include "scheduler.hpp"
#include <iostream>

int main() {
    std::cout << "=== Scheduler Test ===" << std::endl;
    
    // Initialize GPU monitor
    GPUMonitor monitor;
    if (!monitor.initialize()) {
        std::cerr << "Failed to initialize GPU monitor" << std::endl;
        return -1;
    }
    
    // Initialize scheduler
    Scheduler scheduler;
    auto gpu_info = monitor.get_hardware_info();
    if (!scheduler.initialize(gpu_info)) {
        std::cerr << "Failed to initialize scheduler" << std::endl;
        return -1;
    }
    
    // Test different scenarios
    std::vector<std::pair<std::string, ImageSpec>> test_images = {
        {"720p", ImageSpec(1280, 720)},
        {"1080p", ImageSpec(1920, 1080)},
        {"1440p", ImageSpec(2560, 1440)},
        {"4K", ImageSpec(3840, 2160)}
    };
    
    // Define typical kernels
    std::vector<KernelSpec> kernels = {
        KernelSpec(KernelType::MEMORY_BOUND, "canny", 0, 150.0f),
        KernelSpec(KernelType::COMPUTE_BOUND, "harris", 0, 50.0f)
    };
    
    // Get current GPU state
    auto gpu_state = monitor.get_current_state();
    
    std::cout << "\n=== Scheduling Decisions ===" << std::endl;
    
    for (const auto& test_case : test_images) {
        std::cout << "\n--- " << test_case.first << " (" 
                  << test_case.second.width << "x" << test_case.second.height << ") ---" << std::endl;
        
        // Update kernel estimates based on image size
        kernels[0].estimated_blocks = (test_case.second.total_pixels() + 255) / 256;
        kernels[1].estimated_blocks = (test_case.second.total_pixels() + 255) / 256;
        
        auto decision = scheduler.decide_execution_strategy(
            test_case.second, gpu_state, kernels);
        
        scheduler.print_decision_breakdown(decision);
    }
    
    // Test with high GPU load simulation
    std::cout << "\n=== High Load Simulation ===" << std::endl;
    GPUResourceState high_load_state = gpu_state;
    high_load_state.sm_utilization = 85.0f;
    high_load_state.memory_utilization = 75.0f;
    high_load_state.temperature_c = 78.0f;
    
    auto decision = scheduler.decide_execution_strategy(
        ImageSpec(1920, 1080), high_load_state, kernels);
    
    std::cout << "1080p under high GPU load:" << std::endl;
    scheduler.print_decision_breakdown(decision);
    
    return 0;
}