// fully llm generated for sanity check
#include "gpu_monitor.hpp"
#include "scheduler.hpp"
#include "execution_manager.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Full Resource-Aware Scheduling System Test ===" << std::endl;
    
    // Initialize all components
    GPUMonitor monitor;
    if (!monitor.initialize()) {
        std::cerr << "Failed to initialize GPU monitor" << std::endl;
        return -1;
    }
    
    Scheduler scheduler;
    auto gpu_info = monitor.get_hardware_info();
    if (!scheduler.initialize(gpu_info)) {
        std::cerr << "Failed to initialize scheduler" << std::endl;
        return -1;
    }
    
    ExecutionManager executor;
    if (!executor.initialize()) {
        std::cerr << "Failed to initialize execution manager" << std::endl;
        return -1;
    }
    
    std::cout << "All components initialized successfully!" << std::endl;
    std::cout << "GPU: " << gpu_info.name << " (" << gpu_info.architecture << ")" << std::endl;
    
    // Test scenarios
    std::vector<std::tuple<std::string, int, int>> test_cases = {
        {"720p", 1280, 720},
        {"1080p", 1920, 1080},
        {"1440p", 2560, 1440},
        {"4K", 3840, 2160}
    };
    
    // Define workload
    std::vector<KernelSpec> kernels = {
        KernelSpec(KernelType::MEMORY_BOUND, "canny", 0, 150.0f),
        KernelSpec(KernelType::COMPUTE_BOUND, "harris", 0, 50.0f)
    };
    
    std::cout << "\n=== Testing Resource-Aware Scheduling ===" << std::endl;
    
    for (const auto& test_case : test_cases) {
        std::string name = std::get<0>(test_case);
        int width = std::get<1>(test_case);
        int height = std::get<2>(test_case);
        
        std::cout << "\n--- " << name << " (" << width << "x" << height << ") ---" << std::endl;
        
        // Create test image
        cv::Mat test_image = cv::Mat::zeros(height, width, CV_8UC3);
        cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        
        // Convert to grayscale for processing
        cv::Mat gray_image;
        cv::cvtColor(test_image, gray_image, cv::COLOR_BGR2GRAY);
        
        // Get current GPU state
        auto gpu_state = monitor.get_current_state();
        
        // Update kernel specs based on image size
        ImageSpec image_spec(width, height, 1);
        kernels[0].estimated_blocks = (image_spec.total_pixels() + 255) / 256;
        kernels[1].estimated_blocks = (image_spec.total_pixels() + 255) / 256;
        
        // Make scheduling decision
        auto decision = scheduler.decide_execution_strategy(image_spec, gpu_state, kernels);
        
        std::cout << "GPU State: SM=" << gpu_state.sm_utilization 
                  << "%, Mem=" << gpu_state.memory_utilization 
                  << "%, Temp=" << gpu_state.temperature_c << "°C" << std::endl;
        
        std::cout << "Scheduler Decision: " << Scheduler::strategy_to_string(decision.strategy) 
                  << " (confidence: " << std::fixed << std::setprecision(2) 
                  << decision.confidence << ")" << std::endl;
        
        std::cout << "Reasoning: " << decision.reasoning << std::endl;
        
        // Execute with performance comparison
        cv::Mat canny_output, harris_output;
        auto result = executor.execute_with_comparison(gray_image, decision, canny_output, harris_output);
        
        if (result.success) {
            std::cout << "Execution Results:" << std::endl;
            std::cout << "  Strategy Used: " << Scheduler::strategy_to_string(result.strategy_used) << std::endl;
            std::cout << "  Total Time: " << result.total_time_ms << " ms" << std::endl;
            std::cout << "  Canny: " << result.canny_time_ms << " ms" << std::endl;
            std::cout << "  Harris: " << result.harris_time_ms << " ms" << std::endl;
            
            if (result.estimated_sequential_time_ms > 0 && result.estimated_concurrent_time_ms > 0) {
                std::cout << "Performance Comparison:" << std::endl;
                std::cout << "  Sequential would take: " << result.estimated_sequential_time_ms << " ms" << std::endl;
                std::cout << "  Concurrent would take: " << result.estimated_concurrent_time_ms << " ms" << std::endl;
                std::cout << "  Scheduler chose optimal: " << (result.was_optimal_choice ? "YES" : "NO") << std::endl;
                std::cout << "  Performance gain: " << std::showpos << result.performance_gain_percent << "%" << std::endl;
            }
            
            // Provide feedback to scheduler
            scheduler.update_performance_feedback(decision, result.total_time_ms, result.was_optimal_choice);
        } else {
            std::cout << "Execution failed: " << result.error_message << std::endl;
        }
    }
    
    std::cout << "\n=== System Performance Summary ===" << std::endl;
    
    // Final scheduler accuracy
    auto config = scheduler.get_config();
    std::cout << "Scheduler configured for: " << config.gpu_name << std::endl;
    std::cout << "Memory bandwidth threshold: " << config.memory_bandwidth_threshold_gb_s << " GB/s" << std::endl;
    std::cout << "SM utilization threshold: " << config.sm_utilization_threshold_percent << "%" << std::endl;
    
    std::cout << "\nGPU Memory Usage: " << executor.get_allocated_memory_mb() << " MB" << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "Resource-aware scheduling system is fully operational!" << std::endl;
    
    return 0;
}