// fully llm generated for sanity check
#include "gpu_monitor.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>

int main() {
    std::cout << "=== GPU Monitor Test ===" << std::endl;
    
    // Test NVML availability first
    if (!GPUMonitor::is_nvml_available()) {
        std::cerr << "NVML is not available on this system" << std::endl;
        return -1;
    }
    
    std::cout << "NVML is available" << std::endl;
    
    // Initialize GPU monitor
    GPUMonitor monitor;
    if (!monitor.initialize()) {
        std::cerr << "Failed to initialize GPU monitor" << std::endl;
        return -1;
    }
    
    std::cout << "GPU monitor initialized successfully" << std::endl;
    
    // Get hardware information
    auto hw_info = monitor.get_hardware_info();
    std::cout << "\n=== Hardware Information ===" << std::endl;
    std::cout << "GPU Name: " << hw_info.name << std::endl;
    std::cout << "Architecture: " << hw_info.architecture << std::endl;
    std::cout << "Compute Capability: " << hw_info.major_compute_capability 
              << "." << hw_info.minor_compute_capability << std::endl;
    std::cout << "SM Count: " << hw_info.sm_count << std::endl;
    std::cout << "Total Memory: " << hw_info.total_memory_mb << " MB" << std::endl;
    std::cout << "Max Threads/Block: " << hw_info.max_threads_per_block << std::endl;
    std::cout << "Max Blocks/SM: " << hw_info.max_blocks_per_sm << std::endl;
    std::cout << "Memory Bandwidth: " << hw_info.memory_bandwidth_gb_s << " GB/s" << std::endl;
    
    // Monitor real-time state
    std::cout << "\n=== Real-time Monitoring (5 samples) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    
    for (int i = 0; i < 5; i++) {
        auto state = monitor.get_current_state();
        
        if (!state.is_valid) {
            std::cerr << "Invalid state: " << state.error_message << std::endl;
            continue;
        }
        
        std::cout << "Sample " << (i+1) << ":" << std::endl;
        std::cout << "  SM Utilization: " << state.sm_utilization << "%" << std::endl;
        std::cout << "  Memory Utilization: " << state.memory_utilization << "%" << std::endl;
        std::cout << "  Free Memory: " << state.free_memory_mb << " MB" << std::endl;
        std::cout << "  Temperature: " << state.temperature_c << "°C" << std::endl;
        std::cout << "  Active Processes: " << state.active_compute_processes << std::endl;
        std::cout << "  Memory Pressure: " << state.memory_pressure_factor << std::endl;
        std::cout << "  Estimated Bandwidth: " << state.memory_bandwidth_used_gb_s << " GB/s" << std::endl;
        std::cout << "  Thermally Throttled: " << (state.is_thermally_throttled ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
        
        if (i < 4) {  // Don't sleep after last iteration
            sleep(1);
        }
    }
    
    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}