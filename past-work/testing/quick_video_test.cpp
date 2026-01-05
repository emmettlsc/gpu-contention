// fully llm generated for sanity check
#include "video_experiment_framework.hpp"
#include "kernels.hpp"
#include <iostream>

int main() {
    std::cout << "=== Quick Video Test ===" << std::endl;
    
    VideoExperimentRunner runner("quick_test_results");
    if (!runner.initialize()) {
        std::cerr << "Failed to initialize" << std::endl;
        return -1;
    }
    
    // test with a single small video
    VideoConfig config;
    config.input_path = "../test_videos/480p_short_mixed.mp4";
    config.name = "quick_test";
    config.target_fps = 0; // max throughput
    config.max_frames = 60; // 2 seconds
    config.save_output_video = false;
    config.scheduling_strategy = "adaptive";
    config.output_path = "quick_test_output.mp4";
    
    std::cout << "Running quick test with adaptive strategy..." << std::endl;
    auto result = runner.run_single_video_experiment(config);
    
    std::cout << "Results:" << std::endl;
    std::cout << "  Average FPS: " << result.average_fps << std::endl;
    std::cout << "  Total time: " << result.total_processing_time_s << "s" << std::endl;
    std::cout << "  Frames processed: " << result.total_frames_processed << std::endl;
    std::cout << "  Scheduler accuracy: " << result.scheduler_accuracy_percent << "%" << std::endl;
    
    return 0;
}