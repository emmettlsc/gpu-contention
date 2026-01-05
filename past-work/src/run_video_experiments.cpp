#include "video_experiment_framework.hpp"
#include <iostream>
#include <filesystem>

int main() {
    std::cout << "=== Video Processing Experiment Suite ===" << std::endl;
    
    VideoExperimentRunner runner("video_results");
    if (!runner.initialize()) {
        std::cerr << "Failed to initialize video experiment runner" << std::endl;
        return -1;
    }
    
    std::cout << "Experiment runner initialized successfully!" << std::endl;
    
    // Experiment 1: Strategy Comparison on Different Resolutions
    std::cout << "\n=== Experiment 1: Throughput Comparison ===" << std::endl;
    
    std::vector<std::string> test_videos = {
        "../test_videos/480p_short_mixed.mp4",
        "../test_videos/720p_short_mixed.mp4", 
        "../test_videos/1080p_short_mixed.mp4",
        "../test_videos/1440p_short_mixed.mp4"
    };
    
    std::vector<std::string> strategies = {"adaptive", "sequential", "concurrent"};
    
    for (const auto& video_path : test_videos) {
        if (!std::filesystem::exists(video_path)) {
            std::cout << "Skipping missing video: " << video_path << std::endl;
            continue;
        }
        
        std::cout << "\nTesting video: " << video_path << std::endl;
        
        for (const auto& strategy : strategies) {
            VideoConfig config;
            config.input_path = video_path;
            config.output_path = "output_" + strategy + "_" + std::filesystem::path(video_path).stem().string() + ".mp4";
            config.name = std::filesystem::path(video_path).stem().string();
            config.target_fps = 0; // Maximum throughput
            config.max_frames = 300; // Process 10 seconds at 30fps
            config.save_output_video = false; // Don't save for bulk testing
            config.scheduling_strategy = strategy;
            
            std::cout << "  Strategy: " << strategy << std::flush;
            auto result = runner.run_single_video_experiment(config);
            
            std::cout << " -> " << std::fixed << std::setprecision(1) 
                      << result.average_fps << " FPS";
            
            if (strategy == "adaptive") {
                std::cout << " (accuracy: " << result.scheduler_accuracy_percent << "%)";
            }
            std::cout << std::endl;
        }
    }
    
    // Experiment 2: Real-time Performance Test
    std::cout << "\n=== Experiment 2: Real-time Performance ===" << std::endl;
    
    std::vector<int> target_fps_values = {24, 30, 60};
    
    for (int target_fps : target_fps_values) {
        std::cout << "\nTarget FPS: " << target_fps << std::endl;
        
        VideoConfig config;
        config.input_path = "../test_videos/720p_medium_realistic.mp4";
        config.name = "720p_realtime_test";
        config.target_fps = target_fps;
        config.max_frames = target_fps * 30; // 30 seconds
        config.save_output_video = false;
        
        for (const auto& strategy : strategies) {
            config.scheduling_strategy = strategy;
            config.output_path = "realtime_" + strategy + "_" + std::to_string(target_fps) + "fps.mp4";
            
            std::cout << "  " << strategy << ": " << std::flush;
            auto result = runner.run_single_video_experiment(config);
            
            std::cout << result.target_fps_achievement_rate << "% on-time, "
                      << result.average_fps << " avg FPS" << std::endl;
        }
    }
    
    // Experiment 3: Content Complexity Analysis
    std::cout << "\n=== Experiment 3: Content Complexity Analysis ===" << std::endl;
    
    std::vector<std::string> complexity_videos = {
        "../test_videos/720p_short_low_complexity.mp4",
        "../test_videos/720p_short_high_complexity.mp4", 
        "../test_videos/720p_short_mixed.mp4"
    };
    
    for (const auto& video_path : complexity_videos) {
        if (!std::filesystem::exists(video_path)) continue;
        
        std::cout << "\nComplexity test: " << std::filesystem::path(video_path).stem().string() << std::endl;
        
        VideoConfig config;
        config.input_path = video_path;
        config.name = std::filesystem::path(video_path).stem().string();
        config.target_fps = 0;
        config.max_frames = 300;
        config.save_output_video = false;
        config.scheduling_strategy = "adaptive";
        config.output_path = "complexity_test.mp4";
        
        auto result = runner.run_single_video_experiment(config);
        
        std::cout << "  Adaptive scheduler: " << result.average_fps << " FPS, "
                  << result.strategy_switches << " strategy switches, "
                  << result.scheduler_accuracy_percent << "% accuracy" << std::endl;
    }
    
    // Experiment 4: Sustained Performance Test
    std::cout << "\n=== Experiment 4: Sustained Performance ===" << std::endl;
    
    VideoConfig sustained_config;
    sustained_config.input_path = "../test_videos/1080p_medium_mixed.mp4";
    sustained_config.name = "sustained_performance_test";
    sustained_config.target_fps = 0;
    sustained_config.max_frames = 1800; // 60 seconds at 30fps
    sustained_config.save_output_video = false;
    
    for (const auto& strategy : strategies) {
        sustained_config.scheduling_strategy = strategy;
        sustained_config.output_path = "sustained_" + strategy + ".mp4";
        
        std::cout << "  " << strategy << ": " << std::flush;
        auto result = runner.run_single_video_experiment(sustained_config);
        
        std::cout << result.average_fps << " FPS over " << result.total_processing_time_s 
                  << "s processing time" << std::endl;
    }
    
    // Save all results
    std::cout << "\n=== Saving Results ===" << std::endl;
    std::cout << "Total experiments completed: " << runner.get_results_count() << std::endl;
    
    runner.save_results_csv("video_experiment_results.csv");
    runner.save_detailed_results_json("video_detailed_results.json");
    runner.generate_video_processing_report();
    
    std::cout << "\n=== Experiment Suite Complete ===" << std::endl;
    std::cout << "Results saved to video_results/ directory" << std::endl;
    std::cout << "Key findings:" << std::endl;
    std::cout << "- Check video_experiment_results.csv for summary data" << std::endl;
    std::cout << "- Run analysis scripts to generate plots and tables" << std::endl;
    
    return 0;
}