#pragma once
#include "gpu_monitor.hpp"
#include "scheduler.hpp"
#include "execution_manager.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <queue>
#include <thread>
#include <filesystem>
#include <map>
#include <numeric>

struct VideoConfig {
    std::string input_path;
    std::string output_path;
    std::string name;
    int target_fps;
    int max_frames;
    bool save_output_video;
    std::string scheduling_strategy;
};

struct FrameResult {
    int frame_number;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    float sm_utilization;
    float memory_utilization;
    float temperature;
    
    // Scheduling decision
    ExecutionStrategy chosen_strategy;
    float scheduler_confidence;
    std::string reasoning;
    
    // Performance metrics
    float processing_time_ms;
    float canny_time_ms;
    float harris_time_ms;
    float overhead_time_ms;
    
    float alt_strategy_time_ms;
    bool was_optimal_choice;
    float performance_improvement_percent;
    
    // Frame characteristics
    int width, height;
    float frame_complexity_score;
    size_t compressed_size_bytes;
};

struct VideoResult { // only for data collection, not for video processing
    std::string video_name;
    std::string scheduling_strategy;
    VideoConfig config;
    
    // overall performance
    float total_processing_time_s;
    float average_fps;
    float target_fps_achievement_rate; // % of frames meeting real-time target
    int total_frames_processed;
    
    // Scheduler performance
    float scheduler_accuracy_percent;
    float average_performance_improvement;
    int strategy_switches; // How often scheduler changed strategies
    
    // Resource utilization
    float peak_memory_usage_mb;
    float average_sm_utilization;
    float thermal_events; // Times temp exceeded threshold
    
    // Frame-level statistics
    std::vector<FrameResult> frame_results;
    
    float average_processing_quality_score;
};

class VideoProcessor {
private:
    GPUMonitor monitor_;
    Scheduler scheduler_;
    ExecutionManager executor_;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point start_time_;
    std::queue<std::chrono::high_resolution_clock::time_point> frame_timestamps_;
    
    int comparison_sampling_rate_; // only sample every Nth frame for alternative strategy if doing long video processing (like 20 minutes)
    
public:
    VideoProcessor(int comparison_sampling_rate = 10);
    ~VideoProcessor();
    
    bool initialize();
    
    // Main processing functions
    VideoResult process_video(const VideoConfig& config);
    VideoResult process_video_realtime(const VideoConfig& config);
    VideoResult process_video_batch(const VideoConfig& config);
    
    std::vector<VideoResult> compare_strategies(const VideoConfig& base_config);
    
    float calculate_frame_complexity(const cv::Mat& frame);
    bool meets_realtime_requirement(float processing_time_ms, int target_fps);
    
private:
    FrameResult process_single_frame(const cv::Mat& frame, int frame_number, 
                                   const std::string& strategy);
    void save_processed_frame(const cv::Mat& canny, const cv::Mat& harris, 
                            cv::VideoWriter& writer, int frame_number);
    bool should_sample_alternative_strategy(int frame_number);
    void log_frame_processing(const FrameResult& result);
};

class VideoExperimentRunner {
private:
    VideoProcessor processor_;
    std::string results_directory_;
    std::vector<VideoResult> all_results_;
    
public:
    VideoExperimentRunner(const std::string& results_dir = "video_results");
    
    bool initialize();
    
    void run_throughput_comparison_experiment();
    void run_resolution_scaling_experiment();  
    void run_content_complexity_experiment();
    void run_realtime_performance_experiment();
    void run_sustained_processing_experiment();
    
    VideoResult run_single_video_experiment(const VideoConfig& config);
    std::vector<VideoResult> run_strategy_comparison(const VideoConfig& base_config);
    
    // Results management
    void save_results_csv(const std::string& filename);
    void save_detailed_results_json(const std::string& filename);
    void generate_video_processing_report();
    size_t get_results_count() const;
    
    // Test video management
    std::vector<VideoConfig> get_test_video_configs();
    bool download_test_videos(); // Download standard test videos
    VideoConfig create_synthetic_video(int width, int height, int duration_s, 
                                     const std::string& pattern);
};

class VideoResultAnalyzer {
public:
    static void analyze_throughput_comparison(const std::vector<VideoResult>& results);
    static void analyze_realtime_performance(const std::vector<VideoResult>& results);
    static void analyze_scheduler_adaptation(const std::vector<VideoResult>& results);
    
    static void calculate_fps_statistics(const std::vector<VideoResult>& results);
    static void analyze_frame_timing_consistency(const std::vector<VideoResult>& results);
    static void measure_scheduling_overhead(const std::vector<VideoResult>& results);
    
    static void correlate_complexity_with_strategy(const std::vector<VideoResult>& results);
    static void analyze_temporal_patterns(const std::vector<VideoResult>& results);
};

// Plotting and visualization
class VideoPlotGenerator {
public:
    // Main comparison plots
    static void generate_fps_comparison_plot(const std::vector<VideoResult>& results);
    static void generate_processing_time_distribution(const std::vector<VideoResult>& results);
    static void generate_scheduler_decision_timeline(const VideoResult& result);
    
    // Real-time performance plots  
    static void generate_realtime_achievement_plot(const std::vector<VideoResult>& results);
    static void generate_frame_timing_consistency_plot(const std::vector<VideoResult>& results);
    
    // Resource utilization plots
    static void generate_gpu_utilization_over_time(const VideoResult& result);
    static void generate_thermal_performance_plot(const std::vector<VideoResult>& results);
    
    // Content analysis plots
    static void generate_complexity_vs_performance_plot(const std::vector<VideoResult>& results);
    static void generate_strategy_adaptation_heatmap(const VideoResult& result);
    
    // Summary dashboard
    static void generate_video_processing_dashboard(const std::vector<VideoResult>& results);
};