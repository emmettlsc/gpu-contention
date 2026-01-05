#include "video_experiment_framework.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>

VideoProcessor::VideoProcessor(int comparison_sampling_rate) 
    : comparison_sampling_rate_(comparison_sampling_rate) {
}

VideoProcessor::~VideoProcessor() {
}

bool VideoProcessor::initialize() {
    if (!monitor_.initialize()) {
        std::cerr << "Failed to initialize GPU monitor" << std::endl;
        return false;
    }
    
    auto gpu_info = monitor_.get_hardware_info();
    if (!scheduler_.initialize(gpu_info)) {
        std::cerr << "Failed to initialize scheduler" << std::endl;
        return false;
    }
    
    if (!executor_.initialize()) {
        std::cerr << "Failed to initialize execution manager" << std::endl;
        return false;
    }
    
    std::cout << "Video processor initialized for " << gpu_info.name << std::endl;
    return true;
}

VideoResult VideoProcessor::process_video(const VideoConfig& config) {
    VideoResult result;
    result.video_name = config.name;
    result.scheduling_strategy = config.scheduling_strategy;
    result.config = config;
    
    // input video
    cv::VideoCapture cap(config.input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << config.input_path << std::endl;
        return result;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Processing video: " << width << "x" << height << " @ " << fps << " fps, " 
              << total_frames << " frames" << std::endl;
    
    cv::VideoWriter writer;
    if (config.save_output_video) {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(config.output_path, fourcc, fps, cv::Size(width * 2, height)); // Side-by-side output
    }
    
    start_time_ = std::chrono::high_resolution_clock::now();
    int frames_processed = 0;
    int max_frames = (config.max_frames > 0) ? config.max_frames : total_frames;
    
    cv::Mat frame, gray_frame;
    
    std::cout << "Starting video processing with " << config.scheduling_strategy << " strategy..." << std::endl;
    
    while (cap.read(frame) && frames_processed < max_frames) {
        // Convert to grayscale
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        // Process frame
        FrameResult frame_result = process_single_frame(gray_frame, frames_processed, config.scheduling_strategy);
        result.frame_results.push_back(frame_result);
        // Save output if requested
        if (config.save_output_video && writer.isOpened()) {
            cv::Mat output_frame;
            cv::hconcat(gray_frame, gray_frame, output_frame); // Placeholder
            cv::cvtColor(output_frame, output_frame, cv::COLOR_GRAY2BGR);
            writer.write(output_frame);
        }
        // Real-time pacing if needed
        if (config.target_fps > 0) {
            auto target_frame_time = std::chrono::milliseconds(1000 / config.target_fps);
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = current_time - start_time_;
            auto expected_time = std::chrono::milliseconds((frames_processed + 1) * 1000 / config.target_fps);
            
            if (elapsed < expected_time) {
                std::this_thread::sleep_for(expected_time - elapsed);
            }
        }
        
        frames_processed++;
        if (frames_processed % 30 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_s = std::chrono::duration<float>(current_time - start_time_).count();
            float current_fps = frames_processed / elapsed_s;
            
            std::cout << "Processed " << frames_processed << "/" << max_frames 
                      << " frames, Current FPS: " << std::fixed << std::setprecision(1) 
                      << current_fps << "\r" << std::flush;
        }
    }
    
    std::cout << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_processing_time_s = std::chrono::duration<float>(end_time - start_time_).count();
    result.total_frames_processed = frames_processed;
    result.average_fps = frames_processed / result.total_processing_time_s;
    
    int correct_decisions = 0;
    float total_improvement = 0.0f;
    int strategy_changes = 0;
    ExecutionStrategy last_strategy = ExecutionStrategy::SEQUENTIAL;
    
    for (const auto& frame_result : result.frame_results) {
        if (frame_result.was_optimal_choice) correct_decisions++;
        total_improvement += frame_result.performance_improvement_percent;
        
        if (frame_result.chosen_strategy != last_strategy) {
            strategy_changes++;
            last_strategy = frame_result.chosen_strategy;
        }
    }
    
    if (frames_processed > 0) {
        result.scheduler_accuracy_percent = (float)correct_decisions / frames_processed * 100.0f;
        result.average_performance_improvement = total_improvement / frames_processed;
    }
    result.strategy_switches = strategy_changes;
    if (config.target_fps > 0) {
        float target_frame_time_ms = 1000.0f / config.target_fps;
        int frames_meeting_target = 0;
        
        for (const auto& frame_result : result.frame_results) {
            if (frame_result.processing_time_ms <= target_frame_time_ms) {
                frames_meeting_target++;
            }
        }
        
        result.target_fps_achievement_rate = (float)frames_meeting_target / frames_processed * 100.0f;
    }
    
    std::cout << "Video processing complete!" << std::endl;
    std::cout << "Average FPS: " << result.average_fps << std::endl;
    std::cout << "Scheduler accuracy: " << result.scheduler_accuracy_percent << "%" << std::endl;
    std::cout << "Strategy switches: " << result.strategy_switches << std::endl;
    
    return result;
}

FrameResult VideoProcessor::process_single_frame(const cv::Mat& frame, int frame_number, 
                                               const std::string& strategy) {
    FrameResult result;
    result.frame_number = frame_number;
    result.timestamp = std::chrono::high_resolution_clock::now();
    result.width = frame.cols;
    result.height = frame.rows;
    auto gpu_state = monitor_.get_current_state();
    result.sm_utilization = gpu_state.sm_utilization;
    result.memory_utilization = gpu_state.memory_utilization;
    result.temperature = gpu_state.temperature_c;
    
    result.frame_complexity_score = calculate_frame_complexity(frame);
    
    SchedulingDecision decision;
    if (strategy == "adaptive") {
        ImageSpec image_spec(frame.cols, frame.rows, 1);
        std::vector<KernelSpec> kernels = {
            KernelSpec(KernelType::MEMORY_BOUND, "canny", 0, 150.0f),
            KernelSpec(KernelType::COMPUTE_BOUND, "harris", 0, 50.0f)
        };
        
        decision = scheduler_.decide_execution_strategy(image_spec, gpu_state, kernels);
    } else if (strategy == "sequential") {
        decision.strategy = ExecutionStrategy::SEQUENTIAL;
        decision.confidence = 1.0f;
        decision.reasoning = "Always sequential";
    } else if (strategy == "concurrent") {
        decision.strategy = ExecutionStrategy::CONCURRENT;
        decision.confidence = 1.0f;
        decision.reasoning = "Always concurrent";
    }
    
    result.chosen_strategy = decision.strategy;
    result.scheduler_confidence = decision.confidence;
    result.reasoning = decision.reasoning;
    
    cv::Mat canny_output, harris_output;
    auto execution_result = executor_.execute_pipeline(frame, decision, canny_output, harris_output);
    
    result.processing_time_ms = execution_result.total_time_ms;
    result.canny_time_ms = execution_result.canny_time_ms;
    result.harris_time_ms = execution_result.harris_time_ms;
    result.overhead_time_ms = execution_result.overhead_time_ms;
    
    if (should_sample_alternative_strategy(frame_number)) {
        SchedulingDecision alt_decision = decision;
        alt_decision.strategy = (decision.strategy == ExecutionStrategy::SEQUENTIAL) ? 
                               ExecutionStrategy::CONCURRENT : ExecutionStrategy::SEQUENTIAL;
        
        cv::Mat alt_canny, alt_harris;
        auto alt_result = executor_.execute_pipeline(frame, alt_decision, alt_canny, alt_harris);
        
        result.alt_strategy_time_ms = alt_result.total_time_ms;
        result.was_optimal_choice = (execution_result.total_time_ms <= alt_result.total_time_ms);
        
        if (alt_result.total_time_ms > 0) {
            result.performance_improvement_percent = 
                ((alt_result.total_time_ms - execution_result.total_time_ms) / alt_result.total_time_ms) * 100.0f;
        }
    } else {
        result.was_optimal_choice = true;
        result.performance_improvement_percent = 0.0f;
    }
    
    return result;
}

float VideoProcessor::calculate_frame_complexity(const cv::Mat& frame) {
    cv::Mat edges;
    cv::Canny(frame, edges, 50, 150);
    
    int edge_pixels = cv::countNonZero(edges);
    float total_pixels = frame.rows * frame.cols;
    
    return edge_pixels / total_pixels;
}

bool VideoProcessor::should_sample_alternative_strategy(int frame_number) {
    return (frame_number % comparison_sampling_rate_) == 0;
}

bool VideoProcessor::meets_realtime_requirement(float processing_time_ms, int target_fps) {
    if (target_fps <= 0) return true;
    
    float required_time_ms = 1000.0f / target_fps;
    return processing_time_ms <= required_time_ms;
}

VideoExperimentRunner::VideoExperimentRunner(const std::string& results_dir) 
    : results_directory_(results_dir) {
    std::filesystem::create_directories(results_dir);
}

bool VideoExperimentRunner::initialize() {
    return processor_.initialize();
}

VideoResult VideoExperimentRunner::run_single_video_experiment(const VideoConfig& config) {
    VideoResult result = processor_.process_video(config);
        all_results_.push_back(result);
    
    return result;
}

std::vector<VideoResult> VideoExperimentRunner::run_strategy_comparison(const VideoConfig& base_config) {
    std::vector<VideoResult> results;
    std::vector<std::string> strategies = {"adaptive", "sequential", "concurrent"};
    
    for (const auto& strategy : strategies) {
        VideoConfig config = base_config;
        config.scheduling_strategy = strategy;
        config.output_path = results_directory_ + "/" + strategy + "_" + config.name + ".mp4";
        
        VideoResult result = run_single_video_experiment(config);
        results.push_back(result);
        all_results_.push_back(result);
    }
    
    return results;
}

void VideoExperimentRunner::save_results_csv(const std::string& filename) {
    std::string full_path = results_directory_ + "/" + filename;
    std::ofstream file(full_path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to create CSV file: " << full_path << std::endl;
        return;
    }
    
    std::cout << "Writing " << all_results_.size() << " results to " << full_path << std::endl;
    
    // Write CSV header
    file << "video_name,scheduling_strategy,resolution,average_fps,total_processing_time_s,"
         << "scheduler_accuracy_percent,average_performance_improvement,strategy_switches,"
         << "target_fps,target_fps_achievement_rate,total_frames_processed\n";
    
    // Write data rows
    for (const auto& result : all_results_) {
        std::string resolution = std::to_string(result.frame_results.empty() ? 0 : result.frame_results[0].width) + 
                               "x" + std::to_string(result.frame_results.empty() ? 0 : result.frame_results[0].height);
        
        file << result.video_name << ","
             << result.scheduling_strategy << ","
             << resolution << ","
             << result.average_fps << ","
             << result.total_processing_time_s << ","
             << result.scheduler_accuracy_percent << ","
             << result.average_performance_improvement << ","
             << result.strategy_switches << ","
             << result.config.target_fps << ","
             << result.target_fps_achievement_rate << ","
             << result.total_frames_processed << "\n";
    }
    
    file.close();
    
    std::cout << "Results saved to " << full_path << std::endl;
    
    // Verify file was created and has content
    if (std::filesystem::exists(full_path)) {
        std::ifstream check_file(full_path, std::ios::ate);
        std::streamsize size = check_file.tellg();
        std::cout << "CSV file size: " << size << " bytes" << std::endl;
    }
}

void VideoExperimentRunner::save_detailed_results_json(const std::string& filename) {
    // Simple JSON output - you could use a proper JSON library for more complex data
    std::ofstream file(results_directory_ + "/" + filename);
    
    file << "[\n";
    for (size_t i = 0; i < all_results_.size(); ++i) {
        const auto& result = all_results_[i];
        
        file << "  {\n";
        file << "    \"video_name\": \"" << result.video_name << "\",\n";
        file << "    \"scheduling_strategy\": \"" << result.scheduling_strategy << "\",\n";
        file << "    \"average_fps\": " << result.average_fps << ",\n";
        file << "    \"scheduler_accuracy_percent\": " << result.scheduler_accuracy_percent << ",\n";
        file << "    \"strategy_switches\": " << result.strategy_switches << ",\n";
        file << "    \"total_frames_processed\": " << result.total_frames_processed << "\n";
        file << "  }";
        
        if (i < all_results_.size() - 1) file << ",";
        file << "\n";
    }
    file << "]\n";
    
    std::cout << "Detailed results saved to " << results_directory_ + "/" + filename << std::endl;
}

void VideoExperimentRunner::generate_video_processing_report() {
    std::ofstream report(results_directory_ + "/experiment_report.txt");
    
    report << "=== Video Processing Experiment Report ===\n\n";
    
    // Summary statistics
    if (!all_results_.empty()) {
        // Group by strategy
        std::map<std::string, std::vector<float>> fps_by_strategy;
        std::map<std::string, std::vector<float>> accuracy_by_strategy;
        
        for (const auto& result : all_results_) {
            fps_by_strategy[result.scheduling_strategy].push_back(result.average_fps);
            if (result.scheduling_strategy == "adaptive") {
                accuracy_by_strategy[result.scheduling_strategy].push_back(result.scheduler_accuracy_percent);
            }
        }
        
        report << "Performance Summary:\n";
        for (const auto& pair : fps_by_strategy) {
            const auto& fps_values = pair.second;
            float avg_fps = std::accumulate(fps_values.begin(), fps_values.end(), 0.0f) / fps_values.size();
            
            report << "  " << pair.first << ": " << std::fixed << std::setprecision(1) 
                   << avg_fps << " FPS average\n";
        }
        
        if (!accuracy_by_strategy["adaptive"].empty()) {
            float avg_accuracy = std::accumulate(accuracy_by_strategy["adaptive"].begin(), 
                                               accuracy_by_strategy["adaptive"].end(), 0.0f) / 
                                accuracy_by_strategy["adaptive"].size();
            report << "  Adaptive scheduler accuracy: " << avg_accuracy << "%\n";
        }
    }
    
    report << "\nExperiment completed at: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
    report << "Total experiments: " << all_results_.size() << "\n";
    
    std::cout << "Experiment report saved to " << results_directory_ + "/experiment_report.txt" << std::endl;
}

size_t VideoExperimentRunner::get_results_count() const {
    return all_results_.size();
}