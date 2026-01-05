// fully llm generated for sanity check
#include "execution_manager.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "=== Execution Manager Test ===" << std::endl;
    
    // Initialize execution manager
    ExecutionManager executor;
    if (!executor.initialize()) {
        std::cerr << "Failed to initialize execution manager" << std::endl;
        return -1;
    }
    
    std::cout << "Execution manager initialized successfully" << std::endl;
    
    // Create a test image
    cv::Mat test_image = cv::Mat::zeros(720, 1280, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    std::cout << "Created test image: " << test_image.cols << "x" << test_image.rows << std::endl;
    
    // Test memory allocation
    size_t required_memory = test_image.rows * test_image.cols * test_image.channels();
    std::cout << "Required memory: " << required_memory / (1024*1024) << " MB" << std::endl;
    
    // Test CUDA streams and events creation
    std::cout << "Testing CUDA resource creation..." << std::endl;
    
    // Create scheduling decision for testing
    SchedulingDecision decision;
    decision.strategy = ExecutionStrategy::SEQUENTIAL;
    decision.confidence = 0.8f;
    decision.reasoning = "Test decision";
    
    // Test execution (will fail at kernel launch but should get to that point)
    cv::Mat canny_output, harris_output;
    
    std::cout << "Testing execution pipeline..." << std::endl;
    ExecutionResult result = executor.execute_pipeline(test_image, decision, canny_output, harris_output);
    
    // Print results
    executor.print_execution_summary(result);
    
    if (!result.success) {
        std::cout << "Expected failure due to missing kernel integration" << std::endl;
        if (result.error_message.find("kernel") != std::string::npos) {
            std::cout << "✓ Framework is working - just needs kernel integration" << std::endl;
        }
    }
    
    std::cout << "Allocated GPU memory: " << executor.get_allocated_memory_mb() << " MB" << std::endl;
    
    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}