# this does all the synthetic video generation
# i tried using sample-videos but it was impossible finding 
# a good set of videos that met my length, resolution, and complexity requirements
# after i gave up some stack overflow post said using opencv to generate synthetic videos
# was easy so here we are... thank you llms <3
import cv2
import numpy as np
import os

def create_test_video(width, height, fps, duration_s, pattern, output_path):
    """Create synthetic test video with different complexity patterns"""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration_s
    
    print(f"Creating {pattern} video: {width}x{height} @ {fps}fps, {duration_s}s ({total_frames} frames)")
    
    for frame_num in range(total_frames):
        if pattern == "low_complexity":
            # Simple gradient - low edge content
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                intensity = int(255 * y / height)
                frame[y, :] = [intensity, intensity // 2, 255 - intensity]
                
        elif pattern == "high_complexity":
            # Random noise and geometric patterns - high edge content
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            cv2.circle(frame, (width//4, height//4), 50, (255, 255, 255), 2)
            cv2.rectangle(frame, (width//2, height//4), (3*width//4, 3*height//4), (255, 255, 255), 2)
            x_offset = int(50 * np.sin(frame_num * 0.1))
            y_offset = int(30 * np.cos(frame_num * 0.1))
            cv2.circle(frame, (width//2 + x_offset, height//2 + y_offset), 20, (0, 255, 0), -1)
            
        elif pattern == "mixed":
            # Alternating complexity
            if (frame_num // 30) % 2 == 0:  # Change every second
                # Low complexity section
                frame = np.full((height, width, 3), [100, 150, 200], dtype=np.uint8)
                x = int(width * 0.5 + width * 0.3 * np.sin(frame_num * 0.05))
                cv2.circle(frame, (x, height//2), 30, (255, 255, 255), -1)
            else:
                # High complexity section
                frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                for _ in range(20):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
                    
        elif pattern == "realistic":
            # Simulate camera movement over textured surface
            base_texture = np.random.randint(100, 200, (height, width), dtype=np.uint8)
            for i in range(0, width, 20):
                cv2.line(base_texture, (i, 0), (i, height), 255, 1)
            for i in range(0, height, 20):
                cv2.line(base_texture, (0, i), (width, i), 255, 1)
            shake_x = int(5 * np.sin(frame_num * 0.3))
            shake_y = int(3 * np.cos(frame_num * 0.4))

            frame = cv2.merge([base_texture, base_texture, base_texture])

            frame[:, :, 0] = np.clip(frame[:, :, 0].astype(int) + shake_x * 2, 0, 255).astype(np.uint8)
            frame[:, :, 2] = np.clip(frame[:, :, 2].astype(int) + shake_y * 2, 0, 255).astype(np.uint8)
        
        out.write(frame)
        
        if frame_num % (fps * 5) == 0:  # Progress every 5 seconds
            print(f"  Generated {frame_num}/{total_frames} frames...")
    
    out.release()
    print(f"Saved: {output_path}")

def create_test_dataset():
    """Create a comprehensive set of test videos"""
    
    os.makedirs("test_videos", exist_ok=True)
    
    # Resolution and duration combinations
    test_configs = [
        # Format: (width, height, fps, duration_s, description)
        (640, 480, 30, 10, "480p_short"),
        (1280, 720, 30, 10, "720p_short"),
        (1920, 1080, 30, 10, "1080p_short"),
        (2560, 1440, 30, 10, "1440p_short"),
        
        # Longer videos for sustained performance testing
        (1280, 720, 30, 60, "720p_medium"),
        (1920, 1080, 30, 60, "1080p_medium"),
        
        # High frame rate tests
        (1280, 720, 60, 10, "720p_60fps"),
        (1920, 1080, 60, 10, "1080p_60fps"),
    ]
    
    patterns = ["low_complexity", "high_complexity", "mixed", "realistic"]
    
    for width, height, fps, duration, resolution_name in test_configs:
        for pattern in patterns:
            filename = f"test_videos/{resolution_name}_{pattern}.mp4"
            create_test_video(width, height, fps, duration, pattern, filename)
    
    print(f"\nTest dataset created! Generated {len(test_configs) * len(patterns)} videos.")
    
    with open("test_videos/index.txt", "w") as f:
        f.write("# Test Video Dataset\n")
        f.write("# Format: filename, width, height, fps, duration, pattern\n\n")
        
        for width, height, fps, duration, resolution_name in test_configs:
            for pattern in patterns:
                filename = f"{resolution_name}_{pattern}.mp4"
                f.write(f"{filename}, {width}, {height}, {fps}, {duration}, {pattern}\n")

if __name__ == "__main__":
    create_test_dataset()