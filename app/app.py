import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.detection.face_detector import FaceDetector
from src.detection.landmark_extractor import FacialLandmarkExtractor
from src.preprocessing.video_processor import VideoProcessor
from src.analysis.lip_movement import LipMovementAnalyzer
from src.analysis.sync_detector import LipSyncDetector

def process_video(video_path: str, output_dir: str) -> Dict:
    """
    Process a video file to detect lip sync issues.
    
    Args:
        video_path: Path to the video file.
        output_dir: Directory to save processed data.
        
    Returns:
        Dictionary containing analysis results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    face_detector = FaceDetector(min_detection_confidence=0.5)
    landmark_extractor = FacialLandmarkExtractor(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    video_processor = VideoProcessor(face_detector, landmark_extractor)
    lip_analyzer = LipMovementAnalyzer(smoothing_window=15, poly_order=3)
    sync_detector = LipSyncDetector(audio_sampling_rate=16000, window_size=0.1)
    
    # Process video
    print(f"Processing video: {video_path}")
    video_data = video_processor.process_video(
        video_path, 
        output_dir=os.path.join(output_dir, "processed"),
        extract_audio=True,
        fps=None  # Use original fps
    )
    
    # Process frames for landmarks
    print("Extracting facial landmarks...")
    processed_frames = video_processor.process_frames_for_landmarks(
        video_data["frames"],
        output_dir=os.path.join(output_dir, "landmarks")
    )
    
    # Extract lip features
    print("Analyzing lip movements...")
    lip_features = []
    for frame_data in processed_frames:
        if "lip_features" in frame_data:
            lip_features.append(frame_data["lip_features"])
        else:
            lip_features.append({})
    
    # Extract lip time series
    lip_time_series = lip_analyzer.extract_lip_time_series(lip_features)
    
    # Calculate derivatives
    lip_derivatives = lip_analyzer.calculate_derivatives(lip_time_series)
    
    # Get movement features
    movement_features = lip_analyzer.get_movement_features(lip_time_series, lip_derivatives)
    
    # Visualize lip movements
    print("Visualizing lip movements...")
    fig = lip_analyzer.visualize_lip_movements(lip_time_series, fps=video_data["actual_fps"])
    fig.savefig(os.path.join(output_dir, "lip_movements.png"))
    
    # Extract audio features
    print("Analyzing audio...")
    audio_features = sync_detector.extract_audio_features(video_data["audio_path"])
    
    # Align audio and video features
    aligned_audio, aligned_video = sync_detector.align_audio_video_features(
        audio_features, lip_time_series, video_data["actual_fps"]
    )
    
    # Compute correlation
    correlations = sync_detector.compute_correlation(aligned_audio, aligned_video)
    
    # Detect sync issues
    sync_results = sync_detector.detect_sync_issues(correlations, threshold=0.4)
    
    # Create visualization video
    print("Creating visualization video...")
    vis_path = os.path.join(output_dir, "visualization.mp4")
    video_processor.create_visualization_video(
        video_data["frames"], 
        processed_frames, 
        vis_path, 
        fps=video_data["actual_fps"]
    )
    
    # Save results
    results = {
        "video_info": {
            "path": video_path,
            "fps": video_data["actual_fps"],
            "duration": video_data["duration"],
            "frames": video_data["processed_frames"]
        },
        "lip_movement_features": movement_features,
        "sync_analysis": sync_results,
        "correlations": correlations,
        "visualization_path": vis_path
    }
    
    return results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Lip Sync Analysis for AI-Generated Videos")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    # Process video
    start_time = time.time()
    results = process_video(args.video_path, args.output)
    end_time = time.time()
    
    # Print results
    print("\n===== Lip Sync Analysis Results =====")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Video duration: {results['video_info']['duration']:.2f} seconds")
    print(f"Number of frames: {results['video_info']['frames']}")
    
    print("\nSync Analysis:")
    print(f"Is synced: {results['sync_analysis']['is_synced']}")
    print(f"Confidence: {results['sync_analysis']['confidence']:.2f}")
    
    if results['sync_analysis']['issues']:
        print("\nDetected Issues:")
        for issue in results['sync_analysis']['issues']:
            print(f"- {issue}")
    
    print(f"\nVisualization saved to: {results['visualization_path']}")
    print("======================================")

if __name__ == "__main__":
    main()
