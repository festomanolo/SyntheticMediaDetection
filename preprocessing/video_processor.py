import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import tempfile
import subprocess
from tqdm import tqdm

class VideoProcessor:
    """Process video files for face mesh analysis."""
    
    def __init__(self, face_detector=None, landmark_extractor=None):
        """
        Initialize the video processor.
        
        Args:
            face_detector: FaceDetector instance.
            landmark_extractor: FacialLandmarkExtractor instance.
        """
        self.face_detector = face_detector
        self.landmark_extractor = landmark_extractor
        
    def extract_frames(self, video_path: str, output_dir: str = None, 
                       fps: Optional[int] = None, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.
            fps: Frame rate to extract frames at (None for original fps).
            max_frames: Maximum number of frames to extract.
            
        Returns:
            List of extracted frames as numpy arrays.
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine sampling rate
        sampling_rate = 1
        if fps is not None and fps < original_fps:
            sampling_rate = int(original_fps / fps)
        
        # Determine number of frames to extract
        if max_frames is not None:
            num_frames = min(total_frames, max_frames)
        else:
            num_frames = total_frames
        
        # Extract frames
        frames = []
        frame_count = 0
        
        with tqdm(total=num_frames, desc="Extracting frames") as pbar:
            while cap.isOpened() and frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on specified fps
                if frame_count % sampling_rate == 0:
                    frames.append(frame)
                    
                    # Save frame if output directory is specified
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
                        cv2.imwrite(frame_path, frame)
                    
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        return frames
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file.
            output_path: Path to save the extracted audio.
            
        Returns:
            Path to the extracted audio file.
        """
        if output_path is None:
            # Create temporary file if output path is not specified
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-y",  # Overwrite output file if it exists
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error extracting audio: {e.stderr.decode()}")
    
    def process_video(self, video_path: str, output_dir: str = None, 
                      extract_audio: bool = True, fps: Optional[int] = None) -> Dict:
        """
        Process a video file for face mesh analysis.
        
        Args:
            video_path: Path to the video file.
            output_dir: Directory to save processed data.
            extract_audio: Whether to extract audio from the video.
            fps: Frame rate to process at (None for original fps).
            
        Returns:
            Dictionary containing processed video data.
        """
        # Create output directory if not exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames
        frames = self.extract_frames(video_path, 
                                    output_dir=os.path.join(output_dir, "frames") if output_dir else None,
                                    fps=fps)
        
        # Extract audio if required
        audio_path = None
        if extract_audio:
            audio_path = self.extract_audio(video_path, 
                                           output_path=os.path.join(output_dir, "audio.wav") if output_dir else None)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Calculate actual fps if resampling was applied
        actual_fps = original_fps
        if fps is not None:
            actual_fps = fps
        
        return {
            "frames": frames,
            "audio_path": audio_path,
            "original_fps": original_fps,
            "actual_fps": actual_fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "processed_frames": len(frames),
            "duration": total_frames / original_fps
        }
    
    def process_frames_for_landmarks(self, frames: List[np.ndarray], 
                                     output_dir: str = None) -> List[Dict]:
        """
        Process frames to extract facial landmarks.
        
        Args:
            frames: List of video frames.
            output_dir: Directory to save processed frames.
            
        Returns:
            List of dictionaries containing face and landmark data for each frame.
        """
        if self.face_detector is None or self.landmark_extractor is None:
            raise ValueError("Face detector and landmark extractor must be initialized")
        
        processed_data = []
        
        with tqdm(total=len(frames), desc="Processing frames") as pbar:
            for i, frame in enumerate(frames):
                frame_data = {}
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                if faces:
                    frame_data["faces"] = faces
                    
                    # Get the largest face (assuming it's the main subject)
                    largest_face = max(faces, key=lambda x: x["bbox"][2] * x["bbox"][3])
                    
                    # Extract landmarks
                    landmarks = self.landmark_extractor.extract_landmarks(frame)
                    if landmarks:
                        frame_data["landmarks"] = landmarks
                        
                        # Extract lip features
                        lip_features = self.landmark_extractor.get_lip_features(
                            landmarks[0]["lip_landmarks"]
                        )
                        frame_data["lip_features"] = lip_features
                    
                    # Save visualization if output directory is specified
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Draw faces
                        face_vis = self.face_detector.draw_detections(frame, faces)
                        
                        # Draw landmarks
                        if "landmarks" in frame_data:
                            landmark_vis = self.landmark_extractor.draw_landmarks(
                                face_vis, frame_data["landmarks"], draw_lips_only=True
                            )
                            
                            # Save visualization
                            vis_path = os.path.join(output_dir, f"vis_{i:05d}.jpg")
                            cv2.imwrite(vis_path, landmark_vis)
                
                processed_data.append(frame_data)
                pbar.update(1)
        
        return processed_data
    
    def create_visualization_video(self, frames: List[np.ndarray], 
                                  processed_data: List[Dict],
                                  output_path: str, fps: float = 30.0) -> str:
        """
        Create a visualization video with face detection and landmarks.
        
        Args:
            frames: List of original video frames.
            processed_data: List of processed frame data.
            output_path: Path to save the visualization video.
            fps: Frame rate of the output video.
            
        Returns:
            Path to the created visualization video.
        """
        if not frames or not processed_data:
            raise ValueError("Frames and processed data must not be empty")
        
        # Get video dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        with tqdm(total=len(frames), desc="Creating visualization") as pbar:
            for i, (frame, data) in enumerate(zip(frames, processed_data)):
                vis_frame = frame.copy()
                
                # Draw faces if detected
                if "faces" in data:
                    vis_frame = self.face_detector.draw_detections(vis_frame, data["faces"])
                
                # Draw landmarks if detected
                if "landmarks" in data:
                    vis_frame = self.landmark_extractor.draw_landmarks(
                        vis_frame, data["landmarks"], draw_lips_only=True
                    )
                
                # Add frame number
                cv2.putText(vis_frame, f"Frame: {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame to video
                out.write(vis_frame)
                pbar.update(1)
        
        out.release()
        return output_path
