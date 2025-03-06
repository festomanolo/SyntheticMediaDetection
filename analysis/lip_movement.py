import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import librosa
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LipMovementAnalyzer:
    """Analyze lip movements from facial landmarks."""
    
    def __init__(self, smoothing_window=15, poly_order=3):
        """
        Initialize the lip movement analyzer.
        
        Args:
            smoothing_window: Window size for Savitzky-Golay filter.
            poly_order: Polynomial order for Savitzky-Golay filter.
        """
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        
    def extract_lip_time_series(self, video_frames_features: List[Dict]) -> Dict:
        """
        Extract time series data from lip features across video frames.
        
        Args:
            video_frames_features: List of lip feature dictionaries for each frame.
            
        Returns:
            Dictionary containing time series data for different lip measurements.
        """
        # Initialize empty lists for each measurement
        vertical_distances = []
        horizontal_distances = []
        areas = []
        aspect_ratios = []
        
        # Extract measurements from each frame
        for frame_features in video_frames_features:
            if not frame_features:  # Handle missing data (no face detected in some frames)
                vertical_distances.append(0)
                horizontal_distances.append(0) 
                areas.append(0)
                aspect_ratios.append(0)
            else:
                vertical_distances.append(frame_features['vertical_distance'])
                horizontal_distances.append(frame_features['horizontal_distance'])
                areas.append(frame_features['area'])
                aspect_ratios.append(frame_features['aspect_ratio'])
        
        # Apply smoothing filter if there are enough frames
        if len(vertical_distances) > self.smoothing_window:
            # Ensure window is odd
            window = self.smoothing_window if self.smoothing_window % 2 == 1 else self.smoothing_window + 1
            
            # Apply Savitzky-Golay filter for smoothing
            vertical_distances = savgol_filter(vertical_distances, window, self.poly_order)
            horizontal_distances = savgol_filter(horizontal_distances, window, self.poly_order)
            areas = savgol_filter(areas, window, self.poly_order)
            aspect_ratios = savgol_filter(aspect_ratios, window, self.poly_order)
        
        return {
            'vertical_distances': vertical_distances,
            'horizontal_distances': horizontal_distances,
            'areas': areas,
            'aspect_ratios': aspect_ratios
        }
    
    def calculate_derivatives(self, time_series: Dict) -> Dict:
        """
        Calculate derivatives of lip measurements to capture motion.
        
        Args:
            time_series: Dictionary containing time series data.
            
        Returns:
            Dictionary containing derivatives of measurements.
        """
        derivatives = {}
        
        for key, values in time_series.items():
            # Convert to numpy array for vectorized operations
            values_array = np.array(values)
            
            # Calculate first derivative (velocity)
            first_derivative = np.diff(values_array)
            # Pad to maintain the same length
            first_derivative = np.pad(first_derivative, (0, 1), 'edge')
            
            # Calculate second derivative (acceleration)
            second_derivative = np.diff(first_derivative)
            # Pad to maintain the same length
            second_derivative = np.pad(second_derivative, (0, 1), 'edge')
            
            derivatives[f'{key}_velocity'] = first_derivative
            derivatives[f'{key}_acceleration'] = second_derivative
        
        return derivatives
    
    def get_movement_features(self, time_series: Dict, derivatives: Dict) -> Dict:
        """
        Extract statistical features from lip movement time series.
        
        Args:
            time_series: Dictionary containing time series data.
            derivatives: Dictionary containing derivatives of measurements.
            
        Returns:
            Dictionary of statistical features for lip movements.
        """
        # Combine both dictionaries
        all_series = {**time_series, **derivatives}
        features = {}
        
        for key, values in all_series.items():
            values_array = np.array(values)
            
            # Basic statistics
            features[f'{key}_mean'] = np.mean(values_array)
            features[f'{key}_std'] = np.std(values_array)
            features[f'{key}_min'] = np.min(values_array)
            features[f'{key}_max'] = np.max(values_array)
            features[f'{key}_range'] = np.max(values_array) - np.min(values_array)
            
            # Percentiles
            features[f'{key}_25percentile'] = np.percentile(values_array, 25)
            features[f'{key}_median'] = np.median(values_array)
            features[f'{key}_75percentile'] = np.percentile(values_array, 75)
            
            # Zero crossings (for derivatives)
            if 'velocity' in key or 'acceleration' in key:
                zero_crossings = np.where(np.diff(np.signbit(values_array)))[0]
                features[f'{key}_zero_crossings'] = len(zero_crossings)
        
        return features
    
    def visualize_lip_movements(self, time_series: Dict, fps: float = 30.0) -> plt.Figure:
        """
        Visualize lip movements over time.
        
        Args:
            time_series: Dictionary containing time series data.
            fps: Frames per second of the video.
            
        Returns:
            Matplotlib figure with visualizations.
        """
        # Create time axis
        num_frames = len(time_series['vertical_distances'])
        time_axis = np.arange(num_frames) / fps
        
        # Create figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Lip Movement Analysis', fontsize=16)
        
        # Plot vertical distance
        axs[0, 0].plot(time_axis, time_series['vertical_distances'], 'b-')
        axs[0, 0].set_title('Mouth Vertical Opening')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Distance (pixels)')
        axs[0, 0].grid(True)
        
        # Plot horizontal distance
        axs[0, 1].plot(time_axis, time_series['horizontal_distances'], 'r-')
        axs[0, 1].set_title('Mouth Horizontal Width')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Distance (pixels)')
        axs[0, 1].grid(True)
        
        # Plot area
        axs[1, 0].plot(time_axis, time_series['areas'], 'g-')
        axs[1, 0].set_title('Mouth Area')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Area (pixels²)')
        axs[1, 0].grid(True)
        
        # Plot aspect ratio
        axs[1, 1].plot(time_axis, time_series['aspect_ratios'], 'm-')
        axs[1, 1].set_title('Mouth Aspect Ratio (Vertical/Horizontal)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Ratio')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        return fig
