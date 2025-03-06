import numpy as np
import cv2
import librosa
from typing import List, Dict, Tuple, Optional
from scipy.signal import correlate
from scipy.stats import pearsonr

class LipSyncDetector:
    """Detect synchronization between lip movements and audio."""
    
    def __init__(self, audio_sampling_rate=16000, window_size=0.1):
        """
        Initialize the lip sync detector.
        
        Args:
            audio_sampling_rate: Sampling rate for audio processing.
            window_size: Window size in seconds for correlation analysis.
        """
        self.audio_sampling_rate = audio_sampling_rate
        self.window_size = window_size
        
    def extract_audio_features(self, audio_file: str) -> Dict:
        """
        Extract relevant audio features for lip sync analysis.
        
        Args:
            audio_file: Path to the audio file.
            
        Returns:
            Dictionary containing audio features.
        """
        # Load audio file
        y, sr = librosa.load(audio_file, sr=self.audio_sampling_rate)
        
        # Extract audio envelope (amplitude envelope)
        frame_length = int(0.025 * sr)  # 25ms frame
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # MFCC (for speech features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Extract first 3 MFCCs (related to vocal tract configuration)
        mfcc_first_3 = mfcc[:3, :].T
        
        return {
            'rms': rms,
            'zcr': zcr,
            'mfcc': mfcc_first_3,
            'audio': y,
            'sr': sr,
            'hop_length': hop_length,
            'frame_length': frame_length
        }
    
    def align_audio_video_features(self, audio_features: Dict, lip_time_series: Dict, fps: float) -> Tuple[Dict, Dict]:
        """
        Align audio and video features to match time dimensions.
        
        Args:
            audio_features: Dictionary of extracted audio features.
            lip_time_series: Dictionary of lip movement time series.
            fps: Video frames per second.
            
        Returns:
            Tuple of aligned audio and video features.
        """
        # Calculate audio feature rate
        audio_hop_length = audio_features['hop_length']
        audio_sr = audio_features['sr']
        audio_feature_rate = audio_sr / audio_hop_length
        
        # Number of video frames
        num_video_frames = len(lip_time_series['vertical_distances'])
        
        # Calculate alignment factor
        alignment_factor = audio_feature_rate / fps
        
        # Resample audio features to match video frames
        aligned_audio_features = {}
        for key in ['rms', 'zcr']:
            # Use linear interpolation to align audio features with video frames
            audio_times = np.arange(len(audio_features[key])) / audio_feature_rate
            video_times = np.arange(num_video_frames) / fps
            
            aligned_audio_features[key] = np.interp(
                video_times, 
                audio_times[:len(audio_features[key])], 
                audio_features[key]
            )
        
        return aligned_audio_features, lip_time_series
    
    def compute_correlation(self, aligned_audio: Dict, aligned_video: Dict) -> Dict:
        """
        Compute correlation between audio and lip movement features.
        
        Args:
            aligned_audio: Dictionary of aligned audio features.
            aligned_video: Dictionary of aligned video features.
            
        Returns:
            Dictionary of correlation scores.
        """
        correlations = {}
        
        # Calculate correlations between audio RMS and lip movements
        for video_key in ['vertical_distances', 'areas']:
            for audio_key in ['rms', 'zcr']:
                # Pearson correlation
                corr, p_value = pearsonr(
                    aligned_audio[audio_key], 
                    aligned_video[video_key]
                )
                
                correlations[f'{audio_key}_{video_key}_pearson'] = corr
                correlations[f'{audio_key}_{video_key}_p_value'] = p_value
                
                # Cross-correlation (to find potential time offset)
                cross_corr = correlate(
                    aligned_audio[audio_key], 
                    aligned_video[video_key], 
                    mode='full'
                )
                
                # Find max correlation and its offset
                max_corr_idx = np.argmax(cross_corr)
                offset = max_corr_idx - (len(aligned_video[video_key]) - 1)
                
                correlations[f'{audio_key}_{video_key}_max_cross_corr'] = np.max(cross_corr)
                correlations[f'{audio_key}_{video_key}_offset_frames'] = offset
        
        return correlations
    
    def detect_sync_issues(self, correlations: Dict, threshold: float = 0.4) -> Dict:
        """
        Detect lip sync issues based on correlation scores.
        
        Args:
            correlations: Dictionary of correlation scores.
            threshold: Threshold for determining sync issues.
            
        Returns:
            Dictionary with sync analysis results.
        """
        results = {
            'is_synced': True,
            'confidence': 0.0,
            'offset_frames': 0,
            'issues': []
        }
        
        # Check Pearson correlations
        pearson_corrs = [v for k, v in correlations.items() if 'pearson' in k]
        avg_pearson = np.mean(pearson_corrs)
        
        # Check offsets
        offsets = [v for k, v in correlations.items() if 'offset_frames' in k]
        avg_offset = np.mean(offsets)
        
        # Determine if video is synced
        if avg_pearson < threshold:
            results['is_synced'] = False
            results['issues'].append(f"Low correlation between audio and lip movements ({avg_pearson:.3f})")
        
        if abs(avg_offset) > 3:  # More than 3 frames offset
            results['is_synced'] = False
            results['issues'].append(f"Audio and video appear to be offset by {avg_offset:.1f} frames")
        
        # Set confidence based on correlation values
        results['confidence'] = avg_pearson
        results['offset_frames'] = avg_offset
        
        return results
