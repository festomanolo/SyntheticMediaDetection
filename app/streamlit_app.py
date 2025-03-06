import streamlit as st
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.detection.face_detector import FaceDetector
from src.detection.landmark_extractor import FacialLandmarkExtractor
from src.preprocessing.video_processor import VideoProcessor
from src.analysis.lip_movement import LipMovementAnalyzer
from src.analysis.sync_detector import LipSyncDetector
from app.app import process_video

def main():
    st.set_page_config(page_title="Lip Sync Analysis Tool", layout="wide")
    
    st.title("AI-Generated Video Lip Sync Analysis")
    st.markdown("""
    This tool analyzes face mesh landmarks in videos to detect lip synchronization issues.
    Upload an AI-generated video to see if the lip movements match the audio.
    """)
    
    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        # Save uploaded file to temp directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Process video
        with st.spinner("Processing video... This may take a while."):
            output_dir = os.path.join(temp_dir, "output")
            results = process_video(temp_path, output_dir)
        
        # Display results
        st.success("Analysis complete!")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display original video
            st.subheader("Original Video")
            st.video(temp_path)
            
            # Display visualization video
            st.subheader("Visualization with Face Landmarks")
            st.video(results["visualization_path"])
        
        with col2:
            # Display sync analysis results
            st.subheader("Lip Sync Analysis Results")
            
            # Display sync status
            if results["sync_analysis"]["is_synced"]:
                st.success("✅ Video is properly lip-synced")
            else:
                st.error("❌ Video has lip sync issues")
            
            # Display confidence score
            confidence = results["sync_analysis"]["confidence"]
            st.metric("Sync Confidence", f"{confidence:.2f}", 
                     delta=f"{confidence - 0.5:.2f}", delta_color="normal")
            
            # Display detected issues
            if results["sync_analysis"]["issues"]:
                st.subheader("Detected Issues")
                for issue in results["sync_analysis"]["issues"]:
                    st.warning(issue)
        
        # Display lip movement graph
        st.subheader("Lip Movement Analysis")
        lip_movement_img = Image.open(os.path.join(output_dir, "lip_movements.png"))
        st.image(lip_movement_img, caption="Lip Movement Metrics Over Time")
        
        # Display correlation values
        st.subheader("Audio-Visual Correlation Data")
        
        correlation_data = []
        for key, value in results["correlations"].items():
            if "pearson" in key:
                metric_name = key.replace("_", " ").replace("pearson", "correlation")
                correlation_data.append({
                    "Metric": metric_name.title(),
                    "Value": round(value, 3)
                })
        
        st.table(correlation_data)
        
        # Technical details
        with st.expander("Technical Details"):
            st.json({
                "video_info": results["video_info"],
                "sync_analysis": results["sync_analysis"],
                "correlations": {k: round(v, 3) if isinstance(v, float) else v 
                                for k, v in results["correlations"].items()}
            })

if __name__ == "__main__":
    main()
