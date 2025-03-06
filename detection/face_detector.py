import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional

class FaceDetector:
    """Face detection module using MediaPipe."""
    
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize the face detector.
        
        Args:
            min_detection_confidence: Minimum confidence value for face detection.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the given image.
        
        Args:
            image: Input image in RGB format.
            
        Returns:
            List of detected faces with their bounding boxes and landmarks.
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = self.face_detection.process(image_rgb)
        
        faces = []
        # Check if any faces are detected
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )
                
                face_data = {
                    'bbox': bbox,
                    'score': detection.score[0],
                    'keypoints': {}
                }
                
                # Extract keypoints
                for i, keypoint in enumerate(detection.location_data.relative_keypoints):
                    face_data['keypoints'][i] = (int(keypoint.x * iw), int(keypoint.y * ih))
                
                faces.append(face_data)
                
        return faces
    
    def draw_detections(self, image: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw face detections on the image.
        
        Args:
            image: Input image.
            faces: List of detected faces.
            
        Returns:
            Image with drawn face detections.
        """
        image_copy = image.copy()
        for face in faces:
            x, y, w, h = face['bbox']
            # Draw bounding box
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw keypoints
            for keypoint_id, (kx, ky) in face['keypoints'].items():
                cv2.circle(image_copy, (kx, ky), 5, (255, 0, 0), -1)
                
        return image_copy
    
    def crop_face(self, image: np.ndarray, face: Dict, scale=1.5) -> np.ndarray:
        """
        Crop the face from the image with some margin.
        
        Args:
            image: Input image.
            face: Detected face.
            scale: Scaling factor for the bounding box.
            
        Returns:
            Cropped face image.
        """
        x, y, w, h = face['bbox']
        
        # Calculate center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # New top-left corner
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        
        # Ensure we don't go out of bounds
        new_w = min(new_w, image.shape[1] - new_x)
        new_h = min(new_h, image.shape[0] - new_y)
        
        # Crop the image
        face_crop = image[new_y:new_y + new_h, new_x:new_x + new_w]
        
        return face_crop
