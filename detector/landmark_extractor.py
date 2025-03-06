import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional

class FacialLandmarkExtractor:
    """Extract facial landmarks using MediaPipe Face Mesh."""
    
    def __init__(self, static_image_mode=False, max_num_faces=1, 
                min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the facial landmark extractor.
        
        Args:
            static_image_mode: Whether to treat the input images as a batch of static images.
            max_num_faces: Maximum number of faces to detect.
            min_detection_confidence: Minimum confidence value for face detection.
            min_tracking_confidence: Minimum confidence value for landmark tracking.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define lip indices from MediaPipe Face Mesh
        self.lip_indices = list(range(61, 69)) + list(range(291, 299))  # Upper and lower outer lip
        self.lip_indices += list(range(0, 11)) + list(range(267, 278))  # Upper and lower inner lip
        
    def extract_landmarks(self, image: np.ndarray) -> List[Dict]:
        """
        Extract facial landmarks from the image.
        
        Args:
            image: Input image in RGB format.
            
        Returns:
            List of dictionaries containing facial landmarks.
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and extract landmarks
        results = self.face_mesh.process(image_rgb)
        
        all_landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    # Convert landmark to pixel coordinates
                    h, w, _ = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y, landmark.z))
                
                # Extract lip landmarks
                lip_landmarks = [landmarks[i] for i in self.lip_indices]
                
                all_landmarks.append({
                    'all_landmarks': landmarks,
                    'lip_landmarks': lip_landmarks
                })
                
        return all_landmarks
    
    def draw_landmarks(self, image: np.ndarray, all_landmarks: List[Dict], 
                       draw_lips_only=False) -> np.ndarray:
        """
        Draw facial landmarks on the image.
        
        Args:
            image: Input image.
            all_landmarks: List of landmark dictionaries.
            draw_lips_only: Whether to draw only lip landmarks.
            
        Returns:
            Image with drawn landmarks.
        """
        image_copy = image.copy()
        
        if not all_landmarks:
            return image_copy
            
        for face_data in all_landmarks:
            if draw_lips_only:
                # Draw only lip landmarks
                for i, (x, y, _) in enumerate(face_data['lip_landmarks']):
                    cv2.circle(image_copy, (x, y), 2, (0, 0, 255), -1)
            else:
                # Draw all landmarks using MediaPipe's drawing utilities
                face_landmarks_proto = self.mp_face_mesh.FaceMesh().process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).multi_face_landmarks[0]
                
                self.mp_drawing.draw_landmarks(
                    image=image_copy,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
                self.mp_drawing.draw_landmarks(
                    image=image_copy,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                
        return image_copy
    
    def get_lip_features(self, lip_landmarks: List[Tuple[int, int, float]]) -> Dict:
        """
        Extract features from lip landmarks for lip sync analysis.
        
        Args:
            lip_landmarks: List of lip landmark coordinates.
            
        Returns:
            Dictionary of lip features.
        """
        if not lip_landmarks:
            return {}
            
        # Calculate mouth opening (vertical distance)
        top_lip_center = lip_landmarks[3]  # Upper lip center
        bottom_lip_center = lip_landmarks[9]  # Lower lip center
        mouth_vertical = ((top_lip_center[0] - bottom_lip_center[0])**2 + 
                          (top_lip_center[1] - bottom_lip_center[1])**2)**0.5
        
        # Calculate mouth width (horizontal distance)
        left_corner = lip_landmarks[0]  # Left corner
        right_corner = lip_landmarks[6]  # Right corner
        mouth_horizontal = ((left_corner[0] - right_corner[0])**2 + 
                           (left_corner[1] - right_corner[1])**2)**0.5
        
        # Calculate mouth area (approximation using width and height)
        mouth_area = mouth_vertical * mouth_horizontal
        
        # Calculate aspect ratio
        aspect_ratio = mouth_vertical / mouth_horizontal if mouth_horizontal != 0 else 0
        
        return {
            'vertical_distance': mouth_vertical,
            'horizontal_distance': mouth_horizontal,
            'area': mouth_area,
            'aspect_ratio': aspect_ratio,
            'landmarks': lip_landmarks
        }
