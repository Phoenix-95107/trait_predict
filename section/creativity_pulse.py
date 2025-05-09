import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, Tuple, List, Optional, Union

class FacialAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def process_image(self, image_path: str) -> Dict[str, float]:
        """
        Process an image and return eye openness and facial symmetry metrics.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing eye openness and facial symmetry scores
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to RGB (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return {"error": "No face detected in the image"}
        
        # Get image dimensions
        h, w, _ = image.shape
        
        # Calculate metrics
        eye_openness = self.calculate_eye_openness(results.multi_face_landmarks[0], w, h)
        facial_symmetry = self.calculate_facial_symmetry(results.multi_face_landmarks[0], w, h)
        
        return {
            "eye_openness": eye_openness,
            "facial_symmetry": facial_symmetry
        }
    
    def calculate_eye_openness(self, face_landmarks, width: int, height: int) -> float:
        """
        Calculate eye openness based on the ratio of eye height to eye width.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Eye openness score (0-1)
        """
        # Left eye landmarks (vertical)
        left_eye_top = face_landmarks.landmark[159]  # Upper eyelid
        left_eye_bottom = face_landmarks.landmark[145]  # Lower eyelid
        
        # Left eye landmarks (horizontal)
        left_eye_left = face_landmarks.landmark[33]
        left_eye_right = face_landmarks.landmark[133]
        
        # Right eye landmarks (vertical)
        right_eye_top = face_landmarks.landmark[386]  # Upper eyelid
        right_eye_bottom = face_landmarks.landmark[374]  # Lower eyelid
        
        # Right eye landmarks (horizontal)
        right_eye_left = face_landmarks.landmark[362]
        right_eye_right = face_landmarks.landmark[263]
        
        # Calculate eye height
        left_eye_height = math.sqrt(
            ((left_eye_top.x - left_eye_bottom.x) * width) ** 2 + 
            ((left_eye_top.y - left_eye_bottom.y) * height) ** 2
        )
        
        right_eye_height = math.sqrt(
            ((right_eye_top.x - right_eye_bottom.x) * width) ** 2 + 
            ((right_eye_top.y - right_eye_bottom.y) * height) ** 2
        )
        
        # Calculate eye width
        left_eye_width = math.sqrt(
            ((left_eye_left.x - left_eye_right.x) * width) ** 2 + 
            ((left_eye_left.y - left_eye_right.y) * height) ** 2
        )
        
        right_eye_width = math.sqrt(
            ((right_eye_left.x - right_eye_right.x) * width) ** 2 + 
            ((right_eye_left.y - right_eye_right.y) * height) ** 2
        )
        
        # Calculate aspect ratio (height/width)
        left_eye_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
        right_eye_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
        
        # Average eye openness
        avg_eye_openness = (left_eye_ratio + right_eye_ratio) / 2
        
        # Normalize to a 0-1 scale (typical values range from 0.2 to 0.5)
        normalized_openness = min(max((avg_eye_openness - 0.2) / 0.3, 0), 1)
        
        return normalized_openness
    
    def calculate_facial_symmetry(self, face_landmarks, width: int, height: int) -> float:
        """
        Calculate facial symmetry by comparing landmark positions on both sides of the face.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Facial symmetry score (0-1, where 1 is perfectly symmetric)
        """
        # Define pairs of symmetric landmarks
        # Format: (left_point_idx, right_point_idx)
        symmetric_pairs = [
            # Eyebrows
            (70, 300),   # Outer eyebrows
            (63, 293),   # Middle eyebrows
            (105, 334),  # Inner eyebrows
            
            # Eyes
            (33, 263),   # Outer eye corners
            (133, 362),  # Inner eye corners
            
            # Cheeks
            (206, 426),  # Cheekbones
            
            # Mouth
            (61, 291),   # Mouth corners
            (0, 17),     # Upper lip
            (14, 14),    # Lower lip
            
            # Jaw
            (172, 397),  # Jaw angles
        ]
        
        # Calculate asymmetry score
        asymmetry_scores = []
        
        # Vertical line through the middle of the face (nose bridge)
        nose_top = face_landmarks.landmark[168]
        nose_bottom = face_landmarks.landmark[2]
        midline_x = (nose_top.x + nose_bottom.x) / 2
        
        for left_idx, right_idx in symmetric_pairs:
            left_point = face_landmarks.landmark[left_idx]
            right_point = face_landmarks.landmark[right_idx]
            
            # Distance from midline
            left_dist = abs(left_point.x - midline_x)
            right_dist = abs(right_point.x - midline_x)
            
            # Vertical position difference
            y_diff = abs(left_point.y - right_point.y)
            
            # Calculate asymmetry for this pair
            dist_diff = abs(left_dist - right_dist)
            pair_asymmetry = dist_diff + y_diff * 0.5  # Weight vertical differences less
            
            asymmetry_scores.append(pair_asymmetry)
        
        # Average asymmetry score
        avg_asymmetry = sum(asymmetry_scores) / len(asymmetry_scores)
        
        # Convert to symmetry score (0-1, where 1 is perfectly symmetric)
        # Typical asymmetry values range from 0.01 to 0.1
        symmetry_score = max(0, 1 - (avg_asymmetry * 10))
        
        return symmetry_score
    
# Example usage
def calculate_section3(images, au_values):
    analyzer = FacialAnalyzer()
    result = []
    # Process a single image
    for image in images: # Replace with your image path
        metrics = analyzer.process_image(image)
        result.append(metrics)

    collected = {key: [] for key in result[0].keys()}
    for entry in result:
        for key, value in entry.items():
            collected[key].append(value)
    # Compute mean for each key
    mp_result = {key: np.mean(values) for key, values in collected.items()}
    micro_expr_score = 1.0 if any([au > 0.9 for au in [au_values['AU01'], au_values['AU02'], au_values['AU04'], au_values['AU12']]]) else 0.0
    ideation = (au_values['AU12'] + au_values['AU06'])*0.2 + mp_result['eye_openness']*0.3 + micro_expr_score*0.3
    openness = (au_values['AU02'] + au_values['AU05'])*0.3 + mp_result['facial_symmetry']*0.4
    originalty = (1 - mp_result['facial_symmetry'])*0.4 + au_values['AU12']*0.3 + au_values['AU01']*0.3
    attention = mp_result['eye_openness']*0.3 + (1 - mp_result['facial_symmetry'])*0.3 + au_values['AU04']*0.4
    return{
        'ideation': ideation*100,
        'openness': openness*100,
        'originalty': originalty*100,
        'attention':attention*100,
        
    }