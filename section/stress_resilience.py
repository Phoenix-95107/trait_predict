import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, Tuple, List, Optional, Union

class StressResilienceAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def process_image(self, image_path):
        """
        Process an image and return forehead furrows and lip compression metrics.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing forehead furrows and lip compression scores
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
        forehead_furrows = self.calculate_forehead_furrows(results.multi_face_landmarks[0], image_rgb, w, h)
        lip_compression = self.calculate_lip_compression(results.multi_face_landmarks[0], w, h)
        
        return {
            "forehead_furrows": forehead_furrows,
            "lip_compression": lip_compression
        }
    
    def calculate_forehead_furrows(self, face_landmarks, image, width: int, height: int) -> float:
        """
        Calculate forehead furrows based on the texture and landmarks of the forehead region.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image: RGB image
            width: Image width
            height: Image height
            
        Returns:
            Forehead furrows score (0-1, where 1 is high furrow presence)
        """
        # Define forehead region landmarks
        # Top of forehead
        forehead_top_landmarks = [10, 8, 9, 151, 337, 299, 333, 332]
        # Bottom of forehead (near eyebrows)
        forehead_bottom_landmarks = [66, 105, 63, 70, 156, 334, 293, 300]
        
        # Extract forehead region coordinates
        forehead_top_points = []
        for idx in forehead_top_landmarks:
            pt = face_landmarks.landmark[idx]
            forehead_top_points.append((int(pt.x * width), int(pt.y * height)))
        
        forehead_bottom_points = []
        for idx in forehead_bottom_landmarks:
            pt = face_landmarks.landmark[idx]
            forehead_bottom_points.append((int(pt.x * width), int(pt.y * height)))
        
        # Create a mask for the forehead region
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Combine points to form a polygon
        forehead_points = forehead_top_points + forehead_bottom_points[::-1]
        forehead_points_array = np.array(forehead_points, dtype=np.int32)
        
        # Fill the polygon
        cv2.fillPoly(mask, [forehead_points_array], 255)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply mask to get forehead region
        forehead_region = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Apply Sobel filter to detect horizontal edges (furrows)
        sobel_x = cv2.Sobel(forehead_region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        
        # Calculate the average edge intensity in the forehead region
        total_pixels = np.count_nonzero(mask)
        if total_pixels == 0:
            return 0.0
        
        edge_sum = np.sum(sobel_x * (mask > 0) / 255.0)
        avg_edge_intensity = edge_sum / total_pixels
        
        # Normalize to a 0-1 scale (typical values range from 5 to 30)
        normalized_furrows = min(max(avg_edge_intensity / 30.0, 0), 1)
        
        return normalized_furrows
    
    def calculate_lip_compression(self, face_landmarks, width: int, height: int) -> float:
        """
        Calculate lip compression based on the distance between upper and lower lips.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Lip compression score (0-1, where 1 is high compression)
        """
        # Upper lip landmarks (middle)
        upper_lip_top = face_landmarks.landmark[13]  # Upper lip top
        upper_lip_bottom = face_landmarks.landmark[14]  # Upper lip bottom
        
        # Lower lip landmarks (middle)
        lower_lip_top = face_landmarks.landmark[17]  # Lower lip top
        lower_lip_bottom = face_landmarks.landmark[15]  # Lower lip bottom
        
        # Calculate vertical distances
        lip_gap = math.sqrt(
            ((upper_lip_bottom.x - lower_lip_top.x) * width) ** 2 + 
            ((upper_lip_bottom.y - lower_lip_top.y) * height) ** 2
        )
        
        lip_height = math.sqrt(
            ((upper_lip_top.x - lower_lip_bottom.x) * width) ** 2 + 
            ((upper_lip_top.y - lower_lip_bottom.y) * height) ** 2
        )
        
        # Calculate lip compression ratio (gap/height)
        # Lower ratio means more compression
        if lip_height == 0:
            return 0.0
            
        lip_ratio = lip_gap / lip_height
        
        # Normalize to a 0-1 scale (where 1 is high compression)
        # Typical values range from 0.1 (compressed) to 0.5 (relaxed)
        normalized_compression = max(0, 1 - (lip_ratio / 0.5))
        
        return normalized_compression

def calculate_section4(images, au_values):
    analyzer = StressResilienceAnalyzer()
    result = []
    stress_indicator =[]
    emotional_regulation = []
    resilience_score = []
    # Process images
    for i,image in enumerate(images):
        metrics = analyzer.process_image(image)
        # Calculate stress resilience metrics
       
        stress=au_values['AU04'][i]*0.4 + au_values['AU24'][i]*0.3 + au_values['AU04'][i]*0.3
        stress_indicator.append(stress)
       
        emotional=(1 - metrics['forehead_furrows'])*0.5 + (1 - au_values['AU04'][i])*0.3 + (1 - au_values['AU15'][i])*0.2
        emotional_regulation.append(emotional)
        
        resilience=(1-stress)*0.4 + emotional*0.4 + au_values['AU12'][i]*0.2
        resilience_score.append(resilience)
        
    select = np.argmax([stress_indicator,emotional_regulation,resilience_score], axis=1)
    return {
        'stress_indicator': np.mean(stress_indicator)*100,
        'emotional_regulation':np.mean(emotional_regulation)*100,
        'resilience_score': np.mean(resilience_score)*100,
        'select':select
    }