import cv2
import mediapipe as mp
import numpy as np
import math

class AdventureExplorationAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Initialize MediaPipe Holistic for pose detection
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Store previous landmarks for movement analysis
        self.prev_landmarks = None
        self.prev_eye_landmarks = None
        
    def process_image(self, image_path):
        """
        Process an image and extract head movement, micro-expressions, and eye gaze stability.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing head movement, micro-expressions, and eye gaze stability scores
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to RGB (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Face Mesh
        face_results = self.face_mesh.process(image_rgb)
        
        # Process the image with MediaPipe Holistic
        holistic_results = self.holistic.process(image_rgb)
        
        if not face_results.multi_face_landmarks:
            return {"error": "No face detected in the image"}
        
        # Get image dimensions
        h, w, _ = image.shape
        
        # Calculate metrics
        head_movement = self.calculate_head_movement(face_results.multi_face_landmarks[0], w, h)
        micro_expressions = self.detect_micro_expressions(face_results.multi_face_landmarks[0], image_rgb, w, h)
        eye_gaze_stability = self.calculate_eye_gaze_stability(face_results.multi_face_landmarks[0], w, h)
        
        # Update previous landmarks for next frame comparison
        self.prev_landmarks = face_results.multi_face_landmarks[0]
        
        return {
            "head_movement": head_movement,
            "micro_expressions": micro_expressions,
            "eye_gaze_stability": eye_gaze_stability
        }
    
    def calculate_head_movement(self, face_landmarks, width, height):
        """
        Calculate head movement based on the position of facial landmarks.
        If this is the first frame, return a default value.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Head movement score (0-1)
        """
        if self.prev_landmarks is None:
            return 0.5  # Default value for first frame
        
        # Use nose tip as reference point for head position
        nose_tip = face_landmarks.landmark[4]
        prev_nose_tip = self.prev_landmarks.landmark[4]
        
        # Calculate movement distance
        movement_distance = math.sqrt(
            ((nose_tip.x - prev_nose_tip.x) * width) ** 2 +
            ((nose_tip.y - prev_nose_tip.y) * height) ** 2
        )
        
        # Calculate head rotation
        # Use landmarks on opposite sides of face to estimate rotation
        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]
        prev_left_ear = self.prev_landmarks.landmark[234]
        prev_right_ear = self.prev_landmarks.landmark[454]
        
        current_ear_distance = math.sqrt(
            ((left_ear.x - right_ear.x) * width) ** 2 +
            ((left_ear.y - right_ear.y) * height) ** 2
        )
        
        prev_ear_distance = math.sqrt(
            ((prev_left_ear.x - prev_right_ear.x) * width) ** 2 +
            ((prev_left_ear.y - prev_right_ear.y) * height) ** 2
        )
        
        rotation_change = abs(current_ear_distance - prev_ear_distance)
        
        # Combine translation and rotation for overall movement score
        movement_score = (movement_distance / 20.0) + (rotation_change / 10.0)
        
        # Normalize to 0-1 range
        normalized_movement = min(max(movement_score, 0), 1)
        
        return normalized_movement
    
    def detect_micro_expressions(self, face_landmarks, image, width, height):
        """
        Detect micro-expressions by analyzing subtle changes in facial features.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image: RGB image
            width: Image width
            height: Image height
            
        Returns:
            Micro-expression score (0-1)
        """
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Define regions of interest for micro-expressions
        # Eyes, eyebrows, mouth corners
        regions = [
            # Left eyebrow
            [70, 63, 105, 66, 107],
            # Right eyebrow
            [300, 293, 334, 296, 336],
            # Left eye corner
            [33, 133, 160, 144, 145, 153],
            # Right eye corner
            [263, 362, 385, 373, 374, 380],
            # Mouth corners
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        ]
        
        # Extract points for each region
        region_points = []
        for region in regions:
            points = []
            for idx in region:
                pt = face_landmarks.landmark[idx]
                points.append((int(pt.x * width), int(pt.y * height)))
            region_points.append(points)
        
        # Create masks for each region
        masks = []
        for points in region_points:
            mask = np.zeros((height, width), dtype=np.uint8)
            points_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)
            masks.append(mask)
        
        # Apply Laplacian filter to detect edges (changes in expression)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        
        # Calculate micro-expression score based on edge intensity in regions
        micro_expr_scores = []
        for mask in masks:
            region = cv2.bitwise_and(laplacian, laplacian, mask=mask)
            total_pixels = np.count_nonzero(mask)
            if total_pixels > 0:
                edge_sum = np.sum(region) / 255.0
                avg_edge_intensity = edge_sum / total_pixels
                micro_expr_scores.append(avg_edge_intensity)
        
        if not micro_expr_scores:
            return 0.0
            
        # Average micro-expression score
        avg_micro_expr = sum(micro_expr_scores) / len(micro_expr_scores)
        
        # Normalize to 0-1 scale (typical values range from 2 to 15)
        normalized_micro_expr = min(max(avg_micro_expr / 15.0, 0), 1)
        
        return normalized_micro_expr
    
    def calculate_eye_gaze_stability(self, face_landmarks, width, height):
        """
        Calculate eye gaze stability based on the position and orientation of the eyes.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Eye gaze stability score (0-1, where 1 is very stable)
        """
        # Define eye landmarks
        left_eye_landmarks = [33, 133, 160, 144, 145, 153]  # Left eye
        right_eye_landmarks = [263, 362, 385, 373, 374, 380]  # Right eye
        
        # Calculate eye centers
        left_eye_points = []
        for idx in left_eye_landmarks:
            pt = face_landmarks.landmark[idx]
            left_eye_points.append((pt.x * width, pt.y * height))
        
        right_eye_points = []
        for idx in right_eye_landmarks:
            pt = face_landmarks.landmark[idx]
            right_eye_points.append((pt.x * width, pt.y * height))
        
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        
        # Calculate pupil positions (approximated by inner eye corners)
        left_pupil = (face_landmarks.landmark[133].x * width, face_landmarks.landmark[133].y * height)
        right_pupil = (face_landmarks.landmark[362].x * width, face_landmarks.landmark[362].y * height)
        
        # Calculate gaze direction vectors
        left_gaze_vector = (left_pupil[0] - left_eye_center[0], left_pupil[1] - left_eye_center[1])
        right_gaze_vector = (right_pupil[0] - right_eye_center[0], right_pupil[1] - right_eye_center[1])
        
        # Calculate gaze consistency between eyes
        gaze_consistency = 1.0 - min(
            math.sqrt((left_gaze_vector[0] - right_gaze_vector[0])**2 + 
                     (left_gaze_vector[1] - right_gaze_vector[1])**2) / 10.0,
            1.0
        )
        
        # If we have previous eye landmarks, calculate stability over time
        if self.prev_eye_landmarks is not None:
            prev_left_pupil = (self.prev_eye_landmarks[133].x * width, self.prev_eye_landmarks[133].y * height)
            prev_right_pupil = (self.prev_eye_landmarks[362].x * width, self.prev_eye_landmarks[362].y * height)
            
            # Calculate movement of pupils
            left_movement = math.sqrt((left_pupil[0] - prev_left_pupil[0])**2 + (left_pupil[1] - prev_left_pupil[1])**2)
            right_movement = math.sqrt((right_pupil[0] - prev_right_pupil[0])**2 + (right_pupil[1] - prev_right_pupil[1])**2)
            
            avg_movement = (left_movement + right_movement) / 2.0
            
            # Normalize movement (0-1, where 1 is stable)
            temporal_stability = max(0, 1.0 - (avg_movement / 20.0))
            
            # Combine consistency and temporal stability
            stability_score = 0.6 * gaze_consistency + 0.4 * temporal_stability
        else:
            # For first frame, use only consistency
            stability_score = gaze_consistency
        
        # Store current eye landmarks for next comparison
        self.prev_eye_landmarks = face_landmarks.landmark
        
        return stability_score

def calculate_section5(images, au_values):
    analyzer = AdventureExplorationAnalyzer()
    result = []
    
    # Process images
    for image in images:
        metrics = analyzer.process_image(image)
        result.append(metrics)
        
    # Aggregate results
    collected = {key: [] for key in result[0].keys()}
    for entry in result:
        for key, value in entry.items():
            collected[key].append(value)
    
    # Compute mean for each key
    avg_metrics = {key: np.mean(values) for key, values in collected.items()}
    
    # Calculate adventure exploration metrics
    openness_to_experience = avg_metrics['eye_gaze_stability']*0.3 + au_values['AU02']*0.3 + au_values['AU05']*0.4
    novelty_seeking = avg_metrics['head_movement']*0.4 + avg_metrics['micro_expressions']*0.3 + au_values['AU12']*0.3
    risk_tolerance = avg_metrics['head_movement']*0.3 + (1 - avg_metrics['eye_gaze_stability'])*0.3 + au_values['AU06']*0.4
    planning_preference = avg_metrics['eye_gaze_stability']*0.5 + (1 - avg_metrics['head_movement'])*0.3 + (1 - avg_metrics['micro_expressions'])*0.2
    
    return {
        'openness_to_experience': openness_to_experience*100,
        'novelty_seeking': novelty_seeking*100,
        'risk_tolerance': risk_tolerance*100,
        'planning_preference': planning_preference*100,
    }
