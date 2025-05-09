import cv2
import mediapipe as mp
import numpy as np
import math

class LearningGrowthAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Store previous landmarks for movement analysis
        self.prev_landmarks = None
    
    def process_image(self, image_path):
        """
        Process an image and extract eyebrow shape, head nod, and micro-expression frequency.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing eyebrow shape, head nod, and micro-expression frequency scores
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
        eyebrow_shape = self.calculate_eyebrow_shape(results.multi_face_landmarks[0], w, h)
        head_nod = self.calculate_head_nod(results.multi_face_landmarks[0], w, h)
        micro_expression_freq = self.calculate_micro_expression_frequency(results.multi_face_landmarks[0], image_rgb, w, h)
        
        # Update previous landmarks for next frame comparison
        self.prev_landmarks = results.multi_face_landmarks[0]
        
        return {
            "eyebrow_shape": eyebrow_shape,
            "head_nod": head_nod,
            "micro_expression_freq": micro_expression_freq
        }
    
    def calculate_eyebrow_shape(self, face_landmarks, width, height):
        """
        Calculate eyebrow shape based on the curvature and elevation of eyebrows.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Eyebrow shape score (0-1, where higher values indicate more raised/arched eyebrows)
        """
        # Define eyebrow landmarks
        # Left eyebrow (from inner to outer)
        left_eyebrow_landmarks = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        
        # Right eyebrow (from inner to outer)
        right_eyebrow_landmarks = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
        
        # Extract eyebrow points
        left_eyebrow_points = []
        for idx in left_eyebrow_landmarks:
            pt = face_landmarks.landmark[idx]
            left_eyebrow_points.append((pt.x * width, pt.y * height))
        
        right_eyebrow_points = []
        for idx in right_eyebrow_landmarks:
            pt = face_landmarks.landmark[idx]
            right_eyebrow_points.append((pt.x * width, pt.y * height))
        
        # Calculate eyebrow curvature
        # For left eyebrow
        left_inner = np.array(left_eyebrow_points[0])
        left_middle = np.array(left_eyebrow_points[len(left_eyebrow_points) // 2])
        left_outer = np.array(left_eyebrow_points[-1])
        
        # For right eyebrow
        right_inner = np.array(right_eyebrow_points[0])
        right_middle = np.array(right_eyebrow_points[len(right_eyebrow_points) // 2])
        right_outer = np.array(right_eyebrow_points[-1])
        
        # Calculate the height of the middle point relative to the line connecting inner and outer points
        # For left eyebrow
        left_line_vec = left_outer - left_inner
        left_middle_vec = left_middle - left_inner
        left_line_length = np.linalg.norm(left_line_vec)
        left_line_unitvec = left_line_vec / left_line_length
        
        left_projection_length = np.dot(left_middle_vec, left_line_unitvec)
        left_projection = left_inner + left_projection_length * left_line_unitvec
        left_height = np.linalg.norm(left_middle - left_projection)
        
        # For right eyebrow
        right_line_vec = right_outer - right_inner
        right_middle_vec = right_middle - right_inner
        right_line_length = np.linalg.norm(right_line_vec)
        right_line_unitvec = right_line_vec / right_line_length
        
        right_projection_length = np.dot(right_middle_vec, right_line_unitvec)
        right_projection = right_inner + right_projection_length * right_line_unitvec
        right_height = np.linalg.norm(right_middle - right_projection)
        
        # Calculate average height and normalize
        avg_height = (left_height + right_height) / 2
        
        # Calculate eyebrow elevation (vertical position)
        # Use forehead and eye landmarks as reference
        forehead_landmark = face_landmarks.landmark[10]  # Top of forehead
        left_eye_top = face_landmarks.landmark[159]  # Top of left eye
        right_eye_top = face_landmarks.landmark[386]  # Top of right eye
        
        left_eyebrow_bottom = face_landmarks.landmark[336]  # Bottom of left eyebrow
        right_eyebrow_bottom = face_landmarks.landmark[107]  # Bottom of right eyebrow
        
        # Calculate distances
        left_eye_to_eyebrow = abs(left_eyebrow_bottom.y - left_eye_top.y) * height
        right_eye_to_eyebrow = abs(right_eyebrow_bottom.y - right_eye_top.y) * height
        
        # Average distance
        avg_eye_to_eyebrow = (left_eye_to_eyebrow + right_eye_to_eyebrow) / 2
        
        # Normalize distances (typical values range from 5 to 20 pixels)
        normalized_height = min(max(avg_height / 10.0, 0), 1)
        normalized_elevation = min(max(avg_eye_to_eyebrow / 15.0, 0), 1)
        
        # Combine curvature and elevation for overall eyebrow shape score
        eyebrow_shape_score = 0.6 * normalized_height + 0.4 * normalized_elevation
        
        return eyebrow_shape_score
    
    def calculate_head_nod(self, face_landmarks, width, height):
        """
        Calculate head nod based on vertical movement of facial landmarks.
        If this is the first frame, return a default value.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Head nod score (0-1)
        """
        if self.prev_landmarks is None:
            return 0.5  # Default value for first frame
        
        # Use nose tip and chin as reference points for vertical head movement
        nose_tip = face_landmarks.landmark[4]
        chin = face_landmarks.landmark[152]
        
        prev_nose_tip = self.prev_landmarks.landmark[4]
        prev_chin = self.prev_landmarks.landmark[152]
        
        # Calculate vertical movement
        nose_vertical_movement = abs(nose_tip.y - prev_nose_tip.y) * height
        chin_vertical_movement = abs(chin.y - prev_chin.y) * height
        
        # Average vertical movement
        avg_vertical_movement = (nose_vertical_movement + chin_vertical_movement) / 2
        
        # Calculate horizontal movement (for comparison)
        nose_horizontal_movement = abs(nose_tip.x - prev_nose_tip.x) * width
        chin_horizontal_movement = abs(chin.x - prev_chin.x) * width
        
        avg_horizontal_movement = (nose_horizontal_movement + chin_horizontal_movement) / 2
        
        # Calculate ratio of vertical to total movement
        total_movement = avg_vertical_movement + avg_horizontal_movement
        if total_movement < 0.1:  # Very little movement
            vertical_ratio = 0.5
        else:
            vertical_ratio = avg_vertical_movement / total_movement
        
        # Normalize to 0-1 scale
        # Higher values indicate more nodding behavior
        head_nod_score = min(max(vertical_ratio, 0), 1)
        
        return head_nod_score
    
    def calculate_micro_expression_frequency(self, face_landmarks, image, width, height):
        """
        Calculate micro-expression frequency by analyzing subtle changes in facial features.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image: RGB image
            width: Image width
            height: Image height
            
        Returns:
            Micro-expression frequency score (0-1)
        """
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Define regions of interest for micro-expressions
        regions = [
            # Forehead
            [10, 8, 9, 151, 337, 299, 333, 332, 297, 338],
            # Left eye region
            [33, 7, 163, 144, 145, 153, 154, 155, 133],
            # Right eye region
            [362, 382, 381, 380, 374, 373, 390, 249, 263],
            # Mouth region
            [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
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

def calculate_section6(images, au_values):
    analyzer = LearningGrowthAnalyzer()
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
    
    # Calculate learning and growth metrics
    intellectual_curiosity = avg_metrics['eyebrow_shape']*0.4 + au_values['AU01']*0.3 + au_values['AU02']*0.3
    
    reflective_tendency = avg_metrics['head_nod']*0.5 + (1 - avg_metrics['micro_expression_freq'])*0.3 + au_values['AU04']*0.2
    
    structured_learning_preference = (1 - avg_metrics['eyebrow_shape'])*0.3 + (1 - avg_metrics['micro_expression_freq'])*0.4 + (1 - au_values['AU06'])*0.3
    
    return {
        'intellectual_curiosity': intellectual_curiosity*100,
        'reflective_tendency': reflective_tendency*100,
        'structured_learning_preference': structured_learning_preference*100,
    }
