import cv2
import numpy as np
import math
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,refine_landmarks=True, min_detection_confidence=0.5)

# Process the image with MediaPipe for gaze and iris analysis
def calculate_gaze_iris(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get image dimensions
            h, w, _ = image.shape
            
            # Get iris landmarks
            # Left eye iris landmarks (468 is the center)
            left_iris = face_landmarks.landmark[468]
            left_iris_x = int(left_iris.x * w)
            left_iris_y = int(left_iris.y * h)
            
            # Right eye iris landmarks (473 is the center)
            right_iris = face_landmarks.landmark[473]
            right_iris_x = int(right_iris.x * w)
            right_iris_y = int(right_iris.y * h)
            
            # Get iris contour landmarks
            # Left iris contour points (indices 474-477)
            left_iris_contour = []
            for idx in range(474, 478):
                pt = face_landmarks.landmark[idx]
                left_iris_contour.append((int(pt.x * w), int(pt.y * h)))
                
            # Right iris contour points (indices 469-472)
            right_iris_contour = []
            for idx in range(469, 473):
                pt = face_landmarks.landmark[idx]
                right_iris_contour.append((int(pt.x * w), int(pt.y * h)))
            
            # Get eye landmarks for reference
            # Left eye landmarks
            left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            left_eye_points = []
            for idx in left_eye_landmarks:
                pt = face_landmarks.landmark[idx]
                left_eye_points.append((int(pt.x * w), int(pt.y * h)))
                
            # Right eye landmarks
            right_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_points = []
            for idx in right_eye_landmarks:
                pt = face_landmarks.landmark[idx]
                right_eye_points.append((int(pt.x * w), int(pt.y * h)))
            
            # Calculate iris diameter
            # For left eye
            left_iris_top = face_landmarks.landmark[475]
            left_iris_bottom = face_landmarks.landmark[477]
            left_iris_left = face_landmarks.landmark[474]
            left_iris_right = face_landmarks.landmark[476]
            
            left_iris_width = math.sqrt(
                ((left_iris_right.x - left_iris_left.x) * w) ** 2 + 
                ((left_iris_right.y - left_iris_left.y) * h) ** 2
            )
            left_iris_height = math.sqrt(
                ((left_iris_top.x - left_iris_bottom.x) * w) ** 2 + 
                ((left_iris_top.y - left_iris_bottom.y) * h) ** 2
            )
            left_iris_diameter = (left_iris_width + left_iris_height) / 2
            
            # For right eye
            right_iris_top = face_landmarks.landmark[470]
            right_iris_bottom = face_landmarks.landmark[472]
            right_iris_left = face_landmarks.landmark[471]
            right_iris_right = face_landmarks.landmark[469]
            
            right_iris_width = math.sqrt(
                ((right_iris_right.x - right_iris_left.x) * w) ** 2 + 
                ((right_iris_right.y - right_iris_left.y) * h) ** 2
            )
            right_iris_height = math.sqrt(
                ((right_iris_top.x - right_iris_bottom.x) * w) ** 2 + 
                ((right_iris_top.y - right_iris_bottom.y) * h) ** 2
            )
            right_iris_diameter = (right_iris_width + right_iris_height) / 2
            
            # Calculate eye width
            # For left eye
            left_eye_left = face_landmarks.landmark[362]
            left_eye_right = face_landmarks.landmark[263]
            left_eye_width = math.sqrt(
                ((left_eye_right.x - left_eye_left.x) * w) ** 2 + 
                ((left_eye_right.y - left_eye_left.y) * h) ** 2
            )
            
            # For right eye
            right_eye_left = face_landmarks.landmark[133]
            right_eye_right = face_landmarks.landmark[33]
            right_eye_width = math.sqrt(
                ((right_eye_right.x - right_eye_left.x) * w) ** 2 + 
                ((right_eye_right.y - right_eye_left.y) * h) ** 2
            )
            
            # Calculate iris-to-eye ratio
            left_iris_to_eye_ratio = left_iris_diameter / left_eye_width
            right_iris_to_eye_ratio = right_iris_diameter / right_eye_width
            avg_iris_to_eye_ratio = (left_iris_to_eye_ratio + right_iris_to_eye_ratio) / 2
            
            # Estimate diopter based on iris-to-sclera ratio
            # This is a simplified model - actual clinical models would be more complex
            # Typical ratio ranges from ~0.35 (myopic) to ~0.45 (hyperopic)
            # Using a simple linear model for demonstration
            base_ratio = 0.40  # Assumed emmetropic (normal) ratio
            diopter_scale = 10.0  # Scale factor
            estimated_diopter = (avg_iris_to_eye_ratio - base_ratio) * diopter_scale
            
            # Calculate gaze vectors (from eye center to iris center)
            # Left eye center
            left_eye_center_x = int(sum(p[0] for p in left_eye_points) / len(left_eye_points))
            left_eye_center_y = int(sum(p[1] for p in left_eye_points) / len(left_eye_points))
            
            # Right eye center
            right_eye_center_x = int(sum(p[0] for p in right_eye_points) / len(right_eye_points))
            right_eye_center_y = int(sum(p[1] for p in right_eye_points) / len(right_eye_points))
            
            # Gaze vectors
            left_gaze_vector = (left_iris_x - left_eye_center_x, left_iris_y - left_eye_center_y)
            right_gaze_vector = (right_iris_x - right_eye_center_x, right_iris_y - right_eye_center_y)
            
            # Normalize gaze vectors
            left_gaze_magnitude = math.sqrt(left_gaze_vector[0]**2 + left_gaze_vector[1]**2)
            right_gaze_magnitude = math.sqrt(right_gaze_vector[0]**2 + right_gaze_vector[1]**2)
            
            if left_gaze_magnitude > 0:
                left_gaze_normalized = (left_gaze_vector[0]/left_gaze_magnitude, left_gaze_vector[1]/left_gaze_magnitude)
            else:
                left_gaze_normalized = (0, 0)
                
            if right_gaze_magnitude > 0:
                right_gaze_normalized = (right_gaze_vector[0]/right_gaze_magnitude, right_gaze_vector[1]/right_gaze_magnitude)
            else:
                right_gaze_normalized = (0, 0)
            
            # Average the gaze vectors from both eyes
            avg_gaze_vector = ((left_gaze_normalized[0] + right_gaze_normalized[0])/2, 
                            (left_gaze_normalized[1] + right_gaze_normalized[1])/2)
            
            # Determine gaze direction in words
            gaze_x, gaze_y = avg_gaze_vector
            gaze_direction = ""
            
            # Horizontal direction
            if gaze_x > 0.2:
                gaze_direction += "right"
            elif gaze_x < -0.2:
                gaze_direction += "left"
            else:
                gaze_direction += "center"
                
            # Vertical direction
            if gaze_y > 0.2:
                gaze_direction += " down"
            elif gaze_y < -0.2:
                gaze_direction += " up"
            
            if gaze_direction =='center':
                gaze_direction = 1
            else:
                gaze_direction = 0

    return {"gaze_direction": gaze_direction,"diopter": estimated_diopter}

def calculate_section1(images,au_values):
    result = []
    for image in images:
        result.append(calculate_gaze_iris(cv2.imread(image)))
        
    collected = {key: [] for key in result[0].keys()}
    # Collect values for each key
    for entry in result:
        for key, value in entry.items():
            collected[key].append(value)

    # Compute mean for each key
    mp_result = {key: np.mean(values) for key, values in collected.items()}
    iris_score = min(max((mp_result['diopter'] + 1)/2, 0), 1)
    trust = (au_values['AU06'] + au_values['AU12'])*0.2 + mp_result['gaze_direction']*0.4 + iris_score*0.2
    openness = au_values['AU12']*0.5 + au_values['AU06']*0.3 + mp_result['gaze_direction']*0.2
    Empathy = au_values['AU01']*0.5 + au_values['AU06']*0.3 + mp_result['gaze_direction']*0.2
    ConflictAvoid = (1-au_values['AU12'])*0.5 + (1-au_values['AU06'])*0.3 + mp_result['gaze_direction']*0.2
    # EmpathyScore = (Empathy + trust * 0.5)/1.5
    # RelationshipScore = trust*0.4 + openness*0.3 + Empathy*0.3 + (1-ConflictAvoid)*0.1
    return {"trust": trust*100, "openness": openness*100, "Empathy":Empathy*100, "ConflictAvoid":ConflictAvoid*100}