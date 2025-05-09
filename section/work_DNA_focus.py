import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,min_detection_confidence=0.5)

def get_head_pose(image):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None, None
    
    # Get facial landmarks
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Get specific landmarks for jaw and head pose estimation
    # These are indices in the 468-point face mesh
    chin_bottom = landmarks[152]  # Bottom of chin
    chin_left = landmarks[172]    # Left side of jaw
    chin_right = landmarks[397]   # Right side of jaw
    nose_tip = landmarks[4]      # Tip of nose
    forehead = landmarks[10]      # Forehead point
    
    # Calculate jaw angle (horizontal prominence)
    # Using the angle between left-jaw, chin-bottom, right-jaw points
    jaw_vector_left = np.array([chin_left.x - chin_bottom.x, chin_left.y - chin_bottom.y])
    jaw_vector_right = np.array([chin_right.x - chin_bottom.x, chin_right.y - chin_bottom.y])
    
    # Calculate angle between vectors (in degrees)
    unit_vector_left = jaw_vector_left / np.linalg.norm(jaw_vector_left)
    unit_vector_right = jaw_vector_right / np.linalg.norm(jaw_vector_right)
    jaw_angle = np.degrees(np.arccos(np.clip(np.dot(unit_vector_left, unit_vector_right), -1.0, 1.0)))
    
    # Calculate head pitch (forward/backward tilt)
    # Using vertical relationship between nose tip and forehead
    vertical_distance = nose_tip.y - forehead.y
    
    # Convert to approximate degrees (this is a simplified approach)
    # You might need to calibrate this based on your specific needs
    head_pitch = vertical_distance * 100  # Scaling factor - adjust based on your requirements
    
    return jaw_angle, head_pitch
def calculate_section2(images,au_values):
    result=[]
    for image in images:
        jaw_angle, head_pitch = get_head_pose(cv2.imread(image))
        result.append({"jaw":jaw_angle,"head":head_pitch})
    collected = {key: [] for key in result[0].keys()}
    for entry in result:
        for key, value in entry.items():
            collected[key].append(value)
    # Compute mean for each key
    mp_result = {key: np.mean(values) for key, values in collected.items()}
    persistant = (1 - abs(mp_result['jaw']-100)/40)*0.3 + (1 - abs(mp_result['head'])/30)*0.4 + (au_values['AU01']+au_values['AU06'])/2*0.3
    out_focus = (1 - abs(mp_result['head'])/30)*0.4 + (au_values['AU01'] + au_values['AU02'] + au_values['AU06'])*0.2
    out_structure = mp_result['jaw']/140 *0.6 + (1-abs(mp_result['head'])/30)*0.4
    out_risk = (1 if mp_result['head'] > 5 else 0) *0.4 + (1 if mp_result['jaw']>120 else 0)*0.3 + au_values['AU04']*0.3
    # Work_DNA = persistant*0.4 + out_structure*0.3 + out_risk*0.3
    # Focus_Work = out_focus*0.5 + au_values['AU01']*0.2 + (1-au_values['AU04'])*0.1 + au_values['AU06']*0.2
    return {"Persistant":f"{persistant*100:.1f}","Focus":f"{out_focus*100:.1f}","Structure":f"{out_structure*100:.1f}","Risk":f"{out_risk*100:.1f}"}