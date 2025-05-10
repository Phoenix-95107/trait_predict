import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,min_detection_confidence=0.5)

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
    
    def process_image(self, image_path: str):
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
        jaw_angle, head_pitch = self.get_head_pose(results.multi_face_landmarks[0].landmark)
        
        return {
            "jaw_angle": jaw_angle,
            "head_pitch": head_pitch
        }
    
    def get_head_pose(self, landmarks):  
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
    analyzer = FacialAnalyzer()
    persistant_list = []
    out_focus_list = []
    out_structure_list = []
    out_risk_list = []
    for i, image in enumerate(images):
        result = analyzer.process_image(image)
        persistant = (1 - abs(result['jaw_angle']-100)/40)*0.3 + (1 - abs(result['head_pitch'])/30)*0.4 + (au_values['AU01'][i] +au_values['AU06'][i])/2*0.3
        persistant_list.append(persistant)
        
        out_focus = (1 - abs(result['head_pitch'])/30)*0.4 + (au_values['AU01'][i] + au_values['AU02'][i] + au_values['AU06'][i])*0.2
        out_focus_list.append(out_focus)
        
        out_structure = result['jaw_angle']/140 *0.6 + (1-abs(result['head_pitch'])/30)*0.4
        out_structure_list.append(out_structure)
        
        out_risk = (1 if result['head_pitch'] > 5 else 0) *0.4 + (1 if result['jaw_angle']>120 else 0)*0.3 + au_values['AU04'][i] * 0.3
        out_risk_list.append(out_risk)
    
    select = np.argmax([persistant_list,out_focus_list,out_structure_list,out_risk_list], axis=1)
        # Work_DNA = persistant*0.4 + out_structure*0.3 + out_risk*0.3
        # Focus_Work = out_focus*0.5 + au_values['AU01']*0.2 + (1-au_values['AU04'])*0.1 + au_values['AU06']*0.2
    return {
            "Persistant":{"balance": f"{np.mean(persistant_list)*100:.1f}%", "top_image":int(select[0])},
            "Focus":{"balance": f"{np.mean(out_focus_list)*100:.1f}%", "top_image":int(select[1])},
            "Structure":{"balance": f"{np.mean(out_structure_list)*100:.1f}%", "top_image":int(select[2])},
            "Risk":{"balance": f"{np.mean(out_risk_list)*100:.1f}%", "top_image":int(select[3])},
            }