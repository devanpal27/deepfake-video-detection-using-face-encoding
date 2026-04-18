import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

def extract_landmarks(image):
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
    return np.array(landmarks, dtype=np.float32)

def normalize_landmarks(landmarks, left_eye_idx=33, right_eye_idx=263):
    if landmarks is None:
        return None

    if len(landmarks) <= max(left_eye_idx, right_eye_idx):
        return None

    left_eye = landmarks[left_eye_idx]
    right_eye = landmarks[right_eye_idx]

    eye_center = (left_eye + right_eye) / 2.0
    aligned = landmarks - eye_center

    eye_vector = right_eye - left_eye
    eye_distance = np.linalg.norm(eye_vector)

    if eye_distance < 1e-6:
        return None

    angle = np.arctan2(eye_vector[1], eye_vector[0])

    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ], dtype=np.float32)

    aligned = aligned @ rotation_matrix.T
    aligned = aligned / eye_distance

    return aligned.astype(np.float32)