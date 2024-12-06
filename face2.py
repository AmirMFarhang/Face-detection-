import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import mediapipe as mp


IMAGE_PATH = "ex.jpg"
MIN_FACE_SIZE = 50
EMOTION_ACTIONS = ['emotion']  # DeepFace supported actions: ['emotion','age','gender','race']

# For MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Initialize MTCNN face detector
face_detector = MTCNN()


image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError(f"Could not load the image from {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

faces = face_detector.detect_faces(image_rgb)
if len(faces) == 0:
    raise ValueError("No faces detected in the image.")

face = max(faces, key=lambda x: x['box'][2]*x['box'][3])
x, y, w, h = face['box']

x = max(x, 0)
y = max(y, 0)
w = min(w, image.shape[1] - x)
h = min(h, image.shape[0] - y)

if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
    raise ValueError("Detected face is too small.")

face_crop = image[y:y+h, x:x+w]
face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)


results = face_mesh.process(face_crop_rgb)
if not results.multi_face_landmarks:
    raise ValueError("No face landmarks detected.")

face_landmarks = results.multi_face_landmarks[0]  # Only one face

height, width, _ = face_crop.shape
landmarks_xy = []
for lm in face_landmarks.landmark:
    px = int(lm.x * width)
    py = int(lm.y * height)
    landmarks_xy.append((px, py))


left_eyebrow_indices = [70,71,72,73,74,75]
right_eyebrow_indices = [300,301,302,303,304,305]

lips_indices = [78,79,80,81,82,13,312,311,310,415,308]

left_eyebrow_points = [landmarks_xy[i] for i in left_eyebrow_indices if i < len(landmarks_xy)]
right_eyebrow_points = [landmarks_xy[i] for i in right_eyebrow_indices if i < len(landmarks_xy)]

lips_points = [landmarks_xy[i] for i in lips_indices if i < len(landmarks_xy)]

emotions = DeepFace.analyze(face_crop, actions=EMOTION_ACTIONS, enforce_detection=False)

if isinstance(emotions, list) and len(emotions) > 0:
    emotions = emotions[0]
print(emotions)
emotion_result = emotions["emotion"]


if len(landmarks_xy) > 0:

    cheek_index = 234
    if cheek_index < len(landmarks_xy):
        cheek_px, cheek_py = landmarks_xy[cheek_index]
        cheek_px = np.clip(cheek_px, 0, width-1)
        cheek_py = np.clip(cheek_py, 0, height-1)
        skin_color = face_crop[cheek_py, cheek_px, :].tolist()  # BGR color
    else:
        skin_color = face_crop[height//2, width//2, :].tolist()
else:
    skin_color = face_crop[height//2, width//2, :].tolist()


glasses_detected = False  # Just a placeholder

# ---------------------
# Print Results
# ---------------------
print("Detected Face Attributes:")
print(f"Bounding box: (x={x}, y={y}, w={w}, h={h})")
print("Eyebrow landmarks (left):", left_eyebrow_points)
print("Eyebrow landmarks (right):", right_eyebrow_points)
print("Lips landmarks:", lips_points)
print("Emotion:", emotion_result)
print("Estimated skin color (BGR):", skin_color)
print("Glasses detected (placeholder):", glasses_detected)

for px, py in landmarks_xy:
    cv2.circle(face_crop, (px, py), 1, (0,255,0), -1)

cv2.imshow("Face with Landmarks", face_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
