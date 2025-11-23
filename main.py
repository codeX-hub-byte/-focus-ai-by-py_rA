from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from mediapipe import solutions as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe models
mp_pose = mp.pose.Pose(static_image_mode=False, model_complexity=1)
mp_face = mp.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

def head_direction(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]

    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y

    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = (nose.y - (left_eye.y + right_eye.y)/2) * 100
    return yaw, pitch

def detect_writing(body):
    rw = body[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    rs = body[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    motion = abs(rw.y - rs.y)
    return (rw.y > rs.y and motion > 0.01)

def detect_sleep(face):
    top = face[159]
    bottom = face[145]
    eye_gap = abs(top.y - bottom.y)
    return eye_gap < 0.003

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pose_res = mp_pose.process(img_rgb)
    face_res = mp_face.process(img_rgb)

    result = {
        "students": [],
        "total": 0,
        "focused": 0
    }

    if not pose_res.pose_landmarks:
        return result

    body = pose_res.pose_landmarks.landmark
    writing = detect_writing(body)
    focus = True

    if face_res.multi_face_landmarks:
        face = face_res.multi_face_landmarks[0].landmark
        yaw, pitch = head_direction(face)

        if abs(yaw) > 25:
            focus = False
        if pitch > 40:
            focus = False
        if detect_sleep(face):
            focus = False

    if writing:
        focus = True

    result["students"].append({
        "yaw": float(yaw),
        "pitch": float(pitch),
        "writing": writing,
        "sleeping": detect_sleep(face),
        "focused": focus
    })

    result["total"] = 1
    result["focused"] = 1 if focus else 0

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)