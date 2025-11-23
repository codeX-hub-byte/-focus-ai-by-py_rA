from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
import os

from ultralytics import YOLO
from mediapipe import solutions as mp
from google import genai

REVERSE_EYE_LOGIC = False

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model = YOLO("yolov8m-pose.pt")
mp_face = mp.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

STUDENT_MEMORY = {}
MEMORY_TIME = 30
NORMALIZED_MATCH_THRESHOLD = 0.05


def get_yaw_pitch(face):
    left_eye = face[33]
    right_eye = face[263]
    nose = face[1]

    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y

    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = (nose.y - (left_eye.y + right_eye.y) / 2) * 100

    return yaw, pitch


def detect_sleep(face):
    top = face[159]
    bottom = face[145]
    eye_gap = abs(top.y - bottom.y)
    return eye_gap < 0.003


def detect_writing(kp):
    try:
        rw_y = kp[10][1]
        rs_y = kp[6][1]
        rh_y = kp[12][1]
        return (rw_y > rs_y + 30) and (rw_y > rh_y - 30)
    except:
        return False


def detect_posture(kp):
    try:
        hip_y = (kp[11][1] + kp[12][1]) / 2
        knee_y = (kp[13][1] + kp[14][1]) / 2
        ankle_y = (kp[15][1] + kp[16][1]) / 2

        if knee_y - hip_y < 20:
            return "Sitting"
        if ankle_y - hip_y > 100:
            return "Standing"
        return "Unknown/Partial"
    except:
        return "Unknown"


def smooth_focus(student_id, current_focus):
    now = time.time()
    if student_id not in STUDENT_MEMORY:
        STUDENT_MEMORY[student_id] = {"focus": current_focus, "timestamp": now}
        return current_focus
    if now - STUDENT_MEMORY[student_id]["timestamp"] < MEMORY_TIME:
        return STUDENT_MEMORY[student_id]["focus"]
    STUDENT_MEMORY[student_id] = {"focus": current_focus, "timestamp": now}
    return current_focus


@app.post("/process-frame")
async def process_frame(
    file: UploadFile = File(...),
    strictness: int = Query(50, ge=10, le=90)
):
    content = await file.read()

    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    MIN_ANGLE = 10
    MAX_ANGLE = 40
    tol = MIN_ANGLE + ((90 - strictness) / 80) * (MAX_ANGLE - MIN_ANGLE)

    pose_res = yolo_model(img_rgb, verbose=False)[0]
    bodies = pose_res.keypoints

    face_res = mp_face.process(img_rgb)

    result = {"students": [], "total": 0, "focused": 0, "gemini_analysis": ""}

    if not face_res.multi_face_landmarks:
        return result

    for i, face in enumerate(face_res.multi_face_landmarks):
        face_landmarks = face.landmark

        yaw, pitch = get_yaw_pitch(face_landmarks)
        sleeping = detect_sleep(face_landmarks)
        face_nose_norm = np.array([face_landmarks[1].x, face_landmarks[1].y])

        matched_body_kp = None
        min_d = float('inf')

        for kp in bodies:
            try:
                yolo_head = np.array([kp.xy[0][0] / w, kp.xy[0][1] / h])
                d = np.linalg.norm(face_nose_norm - yolo_head)

                if d < min_d and d < NORMALIZED_MATCH_THRESHOLD:
                    min_d = d
                    matched_body_kp = kp.xy[0]
            except:
                continue

        posture = "Unknown"
        writing = False

        if matched_body_kp is not None:
            posture = detect_posture(matched_body_kp)
            writing = detect_writing(matched_body_kp)

        focus = True
        if abs(yaw) > tol:
            focus = False
        if pitch < -tol:
            focus = False

        if REVERSE_EYE_LOGIC:
            if sleeping:
                focus = True
            elif focus:
                focus = False
        else:
            if sleeping:
                focus = False

        if pitch > 20 and matched_body_kp is not None:
            if writing:
                focus = True
            else:
                focus = False

        focus = smooth_focus(i, focus)

        result["students"].append({
            "yaw": float(yaw),
            "pitch": float(pitch),
            "sleeping": sleeping,
            "writing": writing,
            "posture": posture,
            "focused": focus,
            "center_x": float(face_landmarks[1].x),
            "center_y": float(face_landmarks[1].y)
        })

    result["total"] = len(result["students"])
    result["focused"] = sum(1 for s in result["students"] if s["focused"])

    prompt = "Classroom analysis:\n"
    for i, s in enumerate(result["students"]):
        prompt += (
            f"Student {i+1}: focused={s['focused']}, "
            f"posture={s['posture']}, "
            f"sleeping={s['sleeping']}, "
            f"writing={s['writing']}\n"
        )

    try:
        g = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        result["gemini_analysis"] = g.text
    except Exception as e:
        result["gemini_analysis"] = f"Gemini error: {e}"

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
