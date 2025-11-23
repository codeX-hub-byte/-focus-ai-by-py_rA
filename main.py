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

# -----------------------------------------
# CONFIGURATION AND INITIALIZATION
# -----------------------------------------

# --- LOGIC SWITCH ---
# Set to False for Standard Logic (Eyes Closed -> Distracted)
# Set to True for REVERSED Logic (Eyes Closed -> FOCUSED)
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

# Load YOLO Pose Model
yolo_model = YOLO("yolov8m-pose.pt")

# MediaPipe FaceMesh
mp_face = mp.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

# 30-SECOND MEMORY for Temporal Smoothing
STUDENT_MEMORY = {}
MEMORY_TIME = 30  # SECONDS

# Normalized threshold for matching Face (MediaPipe) to Body (YOLO)
# A distance greater than 0.05 (5% of the image) is likely a mismatch.
NORMALIZED_MATCH_THRESHOLD = 0.05


# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

def get_yaw_pitch(face):
    """Calculates head rotation (Yaw) and tilt (Pitch) using normalized coordinates."""
    left_eye = face[33]
    right_eye = face[263]
    nose = face[1]

    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y

    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = (nose.y - (left_eye.y + right_eye.y) / 2) * 100

    return yaw, pitch

def detect_sleep(face):
    """Detects if eyes are closed based on vertical gap."""
    top = face[159]
    bottom = face[145]
    eye_gap = abs(top.y - bottom.y)
    return eye_gap < 0.003

def detect_writing(kp):
    """Detects writing posture using YOLO keypoints (pixel values)."""
    try:
        # Keypoints: rw (10), rs (6), rh (12). Using YOLOv8 Pose indices.
        rw_y = kp[10][1] 
        rs_y = kp[6][1]
        rh_y = kp[12][1]

        # Use absolute pixel difference for reliability, 
        # adjusted slightly from the original to be more robust.
        wrist_below_shoulder = rw_y > rs_y + 30 
        wrist_low = rw_y > rh_y - 30 
        return wrist_below_shoulder and wrist_low
    except:
        return False

def detect_posture(kp):
    """Detects Sitting or Standing using YOLO keypoints (pixel values)."""
    try:
        # Keypoints: lh (11), rh (12), lk (13), rk (14), la (15), ra (16)
        
        # Hip and Knee Y-coordinates
        hip_y = (kp[11][1] + kp[12][1]) / 2
        knee_y = (kp[13][1] + kp[14][1]) / 2
        ankle_y = (kp[15][1] + kp[16][1]) / 2

        hip_to_knee = knee_y - hip_y
        
        # If leg parts are close (short vertical distance), assume sitting
        if hip_to_knee < 20: # Threshold in pixels
            return "Sitting"
            
        # If the full leg length is visible and long, assume standing
        if ankle_y - hip_y > 100: # Threshold in pixels
             return "Standing"
             
        return "Unknown/Partial"
        
    except:
        return "Unknown"


def smooth_focus(student_id, current_focus):
    """Applies 30-second temporal smoothing to the focus state."""
    now = time.time()

    if student_id not in STUDENT_MEMORY:
        STUDENT_MEMORY[student_id] = {"focus": current_focus, "timestamp": now}
        return current_focus

    # Only update the stored focus state every MEMORY_TIME seconds
    if now - STUDENT_MEMORY[student_id]["timestamp"] < MEMORY_TIME:
        return STUDENT_MEMORY[student_id]["focus"]

    STUDENT_MEMORY[student_id] = {"focus": current_focus, "timestamp": now}
    return current_focus


# -----------------------------------------
# CORE LOGIC: ROBUST BODY-FACE ASSOCIATION
# -----------------------------------------
def find_closest_body(face_nose_norm, bodies):
    """
    Finds the best matching YOLO body pose for a MediaPipe face using
    normalized coordinate distance (CRITICAL FIX).
    """
    min_d = float('inf')
    matched_body_kp = None

    for kp in bodies:
        # 1. Get YOLO nose/head keypoint (kp[0] is often the head/nose)
        # We need to normalize the YOLO pixel coordinates to [0.0, 1.0]
        try:
            bx, by = kp.xy[0][0], kp.xy[0][1] # Pixel coordinates
            
            # Note: The image dimensions (w, h) are required here,
            # but since this function is called inside the endpoint, we rely on 
            # the caller to handle normalization if it were outside.
            # Assuming we pass normalized keypoints in a cleaner setup. 
            # For simplicity in this implementation, we will keep the
            # normalization inside the main loop, but note this structure 
            # is typically cleaner if normalized data is passed directly.
            
            # --- USING Normalized Coordinates for Robustness (Fix implemented in main loop) ---
            return kp # Returning the full keypoints object for simplicity 
                     # as the robust check is best done in the main loop.
                     
        except:
            continue
            
    return matched_body_kp


# -----------------------------------------
# MAIN PROCESSING ENDPOINT
# -----------------------------------------
@app.post("/process-frame")
async def process_frame(
    file: UploadFile = File(...),
    strictness: int = Query(50, ge=10, le=90)
):
    content = await file.read()

    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Strictness → angle tolerance (Same logic as before)
    MIN_ANGLE = 10
    MAX_ANGLE = 40
    tol = MIN_ANGLE + ((90 - strictness) / 80) * (MAX_ANGLE - MIN_ANGLE)

    # YOLO pose detection
    pose_res = yolo_model(img_rgb, verbose=False)[0]
    bodies = pose_res.keypoints
    
    # Face detection
    face_res = mp_face.process(img_rgb)

    result = {"students": [], "total": 0, "focused": 0, "gemini_analysis": ""}

    if not face_res.multi_face_landmarks:
        return result

    # For each face
    for i, face in enumerate(face_res.multi_face_landmarks):
        face_landmarks = face.landmark

        # 1. Face Data
        yaw, pitch = get_yaw_pitch(face_landmarks)
        sleeping = detect_sleep(face_landmarks)
        
        # Normalized coordinates of the face's nose for matching
        face_nose_norm = np.array([face_landmarks[1].x, face_landmarks[1].y])

        # 2. Match closest body (ROBUST NORMALIZED CHECK)
        matched_body_kp = None
        min_d = float('inf')

        # Find the best match using normalized distance
        for kp in bodies:
            try:
                # YOLO keypoint (0) in normalized space [0.0, 1.0]
                yolo_head_pixel = kp.xy[0][0] # x (pixel)
                yolo_head_norm = np.array([yolo_head_pixel / w, kp.xy[0][1] / h])
                
                # Calculate Euclidean distance in normalized space
                distance = np.linalg.norm(face_nose_norm - yolo_head_norm)

                if distance < min_d and distance < NORMALIZED_MATCH_THRESHOLD:
                    min_d = distance
                    matched_body_kp = kp.xy[0] # Store pixel keypoints for posture/writing checks
            except:
                continue

        posture = "Unknown"
        writing = False

        if matched_body_kp is not None:
            posture = detect_posture(matched_body_kp)
            writing = detect_writing(matched_body_kp)
            
        # 3. Base Focus (Head Direction)
        focus = True
        if abs(yaw) > tol:
            focus = False # Head turned horizontally
        if pitch < -tol:
            focus = False # Head tilted excessively up
        
        # 4. Eye Focus Logic (Controlled by REVERSE_EYE_LOGIC flag)
        if REVERSE_EYE_LOGIC:
            # REVERSED LOGIC: Eyes Closed -> FOCUSED; Eyes Open -> DISTRACTED (only if head is straight)
            if sleeping:
                focus = True
            elif focus: # If the head is straight, but eyes are open, it's NOT focused (under this reversed logic)
                focus = False
        else:
            # STANDARD LOGIC: Eyes Closed -> DISTRACTED
            if sleeping:
                focus = False
            
        # 5. Looking down override logic
        if pitch > 20 and matched_body_kp is not None:
            if writing:
                focus = True # Override distraction: Looking down + writing = Focused
            else:
                focus = False # Confirm distraction: Looking down + NOT writing = Distracted

        # 6. Temporal smoothing
        student_id = i
        focus = smooth_focus(student_id, focus)

        # Store student
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

    # Gemini summary (same as before)
    # ... (omitted for brevity, assume the original Gemini analysis block is here)
    
    prompt = "Classroom analysis:\n"
    for i, s in enumerate(result["students"]):
        prompt += f"Student {i+1}: focused={s['focused']}, posture={s['posture']}, sleeping={s['sleeping']}, writing={s['writing']}\n"

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
