from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import os
from mediapipe import solutions as mp
from google import genai

# --- GEMINI CLIENT INITIALIZATION ---
# Ensure GEMINI_API_KEY is set in your environment
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe models
# Note: Using model_complexity=1 for a good balance of accuracy and speed
mp_pose = mp.pose.Pose(static_image_mode=False, model_complexity=1) 
mp_face = mp.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

# Helper function to determine head rotation (Yaw) and tilt (Pitch)
def head_direction(landmarks):
    # Landmarks for eyes (33, 263) and nose (1)
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]

    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y

    # Yaw (Horizontal rotation)
    yaw = np.degrees(np.arctan2(dy, dx))
    # Pitch (Vertical tilt - estimate based on nose drop relative to eye line)
    pitch = (nose.y - (left_eye.y + right_eye.y)/2) * 100
    return yaw, pitch

# Helper function to detect if eyes are closed
def detect_sleep(face):
    # Landmarks for right eye (example)
    top = face[159]
    bottom = face[145]
    eye_gap = abs(top.y - bottom.y)
    # Threshold for closed eyes (normalized coordinate space)
    return eye_gap < 0.003

# Helper function to detect if the student is in a writing posture (hand near desk area)
def detect_writing_posture(body_landmarks):
    try:
        # Check Right Hand position (assuming right-handed writing)
        rw = body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        rs = body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        rh = body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        
        # Criteria 1: Right wrist is substantially below the right shoulder (lowered arm)
        wrist_below_shoulder = rw.y > rs.y + 0.05 
        
        # Criteria 2: Right wrist is low enough to be near a desk (below the hip line)
        wrist_low_position = rw.y > rh.y - 0.1 
        
        return wrist_below_shoulder and wrist_low_position
    except (IndexError, AttributeError):
        return False

# Helper function to determine Sitting or Standing posture
def detect_posture(body_landmarks):
    if not body_landmarks:
        return "Sitting (Occluded)"
    
    try:
        hip_y = (body_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y + 
                 body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y) / 2
        knee_y = (body_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y + 
                  body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y) / 2
        ankle_y = (body_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y + 
                   body_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y) / 2
    except (IndexError, AttributeError):
        return "Sitting (Partial View)"

    leg_length = ankle_y - hip_y
    
    # Standing criteria: long, visible leg length
    if leg_length > 0.4:
        return "Standing"
        
    # Sitting criteria: compact lower body
    if knee_y - hip_y < 0.15 or leg_length < 0.4:
        return "Sitting"
        
    return "Unknown"

# Core function: Associates a face with its closest body pose
def find_closest_pose(face_landmarks, all_pose_landmarks):
    if not all_pose_landmarks:
        return None

    face_nose = (face_landmarks[1].x, face_landmarks[1].y)
    min_distance = float('inf')
    closest_pose_landmarks = None
    
    for pose in all_pose_landmarks:
        try:
            pose_nose = (pose.landmark[mp.solutions.pose.PoseLandmark.NOSE].x, 
                         pose.landmark[mp.solutions.pose.PoseLandmark.NOSE].y)
        except (IndexError, AttributeError):
            continue

        distance = np.sqrt((face_nose[0] - pose_nose[0])**2 + (face_nose[1] - pose_nose[1])**2)
        
        # 0.05 is the normalized threshold for a match between face and body head positions
        if distance < min_distance and distance < 0.05: 
            min_distance = distance
            closest_pose_landmarks = pose.landmark

    return closest_pose_landmarks

@app.post("/process-frame")
async def process_frame(
    file: UploadFile = File(...), 
    strictness: int = Query(50, ge=10, le=90) # Accept strictness from frontend (10-90)
):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- DYNAMIC THRESHOLD CALCULATION ---
    # Map the strictness slider (10 to 90) to a Yaw/Pitch angle tolerance (MAX_ANGLE to MIN_ANGLE)
    # High strictness (e.g., 90) = low tolerance (e.g., 10 degrees)
    # Low strictness (e.g., 10) = high tolerance (e.g., 40 degrees)

    MIN_ANGLE = 10  # Strictest threshold (degrees)
    MAX_ANGLE = 40  # Lenient threshold (degrees)

    # Normalized strictness (0 to 1 scale, where 1 means MAX_ANGLE tolerance)
    # Invert the scale: (90 - strictness) / 80
    normalized_tolerance = (90 - strictness) / (90 - 10) 
    
    # Calculate the dynamic angle threshold for Yaw and Pitch down
    DYNAMIC_ANGLE_THRESHOLD = MIN_ANGLE + (MAX_ANGLE - MIN_ANGLE) * normalized_tolerance
    
    # Pitch up (looking at the ceiling) is usually highly distracting, so we keep a tighter threshold.
    # It allows slightly more movement than Yaw, but less than the lenient setting for Yaw.
    DYNAMIC_PITCH_UP_THRESHOLD = min(40, DYNAMIC_ANGLE_THRESHOLD * 1.5) 
    
    # --- END DYNAMIC THRESHOLD CALCULATION ---


    # Perform detection
    pose_res = mp_pose.process(img_rgb)
    face_res = mp_face.process(img_rgb)

    result = {
        "students": [],
        "total": 0,
        "focused": 0,
        "gemini_analysis": "Waiting for data..."
    }

    all_pose_landmarks = pose_res.multi_pose_landmarks or []
    
    if not face_res.multi_face_landmarks:
        return result

    focused_count = 0

    # --- CORE MULTI-STUDENT LOGIC ---
    for face_landmarks in face_res.multi_face_landmarks:
        
        face = face_landmarks.landmark
        focus = True
        is_writing = False
        
        # 1. Association and Posture
        body = find_closest_pose(face, all_pose_landmarks)
        posture = detect_posture(body) if body else "Sitting (No Pose Detected)"

        # 2. Base Focus Detection
        yaw, pitch = head_direction(face)

        # Apply dynamic thresholds for distraction
        if abs(yaw) > DYNAMIC_ANGLE_THRESHOLD:
            focus = False # Head turned horizontally
        if pitch < -DYNAMIC_PITCH_UP_THRESHOLD:
             # Check for head tilted excessively up (Pitch is negative when looking up)
             focus = False 
        if detect_sleep(face):
            focus = False # Eyes closed
            
        # 3. CONTEXTUAL FOCUS OVERRIDE (Writing vs. Phone Use)
        # Looking down (pitch > 20) is the ambiguous state
        if pitch > 20 and body:
            is_writing = detect_writing_posture(body)
            
            if is_writing:
                # OVERRIDE: Looking down + writing hand posture = FOCUSED
                focus = True
            elif detect_sleep(face):
                # If sleeping, keep distraction
                pass 
            else:
                # CONFIRM DISTRACTION: Looking down + NOT writing = DISTRACTED (e.g., phone use)
                focus = False

        # 4. Final Data Compilation
        center_x = face[1].x
        center_y = face[1].y
        
        result["students"].append({
            "yaw": float(yaw),
            "pitch": float(pitch),
            "posture": posture,
            "writing_posture": is_writing, 
            "sleeping": detect_sleep(face),
            "focused": focus,
            "center_x": float(center_x),
            "center_y": float(center_y),
        })

        if focus:
            focused_count += 1

    result["total"] = len(result["students"])
    result["focused"] = focused_count

    # --- GEMINI AI ANALYSIS ---
    # Prepare data for the prompt
    student_summaries = []
    focus_percentage = round((result['focused'] / result['total']) * 100) if result['total'] > 0 else 0
    
    for i, student in enumerate(result["students"]):
        status = "FOCUSED" if student["focused"] else "DISTRACTED"
        reason = []
        if not student["focused"]:
            if student["sleeping"]: reason.append("Sleeping/Eyes Closed")
            # Use the dynamic threshold in the reason summary
            elif abs(student["yaw"]) > DYNAMIC_ANGLE_THRESHOLD: reason.append(f"Head turned away (Yaw: {student['yaw']:.1f}° > {DYNAMIC_ANGLE_THRESHOLD:.1f}°)")
            elif student["pitch"] > 20 and not student["writing_posture"]: reason.append("Looking down, not writing (possible phone use)")
            elif student["pitch"] < -DYNAMIC_PITCH_UP_THRESHOLD: reason.append(f"Looking up (Pitch: {student['pitch']:.1f}°)")
            
        summary = f"Student {i+1} ({student['posture']}) is {status}. Reason: {'; '.join(reason) or 'N/A'}"
        student_summaries.append(summary)

    prompt = f"""
    Analyze the following student focus data for a classroom setting. 
    Total Students Detected: {result['total']}
    Total Focused: {result['focused']}
    Focus Percentage: {focus_percentage}%
    Current Focus Strictness (Tolerance Angle): {DYNAMIC_ANGLE_THRESHOLD:.1f}°

    Data per Student:
    {'\n'.join(student_summaries)}

    Provide a concise, professional summary for the teacher. Highlight the overall focus level and the primary reasons for distraction based on the dynamic tolerance setting. If the focus is low (below 70%), suggest one brief, proactive intervention.
    """

    # Call the Gemini API
    try:
        gemini_response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result["gemini_analysis"] = gemini_response.text
    except Exception as e:
        result["gemini_analysis"] = f"Gemini Error: Check API Key/Network. Details: {e}"

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
