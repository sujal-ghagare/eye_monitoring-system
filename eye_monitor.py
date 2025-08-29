# first make a txt file in the same directory called requirements.txt
# and put the following lines in it:
# opencv-python
# mediapipe
# numpy
# pandas
# streamlit
# then run the following command in your terminal: "pip install -r requirements.txt"
# this will install all the required packages

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh

def eye_aspect_ratio(landmarks, eye_idx):
    """Calculate Eye Aspect Ratio (EAR) for focus detection."""
    points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_idx])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def main(save_csv=None):
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Eye landmark indices (left + right eye)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    focus_history = []
    last_log_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        h, w, _ = frame.shape
        focused = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear_left = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                ear_right = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                ear = (ear_left + ear_right) / 2.0

                # If eyes are not too closed, consider focused
                focused = ear > 0.2

        # Save focus result
        focus_history.append(1 if focused else 0)
        if len(focus_history) > 300:  # keep last ~1 min (5 FPS assumed)
            focus_history.pop(0)

        focus_percent = (sum(focus_history) / len(focus_history)) * 100

        # Show on screen
        cv2.putText(frame, f"Focus: {focus_percent:.1f}%", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if focused else (0, 0, 255), 2)

        cv2.imshow("Eye Monitoring", frame)

        # ✅ Save to CSV every 2 seconds (UTC timestamp)
        if save_csv and (time.time() - last_log_time > 2):
            ts = int(pd.Timestamp.utcnow().timestamp())
            with open(save_csv, "a") as f:
                f.write(f"{ts},{focus_percent:.2f}\n")
            last_log_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to CSV log file", default=None)
    args = parser.parse_args()
    main(save_csv=args.csv)
