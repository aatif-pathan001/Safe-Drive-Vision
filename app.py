import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import YOLO

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

model = YOLO('model/weights/yolov8n.pt')
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpFaceDetection = mp.solutions.face_detection

# Landmark indices
left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]
mouth = [78, 73, 11, 303, 308, 404, 16, 180]

EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.7
fps = 30
rolling_window = 60
rolling_frames = fps * rolling_window
CONSEC_FRAMES = 5


def euclidean(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def calculate_ear(eye_points):
    a = euclidean(eye_points[1], eye_points[5])
    b = euclidean(eye_points[2], eye_points[4])
    c = euclidean(eye_points[0], eye_points[3])
    ear = (a+b)/(2.0*c)
    return ear


def calculate_mar(mouth_points):
    a = euclidean(mouth_points[3], mouth_points[5])
    b = euclidean(mouth_points[2], mouth_points[6])
    c = euclidean(mouth_points[1], mouth_points[7])
    d = euclidean(mouth_points[0], mouth_points[4])
    mar = (a+b+c)/(2.0*d)
    return mar


def boxes_overlap(boxA, boxB, min_iou=0.1):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False
    # Intersection over Union (IoU)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou > min_iou


st.title("🚗 Driver Drowsiness Detection System")
st.write("Real-time detection using computer vision (MediaPipe FaceMesh)")

start_cam = st.button("Start Detection")
stop = st.button('Stop Detection')

FRAME_WINDOW = st.image([])
drowsy_placeholder = st.empty()

if start_cam:
    webcam = cv2.VideoCapture(0)
    eye_closure_status = deque(maxlen=rolling_frames)
    drowse_recode = deque(maxlen=2*fps)
    drowse_count = 0
    yawn_in_progress = False
    frame_counter = 0
    phone_bbox = None
    face_bbox = None

    while webcam.isOpened():
        res, frame = webcam.read()
        if not res:
            st.warning("Camera not responding. Check webcam.")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = faceMesh.process(rgb)
        eyes_closed = 0
        drowse_status = 0
        h, w, _ = frame.shape

        # --- Face Detection ---
        with mpFaceDetection.FaceDetection() as faceDetection:
            results = faceDetection.process(frame)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1_min = int(bbox.xmin * w)
                    y1_min = int(bbox.ymin * h)
                    box_width = int(bbox.width * w)
                    box_height = int(bbox.height * h)
                    x2_min = x1_min + box_width
                    y2_min = y1_min + box_height
                    face_bbox = [x1_min, y1_min, x2_min, y2_min]

        # --- YOLO Phone Detection ---
        phone_detected = False
        phone_bbox = None
        yolo_results = model(frame)[0]

        for det in yolo_results.boxes:
            cls_id = int(det.cls)
            conf = float(det.conf)
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            if cls_id == 67 and conf > 0.2:
                phone_detected = True
                phone_bbox = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, 'Phone Detected!', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if phone_detected and boxes_overlap(face_bbox, phone_bbox, min_iou=0.1):
            cv2.putText(frame, 'ALERT: Using Phone!', (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # --- Eyes Closer Detection ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = faceMesh.process(rgb)
        eyes_closed = 0
        drowse_status = 0
        flag = 0

        if mesh.multi_face_landmarks:
            for facLM in mesh.multi_face_landmarks:
                h, w, _ = frame.shape

                L_eye = [(int(facLM.landmark[i].x * w), int(facLM.landmark[i].y * h)) for i in left_eye]
                R_eye = [(int(facLM.landmark[i].x * w), int(facLM.landmark[i].y * h)) for i in right_eye]
                left_ear = calculate_ear(L_eye)
                right_ear = calculate_ear(R_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                state = "Closed" if avg_ear < EAR_THRESHOLD else "Open"
                eyes_closed = 1 if state == "Closed" else 0

                mouth_mark = [(int(facLM.landmark[i].x * w), int(facLM.landmark[i].y * h)) for i in mouth]
                mar = calculate_mar(mouth_mark)

                # Drowsiness logic
                if mar > MAR_THRESHOLD:
                    drowse_status = 1
                    frame_counter += 1
                    if (not yawn_in_progress) and frame_counter >= CONSEC_FRAMES:
                        drowse_count += 1
                        yawn_in_progress = True
                else:
                    drowse_status = 0
                    frame_counter = 0
                    yawn_in_progress = False

                #cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 100, 55), 2)
                cv2.putText(frame, f'Eyes: {state}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 100, 255), 2)
                #cv2.putText(frame, f'MAR: {mar:.2f}', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 0, 255), 2)
                #for p, q in mouth_mark:
                #    cv2.circle(frame, (p, q), 2, (200, 200, 0), -1)

                eye_closure_status.append(eyes_closed)

        if len(eye_closure_status) > 0:
            perclos = (sum(eye_closure_status) / len(eye_closure_status)) * 100
        else:
            perclos = 0

        drowse_recode.append(drowse_status)
        if yawn_in_progress:
            cv2.putText(frame, f'Drowse Detected', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drowsy_placeholder.info(f'**PERCLOS:** {perclos:.2f}%   |   **Drowsiness Count:** {drowse_count}    |   **Fps:** {fps}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if stop:
            st.info("Stopped detection and released webcam.")
            break

    webcam.release()
    cv2.destroyAllWindows()
    st.success("Detection stopped.")

