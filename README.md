# Safe-Drive-Vision
🚗 DriveGuard AI is an end-to-end computer vision application designed to enhance driver safety. The system detects driver drowsiness (eye closure, yawning) and distracted behavior (mobile phone usage) in real time using a webcam/camera stream.

## Key Features
* Real-Time Eye Closure Detection:
Utilizes facial landmarks and Eye Aspect Ratio (EAR) to monitor and flag drowsiness.

* Yawn Detection:
Computes Mouth Aspect Ratio (MAR) to signal driver yawning—an early indicator of fatigue.

* Phone Usage Monitoring:
Integrates a YOLO object detector to identify smartphones in the frame, with intelligent overlap detection to trigger alerts if the driver is likely talking on the phone.

* Drowsiness Metrics:
Calculates PERCLOS (percentage of eye closure), drowsiness/yawn events, and presents live statistics overlayed on the video stream.

* Visual Alerts:
Instantly warns the user on the display in case of dangerous behavior.

## Technologies Used
* Python

* OpenCV (real-time video processing & visualization)

* MediaPipe FaceMesh (facial landmark detection)

* Ultralytics YOLO (object detection for phone usage)

* NumPy

## How It Works
* The system captures video from a webcam.

* MediaPipe FaceMesh detects facial landmarks to track eye and mouth positions.

* EAR and MAR are computed for drowsiness and yawn detection.

* YOLO detects the presence of a smartphone and, if overlapping with the driver's face, displays an alert for distracted driving.

* Comprehensive statistics (EAR, MAR, PERCLOS, event counts) are displayed live.
