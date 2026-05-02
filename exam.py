import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import time
import sys
import os
from ultralytics import YOLO
import json
from flask import Flask, Response, jsonify
import threading
import joblib

# Constants
EYE_LEFT_THRESHOLD = 0.3    # More sensitive to eye movement
EYE_RIGHT_THRESHOLD = 0.7   # More sensitive to eye movement
HEAD_TILT_THRESHOLD = 6    # More aggressive head tilt detection
TABLE_REGION_THRESHOLD = 0.7  # Hands in lower 30% of frame
SHOULDER_TILT_THRESHOLD = 0.2  # More sensitive body tilt detection
LOOKING_UP_THRESHOLD = 0.25  # Nose in upper 25% of frame for looking up
LOOK_AWAY_THRESHOLD = 15    # Frames to confirm looking away
LOG_COOLDOWN = 2.0         # Seconds between logging events
FRAME_SKIP = 2             # Process every 2nd frame for performance

class CheatingDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Higher confidence
        self.look_away_counters = defaultdict(int)
        self.cheating_start_time = {}
        self.last_cheating_log_time = defaultdict(float)
        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        self.shoulder_history = defaultdict(list)  # For smoothing shoulder positions
        self.cheating_duration = defaultdict(float)  # Track how long cheating persists
        self.score_history = defaultdict(list)  # For smoothing score over time
        self.events = []  # Store structured events for reporting
        self.json_log_file = "events.jsonl"  # JSON Lines file (one JSON per line)
        self.dataset_file = "dataset.csv"  # For ML training data
        self.app = Flask(__name__)
        self.latest_frame = None
        # Load ML model
        try:
            self.ml_model = joblib.load("cheating_model.pkl")
        except Exception as e:
            print("ML model not loaded:", e)
            self.ml_model = None
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera. Check connection or camera index.")

        # Load YOLO model for phone detection
        self.yolo_model = YOLO('yolov5s.pt')
        self.phone_class_id = 67  # Class ID for cell phone in COCO dataset

        # Setup Flask routes
        @self.app.route('/')
        def index():
            return """
            <html>
            <head>
                <title>AI Proctor Dashboard</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.1.0"></script>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #0f172a;
                        color: #e2e8f0;
                        text-align: center;
                    }
                    h2 {
                        margin-top: 20px;
                    }
                    .container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        gap: 20px;
                    }
                    .card {
                        background: #1e293b;
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                        width: 700px;
                    }
                    img {
                        border-radius: 10px;
                    }
                    a {
                        color: #38bdf8;
                        text-decoration: none;
                        font-weight: bold;
                    }
                </style>
            </head>
            <body>
                <h1> AI Proctor Dashboard</h1>

                <div class="container">

                    <div class="card">
                        <h2>Live Feed</h2>
                        <img src='/video_feed' width='640'/>
                    </div>

                    <div class="card">
                        <h2>Cheating Score (Live)</h2>
                        <canvas id="scoreChart" width="600" height="200"></canvas>
                    </div>

                    <div class="card">
                        <h2>Report</h2>
                        <a href='/report'>View Report JSON</a>
                    </div>

                </div>

                <script>
                    const ctx = document.getElementById('scoreChart').getContext('2d');
                    const chart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Score',
                                data: [],
                                borderColor: '#38bdf8',
                                tension: 0.3,
                                borderWidth: 2
                            }]
                        },
                        options: {
                            animation: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100,
                                    ticks: {
                                        callback: function(value) {
                                            if (value < 20) return value + ' (Safe)';
                                            if (value < 50) return value + ' (Suspicious)';
                                            return value + ' (Cheating)';
                                        }
                                    }
                                }
                            },
                            plugins: {
                                annotation: {
                                    annotations: {
                                        safeZone: {
                                            type: 'box',
                                            yMin: 0,
                                            yMax: 20,
                                            backgroundColor: 'rgba(0,255,0,0.1)'
                                        },
                                        suspiciousZone: {
                                            type: 'box',
                                            yMin: 20,
                                            yMax: 50,
                                            backgroundColor: 'rgba(255,255,0,0.1)'
                                        },
                                        cheatingZone: {
                                            type: 'box',
                                            yMin: 50,
                                            yMax: 100,
                                            backgroundColor: 'rgba(255,0,0,0.1)'
                                        }
                                    }
                                }
                            }
                        }
                    });

                    async function fetchScores() {
                        const res = await fetch('/scores');
                        const data = await res.json();

                        chart.data.labels = data.map(d => d.time);
                        chart.data.datasets[0].data = data.map(d => d.score);
                        chart.update();
                    }

                    // Update every 10 seconds
                    setInterval(fetchScores, 10000);
                </script>
            </body>
            </html>
            """

        @self.app.route('/video_feed') # Stream video feed
        def video_feed():
            return Response(self.generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/report') # Return report JSON
        def report():
            try:
                with open('report.json') as f:
                    return jsonify(json.load(f))
            except:
                return jsonify({"error": "No report yet"})

        @self.app.route('/scores') #graph scores
        def scores():
            data = [
                {
                    "score": e.get("score", 0),
                    "time": time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", 0)))
                }
                for e in self.events if "score" in e
            ]
            return jsonify(data[-50:])

        # Run Flask in background thread
        threading.Thread(target=lambda: self.app.run(port=5000, debug=False, use_reloader=False)).start()
    def generate_frames(self):
        while True:
            if self.latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', self.latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def to_pixels(self, landmark, w, h):
        """Convert normalized landmark coordinates to pixel values."""
        try:
            return int(landmark.x * w), int(landmark.y * h)
        except AttributeError:
            return 0, 0  # Default value if landmark is None

    def get_bounding_box(self, pose_landmarks, w, h):
        """Calculate a bounding box around the person using pose landmarks."""
        if not pose_landmarks:
            return None, None, None, None
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for landmark in pose_landmarks.landmark:
            x, y = self.to_pixels(landmark, w, h)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y - int(0.2 * h))  # Move the top boundary 5% higher
            y_max = max(y_max, y)
        return x_min, y_min, x_max, y_max

    def draw_skeleton(self, pose_landmarks, frame):
        """Draw a skeletal structure using pose landmarks."""
        if not pose_landmarks:
            return
        h, w, _ = frame.shape
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
        ]
        for start, end in connections:
            start_point = self.to_pixels(pose_landmarks.landmark[start], w, h)
            end_point = self.to_pixels(pose_landmarks.landmark[end], w, h)
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)  # White lines for skeleton
        for landmark in pose_landmarks.landmark:
            x, y = self.to_pixels(landmark, w, h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red dots for landmarks

    def detect_cheating(self, face_landmarks, pose_landmarks, frame, person_id, hands_landmarks):
        h, w, _ = frame.shape
        try:
            left_eye = self.to_pixels(face_landmarks.landmark[33], w, h)
            right_eye = self.to_pixels(face_landmarks.landmark[263], w, h)
            nose = self.to_pixels(face_landmarks.landmark[1], w, h)

            eye_center_x = (left_eye[0] + right_eye[0]) // 2
            head_tilt = abs(left_eye[1] - right_eye[1])

            if person_id not in self.cheating_start_time:
                self.cheating_start_time[person_id] = None
                self.last_cheating_log_time[person_id] = 0.0
                self.shoulder_history[person_id] = []  # Initialize shoulder history

            # Hand detection (refined to check wrist specifically)
            hand_on_table = False
            if hands_landmarks:
                for hand in hands_landmarks:
                    wrist = hand.landmark[0]  # Wrist landmark
                    if wrist.y > TABLE_REGION_THRESHOLD:
                        hand_on_table = True
                    for idx, landmark in enumerate(hand.landmark):
                        x, y = self.to_pixels(landmark, w, h)
                        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
                        if idx == 0:  # Wrist
                            cv2.putText(frame, "W", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Pose detection with smoothing
            body_tilt_left = body_tilt_right = looking_up = False
            if pose_landmarks:
                # Draw skeletal structure
                self.draw_skeleton(pose_landmarks, frame)

                left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_x, left_y = self.to_pixels(left_shoulder, w, h)
                right_x, right_y = self.to_pixels(right_shoulder, w, h)
                cv2.line(frame, (left_x, left_y), (right_x, right_y), (255, 255, 0), 2)  # Yellow line for shoulders

                # Smooth shoulder positions over last 10 frames
                self.shoulder_history[person_id].append((left_shoulder.x, right_shoulder.x))
                if len(self.shoulder_history[person_id]) > 10:  # Keep last 10 frames
                    self.shoulder_history[person_id].pop(0)
                avg_left_x = np.mean([x[0] for x in self.shoulder_history[person_id]])
                avg_right_x = np.mean([x[1] for x in self.shoulder_history[person_id]])
                shoulder_diff = abs(avg_left_x - avg_right_x)

                # Check for unreasonable x difference (possible landmark error)
                if shoulder_diff > 0.6:
                    body_tilt_left = body_tilt_right = False
                    print(f"Warning: Large shoulder difference ({shoulder_diff:.2f}) detected—possible landmark error for Person {person_id}")
                else:
                    # Detect leaning left/right based on smaller difference (< 0.35)
                    if shoulder_diff < 0.35:
                        if avg_left_x > avg_right_x + SHOULDER_TILT_THRESHOLD:
                            body_tilt_left = True
                        elif avg_right_x > avg_left_x + SHOULDER_TILT_THRESHOLD:
                            body_tilt_right = True

                if nose[1] < h * LOOKING_UP_THRESHOLD:
                    looking_up = True

            # Stricter cheating logic using multiple signals
            cheating = False
            # Initialize score
            score = 0
            eye_ratio = eye_center_x / w
            suspicious_count = 0

            # Ignore tiny eye jitter
            if abs(eye_ratio - 0.5) > 0.15:
                suspicious_count += 1
                self.look_away_counters[person_id] += 1

            if head_tilt > HEAD_TILT_THRESHOLD:
                suspicious_count += 1

            if body_tilt_left or body_tilt_right:
                suspicious_count += 1

            if looking_up:
                suspicious_count += 1

            # Only mark cheating if multiple signals
            if suspicious_count >= 1:
                cheating = True
                score += 25

                if (self.look_away_counters[person_id] >= LOOK_AWAY_THRESHOLD and 
                    not self.cheating_start_time[person_id]):
                    self.cheating_start_time[person_id] = time.time()
            else:
                self.look_away_counters[person_id] = max(0, self.look_away_counters[person_id] - 1)

            # YOLO phone detection
            phone_detected_flag = 0
            results = self.yolo_model(frame)

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls == self.phone_class_id and conf > 0.1:
                        phone_detected_flag = 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = self.yolo_model.names[cls]
                        text = f"{label} {conf:.2f}"
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        cheating = True
                        score += 40
                        log_msg = f"{time.time()}: Person {person_id} - Phone detected with confidence {conf:.2f}"
                        print(log_msg)
                        try:
                            with open("cheating_log.txt", "a") as f:
                                f.write(log_msg + "\n")
                        except IOError as e:
                            print(f"Error writing to log file: {e}")
                        event = {
                            "timestamp": time.time(),
                            "person_id": person_id,
                            "type": "phone_detected",
                            "confidence": float(conf)
                        }
                        self.events.append(event)
                        try:
                            with open(self.json_log_file, "a") as jf:
                                jf.write(json.dumps(event) + "\n")
                        except IOError as e:
                            print(f"Error writing JSON log: {e}")

            # Time-based scoring (continuous cheating increases severity)
            current_time = time.time()
            if cheating:
                if self.cheating_start_time[person_id]:
                    duration = current_time - self.cheating_start_time[person_id]
                    self.cheating_duration[person_id] = duration
                    # Add extra score based on duration
                    score += min(int(duration * 4), 40)  # Max +25 for long cheating
                else:
                    self.cheating_start_time[person_id] = current_time
            else:
                self.cheating_start_time[person_id] = None
                self.cheating_duration[person_id] = 0
            # Normalize score (cap at 100)
            score = min(score, 100)
            # Score smoothing (moving average over last 10 frames)
            self.score_history[person_id].append(score)
            if len(self.score_history[person_id]) > 10:
                self.score_history[person_id].pop(0)
            score = int(np.mean(self.score_history[person_id]))

            # Safe zone correction
            if score < 20:
                cheating = False

            # Controlled logging
            current_time = time.time()
            duration = self.cheating_duration[person_id]
            if cheating and current_time - self.last_cheating_log_time[person_id] >= LOG_COOLDOWN:
                log_msg = (f"{current_time}: Person {person_id} - Score: {score}, Duration: {duration:.2f}s, "
                           f"Eye: {eye_ratio:.2f}, Tilt: {head_tilt}, Body: {(body_tilt_left, body_tilt_right)}, Up: {looking_up}")
                print(log_msg)
                try:
                    with open("cheating_log.txt", "a") as f:
                        f.write(log_msg + "\n")
                except IOError as e:
                    print(f"Error writing to log file: {e}")
                event = {
                    "timestamp": current_time,
                    "person_id": person_id,
                    "type": "behavior",
                    "score": int(score),
                    "duration": float(duration),
                    "eye_ratio": float(eye_ratio),
                    "head_tilt": int(head_tilt),
                    "body_tilt_left": bool(body_tilt_left),
                    "body_tilt_right": bool(body_tilt_right),
                    "looking_up": bool(looking_up)
                }
                self.events.append(event)
                try:
                    with open(self.json_log_file, "a") as jf:
                        jf.write(json.dumps(event) + "\n")
                except IOError as e:
                    print(f"Error writing JSON log: {e}")
                self.last_cheating_log_time[person_id] = current_time

                # --------- ML DATASET LOGGING ---------
                try:
                    smoothed_score = score
                    look_away_counter = self.look_away_counters[person_id]
                    body_tilt = int(body_tilt_left or body_tilt_right)

                    # Score-based labeling
                    label_value = 1 if score >= 60 else 0

                    row = f"{eye_ratio},{head_tilt},{body_tilt},{int(looking_up)},{phone_detected_flag},{look_away_counter},{smoothed_score},{int(hand_on_table)},{label_value},{duration}\n"

                    # Create file with header if it doesn't exist
                    if not os.path.exists(self.dataset_file):
                        with open(self.dataset_file, "w") as df:
                            df.write("eye_ratio,head_tilt,body_tilt,looking_up,phone_detected,look_away_counter,smoothed_score,hand_on_table,label,duration\n")

                    # Append new row
                    with open(self.dataset_file, "a") as df:
                        df.write(row)

                except Exception as e:
                    print(f"Error writing dataset: {e}")

            # Get bounding box and draw it
            x_min, y_min, x_max, y_max = self.get_bounding_box(pose_landmarks, w, h)
            if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box

            # Score-based status
            if score < 30:
                status_text = f"SAFE ({score})"
                color = (0, 255, 0)
            elif score < 60:
                status_text = f"SUSPICIOUS ({score})"
                color = (0, 255, 255)
            else:
                status_text = f"CHEATING ({score})"
                color = (0, 0, 255)

            # -------- ML Prediction --------
            ml_confidence = 0.0
            if self.ml_model is not None:
                try:
                    features = np.array([[
                        eye_ratio,
                        head_tilt,
                        int(body_tilt_left or body_tilt_right),
                        int(looking_up),
                        phone_detected_flag,
                        int(hand_on_table),
                        duration
                    ]])

                    ml_prediction = self.ml_model.predict(features)[0]
                    ml_confidence = self.ml_model.predict_proba(features)[0][1]

                    # Hybrid decision
                    if score >= 60 or ml_prediction == 1:
                        cheating = True
                    else:
                        cheating = False

                except Exception as e:
                    print("ML error:", e)

            status_text = f"{status_text} | ML: {ml_confidence:.2f}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            return (cheating, score, left_eye, right_eye, nose, hand_on_table, 
                    body_tilt_left, body_tilt_right, looking_up)

        except Exception as e:
            print(f"Error in detect_cheating for Person {person_id}: {e}")
            return (False, 0, (0, 0), (0, 0), (0, 0), False, False, False, False)

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera feed ended.")
                    break

                self.frame_count += 1
                if self.frame_count % FRAME_SKIP != 0:  # Skip frames for performance
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_mesh.process(rgb_frame)
                hands_results = self.hands.process(rgb_frame)
                pose_results = self.pose.process(rgb_frame)

                if face_results.multi_face_landmarks:
                    for person_id, face_landmarks in enumerate(face_results.multi_face_landmarks):
                        hands_landmarks = hands_results.multi_hand_landmarks if hands_results.multi_hand_landmarks else []
                        pose_landmarks = pose_results.pose_landmarks if pose_results.pose_landmarks else None
                        cheating, score, left_eye, right_eye, nose, hand_on_table, body_tilt_left, body_tilt_right, looking_up = self.detect_cheating(
                            face_landmarks, pose_landmarks, frame, person_id, hands_landmarks
                        )

                        # Draw landmarks (optional, can remove if not needed)
                        cv2.circle(frame, left_eye, 5, (255, 0, 0), -1)
                        cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)
                        cv2.circle(frame, nose, 5, (0, 0, 255), -1)

                else:
                    cv2.putText(frame, "No faces detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                self.latest_frame = frame.copy()

                cv2.imshow('Cheating Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Stopped by user.")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        # Generate summary report
        try:
            report = {
                "total_events": len(self.events),
                "total_cheating_events": sum(1 for e in self.events if e.get("type") in ["phone_detected", "behavior"] and e.get("score", 0) >= 50),
                "avg_score": float(np.mean([e.get("score", 0) for e in self.events if "score" in e]) or 0),
                "max_score": int(max([e.get("score", 0) for e in self.events if "score" in e], default=0)),
                "duration_seconds": float(sum([e.get("duration", 0) for e in self.events if "duration" in e]))
            }
            with open("report.json", "w") as rf:
                json.dump(report, rf, indent=4)
            print("Report saved to report.json")
        except Exception as e:
            print(f"Error generating report: {e}")
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()

if __name__ == "__main__":
    try:
        detector = CheatingDetector()
        detector.run()
    except RuntimeError as e:
        print(e)
        sys.exit(1)
