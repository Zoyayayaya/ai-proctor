import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import time
import sys
import os
import torch

# Constants
EYE_LEFT_THRESHOLD = 0.3    # Fixed value for looking left
EYE_RIGHT_THRESHOLD = 0.7   # Fixed value for looking right
HEAD_TILT_THRESHOLD = 10    # Pixels for head tilt detection
TABLE_REGION_THRESHOLD = 0.7  # Hands in lower 30% of frame
SHOULDER_TILT_THRESHOLD = 0.3  # Threshold for body tilt (used as a base, but adjusted by diff)
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
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera. Check connection or camera index.")

        # Load YOLO model for phone detection
        yolo_model_path = 'E:\Turin\ML\Project_ml\yolov5s.pt'  # Update this path
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)
        self.phone_class_id = 67  # Class ID for cell phone in COCO dataset

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

            # Cheating logic
            cheating = False
            eye_ratio = eye_center_x / w
            if (eye_ratio < EYE_LEFT_THRESHOLD or 
                eye_ratio > EYE_RIGHT_THRESHOLD or 
                head_tilt > HEAD_TILT_THRESHOLD):
                self.look_away_counters[person_id] += 1
                if (self.look_away_counters[person_id] >= LOOK_AWAY_THRESHOLD and 
                    not self.cheating_start_time[person_id]):
                    self.cheating_start_time[person_id] = time.time()
                cheating = True
            elif body_tilt_left or body_tilt_right or looking_up:
                cheating = True
            else:
                self.look_away_counters[person_id] = max(0, self.look_away_counters[person_id] - 1)

            # YOLO phone detection
            results = self.yolo_model(frame)
            for det in results.pred[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if int(cls) == self.phone_class_id and conf > 0.1:  # Class ID for cell phone and confidence threshold
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # Get label and confidence
                    label = self.yolo_model.names[int(cls)]
                    text = f"{label} {conf:.2f}"
                    # Put label and confidence on the frame
                    cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cheating = True
                    log_msg = f"{time.time()}: Person {person_id} - Phone detected with confidence {conf:.2f}"
                    print(log_msg)
                    try:
                        with open("cheating_log.txt", "a") as f:
                            f.write(log_msg + "\n")
                    except IOError as e:
                        print(f"Error writing to log file: {e}")

            # Controlled logging
            current_time = time.time()
            if cheating and current_time - self.last_cheating_log_time[person_id] >= LOG_COOLDOWN:
                log_msg = (f"{current_time}: Person {person_id} - Eye: {eye_ratio:.2f}, "
                           f"Tilt: {head_tilt}, Body: {(body_tilt_left, body_tilt_right)}, Up: {looking_up}")
                print(log_msg)
                try:
                    with open("cheating_log.txt", "a") as f:
                        f.write(log_msg + "\n")
                except IOError as e:
                    print(f"Error writing to log file: {e}")
                self.last_cheating_log_time[person_id] = current_time

            # Get bounding box and draw it
            x_min, y_min, x_max, y_max = self.get_bounding_box(pose_landmarks, w, h)
            if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box

            # Simplified status text
            status_text = "NOT CHEATING" if not cheating else "CHEATING"
            color = (0, 255, 0) if not cheating else (0, 0, 255)  # Green for not cheating, red for cheating
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            return (cheating, left_eye, right_eye, nose, hand_on_table, 
                    body_tilt_left, body_tilt_right, looking_up)

        except Exception as e:
            print(f"Error in detect_cheating for Person {person_id}: {e}")
            return (False, (0, 0), (0, 0), (0, 0), False, False, False, False)

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
                        cheating, left_eye, right_eye, nose, hand_on_table, body_tilt_left, body_tilt_right, looking_up = self.detect_cheating(
                            face_landmarks, pose_landmarks, frame, person_id, hands_landmarks
                        )

                        # Draw landmarks (optional, can remove if not needed)
                        cv2.circle(frame, left_eye, 5, (255, 0, 0), -1)
                        cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)
                        cv2.circle(frame, nose, 5, (0, 0, 255), -1)

                else:
                    cv2.putText(frame, "No faces detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
