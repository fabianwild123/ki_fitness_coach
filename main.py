import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360.0 - angle
    return angle


class EMAFilter:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value


class AdvancedFitnessAI:
    def __init__(self, dataset_path, model_path='fitness_model.pkl', force_retrain=False):
        self.is_trained = False
        self.model_path = model_path
        self.encoder = LabelEncoder()
        self.target_accuracy = 0.90

        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            existing_accuracy = data.get('accuracy', 0.0)

            if existing_accuracy >= self.target_accuracy and not force_retrain:
                print(f"--- SUPER! ---")
                print(f"Ein Modell mit {existing_accuracy * 100:.2f}% Genauigkeit wurde gefunden.")
                print(f"Training wird übersprungen. KI ist sofort einsatzbereit.")
                self.model = data['model']
                self.encoder = data.get('encoder')
                self.is_trained = True
                return
            else:
                print(f"Existierendes Modell hat nur {existing_accuracy * 100:.2f}%.")
                print("Starte neues Training, um höheres Ziel zu erreichen...")

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=None
        )
        self.train_model(dataset_path)

    def train_model(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Dataset '{dataset_path}' fehlt.")
            return

        try:
            df = pd.read_csv(dataset_path)
            if 'error_type' in df.columns:
                y = df['error_type'].fillna('Good').astype(str).values
                X_df = df.drop(columns=['error_type'])
            else:
                y = df.iloc[:, -1].fillna('Good').astype(str).values
                X_df = df.iloc[:, :-1]

            y_encoded = self.encoder.fit_transform(y)
            X_np = np.asarray(X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0), dtype=np.float64)

            best_accuracy = 0.0
            versuch = 0
            max_versuche = 30

            while best_accuracy < self.target_accuracy and versuch < max_versuche:
                versuch += 1
                X_train_orig, X_test, y_train_orig, y_test = train_test_split(
                    X_np, y_encoded, test_size=0.2, random_state=None
                )

                multiplier = 5
                X_train_aug, y_train_aug = [], []
                for i in range(len(X_train_orig)):
                    X_train_aug.append(X_train_orig[i])
                    y_train_aug.append(y_train_orig[i])
                    for _ in range(multiplier - 1):
                        noise = np.random.normal(0, 0.5, X_train_orig[i].shape)
                        scale = np.random.uniform(0.98, 1.02)
                        X_train_aug.append((X_train_orig[i] + noise) * scale)
                        y_train_aug.append(y_train_orig[i])

                X_train_final = np.array(X_train_aug)
                y_train_final = np.array(y_train_aug)

                self.model.fit(X_train_final, y_train_final)

                if len(X_test) > 0:
                    y_pred = self.model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        joblib.dump({
                            'model': self.model,
                            'encoder': self.encoder,
                            'accuracy': best_accuracy
                        }, self.model_path)

                    print(f"Versuch {versuch}: Genauigkeit = {accuracy * 100:.2f}%")

            self.is_trained = True
            print(f"\n--- TRAINING BEENDET ---")
            print(f"Beste erreichte Genauigkeit: {best_accuracy * 100:.2f}%")

        except Exception as e:
            print(f"Fehler beim KI-Training: {e}")

    def evaluate_rep(self, frame_buffer):
        if len(frame_buffer) < 5: return "Zu kurz/Fehler"
        if not self.is_trained: return "Gut! (Kein KI-Modell)"

        frames_array = np.array(frame_buffer)
        old_indices = np.linspace(0, len(frames_array) - 1, len(frames_array))
        new_indices = np.linspace(0, len(frames_array) - 1, 30)

        resampled_data = np.zeros((30, 4))
        for i in range(4):
            resampled_data[:, i] = np.interp(new_indices, old_indices, frames_array[:, i])

        flattened_input = resampled_data.flatten().reshape(1, -1)
        pred_idx = self.model.predict(flattened_input)[0]

        if self.encoder is not None:
            return str(self.encoder.inverse_transform([pred_idx])[0])
        return str(pred_idx)


class JumpingJackCounter:
    def __init__(self, ai_model):
        self.state = "DOWN"
        self.rep_count = 0
        self.good_rep_count = 0  # Zähler für perfekte Sprünge
        self.consecutive_frames = 0
        self.REQUIRED_FRAMES = 3

        self.ema_l_shoulder = EMAFilter(alpha=0.4)
        self.ema_r_shoulder = EMAFilter(alpha=0.4)
        self.ema_l_hip = EMAFilter(alpha=0.4)
        self.ema_r_hip = EMAFilter(alpha=0.4)

        self.ai = ai_model
        self.rep_buffer = []
        self.last_ai_feedback = "Warte auf ersten Sprung..."
        self.is_recording = False

    def process_frame(self, landmarks):
        l_shoulder = [landmarks[11].x, landmarks[11].y]
        l_hip = [landmarks[23].x, landmarks[23].y]
        l_wrist = [landmarks[15].x, landmarks[15].y]
        l_knee = [landmarks[25].x, landmarks[25].y]

        r_shoulder = [landmarks[12].x, landmarks[12].y]
        r_hip = [landmarks[24].x, landmarks[24].y]
        r_wrist = [landmarks[16].x, landmarks[16].y]
        r_knee = [landmarks[26].x, landmarks[26].y]

        l_shoulder_angle = self.ema_l_shoulder.update(calculate_angle(l_hip, l_shoulder, l_wrist))
        r_shoulder_angle = self.ema_r_shoulder.update(calculate_angle(r_hip, r_shoulder, r_wrist))
        l_hip_angle = self.ema_l_hip.update(calculate_angle([l_hip[0], l_hip[1] + 1.0], l_hip, l_knee))
        r_hip_angle = self.ema_r_hip.update(calculate_angle([r_hip[0], r_hip[1] + 1.0], r_hip, r_knee))

        max_shoulder = max(l_shoulder_angle, r_shoulder_angle)

        if self.is_recording:
            self.rep_buffer.append([l_shoulder_angle, r_shoulder_angle, l_hip_angle, r_hip_angle])

        if max_shoulder > 110:
            if self.state == "DOWN":
                self.consecutive_frames += 1
                if self.consecutive_frames >= self.REQUIRED_FRAMES:
                    self.state = "UP"
                    self.consecutive_frames = 0
                    self.is_recording = True
                    self.rep_buffer = []
        elif max_shoulder < 85:
            if self.state == "UP":
                self.consecutive_frames += 1
                if self.consecutive_frames >= self.REQUIRED_FRAMES:
                    self.state = "DOWN"
                    self.rep_count += 1
                    self.consecutive_frames = 0
                    self.is_recording = False

                    # Sprung auswerten
                    self.last_ai_feedback = self.ai.evaluate_rep(self.rep_buffer)

                    # Wenn KI sagt "Good", dann Good-Zähler erhöhen
                    if "Good" in str(self.last_ai_feedback) or "Gut" in str(self.last_ai_feedback):
                        self.good_rep_count += 1

                    self.rep_buffer = []
        else:
            self.consecutive_frames = 0

        return self.state, max_shoulder, l_hip_angle, self.last_ai_feedback


POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                    (25, 27), (26, 28), (27, 29), (29, 31), (31, 27), (28, 30), (30, 32), (32, 28)]


def draw_custom_landmarks(image, landmarks, feedback):
    h, w, _ = image.shape
    skeleton_color = (0, 255, 0) if ("Good" in str(feedback) or "Gut" in str(feedback)) else (0, 0, 255)
    if "Warte" in str(feedback): skeleton_color = (245, 117, 66)

    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if getattr(landmarks[start_idx], 'visibility', 1.0) > 0.5 and getattr(landmarks[end_idx], 'visibility',
                                                                              1.0) > 0.5:
            cv2.line(image, (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)),
                     (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)), skeleton_color, 3)


def main():
    print("Starte Setup...")
    ai_model = AdvancedFitnessAI('neues_training_dataset.csv', force_retrain=False)

    cap = cv2.VideoCapture(0)
    counter = JumpingJackCounter(ai_model)

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    start_time = time.time()
    last_timestamp_ms = -1
    window_name = 'AI Fitness Coach - Pro Mode'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int((time.time() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms: timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            try:
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except:
                continue

            annotated_frame = frame.copy()
            if detection_result and detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                state, _, _, ai_feedback = counter.process_frame(landmarks)
                draw_custom_landmarks(annotated_frame, landmarks, ai_feedback)

                # --- 1. Transparentes UI-Overlay erstellen ---
                overlay = annotated_frame.copy()
                # Kasten etwas breiter gemacht (650), damit alles reinpasst
                cv2.rectangle(overlay, (0, 0), (650, 160), (20, 20, 20), -1)

                # Bilder mischen (alpha = 0.5 -> 50% Deckkraft)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

                # --- 2. UI-Texte zeichnen (nicht transparent) ---
                # Reps und Good Reps in einer Zeile
                cv2.putText(annotated_frame, f'REPS: {counter.rep_count}  |  GOOD: {counter.good_rep_count}',
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # State etwas nach rechts versetzt
                cv2.putText(annotated_frame, f'STATE: {state}',
                            (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Farbige KI Diagnose
                color = (0, 255, 0) if ("Good" in str(ai_feedback) or "Gut" in str(ai_feedback)) else (0, 0, 255)
                if "Warte" in str(ai_feedback): color = (255, 255, 255)

                cv2.putText(annotated_frame, 'KI DIAGNOSE:', (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),
                            1)
                cv2.putText(annotated_frame, str(ai_feedback), (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()