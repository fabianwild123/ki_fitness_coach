import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import pandas as pd
import os

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


def resample_and_flatten(frame_buffer):
    frames_array = np.array(frame_buffer)
    old_indices = np.linspace(0, len(frames_array) - 1, len(frames_array))
    new_indices = np.linspace(0, len(frames_array) - 1, 30)

    # 4 Werte pro Frame (l_shoulder, r_shoulder, l_hip, r_hip)
    resampled_data = np.zeros((30, 4))
    for i in range(4):
        resampled_data[:, i] = np.interp(new_indices, old_indices, frames_array[:, i])

    return resampled_data.flatten()


def main():
    print("Starte Data Collector (Beide Körperhälften)...")
    cap = cv2.VideoCapture(0)

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    state = "DOWN"
    consecutive_frames = 0
    REQUIRED_FRAMES = 3

    ema_l_shoulder = EMAFilter(alpha=0.4)
    ema_r_shoulder = EMAFilter(alpha=0.4)
    ema_l_hip = EMAFilter(alpha=0.4)
    ema_r_hip = EMAFilter(alpha=0.4)

    is_recording = False
    rep_buffer = []
    dataset = []

    start_time = time.time()
    last_timestamp_ms = -1
    window_name = 'Data Collector - Aufnahme'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
            except Exception:
                continue

            annotated_frame = frame.copy()

            if detection_result and detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]

                # Linke Seite
                l_shoulder = [landmarks[11].x, landmarks[11].y]
                l_hip = [landmarks[23].x, landmarks[23].y]
                l_wrist = [landmarks[15].x, landmarks[15].y]
                l_knee = [landmarks[25].x, landmarks[25].y]

                # Rechte Seite
                r_shoulder = [landmarks[12].x, landmarks[12].y]
                r_hip = [landmarks[24].x, landmarks[24].y]
                r_wrist = [landmarks[16].x, landmarks[16].y]
                r_knee = [landmarks[26].x, landmarks[26].y]

                # Winkel berechnen
                l_shoulder_angle = ema_l_shoulder.update(calculate_angle(l_hip, l_shoulder, l_wrist))
                r_shoulder_angle = ema_r_shoulder.update(calculate_angle(r_hip, r_shoulder, r_wrist))
                l_hip_angle = ema_l_hip.update(calculate_angle([l_hip[0], l_hip[1] + 1.0], l_hip, l_knee))
                r_hip_angle = ema_r_hip.update(calculate_angle([r_hip[0], r_hip[1] + 1.0], r_hip, r_knee))

                # Auslöser: Schaut auf den Arm, der am höchsten ist
                max_shoulder = max(l_shoulder_angle, r_shoulder_angle)

                if is_recording:
                    rep_buffer.append([l_shoulder_angle, r_shoulder_angle, l_hip_angle, r_hip_angle])

                # Hüfte wird ignoriert! Nur Arm löst aus, damit wir auch Bein-Fehler aufnehmen können
                if max_shoulder > 110:
                    if state == "DOWN":
                        consecutive_frames += 1
                        if consecutive_frames >= REQUIRED_FRAMES:
                            state = "UP"
                            consecutive_frames = 0
                            is_recording = True
                            rep_buffer = []
                    else:
                        consecutive_frames = 0

                # Beide Arme müssen wieder unten sein (max < 85)
                elif max_shoulder < 85:
                    if state == "UP":
                        consecutive_frames += 1
                        if consecutive_frames >= REQUIRED_FRAMES:
                            state = "DOWN"
                            consecutive_frames = 0
                            is_recording = False

                            if len(rep_buffer) >= 5:
                                overlay = annotated_frame.copy()
                                cv2.rectangle(overlay, (0, 0), (600, 200), (0, 0, 0), -1)
                                cv2.putText(overlay, 'SPRUNG ERFASST! WIE WAR ER?', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 255), 2)
                                cv2.putText(overlay, '[G] Good', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                                            2)
                                cv2.putText(overlay, '[A] Arm Fehler', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2)
                                cv2.putText(overlay, '[B] Bein Fehler', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2)
                                cv2.putText(overlay, '[X] Verwerfen', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (255, 255, 255), 2)

                                cv2.imshow(window_name, overlay)

                                label = None
                                while True:
                                    key = cv2.waitKey(0) & 0xFF
                                    if key == ord('g'):
                                        label = 'Good';
                                        break
                                    elif key == ord('a'):
                                        label = 'Arm_Error';
                                        break
                                    elif key == ord('b'):
                                        label = 'Leg_Error';
                                        break
                                    elif key == ord('x'):
                                        label = 'Verworfen';
                                        break

                                if label != 'Verworfen':
                                    flat_data = resample_and_flatten(rep_buffer)
                                    row = list(flat_data)
                                    row.append(label)
                                    dataset.append(row)
                                    print(f"Gespeichert als: {label} (In dieser Session: {len(dataset)})")
                                else:
                                    print("Sprung gelöscht.")

                            rep_buffer = []
                    else:
                        consecutive_frames = 0

            cv2.putText(annotated_frame, f'Status: {"AUFNAHME" if is_recording else "Bereit"}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_recording else (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Neue Sprunge: {len(dataset)}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    if len(dataset) > 0:
        print("\nSpeichere Dataset...")
        columns = []
        for i in range(30):
            columns.extend([f'l_shoulder_{i}', f'r_shoulder_{i}', f'l_hip_{i}', f'r_hip_{i}'])
        columns.append('error_type')

        df = pd.DataFrame(dataset, columns=columns)
        filename = 'neues_training_dataset.csv'

        file_exists = os.path.isfile(filename)
        df.to_csv(filename, mode='a', index=False, header=not file_exists)

        print(f"ERFOLG! {len(dataset)} neue Sprünge wurden an '{filename}' angehängt.")
    else:
        print("Nichts aufgezeichnet.")


if __name__ == '__main__':
    main()