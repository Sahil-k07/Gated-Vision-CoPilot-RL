import os
import cv2
import csv
import math
import urllib.request
import numpy as np
from tqdm import tqdm

DATASET_DIR = os.path.join("dataset", "driving_sequence")
FRAMES_DIR = os.path.join(DATASET_DIR, "frames")
TELEMETRY_FILE = os.path.join(DATASET_DIR, "telemetry.csv")
SAMPLE_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
VIDEO_PATH = os.path.join("dataset", "sample_driving.mp4")


def download_or_generate_dataset(num_frames=600):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    video_loaded = False

    if not os.path.exists(VIDEO_PATH):
        print("[Dataset Setup] Attempting to download real driving video...")
        try:
            urllib.request.urlretrieve(SAMPLE_VIDEO_URL, VIDEO_PATH)
            print("[Dataset Setup] Download complete.")
        except Exception as e:
            print(f"[Dataset Setup] Network download skipped ({e}). Building synthetic RL driving dataset...")

    cap = cv2.VideoCapture(VIDEO_PATH) if os.path.exists(VIDEO_PATH) else None
    if cap and cap.isOpened():
        video_loaded = True

    print(f"[Dataset Setup] Preparing {num_frames} frames and synchronized vehicle telemetry...")
    with open(TELEMETRY_FILE, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_id", "speed_kmh", "steering_angle_deg",
            "acceleration_ms2", "gt_obstacle_dist_m", "hazard_event"
        ])

        lead_vehicle_dist = 45.0
        ego_speed = 65.0

        for idx in tqdm(range(num_frames), desc="Building Dataset"):
            # Simulate realistic highway kinematics & sudden braking hazards
            hazard = 1 if (120 <= idx <= 160 or 350 <= idx <= 400) else 0
            if hazard:
                lead_vehicle_dist = max(8.0, lead_vehicle_dist - 1.2)
                ego_speed = max(25.0, ego_speed - 1.5)
                accel = -3.5 + np.random.normal(0, 0.2)
            else:
                lead_vehicle_dist = min(60.0, lead_vehicle_dist + 0.5 + np.sin(idx * 0.05))
                ego_speed = np.clip(65.0 + 10.0 * np.sin(idx * 0.03), 40.0, 90.0)
                accel = np.cos(idx * 0.03) * 0.8 + np.random.normal(0, 0.1)

            steering = 8.0 * math.sin(idx * 0.04) + np.random.normal(0, 0.5)

            # Frame extraction or synthesis
            frame = None
            if video_loaded:
                ret, raw_frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap.read()
                if ret:
                    frame = cv2.resize(raw_frame, (320, 180))

            if frame is None:
                frame = _generate_synthetic_road_frame(idx, lead_vehicle_dist, steering, hazard)

            frame_path = os.path.join(FRAMES_DIR, f"frame_{idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)

            writer.writerow([
                idx,
                round(float(ego_speed), 2),
                round(float(steering), 2),
                round(float(accel), 2),
                round(float(lead_vehicle_dist), 2),
                hazard
            ])

    if cap:
        cap.release()
    print(f"[Dataset Setup] Dataset ready at '{DATASET_DIR}/'")


def _generate_synthetic_road_frame(idx, dist, steering, hazard):
    """Generates a 320x180 driving view with moving lane markers and a lead vehicle."""
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    # Sky and road
    img[:75, :] = (180, 130, 70)
    img[75:, :] = (50, 50, 50)

    # Horizon vanishing point shifts with steering
    vp_x = int(160 + steering * 2.5)
    cv2.line(img, (vp_x - 20, 75), (30, 180), (220, 220, 220), 2)
    cv2.line(img, (vp_x + 20, 75), (290, 180), (220, 220, 220), 2)

    # Lead vehicle bounding box scaled by distance
    scale = np.clip(45.0 / max(dist, 5.0), 0.3, 2.5)
    w, h = int(36 * scale), int(26 * scale)
    cx = int(vp_x + math.sin(idx * 0.05) * 15)
    cy = int(85 + scale * 28)

    color = (0, 0, 255) if hazard else (200, 180, 40)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, -1)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (255, 255, 255), 1)
    return img


if __name__ == "__main__":
    download_or_generate_dataset()